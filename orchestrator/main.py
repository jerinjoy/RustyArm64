"""
LangGraph orchestrator for incremental ARM64 Rust simulator development.

Virtual workspace pattern — each node mutates a Dict[str,str] filesystem so the
graph never ships a monolithic string blob through LLM context windows.

Run from the orchestrator/ directory:
    uv run python main.py -s deepseek --spec-dir /path/to/specs "Build an ARM64 decoder"
    uv run python main.py --thread-id my-run --spec-dir /path/to/specs
    uv run python main.py --reset "Focus on integer ALU instructions"

Model defaults to deepseek-v4-pro.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict
import operator

# --- optional dotenv (safe if not installed) ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field


# ============================================================================
# Model configuration
# ============================================================================

SERVICE_DEFAULTS: dict[str, dict] = {
    "deepseek": {
        "model": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com",
        "cls": ChatDeepSeek,
        "env_key": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "model": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "cls": ChatOpenAI,
        "env_key": "OPENAI_API_KEY",
    },
}


@dataclass
class LLMConfig:
    """Holds the resolved model, API key, and base URL for one service."""
    model: str
    api_key: str
    base_url: str
    service: str

    def create_chat_model(self):
        if self.service == "deepseek":
            return ChatDeepSeek(
                model=self.model,
                api_key=self.api_key,
                temperature=0.2,
                extra_body={"thinking": {"type": "disabled"}},
            )
        return ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.2,
        )

    def with_structured_output(self, schema_cls: type[BaseModel], **kwargs):
        """Return a model+parser chain compatible with this service."""
        llm = self.create_chat_model()
        method = "json_mode" if self.service == "deepseek" else "json_schema"
        return llm.with_structured_output(schema_cls, method=method, **kwargs)


# Mutable module-level config populated by CLI at startup.
_llm: LLMConfig | None = None
_spec_root: Path | None = None


def configure_llm(service: str, api_key: str, model: str | None = None) -> LLMConfig:
    global _llm
    svc = SERVICE_DEFAULTS[service]
    resolved_model = model or svc["model"]
    _llm = LLMConfig(
        model=resolved_model,
        api_key=api_key,
        base_url=svc["base_url"],
        service=service,
    )
    return _llm


def configure_specs(spec_dir: str) -> None:
    global _spec_root
    _spec_root = Path(spec_dir).expanduser().resolve()


def _get_llm() -> LLMConfig:
    if _llm is None:
        raise RuntimeError("LLM not configured. Call configure_llm() first.")
    return _llm


# ============================================================================
# Pydantic output models for structured LLM responses
# ============================================================================

class ArchitectOutput(BaseModel):
    plan: List[str] = Field(description="Complete list of remaining implementation steps")
    current_step: str = Field(description="Single step to execute next (full text, not index)")
    need_spec: bool = Field(description="Whether spec data is needed before implementation")
    target_file: str = Field(description="Which workspace file this step modifies (e.g. src/lib.rs)")


class SpecReaderOutput(BaseModel):
    instruction_id: str = Field(default="")
    mnemonic: str = Field(default="")
    summary: str = Field(default="")
    bitfields: List[dict] = Field(default_factory=list)
    decode_pseudocode: str = Field(default="")
    execute_pseudocode: str = Field(default="")
    constraints: List[str] = Field(default_factory=list)


# ============================================================================
# Virtual workspace merge reducer
# ============================================================================

def _merge_workspace(current: dict, update: dict) -> dict:
    """Accumulate file updates — source of truth for the virtual filesystem."""
    return {**current, **update}


# ============================================================================
# Graph state
# ============================================================================

class SimulatorState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    progress_state: str                      # json-serialised progress.yaml contents
    spec_context: str                        # extracted MRA spec context
    current_plan: str                        # architect's current single-step task
    workspace: Annotated[Dict[str, str], _merge_workspace]  # virtual filesystem
    current_target_file: str                 # which file coder/test_writer should mutate
    test_feedback: str                       # cargo test output / error details
    failure_source: str                      # "logic" | "tests" | ""
    human_feedback: str                      # user feedback from HIL gates
    approved: bool                           # whether last HIL gate approved
    startup_directive: str                   # user prompt; consumed by architect on first run
    retry_count: int                         # consecutive test failures on same step


# ============================================================================
# Helpers
# ============================================================================

def extract_markdown_code(text: str, lang: str = "rust") -> str:
    """Pull code out of ```lang ... ``` fences. Falls back to any fence, then raw."""
    pattern = rf'```{lang}\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return "\n\n".join(m.strip("\n") for m in matches)
    # Fallback: any code fence
    pattern = r'```(?:\w*)\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(m.strip("\n") for m in matches)
    return text.strip()


# --- spec index helpers ---

def _resolve_version_dir(parent: Path, pattern: str) -> Path | None:
    """Find newest versioned directory inside *parent* matching *pattern*."""
    candidates = sorted(
        p for p in parent.glob(pattern)
        if p.is_dir() and not p.name.endswith("_diff") and "diff" not in p.stem
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        print(
            f"  [spec] Multiple version dirs, picking newest: "
            f"{candidates[-1].name}",
            file=sys.stderr,
        )
    return candidates[-1]


def _load_xml_spec_for_task(current_plan: str) -> str:
    """Load the most relevant ARM64 MRA XML for the current step."""
    if _spec_root is None or not _spec_root.exists():
        return ""

    orch_dir = Path(__file__).resolve().parent
    isa_dir = _resolve_version_dir(_spec_root / "ISA_A64", "ISA_A64_xml_A_profile-*")
    sysreg_dir = _resolve_version_dir(_spec_root / "SysReg", "SysReg_xml_A_profile-*")

    plan_upper = current_plan.upper()
    plan_words = set(current_plan.replace("-", " ").replace("_", " ").split())

    # Instruction index lookup
    if isa_dir:
        idx_path = orch_dir / "instruction_index.json"
        if idx_path.exists():
            try:
                idx_data = json.loads(idx_path.read_text())
                entries = idx_data.get("instructions", {})
                candidates: list[tuple[int, str]] = []
                for instr_id, rec in entries.items():
                    mnem = rec.get("mnemonic", "").upper()
                    if mnem and mnem in plan_upper:
                        candidates.append((0, instr_id))
                    if any(w.lower() in instr_id.lower() for w in plan_words):
                        candidates.append((1, instr_id))
                candidates.sort()
                if candidates:
                    xml_path = isa_dir / entries[candidates[0][1]]["file"]
                    if xml_path.exists():
                        return xml_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass

    # SysReg index lookup
    if sysreg_dir:
        idx_path = orch_dir / "sysreg_index.json"
        if idx_path.exists():
            try:
                idx_data = json.loads(idx_path.read_text())
                for reg_name, rec in idx_data.get("registers", {}).items():
                    if reg_name.upper() in plan_upper or reg_name.upper() in plan_words:
                        xml_path = sysreg_dir / rec["file"]
                        if xml_path.exists():
                            return xml_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass

    # Fallback: scan XML files by name
    for xml_dir in (d for d in (isa_dir, sysreg_dir) if d):
        for xml_file in sorted(xml_dir.glob("*.xml")):
            if any(w.lower() in xml_file.stem.lower() for w in plan_words):
                return xml_file.read_text(encoding="utf-8", errors="replace")

    return ""


def _save_progress_yaml(completed: list[str], pending: list[str], current: str) -> None:
    """Write progress.yaml to the repo root."""
    import yaml

    repo_root = Path(__file__).resolve().parent.parent
    data = {
        "completed_steps": completed,
        "pending_plan": pending,
        "current_step": current,
    }
    (repo_root / "progress.yaml").write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _load_progress_yaml() -> dict:
    """Read progress.yaml from repo root."""
    import yaml

    path = Path(__file__).resolve().parent.parent / "progress.yaml"
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return {
            "completed_steps": data.get("completed_steps", []),
            "pending_plan": data.get("pending_plan", []),
            "current_step": data.get("current_step", ""),
        }
    except Exception:
        return {}


# ============================================================================
# Nodes
# ============================================================================

# --- Architect ---

_ARCHITECT_SYSTEM = """You are the architect for a Rust ARM64 CPU simulator MVP.
Manage the implementation plan. Plan steps that build a minimal but functional
simulator capable of:

- Parsing and decoding a subset of ARM64 instructions
- Modelling CPU registers (X0–X30, SP, PC, NZCV flags)
- Executing basic data-processing instructions (ADD, SUB, MOV, AND, ORR, EOR…)
- Handling memory load/store operations (LDR, STR)
- Running a simple test program end-to-end

The simulator lives in a virtual workspace (Dict[str,str]). Each step targets
exactly one file — set target_file to its path.  When a new submodule file
(e.g. src/cpu.rs) appears in the plan, add a preceding step that inserts the
module declaration (e.g. pub mod cpu;) into src/lib.rs — the Rust compiler
needs it to discover the file during cargo test.

Output JSON. Return: plan (list of remaining step strings), current_step (the
single next step to execute), need_spec (bool), target_file (string path)."""


def architect_node(state: SimulatorState) -> dict:
    """Create or revise the implementation plan."""
    llm = _get_llm()
    model = llm.with_structured_output(ArchitectOutput)

    progress_raw = state.get("progress_state", "")
    human_fb = state.get("human_feedback", "")
    test_fb = state.get("test_feedback", "")
    failure_src = state.get("failure_source", "")
    spec_ctx = state.get("spec_context", "")
    startup = state.get("startup_directive", "")
    retry = state.get("retry_count", 0)

    # Hydrate progress from disk on cold start
    progress: dict = {}
    if progress_raw:
        try:
            progress = json.loads(progress_raw)
        except (json.JSONDecodeError, TypeError):
            pass
    if not progress:
        disk_progress = _load_progress_yaml()
        if disk_progress:
            progress = disk_progress

    completed = progress.get("completed_steps", [])
    plan = progress.get("pending_plan", [])

    directive_section = ""
    if startup:
        directive_section = (
            "Startup directive from user (MUST follow this guidance for ALL steps):\n"
            f"  {startup}\n\n"
        )

    prompt = f"""{directive_section}CURRENT STATE
  Completed steps:       {json.dumps(completed)}
  Pending plan:          {json.dumps(plan)}
  Consecutive retries:   {retry} (max 3 before architect intervenes)
  Spec context:          {spec_ctx[:400] if spec_ctx else "none"}
  Human feedback:        {human_fb if human_fb else "none"}
  Test feedback:         {test_fb[:600] if test_fb else "none"}
  Failure source:        {failure_src if failure_src else "none"}

RULES (apply in order):
0. If startup_directive is present, it guides ALL steps (scope, style,
   constraints). Clear it after applying to this plan generation.
1. If human_feedback is present, interpret it literally and revise. NEVER
   return the same plan when feedback is present.
   - If feedback says something is already done/created/exists/have/→ REMOVE
     the current step from the plan and pick the next.
   - If feedback gives a specific instruction → change current_step to match.
   - If unsure what the feedback means, still change the plan (pick next step).
2. If retry_count >= 3 and test_feedback is present → the coder has failed
   repeatedly. CHANGE THE APPROACH: split the step into smaller sub-steps,
   simplify the task, or skip it entirely. Do NOT keep the same current_step.
3. If test_feedback is present and failure_source is "logic" (retries < 3):
   keep the same current_step (the implementation is being fixed).
4. If test_feedback is present and failure_source is "tests" (retries < 3):
   keep the same current_step (the tests are being fixed).
5. If the plan is empty and no feedback requires action, generate a fresh
   step-by-step plan following the startup directive.
6. Set need_spec=true if you need ARM64 instruction-set details for the
   current step. Set false if you already have spec data or it's not needed.
7. Set target_file to the workspace path this step mutates (e.g. src/lib.rs,
   src/cpu.rs).  When introducing a new file, the PREVIOUS step must insert
   its module declaration into src/lib.rs (target_file=src/lib.rs)."""

    response = model.invoke(_ARCHITECT_SYSTEM + "\n\n" + prompt)

    # Persist full plan in progress_state; current_plan is the step text
    progress_state = json.dumps({
        "completed_steps": completed,
        "pending_plan": response.plan,
        "current_step": response.current_step,
    })

    return {
        "progress_state": progress_state,
        "current_plan": response.current_step,
        "current_target_file": response.target_file,
        "spec_context": "",                      # reset; re-requested if needed
        "human_feedback": "",
        "test_feedback": "",
        "failure_source": "",
        "startup_directive": "",  # consumed
        "retry_count": 0,         # reset for new/fixed step
    }


# --- Spec Reader ---

_SPEC_READER_SYSTEM = """You are an ARM64 (AArch64) ISA expert. Parse the XML
specification and extract structured information for the Rust coder.

For each <box> element:
- If it has usename="1", it's a variable field (operand to decode).
- If it has settings="N", it's a constant field (opcode bits to verify).
- hibit="N" is the MSB of the slice; width="W" is the bit count.
- Extract fields in MSB-to-LSB order.

Decode pseudocode: inside <ps_section> with secttype="noheading" inside
<iclass>. Extract verbatim.

Execute pseudocode: inside <ps_section> with secttype="Operation" as a
direct child of <instructionsection>. Extract verbatim.

Return clean JSON."""


def spec_reader_node(state: SimulatorState) -> dict:
    """Load ARM64 spec XML and extract structured context."""
    llm = _get_llm()
    model = llm.with_structured_output(SpecReaderOutput)

    current_plan = state.get("current_plan", "")

    xml_text = _load_xml_spec_for_task(current_plan)
    if not xml_text:
        return {"spec_context": ""}

    prompt = f"""{_SPEC_READER_SYSTEM}

CURRENT TASK: {current_plan}

RAW ARM64 SPEC XML:
{xml_text[:12000]}"""

    try:
        response = model.invoke(prompt)
    except Exception:
        return {"spec_context": f"[Spec XML loaded, {len(xml_text)} bytes]"}

    # Build spec_context string
    parts = []
    if response.instruction_id:
        parts.append(f"Instruction: {response.instruction_id} ({response.mnemonic})")
    if response.summary:
        parts.append(f"Summary: {response.summary}")
    if response.bitfields:
        parts.append("\nBit-field encoding:")
        for bf in response.bitfields:
            kind = "variable" if bf.get("is_variable") else "constant"
            name = bf.get("name") or "(const)"
            val = bf.get("default_value", "")
            extra = f" = {val}" if val and not bf.get("is_variable") else ""
            desc = bf.get("description", "")
            desc_str = f"  // {desc}" if desc else ""
            parts.append(f"  [{bf.get('bits')}] {name} ({kind}){extra}{desc_str}")
    if response.decode_pseudocode:
        parts.append(f"\nDecode pseudocode:\n{response.decode_pseudocode}")
    if response.execute_pseudocode:
        parts.append(f"\nExecute pseudocode:\n{response.execute_pseudocode}")
    if response.constraints:
        parts.append("\nConstraints:")
        for c in response.constraints:
            parts.append(f"  - {c}")

    return {"spec_context": "\n".join(parts)}


# --- Coder ---

_CODER_SYSTEM = """You are a Rust systems programmer building an ARM64 CPU simulator.
Write production-quality, idiomatic Rust. Model ARM64 registers in a struct;
use enums for instruction types. Keep functions small and testable.

Output the COMPLETE file content inside a single ```rust code fence.
Include ALL imports, types, and implementations — output the full file.
Do NOT output JSON — output raw Rust code in a markdown code block."""


def coder_node(state: SimulatorState) -> dict:
    """Generate Rust implementation for the current step on a single target file."""
    llm = _get_llm()
    model = llm.create_chat_model()

    plan = state.get("current_plan", "")
    target = state.get("current_target_file", "src/lib.rs")
    workspace = state.get("workspace", {})
    spec_ctx = state.get("spec_context", "")
    test_fb = state.get("test_feedback", "")
    failure_src = state.get("failure_source", "")

    existing = workspace.get(target, "")

    debug_section = ""
    if test_fb and failure_src == "logic":
        errors_head = "\n".join(test_fb.splitlines()[:60])
        debug_section = (
            f"\nCOMPILER ERRORS (fix these — this is a retry):\n"
            f"{errors_head[:3000]}\n"
        )

    prompt = f"""CURRENT TASK: {plan}
TARGET FILE: {target}

ARM64 SPEC CONTEXT: {spec_ctx[:3000] if spec_ctx else "Use your ARM64 knowledge."}

EXISTING FILE CONTENT:
{existing[:5000] if existing else "(new file — create from scratch)"}
{debug_section}
Return the COMPLETE file inside a SINGLE ```rust code fence. Output NOTHING else."""

    messages = [
        SystemMessage(content=_CODER_SYSTEM),
        HumanMessage(content=prompt),
    ]
    response = model.invoke(messages)
    code = extract_markdown_code(response.content)
    return {"workspace": {target: code}}


# --- Test Writer ---

_TESTWRITER_SYSTEM = """You are a Rust test engineer. Write comprehensive inline
unit tests for ARM64 simulator code. Append your tests directly to the target file.

Output the COMPLETE file (all existing code PLUS new tests at the bottom)
inside a single ```rust code fence.  Add or update a #[cfg(test)] mod tests { ... }
block covering happy-path, edge cases, boundary values, and error paths.
Do NOT output JSON — output raw Rust code in a markdown code block."""


def test_writer_node(state: SimulatorState) -> dict:
    """Generate or revise inline tests on the same target file."""
    llm = _get_llm()
    model = llm.create_chat_model()

    plan = state.get("current_plan", "")
    target = state.get("current_target_file", "src/lib.rs")
    workspace = state.get("workspace", {})
    spec_ctx = state.get("spec_context", "")
    test_fb = state.get("test_feedback", "")
    failure_src = state.get("failure_source", "")

    existing = workspace.get(target, "")

    debug_section = ""
    if test_fb and failure_src == "tests":
        debug_section = f"\nTEST FAILURES (fix these):\n{test_fb[:2000]}\n"

    prompt = f"""CURRENT TASK: {plan}
TARGET FILE: {target}

ARM64 SPEC CONTEXT: {spec_ctx[:2000] if spec_ctx else ""}

EXISTING FILE CONTENT:
{existing[:5000]}
{debug_section}
Add or update #[cfg(test)] mod tests {{ ... }} at the bottom of this file.
Return the COMPLETE file (all existing code + new tests) inside a SINGLE ```rust code fence.
Output NOTHING else."""

    messages = [
        SystemMessage(content=_TESTWRITER_SYSTEM),
        HumanMessage(content=prompt),
    ]
    response = model.invoke(messages)
    code = extract_markdown_code(response.content)
    return {"workspace": {target: code}}


# --- Test Runner ---

def _determine_failure_source(cargo_output: str) -> str:
    """Heuristic: cargo error → logic, test failure → tests."""
    output_lower = cargo_output.lower()
    if "error[" in output_lower or "error:" in output_lower:
        return "logic"
    if "panicked" in output_lower or "assertion" in output_lower or "thread" in output_lower:
        return "tests"
    return "logic"


def test_runner_node(state: SimulatorState) -> dict:
    """Write the virtual workspace to a temp sandbox and run cargo test."""
    workspace = state.get("workspace", {})
    retry = state.get("retry_count", 0)

    if not workspace:
        return {"test_feedback": "", "failure_source": "", "retry_count": 0}

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmp = Path(tmpdir_str)
        src_dir = tmp / "src"
        src_dir.mkdir()

        # Write every workspace file into the sandbox
        for rel_path, content in workspace.items():
            file_path = tmp / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

        # Create Cargo.toml if the workspace doesn't supply one
        cargo_toml = tmp / "Cargo.toml"
        if not cargo_toml.exists():
            cargo_toml.write_text(
                '[package]\nname = "arm64_sim"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\n',
                encoding="utf-8",
            )

        # Run cargo test
        try:
            proc = subprocess.run(
                ["cargo", "test"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(tmp),
            )
            output = (proc.stdout + "\n" + proc.stderr).strip()
            if proc.returncode == 0:
                return {"test_feedback": output, "failure_source": "", "retry_count": 0}
            return {
                "test_feedback": output,
                "failure_source": _determine_failure_source(output),
                "retry_count": retry + 1,
            }
        except subprocess.TimeoutExpired:
            return {
                "test_feedback": "cargo test timed out after 120 seconds",
                "failure_source": "logic",
                "retry_count": retry + 1,
            }
        except FileNotFoundError:
            return {
                "test_feedback": "cargo not found — is the Rust toolchain installed?",
                "failure_source": "logic",
                "retry_count": retry + 1,
            }


# --- Committer ---

def committer_node(state: SimulatorState) -> dict:
    """Flush workspace to disk, update progress.yaml, and commit to git."""
    progress_raw = state.get("progress_state", "")
    current_plan = state.get("current_plan", "")
    workspace = state.get("workspace", {})
    repo_root = Path(__file__).resolve().parent.parent

    # Flush virtual workspace to the real repository (does NOT clear workspace)
    if workspace:
        sim_dir = repo_root / "simulator"
        for rel_path, content in workspace.items():
            file_path = sim_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
        # Also write Cargo.toml if the workspace provides one
        cargo_toml = sim_dir / "Cargo.toml"
        if not cargo_toml.exists():
            cargo_toml.write_text(
                '[package]\nname = "arm64_sim"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\n',
                encoding="utf-8",
            )

    # Parse progress
    completed: list[str] = []
    pending: list[str] = []
    current_step = ""
    if progress_raw:
        try:
            p = json.loads(progress_raw)
            completed = p.get("completed_steps", [])
            pending = p.get("pending_plan", [])
            current_step = p.get("current_step", "")
        except (json.JSONDecodeError, TypeError):
            pass

    if isinstance(current_plan, str):
        current_step = current_plan

    # Move completed step
    if current_step and current_step in pending:
        pending.remove(current_step)
    if current_step and current_step not in completed:
        completed.append(current_step)

    _save_progress_yaml(completed, pending, pending[0] if pending else "")

    # Git commit
    try:
        subprocess.run(
            ["git", "add", "-A", "simulator/", "progress.yaml"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"feat: {current_step}"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        pass  # non-fatal — maybe nothing to commit

    next_plan = pending[0] if pending else ""
    return {
        "progress_state": json.dumps({
            "completed_steps": completed,
            "pending_plan": pending,
            "current_step": next_plan,
        }),
        "current_plan": next_plan,
        "test_feedback": "",
        "failure_source": "",
        "retry_count": 0,
    }


# --- HIL nodes (interrupt() gates) ---

def human_plan_approval_node(state: SimulatorState) -> dict:
    """Interrupt gate: present the plan to the user for approval."""
    current_plan = state.get("current_plan", "")
    progress_raw = state.get("progress_state", "")
    spec_ctx = state.get("spec_context", "")
    target = state.get("current_target_file", "")

    plan_count = 0
    if progress_raw:
        try:
            p = json.loads(progress_raw)
            plan_count = len(p.get("pending_plan", []))
        except (json.JSONDecodeError, TypeError):
            pass

    user_response = interrupt({
        "type": "plan_approval",
        "question": "Approve this plan step?",
        "step": current_plan,
        "plan_left": plan_count,
        "spec_context": spec_ctx[:500],
        "target_file": target,
    })

    if user_response == "y":
        return {"human_feedback": "", "approved": True}
    else:
        return {"human_feedback": str(user_response), "approved": False}


def human_code_approval_node(state: SimulatorState) -> dict:
    """Interrupt gate: present passing code/tests to the user for approval."""
    current_plan = state.get("current_plan", "")
    test_fb = state.get("test_feedback", "")
    workspace = state.get("workspace", {})
    target = state.get("current_target_file", "")

    # Build a compact file listing
    file_summary = "\n".join(
        f"  {p} ({len(c)} bytes)" for p, c in sorted(workspace.items())
    )

    user_response = interrupt({
        "type": "code_approval",
        "question": "Approve the code and commit?",
        "step": current_plan,
        "cargo_output": test_fb[-600:] if test_fb else "PASS",
        "target_file": target,
        "files": file_summary,
    })

    if user_response == "y":
        return {"human_feedback": "", "approved": True}
    else:
        return {"human_feedback": str(user_response), "approved": False}


# ============================================================================
# Routing functions
# ============================================================================

def route_after_architect(state: SimulatorState) -> str:
    """If spec needed, go to spec_reader; otherwise to plan approval."""
    spec_ctx = state.get("spec_context", "")
    current_plan = state.get("current_plan", "")
    if not current_plan:
        return END

    plan_words = set(
        current_plan.upper()
        .replace("-", " ")
        .replace("_", " ")
        .replace(",", " ")
        .replace(".", " ")
        .replace("(", " ")
        .replace(")", " ")
        .split()
    )
    spec_keywords = {"ADD", "SUB", "MOV", "LDR", "STR", "INSTRUCTION", "DECODE", "ENCODING",
                     "INSTRUCTIONS", "ALU", "IMMEDIATE", "REGISTER", "BITFIELD",
                     "SIGNED", "UNSIGNED", "OFFSET", "SHIFT", "EXTEND"}
    if not spec_ctx and plan_words & spec_keywords:
        return "spec_reader"
    return "human_plan_approval"


def route_after_spec_reader(state: SimulatorState) -> str:
    """After loading spec, go to plan approval."""
    return "human_plan_approval"


def route_after_plan_approval(state: SimulatorState) -> str:
    """If feedback → architect, else start serial generation with coder."""
    if state.get("human_feedback"):
        return "architect"
    return "coder"


def route_after_test_runner(state: SimulatorState) -> str:
    """Route based on test results. Break infinite retry loops."""
    failure_src = state.get("failure_source", "")
    retry = state.get("retry_count", 0)

    # Tests passed → code approval gate
    if not failure_src:
        return "human_code_approval"

    # Max 3 consecutive retries — escalate to architect
    if retry >= 3:
        return "architect"

    if failure_src == "logic":
        return "coder"
    if failure_src == "tests":
        return "test_writer"
    return "human_code_approval"


def route_after_code_approval(state: SimulatorState) -> str:
    """If rejected with feedback, route to architect. If approved, commit."""
    if state.get("human_feedback"):
        return "architect"
    return "committer"


# ============================================================================
# Build the graph
# ============================================================================

def build_graph() -> StateGraph:
    builder = StateGraph(SimulatorState)

    # Add nodes
    builder.add_node("architect", architect_node)
    builder.add_node("spec_reader", spec_reader_node)
    builder.add_node("human_plan_approval", human_plan_approval_node)
    builder.add_node("coder", coder_node)
    builder.add_node("test_writer", test_writer_node)
    builder.add_node("test_runner", test_runner_node)
    builder.add_node("human_code_approval", human_code_approval_node)
    builder.add_node("committer", committer_node)

    # Edges — strict serial chain with retry loops
    builder.add_edge(START, "architect")

    builder.add_conditional_edges(
        "architect",
        route_after_architect,
        {
            "spec_reader": "spec_reader",
            "human_plan_approval": "human_plan_approval",
            END: END,
        },
    )

    builder.add_edge("spec_reader", "human_plan_approval")

    builder.add_conditional_edges(
        "human_plan_approval",
        route_after_plan_approval,
        ["architect", "coder"],
    )

    # Serial: coder → test_writer → test_runner
    builder.add_edge("coder", "test_writer")
    builder.add_edge("test_writer", "test_runner")

    builder.add_conditional_edges(
        "test_runner",
        route_after_test_runner,
        ["coder", "test_writer", "human_code_approval", "architect"],
    )

    builder.add_conditional_edges(
        "human_code_approval",
        route_after_code_approval,
        ["architect", "committer"],
    )

    builder.add_edge("committer", "architect")

    return builder


# ============================================================================
# CLI
# ============================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LangGraph orchestrator for ARM64 Rust simulator MVP"
    )
    p.add_argument(
        "-s", "--model", default="deepseek",
        choices=["deepseek", "openai"],
        help="LLM service (default: deepseek)",
    )
    p.add_argument(
        "--spec-dir", default=os.environ.get("ARM64_SPEC_DIR", ""),
        help="Path to ARM64 MRA XML specs (ISA_A64/ and SysReg/; also ARM64_SPEC_DIR env var)",
    )
    p.add_argument(
        "--thread-id", default=str(uuid.uuid4()),
        help="Thread ID for checkpoint persistence (new UUID if not provided)",
    )
    p.add_argument(
        "--model-name", default=None,
        help="Override the default model name (e.g. 'deepseek-v4-flash')",
    )
    p.add_argument(
        "--reset", action="store_true",
        help="Delete the checkpoint DB before starting (force cold start)",
    )
    p.add_argument(
        "prompt", nargs="?", default="",
        help="Optional user directive for the architect on cold boot",
    )
    return p.parse_args(argv)


def _print_banner() -> None:
    print("=" * 50)
    print("  ARM64 Simulator — LangGraph Orchestrator")
    print("=" * 50)
    print()


def _resolve_api_key(service: str) -> str:
    info = SERVICE_DEFAULTS[service]
    key = os.environ.get(info["env_key"], "")
    if not key:
        key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print(
            f"Error: {info['env_key']} not set. "
            f"Export it or add it to a .env file.",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    api_key = _resolve_api_key(args.model)
    llm = configure_llm(args.model, api_key, args.model_name)

    if args.spec_dir:
        configure_specs(args.spec_dir)

    _print_banner()
    print(f"Service:  {llm.service}")
    print(f"Model:    {llm.model}")
    print(f"Thread:   {args.thread_id}")
    if args.spec_dir:
        print(f"Specs:    {args.spec_dir}")
    if args.prompt:
        print(f"Directiv: {args.prompt}")
    print()

    # SQLite checkpointer
    db_path = Path(__file__).resolve().parent / "state.db"
    if args.reset:
        for suffix in ("", "-shm", "-wal"):
            p = Path(str(db_path) + suffix)
            if p.exists():
                p.unlink()
        print("[init] Checkpoint DB reset — cold start")

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    builder = build_graph()
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": args.thread_id}}

    # Check if we're resuming an existing thread
    try:
        snapshot = checkpointer.get_tuple(config)
        is_resume = snapshot is not None and snapshot.checkpoint.get("channel_versions", {})
    except Exception:
        is_resume = False

    if args.prompt and not is_resume and not args.reset:
        current_input: dict = {
            "startup_directive": args.prompt,
        }
    else:
        current_input = {}

    # Event loop
    while True:
        interrupted = False

        try:
            for event in graph.stream(current_input, config):
                if "__interrupt__" in event:
                    interrupted = True
                    interrupt_value = event["__interrupt__"][0].value

                    intr_type = interrupt_value.get("type", "")
                    print(f"\n{'─' * 50}")
                    if intr_type == "plan_approval":
                        step = interrupt_value.get("step", "")
                        plan_left = interrupt_value.get("plan_left", 0)
                        target = interrupt_value.get("target_file", "")
                        print(f"Step:       {step}")
                        print(f"Target:     {target}")
                        print(f"Plan left:  {plan_left} step(s)")
                        spec = interrupt_value.get("spec_context", "")
                        if spec:
                            print(f"Spec ctx:   {spec[:300]}")
                    elif intr_type == "code_approval":
                        print(f"Step:       {interrupt_value.get('step', '')}")
                        print(f"Target:     {interrupt_value.get('target_file', '')}")
                        print(f"Cargo:      PASS")
                        print(f"Files:      {interrupt_value.get('files', '')[:400]}")
                        print(f"Output:     {interrupt_value.get('cargo_output', '')[:400]}")
                    print(f"{'─' * 50}")

                    while True:
                        raw = input(
                            "y=approve  y:=approve+note  <anything>=reject & revise: "
                        ).strip()
                        if not raw:
                            continue
                        if raw.lower() in ("y", "yes"):
                            user_response = "y"
                            break
                        if raw.lower().startswith("y:") or raw.lower().startswith("yes:"):
                            user_response = "y"
                            break
                        user_response = raw
                        break

                    current_input = Command(resume=user_response)
                    break  # exit event loop, will re-enter with Command

                for node_name in event:
                    print(f"[{node_name}] ✓", end="", flush=True)

                    if node_name == "test_runner":
                        s = graph.get_state(config).values
                        failure_src = s.get("failure_source", "")
                        if failure_src:
                            print(f"  FAIL ({failure_src})")
                            tail = s.get("test_feedback", "")[-400:]
                            print(f"    {tail}")
                        else:
                            print("  PASS")
                    elif node_name in ("architect", "committer"):
                        s = graph.get_state(config).values
                        cp = s.get("current_plan", "")
                        tf = s.get("current_target_file", "")
                        print(f"  → {cp[:80]}")
                        if tf:
                            print(f"    file: {tf}")
                    elif node_name == "coder":
                        s = graph.get_state(config).values
                        tf = s.get("current_target_file", "")
                        if tf:
                            print(f"  → {tf}")
                    elif node_name == "test_writer":
                        s = graph.get_state(config).values
                        tf = s.get("current_target_file", "")
                        if tf:
                            print(f"  → {tf}")
                    else:
                        print()

        except Exception as exc:
            print(f"\n[ERROR] {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            break

        if not interrupted:
            print("\nDone.")
            break

        current_input = current_input if current_input is not None else {}

    conn.close()


if __name__ == "__main__":
    run(sys.argv[1:])
