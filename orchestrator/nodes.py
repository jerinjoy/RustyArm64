import json
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, get_args

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator, model_validator

from state import SimulatorState, clear_feedback

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Module-level config set by main.py before the graph runs.
_model_config: "ModelConfig | None" = None
_spec_root: Optional[Path] = None

# Agent-level configs and LLM instances (populated by configure_model).
_agent_configs: dict[str, ModelConfig] = {}
_agent_llms: dict[str, object] = {}

# Edit this dict to change DeepSeek model/thinking per agent.
# Every LLM-backed node reads its own entry by agent name.
AGENT_MODEL_DEFAULTS: dict[str, dict[str, str]] = {
    "architect":    {"model": "deepseek-v4-pro",  "thinking": "enabled"},
    "debugger":     {"model": "deepseek-v4-pro",  "thinking": "enabled"},
    "rust_coder":   {"model": "deepseek-v4-pro",  "thinking": "disabled"},
    "spec_reader":  {"model": "deepseek-v4-flash", "thinking": "disabled"},
    "test_writer":  {"model": "deepseek-v4-flash", "thinking": "disabled"},
}

SERVICE_DEFAULTS: dict[str, dict[str, str]] = {
    "deepseek": {
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
    },
    "openai": {
        "model": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
    },
    "gemini": {
        "model": "gemini-2.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
}

VALID_SERVICES = frozenset(SERVICE_DEFAULTS.keys())


_ThinkingMode = Literal["enabled", "disabled"]


@dataclass
class ModelConfig:
    service: str
    api_key: str
    model: str | None = None
    thinking: _ThinkingMode = "disabled"

    def __post_init__(self):
        if self.service not in VALID_SERVICES:
            raise ValueError(
                f"Unknown service '{self.service}'. Choose: {', '.join(sorted(VALID_SERVICES))}"
            )
        if self.thinking not in get_args(_ThinkingMode):
            raise ValueError(
                f"Invalid thinking '{self.thinking}'. Choose: {get_args(_ThinkingMode)}"
            )

    @property
    def resolved_model(self) -> str:
        return self.model or SERVICE_DEFAULTS[self.service]["model"]

    @property
    def base_url(self) -> str:
        return SERVICE_DEFAULTS[self.service]["base_url"]

    @property
    def structured_output_method(self) -> str:
        """Return the structured output method compatible with this service.

        DeepSeek uses json_mode because its thinking models reject tool_choice.
        OpenAI uses json_schema (native).
        """
        if self.service == "deepseek":
            return "json_mode"
        return "json_schema"

    def create_chat_model(self):
        if self.service == "deepseek":
            return self._create_deepseek()
        if self.service == "gemini":
            return self._create_gemini()
        return ChatOpenAI(
            model=self.resolved_model,
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def _create_deepseek(self):
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(
            model=self.resolved_model,
            api_key=self.api_key,
            temperature=0.2,
            extra_body={"thinking": {"type": self.thinking}},
        )

    def _create_gemini(self):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "Gemini requires langchain-google-genai. Run: uv add langchain-google-genai"
            ) from None
        return ChatGoogleGenerativeAI(
            model=self.resolved_model,
            google_api_key=self.api_key,
        )


def configure_model(service: str, api_key: str, model: str | None = None):
    """Set up one LLM instance per agent.

    For DeepSeek, reads model/thinking from AGENT_MODEL_DEFAULTS.
    For other services all agents share the same model.
    """
    global _model_config, _agent_configs, _agent_llms

    _model_config = ModelConfig(service=service, api_key=api_key, model=model)

    if service == "deepseek":
        for agent, cfg in AGENT_MODEL_DEFAULTS.items():
            _agent_configs[agent] = ModelConfig(
                service="deepseek",
                api_key=api_key,
                model=cfg["model"],
                thinking=cfg.get("thinking", "disabled"),
            )
    else:
        for agent in AGENT_MODEL_DEFAULTS:
            _agent_configs[agent] = ModelConfig(service=service, api_key=api_key, model=model)

    for agent, cfg in _agent_configs.items():
        _agent_llms[agent] = cfg.create_chat_model()


def configure_specs(spec_root: str | Path) -> None:
    """Set the root directory for ARM64 MRA specification files.

    Must contain ISA_A64/ and SysReg/ subdirectories, each holding a single
    versioned subdirectory (e.g. ISA_A64_xml_A_profile-YYYY-MM).
    """
    global _spec_root
    _spec_root = Path(spec_root).expanduser().resolve()


def _resolve_version_dir(parent: Path, pattern: str) -> Path:
    """Find the versioned directory inside *parent* matching *pattern*.

    Picks the newest (highest sort order) when multiple directories match.
    Raises FileNotFoundError if none are found.
    """
    candidates = sorted(
        p for p in parent.glob(pattern)
        if p.is_dir() and not p.name.endswith("_diff") and "diff" not in p.stem
    )
    if not candidates:
        raise FileNotFoundError(
            f"No directory matching '{pattern}' found inside {parent}"
        )
    if len(candidates) > 1:
        chosen = candidates[-1]  # highest sort order = newest date
        names = "\n  ".join(c.name for c in candidates)
        print(
            f"[spec_reader] Multiple version dirs found, picking newest:\n"
            f"  {names}\n"
            f"  → {chosen.name}\n"
            f"  Remove old directories to silence this warning.",
            file=sys.stderr,
        )
        return chosen
    return candidates[0]


# ---------------------------------------------------------------------------
# Pydantic output models for structured LLM responses
# ---------------------------------------------------------------------------


_ARCHITECT_PLAN_ALIASES = frozenset(
    {"plan", "remaining_plan", "steps", "tasks", "implementation_plan"}
)
_ARCHITECT_STEP_ALIASES = frozenset(
    {"current_step", "next_step", "current_task", "first_step", "step"}
)


class ArchitectOutput(BaseModel):
    plan: List[str] = Field(
        default=...,
        description="Complete list of remaining implementation steps (each a short string)",
    )
    current_step: str = Field(
        default=...,
        description="Exactly one step from the plan to execute next — full step text, not an index",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        # Map known alternative LLM-chosen keys to canonical names
        for key in list(data):
            if key in _ARCHITECT_PLAN_ALIASES and "plan" not in data:
                data["plan"] = data.pop(key)
                break
        for key in list(data):
            if key in _ARCHITECT_STEP_ALIASES and "current_step" not in data:
                data["current_step"] = data.pop(key)
                break
        return data

    @field_validator("plan", mode="before")
    @classmethod
    def _coerce_list_str(cls, v: object) -> List[str]:
        return [str(item) for item in v]

    @field_validator("current_step", mode="before")
    @classmethod
    def _coerce_step(cls, v: object) -> str:
        return str(v)


class RustCoderOutput(BaseModel):
    files: Dict[str, str] = Field(
        description="Map of relative file path -> complete Rust source content"
    )


class TestWriterOutput(BaseModel):
    files: Dict[str, str] = Field(
        description="Map of relative file path -> complete Rust source (with tests inline)"
    )


class SpecReaderOutput(BaseModel):
    instruction_id: str = Field(
        default="",
        description="The ARM64 instruction identifier (e.g. 'ADD_addsub_imm', 'LDR_imm_gen')",
    )
    mnemonic: str = Field(
        default="",
        description="The assembly mnemonic (e.g. 'ADD', 'LDR')",
    )
    summary: str = Field(
        default="",
        description="One-sentence description of what the instruction does",
    )
    bitfields: List[dict] = Field(
        default_factory=list,
        description=(
            "Ordered list of instruction encoding fields. Each dict has: "
            "name (str), bits (str like '31'), hi (int), lo (int), "
            "width (int), is_variable (bool), default_value (str or null). "
            "Variable fields (usename='1') are operands to decode; "
            "constant fields define the opcode bit-pattern to match."
        ),
    )
    decode_pseudocode: str = Field(
        default="",
        description="Decode-stage pseudocode. Extracts operands from bitfields, "
        "validates encoding constraints. Runs at decode time.",
    )
    execute_pseudocode: str = Field(
        default="",
        description="Execute-stage pseudocode. Performs the actual operation "
        "using decoded operands. Runs at execute time.",
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="List of constraints or edge cases (e.g. UNDEFINED conditions, "
        "register restrictions, alignment requirements)",
    )


class DebuggerOutput(BaseModel):
    analysis: str = Field(description="Root-cause analysis of the cargo failure")
    repair_target: str = Field(
        description="Either 'code' (implementation bug) or 'test' (test bug)"
    )

    @field_validator("analysis", mode="before")
    @classmethod
    def _coerce_analysis(cls, v: object) -> str:
        return str(v)

    @field_validator("repair_target", mode="before")
    @classmethod
    def _normalise_target(cls, v: object) -> str:
        s = str(v).strip().lower()
        if s in ("code", "test"):
            return s
        if "code" in s:
            return "code"
        if "test" in s:
            return "test"
        return "code"  # safe default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_agent(agent_name: str) -> None:
    """Print which service/model an agent node is about to call."""
    cfg = _agent_configs.get(agent_name)
    if cfg is None:
        return
    print(f"[{agent_name}]  service={cfg.service}  model={cfg.resolved_model}")


def _get_agent_model(agent_name: str):
    """Return the chat model instance for *agent_name*."""
    if agent_name not in _agent_llms:
        raise RuntimeError(
            f"Agent '{agent_name}' not configured. Call configure_model() first."
        )
    return _agent_llms[agent_name]


def _get_agent_structured_model(agent_name: str, output_schema):
    """Return a model with structured output for *agent_name*.

    Uses the structured-output method that is compatible with the agent's
    service (json_mode for DeepSeek, json_schema for others).
    """
    model = _get_agent_model(agent_name)
    cfg = _agent_configs[agent_name]
    method = cfg.structured_output_method
    structured = model.with_structured_output(output_schema, method=method)

    if method == "json_mode":
        from langchain_core.runnables import RunnableLambda

        def _inject_json(prompt: str) -> str:
            if isinstance(prompt, str) and "json" not in prompt.lower():
                return "Output JSON.\n" + prompt
            return prompt

        structured = RunnableLambda(_inject_json) | structured

    return structured


def _codebase_to_str(codebase: Dict[str, str]) -> str:
    if not codebase:
        return "(empty – no files yet)"
    parts = []
    for path in sorted(codebase):
        parts.append(f"\n=== {path} ===\n{codebase[path]}")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def architect_node(state: SimulatorState) -> dict:
    """Create or revise the implementation plan and select the next task.

    Processes human_feedback and debugger_feedback if present, removes
    successfully-completed steps from the plan, and generates a fresh plan
    when none exists.
    """
    _log_agent("architect")
    result = clear_feedback(state)
    model = _get_agent_structured_model("architect", ArchitectOutput)

    plan = state.get("plan", [])
    current_step = state.get("current_step", "")
    completed = state.get("completed_steps", [])
    human_fb = state.get("human_feedback", "")
    debugger_fb = state.get("debugger_feedback", "")
    repair_target = state.get("repair_target", "")
    cargo_success = state.get("cargo_success", False)
    codebase_keys = sorted(state.get("codebase", {}).keys())

    completed_summary = ""
    if completed:
        completed_summary = "  Completed steps:\n"
        for cs in completed:
            completed_summary += f"    - {cs}\n"

    prompt = f"""You are the architect for a Rust ARM64 CPU simulator MVP.
Manage the implementation plan.

CURRENT STATE
  Remaining plan: {json.dumps(plan)}
  Current step:   {current_step}
{completed_summary}  Cargo success:  {cargo_success}
  Existing files: {json.dumps(codebase_keys) if codebase_keys else "none yet"}
  Human feedback: {human_fb if human_fb else "none"}
  Debug feedback: {debugger_fb if debugger_fb else "none"}
  Repair target:  {repair_target if repair_target else "n/a"}

RULES (apply in order):
1. If human_feedback is present, use it to revise the plan and current step.
2. If debugger_feedback is present, adjust the plan to fix the reported issue
   (keep current_step the same so the fix is applied).
3. If cargo_success is True and the current step is still in the plan,
   remove it from the plan (it is done). Then pick the next step from the
   remaining plan.
4. If the plan is empty and no feedback requires action, return an empty plan
   and empty current_step.
5. If the plan is empty, generate an initial step-by-step plan for building a
   minimal but functional ARM64 simulator that can:
   - Parse and decode a subset of ARM64 instructions
   - Model CPU registers (X0-X30, SP, PC, NZCV flags)
   - Execute basic data-processing instructions (ADD, SUB, MOV, AND, ORR, EOR…)
   - Handle memory load/store operations (LDR, STR)
   - Run a simple test program end-to-end
   IMPORTANT: when files already exist in the codebase, generate a plan that
   builds on top of what is already there — do NOT plan to rebuild from scratch.

Return the COMPLETE remaining plan and the SINGLE next step to execute."""

    response = model.invoke(prompt)
    result["plan"] = response.plan
    result["current_step"] = response.current_step
    return result


def _load_xml_spec_for_task(current_step: str) -> str:
    """Find and load the most relevant ARM64 spec XML file for *current_step*.

    Uses the module-level ``_spec_root`` (set by ``configure_specs()``) and
    discovers the versioned sub-directories via glob patterns.  Falls back to
    the project ``specs/`` directory if no XML source is configured.
    """
    if _spec_root is None or not _spec_root.exists():
        # Fallback: project specs/ directory
        specs_dir = Path(__file__).resolve().parent.parent / "specs"
        if specs_dir.exists():
            parts: list[str] = []
            for sf in sorted(specs_dir.iterdir()):
                if sf.suffix in (".txt", ".md"):
                    parts.append(sf.read_text(encoding="utf-8", errors="replace"))
            return "\n".join(parts)
        return ""

    orch_dir = Path(__file__).resolve().parent

    # Derive the versioned XML directories (glob, single match)
    try:
        isa_xml_dir = _resolve_version_dir(
            _spec_root / "ISA_A64", "ISA_A64_xml_A_profile-*"
        )
    except FileNotFoundError:
        isa_xml_dir = None

    try:
        sysreg_xml_dir = _resolve_version_dir(
            _spec_root / "SysReg", "SysReg_xml_A_profile-*"
        )
    except FileNotFoundError:
        sysreg_xml_dir = None

    if isa_xml_dir is None and sysreg_xml_dir is None:
        return ""

    task_upper = current_step.upper()
    task_words = set(current_step.replace("-", " ").replace("_", " ").split())

    # --- instruction index lookup ---
    if isa_xml_dir is not None:
        index_path = orch_dir / "instruction_index.json"
        if index_path.exists():
            try:
                with open(index_path) as f:
                    idx_data = json.load(f)
                entries = idx_data.get("instructions", {})

                candidates: list[tuple[int, str]] = []
                for instr_id, rec in entries.items():
                    id_lower = instr_id.lower()
                    mnem = rec.get("mnemonic", "").upper()
                    if mnem and mnem in task_upper:
                        candidates.append((0, instr_id))
                    for w in task_words:
                        if w.lower() in id_lower:
                            candidates.append((1, instr_id))
                            break
                candidates.sort()
                if candidates:
                    best_id = candidates[0][1]
                    rec = entries[best_id]
                    xml_path = isa_xml_dir / rec["file"]
                    if xml_path.exists():
                        return xml_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass

    # --- sysreg index lookup ---
    if sysreg_xml_dir is not None:
        idx_path = orch_dir / "sysreg_index.json"
        if idx_path.exists():
            try:
                with open(idx_path) as f:
                    idx_data = json.load(f)
                entries = idx_data.get("registers", {})
                for reg_name, rec in entries.items():
                    if reg_name.upper() in task_upper or reg_name.upper() in task_words:
                        xml_path = sysreg_xml_dir / rec["file"]
                        if xml_path.exists():
                            return xml_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass

    # --- fallback: scan the version dirs for any file whose name hints at the task ---
    for xml_dir in (d for d in (isa_xml_dir, sysreg_xml_dir) if d is not None):
        # os.walk-like scan, but simple since these dirs are flat
        task_lower = current_step.lower()
        for xml_file in sorted(xml_dir.glob("*.xml")):
            if any(w.lower() in xml_file.stem.lower() for w in task_words):
                return xml_file.read_text(encoding="utf-8", errors="replace")

    return ""


_SPEC_READER_SYSTEM_PROMPT = """You are an ARM64 (AArch64) Instruction Set Architecture expert. Your job is to
read raw ARM Machine-Readable Architecture (MRA) XML specification files and
extract the precise, structured information that a downstream Rust coding agent
needs to implement an instruction-accurate simulator.

================================================================================
INPUT
================================================================================

You will receive raw XML from an ARM64 instruction specification file
(ISA_A64/*.xml). Your task is to parse it and produce a clean JSON object with
the fields described below.

================================================================================
HOW TO READ THE XML
================================================================================

1. IDENTITY — look at the root element <instructionsection>:
   - @id        → instruction identifier (e.g. "ADD_addsub_imm", "LDR_imm_gen")
   - @title     → human-readable title
   - <docvars>/<docvar key="mnemonic"> → assembly mnemonic (e.g. "ADD")
   - <desc>/<brief>/<para> → one-line summary

2. BIT-FIELD DEFINITIONS — look inside <classes> → <iclass> → <regdiagram>:

   Each <box> element defines one contiguous slice of the 32-bit instruction
   encoding, read from MSB (bit 31) down to LSB (bit 0).

   <box hibit="N" width="W" name="fieldname" usename="1">
   Defines a NAMED variable field. The slice occupies bits [N, N-W+1] inclusive.

   <box hibit="N" width="W" settings="W">
   Defines a FIXED constant field. The <c> children give the expected bit values
   (listed MSB-first). These form the opcode pattern that must be matched during
   decoding.

   Key attributes on <box>:
   - hibit="N"      → highest bit index of this slice
   - width="W"      → number of bits (default 1 if absent)
   - name="X"       → field identifier used in pseudocode (absent for constants)
   - usename="1"    → this is a variable field; must be decoded
   - settings="W"   → this is a constant field of width W; must be verified

   Example: <box hibit="21" width="12" name="imm12" usename="1">
   → variable field "imm12" occupying bits [21:10]

   Example: <box hibit="28" width="6" settings="6">
   → 6-bit constant occupying bits [28:23], values given by <c> children

   ENCODING VARIANTS: Some instructions have separate <encoding> elements for
   32-bit and 64-bit forms (toggled by the "sf" bit at bit 31). Each variant
   may have different assembler templates. Note the sf=0 (32-bit) and sf=1
   (64-bit) variants separately if both exist.

3. PSEUDOCODE — two blocks, found in different locations:

   DECODE PSEUDOCODE — inside <iclass>:
     <ps_section howmany="1">
       <ps name="..." sections="1" secttype="noheading">
         <pstext section="Decode" rep_section="decode">
           ... decode logic here ...
         </pstext>
       </ps>
     </ps_section>
   The secttype is "noheading". This runs at decode time: it extracts operands
   from bit-fields and validates encoding conditions. Failure → UNDEFINED.

   EXECUTE PSEUDOCODE — direct child of <instructionsection> (outside
   <classes>):
     <ps_section howmany="1">
       <ps name="..." sections="1" secttype="Operation">
         <pstext section="Execute" rep_section="execute">
           ... execution logic here ...
         </pstext>
       </ps>
     </ps_section>
   The secttype is "Operation". Extract the EXACT pseudocode text. Do not
   paraphrase. Do not interpret. Copy it verbatim (with HTML entities
   decoded: &lt; → <, &gt; → >, &amp; → &, &quot; → ").

   IMPORTANT: If the execute pseudocode references external functions via
   <a link="func_AddWithCarry_4" file="shared_pseudocode.xml">AddWithCarry</a>,
   preserve the function name as plain text: "AddWithCarry". The Rust coder
   will recognise these standard ARM pseudocode functions.

4. CONSTRAINTS — gather from:
   - <operationalnotes> and <operationalnote> → special execution constraints
   - <desc>/<authored> → behavioral notes
   - <alias_list>/<aliasref> → alias relationships (e.g. MOV is an alias of ADD)
   - Pseudocode conditions that result in UNDEFINED behavior

================================================================================
OUTPUT JSON SCHEMA
================================================================================

Return EXACTLY this JSON structure (no extra text, no markdown wrapping):

{
  "instruction_id": "...",
  "mnemonic": "...",
  "summary": "...",
  "bitfields": [
    {
      "name": "sf",
      "bits": "31",
      "hi": 31,
      "lo": 31,
      "width": 1,
      "is_variable": true,
      "default_value": null,
      "description": "32-bit (0) / 64-bit (1) size flag"
    },
    {
      "name": "imm12",
      "bits": "21:10",
      "hi": 21,
      "lo": 10,
      "width": 12,
      "is_variable": true,
      "default_value": null,
      "description": "12-bit unsigned immediate"
    },
    {
      "name": null,
      "bits": "28:23",
      "hi": 28,
      "lo": 23,
      "width": 6,
      "is_variable": false,
      "default_value": "100010",
      "description": "Opcode constant"
    }
  ],
  "decode_pseudocode": "let d : integer{} = UInt(Rd);\\nlet datasize : integer{} = 32 << UInt(sf);\\n...",
  "execute_pseudocode": "let operand1 : bits(datasize) = X{}(n);\\n...",
  "constraints": [
    "Register 31 is the stack pointer (SP) rather than XZR when used in this context",
    "The shift amount must be 0 or 12"
  ]
}

RULES FOR bitfields:
- List in MSB-to-LSB order (highest hi first).
- For variable fields: is_variable=true, default_value=null.
- For constant fields: is_variable=false, default_value is the bit pattern
  string (MSB-first, e.g. "100010").
- Field "bits" is a human-readable string: "N" for single-bit, "Hi:Lo" for
  multi-bit.
- Include a concise description for each named field based on the instruction
  context (e.g. "destination register").

RULES FOR pseudocode:
- Preserve exact whitespace structure (line breaks matter).
- Do NOT add or remove any logic.
- Do NOT expand links or resolve cross-references.
- If a pseudocode block is absent, use an empty string "".

RULES FOR constraints:
- Be specific, not generic.
- Omit constraints that are already obvious from the bitfield layout.
- Include UNDEFINED conditions explicitly."""


def spec_reader_node(state: SimulatorState) -> dict:
    """Gather ARM architecture specification context for the current task.

    Loads the relevant ARM64 MRA XML file, extracts bit-field definitions and
    pseudocode via the LLM, and returns structured JSON spec_context.
    """
    _log_agent("spec_reader")
    current_step = state.get("current_step", "")

    # Load the most relevant spec XML file
    xml_text = _load_xml_spec_for_task(current_step)

    # Build the prompt — the system prompt is long, so put it in the role
    # preamble and leave the user message as the task + data.
    model = _get_agent_structured_model("spec_reader", SpecReaderOutput)

    if xml_text:
        # Truncate if extremely large (SysReg files can be big)
        xml_snippet = xml_text[:12000]
        user_prompt = f"""{_SPEC_READER_SYSTEM_PROMPT}

================================================================================
INPUT
================================================================================

CURRENT TASK
  {current_step}

RAW ARM64 SPECIFICATION XML
================================================================================
  {xml_snippet}
"""
    else:
        user_prompt = f"""You are an ARM64 (AArch64) architecture expert.
Provide a clean JSON response with instruction_id, mnemonic, summary,
bitfields, decode_pseudocode, execute_pseudocode, and constraints.

CURRENT TASK
  {current_step}

No ARM64 specification file was found for this task. Use your built-in
ARM64 architecture knowledge to provide the relevant instruction format,
register encodings, and pseudocode."""

    response = model.invoke(user_prompt)

    # Build the spec_context string from the structured output
    context_parts = []
    if response.instruction_id:
        context_parts.append(f"Instruction: {response.instruction_id} ({response.mnemonic})")
    if response.summary:
        context_parts.append(f"Summary: {response.summary}")
    if response.bitfields:
        context_parts.append("\nBit-field encoding:")
        for bf in response.bitfields:
            kind = "variable" if bf.get("is_variable") else "constant"
            val = bf.get("default_value") or ""
            desc = bf.get("description", "")
            extra = f" = {val}" if val and not bf.get("is_variable") else ""
            desc_str = f"  // {desc}" if desc else ""
            context_parts.append(f"  [{bf.get('bits')}] {bf.get('name') or '(const)'} ({kind}){extra}{desc_str}")
    if response.decode_pseudocode:
        context_parts.append(f"\nDecode pseudocode:\n{response.decode_pseudocode}")
    if response.execute_pseudocode:
        context_parts.append(f"\nExecute pseudocode:\n{response.execute_pseudocode}")
    if response.constraints:
        context_parts.append("\nConstraints:")
        for c in response.constraints:
            context_parts.append(f"  - {c}")

    spec_context = "\n".join(context_parts)

    return {"spec_context": spec_context}


def human_approval_node(state: SimulatorState) -> dict:
    """Passthrough node gated by interrupt_before in the compiled graph.

    The graph pauses *before* this node.  The interactive CLI inspects
    current_step / spec_context, optionally writes human_feedback into the
    state, and resumes.  This node's conditional edge then routes to either
    architect (feedback present) or rust_coder (approved).
    """
    return {}


def rust_coder_node(state: SimulatorState) -> dict:
    """Generate or update in-memory Rust source files based on spec and feedback."""
    _log_agent("rust_coder")
    current_step = state.get("current_step", "")
    spec_context = state.get("spec_context", "")
    codebase = state.get("codebase", {})
    human_fb = state.get("human_feedback", "")
    dbg_fb = state.get("debugger_feedback", "")
    repair_target = state.get("repair_target", "")

    model = _get_agent_structured_model("rust_coder", RustCoderOutput)

    debug_section = ""
    if dbg_fb and repair_target == "code":
        debug_section = f"""
DEBUGGER FEEDBACK (fix the implementation – this is the root cause):
  {dbg_fb[:4000]}
"""

    prompt = f"""You are a Rust systems programmer building an ARM64 CPU simulator.
Write or update Rust code under simulator/src/.

CURRENT TASK
  {current_step}

ARM64 SPEC CONTEXT
  {spec_context[:6000]}

HUMAN FEEDBACK (if any)
  {human_fb if human_fb else "none"}
{debug_section}
EXISTING CODEBASE
{_codebase_to_str(codebase)}

GUIDELINES
- Produce production-quality, idiomatic Rust with proper error handling.
- Model ARM64 registers in a struct; use enums for instruction types.
- Keep functions small and testable (pub interfaces, clear ownership).
- Only return files you are creating or modifying (not the whole codebase).
- Each file must contain its COMPLETE final content, not a diff."""

    response = model.invoke(prompt)
    return {
        "codebase": response.files,
        "human_feedback": "",
        "debugger_feedback": "",
        "repair_target": "",
    }


def test_writer_node(state: SimulatorState) -> dict:
    """Add inline Rust unit tests to the codebase for the current task."""
    _log_agent("test_writer")
    current_step = state.get("current_step", "")
    spec_context = state.get("spec_context", "")
    codebase = state.get("codebase", {})
    dbg_fb = state.get("debugger_feedback", "")
    repair_target = state.get("repair_target", "")

    model = _get_agent_structured_model("test_writer", TestWriterOutput)

    debug_section = ""
    if dbg_fb and repair_target == "test":
        debug_section = f"""
DEBUGGER FEEDBACK (fix the TESTS – tests are incorrect, not the implementation):
  {dbg_fb[:4000]}
"""

    prompt = f"""You are a Rust test engineer. Add comprehensive inline unit tests
to the ARM64 simulator code.

CURRENT TASK
  {current_step}

ARM64 SPEC CONTEXT
  {spec_context[:4000]}
{debug_section}
CURRENT CODEBASE
{_codebase_to_str(codebase)}

REQUIREMENTS
- Add #[cfg(test)] mod tests {{ ... }} blocks to files that have implementation
  code relevant to the current task.
- Cover: happy-path execution, edge cases (boundary values, overflow), and
  error paths where applicable.
- Every ARM64 instruction implemented in the current task must have at least
  one test case.
- If debugger feedback is present, fix the tests based on that feedback.
- Return only files you modified, with their COMPLETE content including tests."""

    response = model.invoke(prompt)
    return {
        "codebase": response.files,
        "debugger_feedback": "",
        "repair_target": "",
    }


def cargo_tool_node(state: SimulatorState) -> dict:
    """Pure-Python node: flush codebase to disk, validate paths, run cargo test.

    No LLM calls – completely deterministic.
    """
    codebase: Dict[str, str] = state.get("codebase", {})
    simulator_dir = Path(__file__).resolve().parent.parent / "simulator"
    src_dir = simulator_dir / "src"

    # --- path-traversal guard ---
    src_root = src_dir.resolve()
    for rel in codebase:
        if ".." in Path(rel).parts:
            return {
                "cargo_output": f"SECURITY: path traversal in '{rel}'",
                "cargo_success": False,
            }
        resolved = (src_dir / rel).resolve()
        if not str(resolved).startswith(str(src_root)):
            return {
                "cargo_output": f"SECURITY: escape for '{rel}' -> {resolved}",
                "cargo_success": False,
            }

    # --- write files ---
    src_dir.mkdir(parents=True, exist_ok=True)
    for rel, content in codebase.items():
        dest = src_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)

    # --- run cargo test ---
    manifest = simulator_dir / "Cargo.toml"
    try:
        proc = subprocess.run(
            ["cargo", "test", "--manifest-path", str(manifest)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(simulator_dir),
        )
        output = proc.stdout + "\n" + proc.stderr
        success = proc.returncode == 0
    except subprocess.TimeoutExpired:
        output = "cargo test timed out after 120 seconds"
        success = False
    except FileNotFoundError:
        output = "cargo not found – is the Rust toolchain installed?"
        success = False

    return {"cargo_output": output.strip(), "cargo_success": success}


def debugger_node(state: SimulatorState) -> dict:
    """Analyse cargo test failures and route repair to code or tests."""
    _log_agent("debugger")
    cargo_output = state.get("cargo_output", "")
    codebase = state.get("codebase", {})
    current_step = state.get("current_step", "")

    model = _get_agent_structured_model("debugger", DebuggerOutput)

    code_summary = "\n".join(f"  {p}" for p in sorted(codebase))

    prompt = f"""You are a Rust compiler/test debugger. Analyse the cargo output
and decide whether the root cause is in implementation code or test code.

CURRENT TASK
  {current_step}

FILES IN CODEBASE
  {code_summary if code_summary else "(none)"}

CARGO OUTPUT
  {cargo_output[:8000]}

Decide:
- 'code' → implementation logic is wrong (register behaviour, instruction
  semantics, decoding errors, etc.).
- 'test' → tests are wrong (incorrect expected values, bad assertions, testing
  undefined behaviour, etc.).

Provide a concise, actionable analysis so the target node can fix the issue."""

    response = model.invoke(prompt)
    return {
        "debugger_feedback": response.analysis,
        "repair_target": response.repair_target,
    }


def human_test_approval_node(state: SimulatorState) -> dict:
    """Passthrough node gated by interrupt_before.

    The graph pauses *before* this node so the human can inspect the passing
    cargo test results.  If approved the progress_writer and committer run;
    if rejected the architect receives the feedback.
    """
    return {}


def progress_writer_node(state: SimulatorState) -> dict:
    """Write completed/pending steps to progress.yaml.

    Pure Python — no LLM.  Flipped before committer so progress.yaml lands
    in the same commit as the code changes.
    """
    import yaml

    completed = list(state.get("completed_steps", []))
    current_step = state.get("current_step", "")
    plan = state.get("plan", [])
    codebase_keys = sorted(state.get("codebase", {}).keys())

    # Move the just-completed step from plan into completed list
    if current_step and current_step in plan:
        plan.remove(current_step)
    if current_step and current_step not in completed:
        completed.append(current_step)

    progress_path = Path(__file__).resolve().parent.parent / "progress.yaml"

    progress_path.write_text(
        yaml.dump(
            {
                "completed_steps": completed,
                "pending_plan": plan,
                "current_step": plan[0] if plan else "",
                "files": codebase_keys,
            },
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    print(f"[progress_writer]  wrote {len(completed)} completed, "
          f"{len(plan)} pending → {progress_path.name}")

    return {
        "completed_steps": completed,
        "plan": plan,
        "current_step": plan[0] if plan else "",
    }


def committer_node(state: SimulatorState) -> dict:
    """Stage and commit simulator/src/ and progress.yaml.

    Pure Python — no LLM.  Runs after progress_writer so both the code and
    the updated YAML land in one atomic commit.
    """
    repo_root = Path(__file__).resolve().parent.parent
    current_step = state.get("current_step", "")

    try:
        subprocess.run(
            ["git", "add", "-A", "simulator/src/", "progress.yaml"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"feat: {current_step}"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"[committer]  committed → {current_step}")
    except subprocess.CalledProcessError as exc:
        print(f"[committer]  commit failed: {exc.stderr[:400]}", file=sys.stderr)

    return {}
