import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, get_args

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator, model_validator

from state import SimulatorState, clear_feedback

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Module-level config set by main.py before the graph runs.
_model_config: "ModelConfig | None" = None

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
    human_fb = state.get("human_feedback", "")
    debugger_fb = state.get("debugger_feedback", "")
    repair_target = state.get("repair_target", "")
    cargo_success = state.get("cargo_success", False)
    codebase_keys = sorted(state.get("codebase", {}).keys())

    prompt = f"""You are the architect for a Rust ARM64 CPU simulator MVP.
Manage the implementation plan.

CURRENT STATE
  Remaining plan: {json.dumps(plan)}
  Current step:   {current_step}
  Cargo success:  {cargo_success}
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

Return the COMPLETE remaining plan and the SINGLE next step to execute."""

    response = model.invoke(prompt)
    result["plan"] = response.plan
    result["current_step"] = response.current_step
    return result


def spec_reader_node(state: SimulatorState) -> dict:
    """Gather ARM architecture specification context for the current task."""
    _log_agent("spec_reader")
    current_step = state.get("current_step", "")

    specs_dir = Path(__file__).resolve().parent.parent / "specs"
    spec_text = ""
    if specs_dir.exists():
        for sf in sorted(specs_dir.iterdir()):
            if sf.suffix in (".txt", ".md"):
                spec_text += sf.read_text() + "\n"

    model = _get_agent_model("spec_reader")
    prompt = f"""You are an ARM64 (AArch64) architecture expert. Given the current
implementation task, provide the relevant ARM64 specification details.

CURRENT TASK
  {current_step}

REFERENCE MATERIAL
  {spec_text[:5000] if spec_text else "Use your built-in ARM64 knowledge."}

Provide concise, technically accurate ARM64 architectural context needed to
implement this specific task. Include register encodings, instruction formats,
addressing modes, and any edge cases or constraints relevant to the task."""

    response = model.invoke(prompt)
    return {"spec_context": str(response.content)}


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
