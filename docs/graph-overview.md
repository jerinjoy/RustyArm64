# LangGraph Orchestrator Overview

## Graph topology

```
START
  │
  ▼
architect        ◄─────────────────────────────┐
  │                                              │
  ▼                                              │
spec_reader                                       │
  │                                              │
  ▼                                              │
human_approval ───(feedback)─────────────────────┘
  │
  │ (approved)
  ▼
rust_coder
  │
  ▼
test_writer
  │
  ▼
cargo_tool ───(success + remaining plan)────────► architect
  │              (success + empty plan)─────────► END
  │              (failure)──────────────────────► debugger
  │                                               │
  │                          ┌──(code bug)────────┘
  │                          │
  │                          └──(test bug)───────► test_writer
  │
  ▼
 (back to architect or END)
```

## State (`SimulatorState`, defined in `orchestrator/state.py`)

```
plan:             List[str]           — remaining implementation steps
current_step:     str                 — the step currently being worked on
codebase:         Dict[str, str]      — key=relative path under simulator/src, value=file content
                                         Uses Annotated[...] reducer to merge partial updates
spec_context:      str                — ARM64 spec information for the current step
cargo_output:      str                — stdout+stderr from last cargo test run
cargo_success:     bool               — whether last cargo test passed
debugger_feedback: str                — failure analysis from the debugger node
repair_target:     str                — "code" or "test", set by debugger
human_feedback:    str                — rejection feedback typed by the human
```

## Nodes

| Node | LLM? | Pydantic output | What it does |
|------|------|-----------------|--------------|
| `architect` | LLM | `ArchitectOutput(plan, current_step)` | Creates/revises plan, picks next step. Calls `clear_feedback()` at start of every iteration. |
| `spec_reader` | LLM | (raw string → `spec_context`) | Gathers ARM64 spec context for the current step. Reads local `specs/` if files exist. |
| `human_approval` | — | (passthrough) | Gated by `interrupt_before`. The CLI inspects state here and injects `human_feedback`. Routes to architect if feedback present, otherwise to rust_coder. |
| `rust_coder` | LLM | `RustCoderOutput(files: dict)` | Generates/updates Rust source files. Reads `debugger_feedback` if `repair_target == "code"`. Clears feedback after use. |
| `test_writer` | LLM | `TestWriterOutput(files: dict)` | Adds inline `#[cfg(test)]` blocks. Reads `debugger_feedback` if `repair_target == "test"`. Clears feedback after use. |
| `cargo_tool` | — | — | Writes codebase to `../simulator/src/`, validates paths (no `..` traversal), runs `cargo test --manifest-path ../simulator/Cargo.toml`. |
| `debugger` | LLM | `DebuggerOutput(analysis, repair_target)` | Analyzes cargo failures. Sets `repair_target` to `"code"` or `"test"`. |

## Routing rules

- **`route_after_human`**: `human_feedback` non-empty → architect. Empty → rust_coder.
- **`route_after_cargo`**: `cargo_success` + plan non-empty → architect. `cargo_success` + plan empty → END. Failure → debugger.
- **`route_after_debugger`**: `repair_target == "test"` → test_writer. Otherwise → rust_coder.

## Human-in-the-loop

- Graph compiled with `interrupt_before=["human_approval"]`.
- At the pause, the CLI calls `graph.get_state(config).values` to show `current_step` and `spec_context`.
- **Approve** (`y`): resumes with `None` input; human_approval is a passthrough, routes to rust_coder.
- **Reject** (any other text): calls `graph.update_state(config, values={"human_feedback": text})`, then resumes. Human_approval routes back to architect, which processes the feedback.

## LLM configuration (`orchestrator/nodes.py`)

- `configure_model(service, api_key, model=None)` sets a module-level `_model_config`.
- `ModelConfig` resolves per-service defaults:
  - deepseek → `deepseek-v4-flash`, api.deepseek.com, `ChatDeepSeek` with `extra_body={"thinking": {"type": "disabled"}}`
  - openai → `gpt-4o`, api.openai.com/v1, `ChatOpenAI`
  - gemini → `gemini-2.5-flash`, `ChatGoogleGenerativeAI` (needs `langchain-google-genai`)
- Structured output method:
  - deepseek → `json_mode` (avoids tool_choice conflict with thinking models). A `RunnableLambda` prepends `"Output JSON.\n"` to prompts because DeepSeek's json_object mode requires the literal word "json" in the prompt.
  - openai → `json_schema`
- `ArchitectOutput` and `DebuggerOutput` have `@model_validator(mode="before")` and `@field_validator(mode="before")` to handle LLM field-name variations (e.g. `remaining_plan` → `plan`, `first_step` → `current_step`) and type coercions.

## Feedback clearing

`clear_feedback()` resets `human_feedback`, `debugger_feedback`, `repair_target`, and `cargo_output` to empty strings. Called by `architect_node` at the start of every new iteration. Nodes that consume feedback (`rust_coder`, `test_writer`) also clear the fields they read.

## Checkpointer

`SqliteSaver` backed by `orchestrator/checkpoints.db`. Thread ID: `"main"`. Allows pause/resume across process restarts.

## CLI invocation

```
cd orchestrator
uv run python main.py -s deepseek -k sk-xxx
uv run python main.py -s openai  -k sk-xxx -m gpt-4.1
uv run python main.py -s deepseek -k sk-xxx -m deepseek-v4-pro
```

Environment variable fallbacks: `LLM_SERVICE`, `OPENAI_API_KEY`, `LLM_MODEL`.
