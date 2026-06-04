# LangGraph Orchestrator Overview

## Graph topology

```
START
  │
  ▼
architect ◄──────────────────────────────┐
  │                                       │
  ▼                                       │
spec_reader                               │
  │                                       │
  ▼                                       │
human_approval ───(feedback)──────────────┘
  │
  │ (approved)
  ▼
rust_coder ◄────────────────────┐
  │                              │
  ▼                              │
test_writer ◄────────┐           │
  │                  │           │
  ▼                  │           │
cargo_tool            │           │
  ├─ fail ────────────┤           │
  │                   ▼           │
  │              debugger         │
  │                │  │           │
  │           (code) (test)       │
  │              │     └──────────┘
  │              │          ▲
  │              └──────────┘
  │            (both flow back up)
  ├─ success+empty ─────────► END
  │
  └─ success+plan ──────────► architect
```

## State

State is a shared dictionary that flows through every node. Each node reads from it and returns a partial update — LangGraph merges the updates automatically.

```
plan:             List[str]           remaining implementation steps
current_step:     str                 the step currently being worked on
codebase:         Dict[str, str]      key=relative path under simulator/src, value=file content
                                       Uses an annotated reducer to merge partial updates (nodes
                                       only return files they touched, not the whole codebase)
spec_context:     str                 ARM64 spec information for the current step
cargo_output:     str                 stdout+stderr from last cargo test run
cargo_success:    bool                whether last cargo test passed
debugger_feedback: str                failure analysis from the debugger node
repair_target:    str                 "code" or "test", set by debugger
human_feedback:   str                 rejection feedback typed by the human
```

## Nodes

| Node | LLM? | Output | What it does |
|------|------|--------|--------------|
| `architect` | Yes | plan list + current step | Creates or revises the implementation plan and picks the next step. |
| `spec_reader` | Yes | ARM64 spec context string | Pulls ARM64 spec details relevant to the current step. |
| `human_approval` | No | (passthrough) | Pause point for human review; routes on approval or feedback. |
| `rust_coder` | Yes | file path → content map | Writes or patches Rust implementation files. |
| `test_writer` | Yes | file path → content map | Adds unit tests. |
| `cargo_tool` | No | cargo stdout/stderr + pass/fail | Writes files to disk, validates paths, and runs `cargo test`. |
| `debugger` | Yes | analysis + repair target | Diagnoses test failures and tags the fix target as code or test. |

## Human-in-the-loop

- The graph pauses before `human_approval` and shows the current step with its ARM64 spec context.
- **Approve** — type `y` to continue to the Rust coder.
- **Reject** — type any feedback; it's injected into state and the architect revises the plan.

## LLM configuration

Each agent gets its own DeepSeek model and thinking mode, set in the `AGENT_MODEL_DEFAULTS` dict. Edit that dict to swap models or toggle thinking per agent.

| Agent | Model | Thinking |
|-------|-------|----------|
| `architect` | `deepseek-v4-pro` | **enabled** |
| `debugger` | `deepseek-v4-pro` | **enabled** |
| `rust_coder` | `deepseek-v4-pro` | **disabled** |
| `spec_reader` | `deepseek-v4-flash` | disabled |
| `test_writer` | `deepseek-v4-flash` | disabled |

### Output model validators

`ArchitectOutput` and `DebuggerOutput` include alias-remapping and type-coercion validators to handle LLM field-name variations (e.g. `remaining_plan` → `plan`, `first_step` → `current_step`).

## Feedback clearing

At the start of every new iteration the architect resets `human_feedback`, `debugger_feedback`, `repair_target`, and `cargo_output` so stale data doesn't leak across iterations. Nodes that consume feedback also clear the fields they read.

## Checkpointer

SQLite-backed via `orchestrator/checkpoints.db`, thread ID `"main"`. Lets you pause and resume across process restarts.

## CLI invocation

```
cd orchestrator
uv run python main.py -s deepseek -k sk-xxx
uv run python main.py -s deepseek -k sk-xxx -m deepseek-v4-pro
```
