# LangGraph Orchestrator Overview

## Graph topology

```
START
  │
  ▼
architect ──(empty plan)──► END
  │
  │ (pending plan)
  ▼
spec_reader
  │
  ▼
[INTERRUPT ①] human_plan_approval ◄─────────────┐
  │                                               │
  │ (approved)                                    │
  ▼                                               │
rust_coder ◄──────────────────────┐               │
  │                                │               │
  ▼                                │               │
test_writer ◄──────────┐           │               │
  │                    │           │               │
  ▼                    │           │               │
cargo_tool              │           │               │
  ├─ fail ──────────────┤           │               │
  │                     ▼           │               │
  │                 debugger        │               │
  │                   │  │          │               │
  │              (code)  (test)     │               │
  │                 │     └─────────┘               │
  │                 │          ▲                    │
  │                 └──────────┘                    │
  │            (both flow back up)                  │
  │                                                 │
  ├─ pass + pending plan ──►                       │
  │                     [INTERRUPT ②]              │
  │                 human_test_approval ◄───────────┘
  │                      │ (rejected — feedback)
  │                      │
  │                      │ (approved)
  │                      ▼
  │                 progress_writer
  │                      │
  │                      ▼
  │                 committer
  │                      │
  └──────────────────────┘ (back to architect)
```

Two interrupt gates:

1. **Plan gate** (after `spec_reader`) — review the planned step and ARM64
   spec context before any code is written.
2. **Test gate** (after `cargo_tool` passes) — inspect the test results
   before the step is committed as done.

## State

State is a shared dictionary that flows through every node. Each node
reads from it and returns a partial update — LangGraph merges the updates
automatically.

```
plan:              List[str]           remaining implementation steps
current_step:      str                 the step currently being worked on
completed_steps:   List[str]           steps that have passed test approval
codebase:          Dict[str, str]      key=relative path under simulator/src, value=file content
                                       Uses an annotated reducer to merge partial updates
spec_context:      str                 ARM64 spec information for the current step
cargo_output:      str                 stdout+stderr from last cargo test run
cargo_success:     bool                whether last cargo test passed
debugger_feedback: str                 failure analysis from the debugger node
repair_target:     str                 "code" or "test", set by debugger
human_feedback:    str                 rejection feedback typed by the human
```

## Nodes

| Node | LLM? | Output | What it does |
|------|------|--------|--------------|
| `architect` | Yes | plan list + current step | Creates/revises the implementation plan. On restart reads existing codebase and plans from there. |
| `spec_reader` | Yes | spec_context string | Loads the relevant ARM64 MRA XML file, extracts bit-field definitions and pseudocode via the LLM. |
| `human_plan_approval` | No | (passthrough) | Pause point ① — human reviews plan + spec before coding begins. |
| `rust_coder` | Yes | file path → content map | Writes or patches Rust implementation files. |
| `test_writer` | Yes | file path → content map | Adds inline unit tests to existing source files. |
| `cargo_tool` | No | cargo stdout/stderr + pass/fail | Writes files to disk, validates paths, runs `cargo test`. |
| `debugger` | Yes | analysis + repair target | Diagnoses test failures and tags the fix target as code or test. |
| `human_test_approval` | No | (passthrough) | Pause point ② — human inspects passing test output before committing. |
| `progress_writer` | No | updates progress.yaml | Writes completed steps + pending plan to `progress.yaml`. Runs *before* committer so both land in one commit. |
| `committer` | No | git commit | `git add -A simulator/src/ progress.yaml` then `git commit`. |

## Cross-machine persistence

The source of truth for progress is the filesystem, not the checkpoint DB:

- **`progress.yaml`** (committed) — tracks completed steps, pending plan,
  and current step in human-readable YAML.
- **`simulator/src/`** (committed) — the actual generated Rust code.
- **`checkpoints.db`** (gitignored) — ephemeral LangGraph state for the
  current session only (in-progress iteration state, which interrupt gate
  you're paused at, etc.).

On startup, `_initial_state()` checks for an existing checkpoint:
- **Warm resume** — checkpoint exists, graph resumes from saved state.
- **Cold start** — no checkpoint; hydrates `plan`, `current_step`,
  `completed_steps` from `progress.yaml` and `codebase` from
  `simulator/src/`.

## Routing

| From | Condition | To |
|------|-----------|----|
| `architect` | plan is empty | `END` |
| `architect` | plan is non-empty | `spec_reader` |
| `human_plan_approval` | feedback present | `architect` |
| `human_plan_approval` | approved | `rust_coder` |
| `cargo_tool` | passed + pending plan | `human_test_approval` |
| `cargo_tool` | passed + empty plan | `END` |
| `cargo_tool` | failed | `debugger` |
| `human_test_approval` | feedback present | `architect` |
| `human_test_approval` | approved | `progress_writer` |
| `debugger` | repair_target = "test" | `test_writer` |
| `debugger` | repair_target = "code" | `rust_coder` |

## Human-in-the-loop

- **Gate ① (plan):** The graph pauses before `human_plan_approval` and shows
  the current step with its ARM64 spec context.
- **Gate ② (test):** The graph pauses before `human_test_approval` and shows
  the cargo test output.
- At either gate:
  - **Approve** — type `y` to continue.
  - **Reject** — type any feedback; it's injected into state and the
    architect revises the plan.

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

SQLite-backed via `orchestrator/checkpoints.db`, thread ID `"main"`. Lets you pause and resume within a single session. Not committed to git.

## CLI invocation

```bash
cd orchestrator
uv run python main.py -s deepseek -k sk-xxx --spec-dir /path/to/arm64_specs
uv run python main.py -s deepseek -k sk-xxx -m deepseek-v4-pro --spec-dir /path/to/arm64_specs
```

Or set `ARM64_SPEC_DIR` in your environment to skip `--spec-dir` on every invocation.

## Spec index builders

Before running the orchestrator, build the instruction and system register
indexes once:

```bash
ARM64_SPEC_DIR=/path/to/arm64_specs uv run python build_instruction_index.py
ARM64_SPEC_DIR=/path/to/arm64_specs uv run python build_sysreg_index.py
```

These produce `instruction_index.json` and `sysreg_index.json` in the
`orchestrator/` directory. Both use glob patterns (`*_xml_A_profile-*`)
to auto-discover the latest spec version directory.
