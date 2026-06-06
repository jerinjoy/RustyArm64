# LangGraph Workflow

The orchestrator (`orchestrator/main.py`) uses LangGraph to coordinate an LLM-driven
code-and-test loop for building a Rust ARM64 simulator.

## State

The graph carries a shared `WorkflowState` dict with these keys:

| Key | Type | Purpose |
|---|---|---|
| `goal` | `str` | High-level objective (fixed) |
| `todo_tasks` | `List[str]` | Queue of tasks the LLM must implement |
| `completed_tasks` | `List[str]` | Tasks finished so far (reducer: append) |
| `architecture_spec` | `str` | System prompt describing the target ISA |
| `compiler_logs` | `str` | Clippy output (stored, not used in routing) |
| `test_results` | `str` | Latest cargo-test output |
| `messages` | `List[BaseMessage]` | LLM conversation (reducer: add_messages) |
| `tests_passed` | `bool` | Whether the last test run passed |

## Nodes

```
           ┌──────────┐
           │  coder   │  LLM writes Rust code (tools: write_rust_file, run_clippy,
           └────┬─────┘  add_rust_dependency). Decides when a task is complete.
                │
        tools_condition
        ┌───────┴───────┐
        ▼               ▼
   ┌─────────┐    ┌─────────┐
   │  tools  │    │ tester  │  Runs `cargo test`. Sets `tests_passed`.
   └────┬────┘    └────┬────┘
        │              │
        │         test_evaluator
        │         ┌──────┴──────┐
        │    "fail"▼        "pass"▼
        │   ┌─────────┐  ┌──────────────┐
        └──▶│  coder  │  │ queue_manager│  Pops the current task from
            └─────────┘  └──────┬───────┘  `todo_tasks`, clears messages.
                                │
                          queue_evaluator
                          ┌───────┴───────┐
                    "continue"▼       "done"▼
                     ┌─────────┐        END
                     │  coder  │
                     └─────────┘
```

### coder

The entry point. Sends the architecture spec and current task to an LLM (DeepSeek Coder).
Returns an LLM response, which may contain tool calls.

### tools

LangGraph `ToolNode` that executes `write_rust_file`, `run_clippy`, or `add_rust_dependency`.
Results are appended to `messages` and the graph loops back to `coder`.

### tester

Runs `cargo test` in the simulator directory. Sets `tests_passed` and appends the
raw test output to both `test_results` and `messages`.

### queue_manager

Called when tests pass. Pops the first task from `todo_tasks`, appends it to
`completed_tasks`, and clears the `messages` list so the LLM starts fresh on the
next task.

## Edges

| From | Condition | To | Why |
|---|---|---|---|
| `coder` | LLM requested tools | `tools` | Execute file writes / clippy |
| `coder` | LLM responded with text | `tester` | Task is done — validate with tests |
| `tools` | (always) | `coder` | Let LLM see tool results |
| `tester` | `tests_passed == false` | `coder` | Fix failing code |
| `tester` | `tests_passed == true` | `queue_manager` | Advance to next task |
| `queue_manager` | tasks remain | `coder` | Implement next task |
| `queue_manager` | no tasks remain | `END` | Workflow complete |

## Full Loop (one task)

```
coder ──tools──▶ coder ──tools──▶ coder ──(no tools)──▶ tester
                                                           │
                                              pass ──▶ queue_manager
                                              fail ──▶ coder (fix)
```

The coder–tools loop repeats until the LLM produces a text-only response (no tool calls),
which signals it believes the current task is complete.
