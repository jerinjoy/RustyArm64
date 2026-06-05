# Orchestrator: Graph Overview

The orchestrator (`orchestrator/main.py`) uses [LangGraph](https://langchain-ai.github.io/langgraph/)
to drive an LLM through a write → lint → test → advance loop until all tasks are done.

**LangGraph basics:** you define a graph of *nodes* (Python functions) connected by *edges*.
A shared *state* dict is passed between nodes — each node reads from it and returns a
partial update. Conditional edges let you branch based on state values.

---

## State

All nodes read from and write to a single `WorkflowState` dict:

| Key | Type | Purpose |
|---|---|---|
| `goal` | `str` | High-level objective (never changes) |
| `todo_tasks` | `List[str]` | Remaining tasks (treated as a queue) |
| `completed_tasks` | `List[str]` | Finished tasks (append-only) |
| `architecture_spec` | `str` | ISA description injected into every coder prompt |
| `test_results` | `str` | Raw output from the last `cargo test` run |
| `messages` | `List[BaseMessage]` | Conversation history passed to the LLM |
| `tests_passed` | `bool` | Set by `tester`; cleared by `queue_manager` |
| `compiler_logs` | `str` | Reserved — currently unused |

---

## Graph

```
START ──► coder ◄─────────────────────────────────┐
            │                                      │
    [tools_condition]                              │
      ┌─────┴──────┐                               │
  tool call    text reply                          │
      │             │                              │
      ▼             ▼                              │
   tools         tester                            │
      │             │                              │
      └──► coder  [test_evaluator]                 │
                ┌───┴────┐                         │
              fail      pass                       │
                │         │                        │
              coder   queue_manager                │
                        │                          │
                [queue_evaluator]                  │
                ┌────────┴───────┐                 │
            continue            done               │
                │                 │                │
                └─────────────────┘              END
```

---

## Nodes

**`coder`** — Sends the architecture spec, the current task, and any prior messages
(tool results, test failures) to DeepSeek Coder. The LLM either calls a tool or
returns a plain text reply.

**`tools`** — LangGraph's built-in `ToolNode`. Executes whichever tool the LLM called
(`write_rust_file`, `run_clippy`, or `add_rust_dependency`) and appends the result to
`messages`.

**`tester`** — Runs `cargo test` and updates `tests_passed` and `test_results`.

**`queue_manager`** — Moves the finished task from `todo_tasks` to `completed_tasks`,
then clears `messages` so the next task starts with a fresh context window.

---

## Tools available to the LLM

| Tool | What it does |
|---|---|
| `write_rust_file(filepath, content)` | Writes a file inside `../simulator/` |
| `run_clippy()` | Runs `cargo clippy` and returns warnings/errors |
| `add_rust_dependency(crate_name)` | Runs `cargo add <crate>` |

The coder's system prompt requires it to call `run_clippy` after every file write and fix
all warnings before declaring a task done.

---

## One task, end to end

```
coder → (tool call) → tools → coder   # write code, lint, repeat
coder → (text reply) → tester         # signal "done", run tests
  tester fail → coder                 # fix the code
  tester pass → queue_manager         # advance to next task
```
