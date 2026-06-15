# Orchestrator: LangGraph Design

This orchestrator uses [LangGraph](https://langchain-ai.github.io/langgraph/) to semi-autonomously build an ARM64 simulator in Rust. You give it a high-level goal; the graph plans, codes, and tests — pausing for your review before execution begins.

---

## Mental model

A **graph** in LangGraph is a set of **nodes** (Python functions) connected by **edges** (routing decisions). Every node reads from and writes to a shared **state** dict. LangGraph snapshots state after every node, so a run can be resumed from the last completed node after any crash.

This graph runs in two phases:

1. **Planning** — the LLM proposes a multi-step plan; you review it and either approve or give feedback. The graph loops until you approve.
2. **Execution** — the graph works through tasks one at a time: code → test → retry or advance.

---

## Graph

```
START ──► planner ◄──┐
             │       │ (feedback loop)
   [plan_evaluator]  │
       (approved) ───┘
             │
             ▼
          coder ◄───────────────────────────────────────┐
             │                                           │
       [coder_router]                                    │
        ┌────┴─────────────────────────┐                 │
   tool calls                    no more tool calls      │
        │                             │                  │
        ▼                             ▼                  │
  coder_tools ──► coder            tester                │
                              [test_evaluator]            │
                           ┌──────┬──────────┐           │
                         pass  too many     fail ─────────┘
                           │    retries
                           ▼       │
                     queue_manager ▼
                           │    give_up ──► END
                   [queue_evaluator]
                    ┌──────┴──────┐
               tasks left     all done
                    │              │
                  coder           END
```

### Routing

| Router | Logic |
|---|---|
| `plan_evaluator` | `plan_approved` → `coder`; else → `planner` (self-loop) |
| `coder_router` | last message has tool calls AND under budget → `coder_tools`; else → `tester` |
| `test_evaluator` | pass → `queue_manager`; `retry_count >= MAX_TEST_RETRIES` → `give_up`; else → `coder` |
| `queue_evaluator` | tasks remaining → `coder`; empty → `END` |

`coder_tools → coder` is an **unconditional** edge — declared with `workflow.add_edge`, not `add_conditional_edges`.

---

## State

All nodes share a single `WorkflowState` TypedDict. LangGraph merges each node's return dict into state using **reducers** — functions that control how values combine rather than simply replace.

| Field | Type | Reducer | Purpose |
|---|---|---|---|
| `goal` | `str` | last-write-wins | Top-level objective; set at startup, never changes |
| `plan_draft` | `Optional[str]` | last-write-wins | Current draft surfaced to the human; `None` after approval |
| `plan_approved` | `bool` | last-write-wins | Gates entry to Phase 2 |
| `plan_feedback` | `Optional[str]` | last-write-wins | Human correction text; cleared before each LLM call |
| `todo_tasks` | `List[str]` | last-write-wins | Remaining step IDs; planner must always return the full list |
| `completed_tasks` | `List[str]` | `operator.add` | Finished step IDs; each write *appends* rather than replaces |
| `step_plans` | `Dict[str, str]` | last-write-wins | Step ID → YAML content; planner must always return the full dict |
| `architecture_spec` | `str` | last-write-wins | ISA description injected into every LLM prompt; set at startup |
| `messages` | `List[BaseMessage]` | `add_messages` | Coder conversation history; wiped between tasks |
| `test_results` | `str` | last-write-wins | Raw `cargo test` output; injected into coder's system prompt on failure |
| `tests_passed` | `bool` | last-write-wins | Set by `tester_node`; cleared by `queue_manager_node` |
| `retry_count` | `int` | last-write-wins | Test failure count for the current task |
| `tool_call_count` | `int` | last-write-wins | Tool calls used in the current task; enforces the per-node budget |

**Two fields use non-default reducers:**

- `completed_tasks` uses `operator.add` — each node's return is appended, not replaced. This is how LangGraph accumulates a list across multiple nodes without one write clobbering another.
- `messages` uses `add_messages` — LangGraph's built-in that merges by ID and handles `RemoveMessage` tombstones. `queue_manager_node` wipes the conversation between tasks with `[RemoveMessage(id=m.id) for m in state["messages"]]`.

---

## Nodes

### `planner_node`

The only node that pauses for human input. It:

1. Calls the LLM with `goal`, `architecture_spec`, any prior `plan_draft`, and any `plan_feedback`.
2. Calls `interrupt(revised_draft)` — this suspends the graph and surfaces the draft to the caller. The **return value** of `interrupt()` is whatever the human passes via `Command(resume=...)`.
3. On `"APPROVED"`: parses the draft YAML into `step_plans` and `todo_tasks`, sets `plan_approved = True`.
4. On any other input: stores it in `plan_feedback`. `plan_evaluator` routes back to `planner` for another revision.

One `interrupt()` call per node execution is the key constraint — calling it multiple times causes LangGraph to replay prior LLM calls on re-entry.

### `coder_node`

Reads `todo_tasks[0]` and its YAML plan from `step_plans`, then invokes the LLM with tools bound. The system prompt is rebuilt on every call: it includes the static `base_system_prompt` from config, `architecture_spec`, the current step plan, and — on failure — the last `test_results`. This keeps failure context out of the message history and in the system prompt where it belongs.

On the first call for a task, seeds `messages` with `HumanMessage("Begin the current task.")`. On subsequent calls, passes the full history so the LLM sees its prior tool use.

### `coder_tools`

LangGraph's built-in `ToolNode`. Executes the tool the coder requested and appends the result to `messages`. The unconditional edge back to `coder` means the LLM always sees the tool result before deciding its next action.

Available tools: `read_rust_file`, `write_rust_file`, `run_clippy`, `add_rust_dependency`.

### `tester_node`

Runs `cargo test --color=never`. Writes `test_results` and sets `tests_passed`. On failure, increments `retry_count`. Does not touch `messages` — failure context reaches the coder via the system prompt on the next call.

### `queue_manager_node`

Advances the queue: moves `todo_tasks[0]` to `completed_tasks`, resets counters, wipes `messages`, and runs `git commit`. The commit is guarded by `git status --porcelain` — if the working tree is already clean the commit already happened (LangGraph gives at-least-once execution), so the node skips it safely.

### `give_up_node`

Fires when `retry_count >= MAX_TEST_RETRIES`. Prints the step's YAML plan, last test output, and `git diff`, then routes to `END`. The run must be restarted manually after inspecting the output.

---

## Configuration

Model selection, prompts, and node settings live in `orchestrator/config.toml` rather than in the script.

```toml
[llms.deepseek-v4-pro]
model       = "deepseek-v4-pro"
api_key_env = "DEEPSEEK_API_KEY"
base_url    = "https://api.deepseek.com/v1"

[nodes.planner]
llm           = "deepseek-v4-pro"
system_prompt = "..."

[nodes.coder]
llm                = "deepseek-v4-pro"
max_tool_calls     = 20
base_system_prompt = "..."
```

`load_config()` builds an `llm_pool` dict and validates that every `llm` reference in `[nodes.*]` exists in `[llms.*]`, failing fast before any graph work begins.

---

## Persistence & Resumption

The graph is compiled with a **SqliteSaver checkpointer**:

```python
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    app = workflow.compile(checkpointer=checkpointer)
```

LangGraph writes a checkpoint after every node. If the process crashes, restarting with the same `thread_id` resumes from the last completed node.

The thread ID is derived from the goal string so the same goal always resumes the same run:

```python
slug = hashlib.sha1(" ".join(goal.split()).encode()).hexdigest()[:8]
thread_id = f"arm64-sim-{slug}"
```

---

## Human-in-the-Loop

`interrupt()` inside `planner_node` suspends the graph. The driver loop detects this via the `"__interrupt__"` key in the stream and collects human input:

```python
stream_input = initial_state

while True:
    interrupted = False
    for chunk in app.stream(stream_input, config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            interrupted = True
            interrupt_payload = chunk["__interrupt__"][0].value  # the plan draft

    if not interrupted:
        break  # graph reached END

    print(interrupt_payload)
    user_response = input("Response: ")           # "APPROVED" or feedback text
    stream_input = Command(resume=user_response)  # resume the suspended node
```

---

## Step Plan Schema

Each step the planner produces is a YAML document. The schema is prescriptive by design — the more the planner specifies, the less the coder hallucinates.

```yaml
id: "step_03_elf_loader"
title: "Implement ELF loader"
depends_on: ["step_01_cpu", "step_02_memory"]

deliverables:
  files:
    - path: "src/elf_loader.rs"
      action: create              # create | modify
    - path: "src/lib.rs"
      action: modify
      expected_additions:        # exact lines to add (required for modify)
        - "pub mod elf_loader;"

  types:
    - name: "ElfLoadError"
      kind: enum
      variants:
        - name: "InvalidMagic"
          description: "File does not start with ELF magic bytes."

  functions:
    - name: "ElfLoader::load"
      signature: "pub fn load(bytes: &[u8], cpu: &mut Cpu) -> Result<(), ElfLoadError>"
      constraints:
        - "Use the `goblin` crate — do not hand-roll ELF parsing."

interface_contracts:
  imports: ["Cpu from crate::cpu"]
  exports:  ["ElfLoader::load", "ElfLoadError — must implement Debug"]

cargo_dependencies: []

tests:
  - name: "test_load_valid_elf"
    description: "Build a minimal AArch64 ELF, call load, assert pc == entry point."
  - name: "test_load_invalid_magic"
    description: "Pass 16 zero bytes, assert Err(ElfLoadError::InvalidMagic)."

acceptance_criteria:
  - "cargo clippy passes with no warnings"
  - "All tests pass"
```

---

## Running

```
python main.py "<goal>"           # fresh run, or resume if a checkpoint exists
python main.py "<goal>" --reset   # discard the existing checkpoint and start fresh
```

`--reset` deletes rows for that thread from `checkpoints.db`. Other threads are unaffected.
