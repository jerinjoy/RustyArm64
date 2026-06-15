# Orchestrator: LangGraph Design

The orchestrator (`orchestrator/main.py`) uses [LangGraph](https://langchain-ai.github.io/langgraph/)
to semi-autonomously build the ARM64 functional simulator in Rust. A human provides a high-level
goal; the system plans, codes, tests, and iterates — pausing for human review at key decision points.

---

## Phases

The system operates in two sequential phases within a single graph run:

**Phase 1 — Planning**
The planner node takes the goal, produces a structured multi-step plan, and enters a
human-in-the-loop review loop. The human iterates with the planner (providing feedback,
corrections, additional context) until the plan is approved. On approval the plan is committed
to state and Phase 2 begins.

**Phase 2 — Execution**
The graph works through the approved task queue one step at a time. Each step is fully
specified by a YAML plan (see [Step Plan Schema](#step-plan-schema)) that constrains the
coder and minimises hallucination. After coding, tests are run. On failure the coder retries
up to `MAX_TEST_RETRIES` times; if still failing, the run terminates with a structured failure summary for manual inspection.

---

## Constants

```python
MAX_TEST_RETRIES = 5   # max cargo test failures per task before terminating
MAX_TOOL_CALLS = 20  # max coder→coder_tools cycles per task before forcing evaluation
```

---

## State

All nodes read from and write to a single `WorkflowState` dict. The checkpointer snapshots
this state after every node so the run can be resumed after any failure.

| Key | Type | Purpose |
|---|---|---|
| `goal` | `str` | Top-level objective provided at invocation; never changes |
| `plan_draft` | `Optional[str]` | Current plan draft held between planner interrupt rounds |
| `plan_approved` | `bool` | Set to `True` by `planner_node` on approval; gates entry to Phase 2 |
| `plan_feedback` | `Optional[str]` | Human feedback from the last planner interrupt; written by `planner_node` on the feedback path, read and cleared on the next re-entry before the LLM call |
| `todo_tasks` | `List[str]` | Step IDs remaining in the queue (last-write-wins — planner must always return the **full** list, not a partial update) |
| `completed_tasks` | `List[str]` | Step IDs finished (append-only, reducer: `operator.add`) |
| `step_plans` | `Dict[str, str]` | Maps step ID → raw YAML plan content (last-write-wins — planner must always return the **full** dict, not a partial update) |
| `architecture_spec` | `str` | ISA description injected into every coder and planner prompt; provided at invocation time alongside `goal` and never modified by the graph |
| `test_results` | `str` | Raw output from the last `cargo test` run |
| `messages` | `List[BaseMessage]` | Conversation history for the current task (reducer: `add_messages`) |
| `tests_passed` | `bool` | Set by `tester_node`; cleared by `queue_manager_node` |
| `retry_count` | `int` | Test failure count for the current task (last-write-wins, no reducer) |
| `tool_call_count` | `int` | Tool calls made during the current task (last-write-wins, no reducer) |

> **`messages` lifecycle:** On a fresh task the list is empty. `coder_node` seeds it with a
> `HumanMessage` so the full conversation is visible in checkpoints. `queue_manager_node` wipes
> it with `RemoveMessage` before advancing on a passing task.
>
> **`RemoveMessage` pattern:** The wipe must use
> `[RemoveMessage(id=m.id) for m in state.get("messages", [])]`. LangGraph initialises all
> TypedDict state fields to their zero values before any node runs, so `state["messages"]` is
> always `[]` and never raises a `KeyError`; the `.get()` form is kept as defensive style.
> Returning an empty list from the comprehension is a no-op. All messages that have passed
> through the `add_messages` reducer are guaranteed to have an ID (the reducer assigns a UUID
> if one is absent), so `m.id` is always safe to access.

> **Reset responsibilities:** `queue_manager_node` resets `retry_count`, `tool_call_count`,
> and `tests_passed` on a passing task. `planner_node` also resets `retry_count` and
> `tool_call_count` on plan approval so that Phase 2 starts with a clean budget.

---

## Nodes

### `planner_node` *(planned)*

Reads `goal` and `architecture_spec` from state and uses an LLM to produce a structured
multi-step plan. On each execution the node:

1. Reads `plan_draft` and `plan_feedback` from state (`plan_feedback` is `None` on the
   first call; human correction text on re-entry after feedback).
2. Calls the LLM with `goal`, `architecture_spec`, the prior `plan_draft`, and any
   `plan_feedback` to produce a revised draft; clears `plan_feedback` to `None`.
3. Writes the revised draft to `plan_draft` in state.
4. Calls `interrupt(plan_draft)` **once** to surface the draft to the human;
   the **return value** of `interrupt()` is the human's resume input — whatever
   was passed as `Command(resume=...)` when the graph was resumed.
5. On resume, checks the human's response:
   - **If the response is `"APPROVED"` (case-insensitive):** writes the finalised
     `todo_tasks` and `step_plans` to state, sets `plan_approved = True`, clears
     `plan_draft` to `None`, and resets `retry_count` and `tool_call_count` to 0.
   - **Otherwise (any other string):** treats the response as feedback — writes it to
     `plan_feedback` in state and returns. `plan_evaluator` routes the node back to itself.

Calling `interrupt()` exactly once per node execution avoids LangGraph's replay behaviour,
where re-entering a node replays all LLM calls made before previous `interrupt()` calls.

### `coder_node`

Reads `todo_tasks[0]` to identify the current step, then looks up its YAML plan in
`step_plans`. Sends that plan, `architecture_spec`, and the conversation history to the LLM.
The LLM either calls a tool or returns a plain-text reply signalling it is done. Increments
`tool_call_count` on each tool-calling response.

The system prompt is rebuilt from state on every call via `_build_system_prompt()`, which
includes `architecture_spec` and appends `test_results` when tests have not yet passed — so
the coder always has the ISA constraints and failure context without them being injected as
message roles.

### `coder_tools`

LangGraph's built-in `ToolNode`. Executes whichever tool the coder requested and appends the
result to `messages`. Has an unconditional return edge to `coder` — declare this explicitly
with `workflow.add_edge("coder_tools", "coder")` in the graph definition (it does not appear
in the routing table because it is not conditional).

### `tester_node`

Runs `cargo test --color=never`. On every execution writes `test_results` with the raw cargo
output and sets `tests_passed = True` if the exit code is 0, `False` otherwise. On failure,
also increments `retry_count`. Does not write to `messages`; failure context reaches the coder
via `test_results` in the system prompt.

### `queue_manager_node`

Moves the completed step from `todo_tasks` to `completed_tasks`, resets `retry_count`,
`tool_call_count`, and `tests_passed` to their zero values, and wipes `messages` via
`RemoveMessage`. Performs a `git commit` of the simulator code so each completed step has a
clean recovery point.

> **Idempotency:** LangGraph gives at-least-once execution — if the SQLite checkpoint write
> fails after the node's Python logic completes, the node re-runs on restart. The git commit
> must be guarded:
> ```
> if git status --porcelain returns nothing → working tree already clean → skip commit
> else → git add -A && git commit -m "complete {step_id}: {title}"
> ```
> `step_id` and `title` are read from the completed step's YAML plan. This makes the node
> safe to re-run: a clean tree means the commit already happened, so the node skips the
> commit and advances the queue.

### `give_up_node`

Fires when `retry_count >= MAX_TEST_RETRIES`. Reads `todo_tasks[0]` to identify the current step,
then prints a structured failure summary — that step's YAML plan (from `step_plans`), the
full `test_results`, and `git diff` of what the coder wrote — then routes to `END`. The run
must be restarted manually after inspecting the output.

---

## Graph

```
START ──► planner ◄──┐
             │       │(feedback — write plan_feedback, re-enter)
    [plan_evaluator]─┘
         (APPROVED)
             │
             ▼
          coder ◄────────────────────────────────────────────┐
             │                                               │
       [coder_router]                                        │
        ┌────┴──────────────────────────┐                    │
  tool_calls present             no tool_calls               │
  AND count < MAX_TOOL_CALLS     OR count >= MAX_TOOL_CALLS  │
        │                        (forced evaluation)         │
        ▼                               │                    │
  coder_tools                           ▼                    │
  │ (unconditional)                  tester                  │
  └──────────────────► coder    [test_evaluator]             │
                          (evaluated in order: 1→2→3)        │
                   ┌──────────┬──────────────────┐           │
                  (1)        (2)                 (3)          │
                 pass    retry_count >= MAX      fail         │
                   │       (give_up)               │          │
                   ▼           │                   └──────────┘
            queue_manager      ▼
                   │         give_up
           [queue_evaluator]   │
            ┌──────┴──────┐   END
         continue        done
            │              │
           coder           END
```

---

## Routing Functions

| Function | Source node | Logic |
|---|---|---|
| `plan_evaluator` | `planner` | `plan_approved` → `"coder"`, else `"planner"` (self-loop) |
| `coder_router` | `coder` | (1) `tool_call_count >= MAX_TOOL_CALLS` → `"tester"`, (2) `last.tool_calls` → `"coder_tools"`, (3) else → `"tester"` (evaluated in this order) |
| *(unconditional)* | `coder_tools` | always → `"coder"` (use `workflow.add_edge`, not `add_conditional_edges`) |
| `test_evaluator` | `tester` | (1) `tests_passed` → `"queue_manager"`, (2) `retry_count >= MAX_TEST_RETRIES` → `"give_up"`, (3) else → `"coder"` (evaluated in this order) |
| `queue_evaluator` | `queue_manager` | `todo_tasks` non-empty → `"coder"`, else `END` |

---

## Tools Available to the Coder

| Tool | What it does |
|---|---|
| `read_rust_file(filepath)` | Returns the current contents of a file inside `../simulator/` |
| `write_rust_file(filepath, content)` | Writes a file inside `../simulator/` |
| `run_clippy()` | Runs `cargo clippy` and returns warnings/errors |
| `add_rust_dependency(crate_name)` | Runs `cargo add <crate>` |

The coder's system prompt requires it to:
1. Call `read_rust_file` before any `action: modify` operation — never reconstruct an existing file from memory.
2. Call `run_clippy` after every `write_rust_file` and fix all warnings before signalling done.

---

## Step Plan Schema

Each entry in `step_plans` is a YAML string with the following structure. The schema is
intentionally prescriptive to minimise coder hallucination.

```yaml
id: "step_03_elf_loader"
title: "Implement ELF loader"
depends_on:
  - "step_01_cpu"
  - "step_02_memory"

deliverables:
  files:
    - path: "src/elf_loader.rs"
      action: create          # create | modify
    - path: "src/lib.rs"
      action: modify
      expected_additions:     # required for action: modify; list exact lines to add
        - "pub mod elf_loader;"
        - "pub use crate::elf_loader::{ElfLoader, ElfLoadError};"

  types:
    - name: "ElfLoader"
      kind: struct
      module: "crate::elf_loader"
      fields:                   # required for structs; omit for enums
        - name: "data"
          type: "&'a [u8]"
          visibility: pub
    - name: "ElfLoadError"
      kind: enum
      module: "crate::elf_loader"
      variants:
        - name: "InvalidMagic"
          description: "File does not start with ELF magic bytes."
        - name: "UnsupportedArch"
          description: "ELF target is not AArch64."
        - name: "SegmentOutOfBounds"
          description: "A LOAD segment exceeds the Memory address space."

  functions:
    - name: "ElfLoader::load"
      module: "crate::elf_loader"
      signature: "pub fn load(bytes: &[u8], cpu: &mut Cpu) -> Result<(), ElfLoadError>"
      description: >
        Parse the ELF header, verify magic and AArch64 machine type,
        iterate LOAD segments and copy them into cpu.memory, then set
        cpu.pc to the ELF entry point.
      constraints:
        - "Use the `goblin` crate — do not hand-roll ELF parsing."
        - "Segments that map to address 0x0 are valid; do not treat as errors."

interface_contracts:
  imports:
    - "Cpu from crate::cpu — use cpu.memory and cpu.pc directly"
    - "Memory from crate::memory — assumed to be [u8; 65536]"
  exports:
    - "ElfLoader::load — called by main.rs to bootstrap the simulator"
    - "ElfLoadError — must implement Debug"

cargo_dependencies: []

tests:
  - name: "test_load_valid_elf"
    description: >
      Build a minimal AArch64 ELF binary in the test, call ElfLoader::load,
      assert cpu.pc equals the entry point and memory contains the segment bytes.
  - name: "test_load_invalid_magic"
    description: "Pass 16 zero bytes, assert Err(ElfLoadError::InvalidMagic)."
  - name: "test_load_wrong_arch"
    description: "Pass a valid x86-64 ELF, assert Err(ElfLoadError::UnsupportedArch)."

acceptance_criteria:
  - "cargo clippy passes with no warnings"
  - "All tests pass"
  - "ElfLoader::load correctly sets cpu.pc to the ELF entry point"
```

---

## Persistence

The graph is compiled with a **SqliteSaver checkpointer**. LangGraph writes a checkpoint after
every node completes, so a restart with the same `thread_id` resumes from the last completed
node rather than from scratch.

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    app = workflow.compile(checkpointer=checkpointer)
    # run the driver loop inside this block so the connection stays open
```

Step plan YAMLs are stored in `WorkflowState.step_plans` (a plain dict) rather than in a
separate LangGraph Store. This keeps everything in one persistence backend and means plan
content is automatically captured in every checkpoint.

### Thread ID strategy

The `thread_id` is the persistent cursor into the checkpoint DB. Use a stable, human-readable
ID per project run:

```python
config = {"configurable": {"thread_id": "arm64-sim-mvp-run-1"}}
```

Reusing the same ID resumes the existing run. Using a new ID starts fresh.

### Initial state

`goal` and `architecture_spec` are the only fields the caller must supply. Everything else
starts at its zero value and is populated by the graph.

```python
initial_state = {
    "goal": "Build an MVP ARM64 functional simulator that can load a bare-metal ELF, "
            "execute a few arithmetic instructions, and stop on a halt instruction.",
    "architecture_spec": (
        "Target: ARMv8-A AArch64 (64-bit execution state). "
        "Registers: 31 general-purpose 64-bit registers (X0–X30), SP, PC, PSTATE. "
        "Memory model: flat, byte-addressable. "
        "Instruction encoding: fixed 32-bit little-endian words. "
        "Relevant instructions for MVP: ADD, SUB, MOV (wide immediate), LDR, STR, B, BL, RET, HLT."
    ),
    # zero values — populated by the graph
    "plan_draft": None,
    "plan_approved": False,
    "plan_feedback": None,
    "todo_tasks": [],
    "completed_tasks": [],
    "step_plans": {},
    "test_results": "",
    "messages": [],
    "tests_passed": False,
    "retry_count": 0,
    "tool_call_count": 0,
}
```

---

## Human-in-the-Loop Interactions

The graph pauses using `interrupt()` inside `planner_node` for plan approval:

The planner generates a draft, writes it to `plan_draft`, then calls `interrupt()` once to
surface it. The human responds with feedback or the approval signal `"APPROVED"` (case-
insensitive). Any other string is treated as feedback — written to `plan_feedback` and fed
into the next LLM call. `plan_evaluator` routes back to `planner` on feedback or forwards to
`coder` on approval.

### Driver loop

`app.stream()` with `stream_mode="updates"` is the correct method for HITL interrupt
detection. When a node calls `interrupt()`, the stream yields a chunk whose key is
`"__interrupt__"`, then the stream ends. To resume, pass `Command(resume=...)` as the
next input to `app.stream()`.

```python
from langgraph.types import Command

stream_input = initial_state

while True:
    interrupted = False
    interrupt_payload = None

    for chunk in app.stream(stream_input, config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            interrupted = True
            interrupt_payload = chunk["__interrupt__"][0].value

    if not interrupted:
        break  # graph reached END

    print(interrupt_payload)                  # show plan draft or failure summary
    user_response = input("Response: ")       # collect feedback or approval
    stream_input = Command(resume=user_response)
```

---

## Implementation Status

All components are implemented.

| Component | Status |
|---|---|
| `coder_node` | Implemented |
| `coder_tools` (`read_rust_file`, `write_rust_file`, `run_clippy`, `add_rust_dependency`) | Implemented |
| `tester_node` | Implemented |
| `queue_manager_node` (message wipe, git commit with idempotency guard) | Implemented |
| `give_up_node` (step YAML, test output, git diff) | Implemented |
| `test_evaluator` with `MAX_TEST_RETRIES` guard | Implemented |
| `coder_router` with `MAX_TOOL_CALLS` guard | Implemented |
| `SqliteSaver` checkpointer | Implemented |
| `planner_node` with single-interrupt HITL loop | Implemented |
| `plan_evaluator` router (self-loop until approved) | Implemented |
| `plan_draft` / `plan_approved` / `plan_feedback` / `tool_call_count` / `step_plans` in state | Implemented |
| Driver loop with `app.stream()` and `Command(resume=...)` | Implemented |
