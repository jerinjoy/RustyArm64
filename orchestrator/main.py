import argparse
import hashlib
import operator
import os
import sqlite3
import subprocess
import sys
import tomllib
import yaml
from pathlib import Path
from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

SIMULATOR_DIR = "../simulator"
MAX_TEST_RETRIES = 5
DB_PATH = "checkpoints.db"


# ==========================================
# 1. CLI HELPERS
# ==========================================
def make_thread_id(goal: str) -> str:
    """Derive a stable thread ID from the goal so each distinct goal gets its own checkpoint."""
    normalized = " ".join(goal.split())
    slug = hashlib.sha1(normalized.encode()).hexdigest()[:8]
    return f"arm64-sim-{slug}"


def reset_thread(thread_id: str) -> None:
    """Delete all checkpoint rows for thread_id from the SQLite DB."""
    if not os.path.exists(DB_PATH):
        return
    with sqlite3.connect(DB_PATH) as conn:
        for table in ("checkpoints", "writes"):
            try:
                conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
            except sqlite3.OperationalError:
                pass  # table doesn't exist yet


# ==========================================
# 2. CONFIG
# ==========================================
def load_config(path: Path = Path(__file__).parent / "config.toml"):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    llm_pool: dict[str, ChatOpenAI] = {}
    for name, params in cfg.get("llms", {}).items():
        llm_pool[name] = ChatOpenAI(
            model=params["model"],
            api_key=os.getenv(params["api_key_env"]),
            base_url=params["base_url"],
        )

    for node_name, node_cfg in cfg.get("nodes", {}).items():
        ref = node_cfg.get("llm")
        if ref and ref not in llm_pool:
            raise ValueError(f"Node '{node_name}' references unknown LLM '{ref}'")

    return cfg, llm_pool


def resolve_node(name: str, cfg: dict, llm_pool: dict[str, ChatOpenAI]) -> dict:
    """Return node config with the llm key resolved to its ChatOpenAI object and strings stripped."""
    node_cfg = {
        k: (v.strip() if isinstance(v, str) else v)
        for k, v in cfg["nodes"][name].items()
    }
    if "llm" in node_cfg:
        node_cfg["llm"] = llm_pool[node_cfg["llm"]]
    return node_cfg


# ==========================================
# 3. STATE DEFINITION
# ==========================================
class WorkflowState(TypedDict):
    goal: str
    plan_draft: Optional[str]
    plan_approved: bool
    plan_feedback: Optional[str]
    todo_tasks: List[str]
    completed_tasks: Annotated[List[str], operator.add]
    step_plans: Dict[str, str]
    architecture_spec: str
    test_results: str
    messages: Annotated[List[BaseMessage], add_messages]
    tests_passed: bool
    retry_count: int
    tool_call_count: int


# ==========================================
# 4. TOOLS
# ==========================================
@tool
def read_rust_file(filepath: str) -> str:
    """Reads and returns the contents of a file relative to the Rust project root."""
    full_path = os.path.join(SIMULATOR_DIR, filepath)
    try:
        with open(full_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {full_path}. Use write_rust_file to create it first."


@tool
def write_rust_file(filepath: str, content: str) -> str:
    """Writes Rust code to the specified file path relative to the Rust project root."""
    full_path = os.path.join(SIMULATOR_DIR, filepath)
    os.makedirs(
        os.path.dirname(full_path) if os.path.dirname(full_path) else ".", exist_ok=True
    )
    try:
        with open(full_path, "w") as f:
            f.write(content)
    except OSError as e:
        return f"Failed to write {full_path}: {e}"
    return f"Successfully wrote to {full_path}"


@tool
def run_clippy() -> str:
    """Runs `cargo clippy` to lint the current Rust project."""
    try:
        result = subprocess.run(
            ["cargo", "clippy", "--message-format=short"],
            cwd=SIMULATOR_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return "Clippy passed with no issues."
        return f"Clippy warnings/errors:\n{result.stderr}\n{result.stdout}"
    except Exception as e:
        return f"Failed to run clippy: {e}"


@tool
def run_tests() -> str:
    """Runs `cargo test` and returns the full output. Use this to verify your implementation
    is correct — clippy only checks compilation and linting, not test results."""
    try:
        result = subprocess.run(
            ["cargo", "test", "--color=never"],
            cwd=SIMULATOR_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        combined = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        if result.returncode == 0:
            return f"All tests passed.\n{combined}"
        return f"Tests FAILED (exit {result.returncode}).\n{combined}"
    except Exception as e:
        return f"Failed to run tests: {e}"


@tool
def add_rust_dependency(crate_name: str) -> str:
    """Adds a dependency to the Rust project using cargo add."""
    try:
        subprocess.run(["cargo", "add", crate_name], cwd=SIMULATOR_DIR, check=True)
        return f"Successfully added {crate_name}."
    except Exception as e:
        return f"Failed to add dependency: {e}"


# ==========================================
# 5. NODES (non-LLM — module-level)
# ==========================================
def _build_system_prompt(state: WorkflowState, base_prompt: str) -> str:
    architecture = state.get("architecture_spec", "No architecture spec provided.")
    current_step_id = state["todo_tasks"][0] if state["todo_tasks"] else None
    if current_step_id:
        step_yaml = state.get("step_plans", {}).get(current_step_id, "No step plan found.")
    else:
        step_yaml = "No tasks left."
    prompt = (
        f"{base_prompt}\n\n"
        f"Architecture Spec:\n{architecture}\n\n"
        f"Current Step Plan:\n{step_yaml}"
    )
    test_results = state.get("test_results", "")
    if test_results and not state.get("tests_passed", False):
        prompt += f"\n\nPrevious test run FAILED — you must fix these errors:\n{test_results}"
    return prompt


def plan_evaluator(state: WorkflowState):
    return "coder" if state.get("plan_approved", False) else "planner"


def tester_node(state: WorkflowState):
    print("\n>>> [tester] running cargo test...")
    try:
        result = subprocess.run(
            ["cargo", "test", "--color=never"],
            cwd=SIMULATOR_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print(result.stdout)
            return {
                "test_results": f"Tests passed successfully.\n{result.stdout}",
                "tests_passed": True,
                "retry_count": 0,
            }
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        return {
            "test_results": f"Tests failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            "tests_passed": False,
            "retry_count": state.get("retry_count", 0) + 1,
        }
    except Exception as e:
        return {
            "test_results": f"System Error running tests: {e}",
            "tests_passed": False,
            "retry_count": state.get("retry_count", 0) + 1,
        }


def test_evaluator(state: WorkflowState):
    if state.get("tests_passed", False):
        return "pass"
    if state.get("retry_count", 0) >= MAX_TEST_RETRIES:
        return "give_up"
    return "fail"


def queue_manager_node(state: WorkflowState):
    print("\n>>> [queue_manager] starting...")
    todo = state.get("todo_tasks", [])
    if not todo:
        return {}

    completed_task = todo[0]
    remaining_tasks = todo[1:]

    print(f"\n>>> Queue Manager: Finished '{completed_task}'")
    print(f">>> Tasks remaining: {len(remaining_tasks)}")

    git_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=SIMULATOR_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if git_status.stdout.strip():
        step_yaml = state.get("step_plans", {}).get(completed_task, "")
        title = completed_task
        try:
            parsed = yaml.safe_load(step_yaml)
            if parsed and isinstance(parsed, dict):
                title = parsed.get("title", completed_task)
        except Exception:
            pass
        subprocess.run(["git", "add", "-A"], cwd=SIMULATOR_DIR, check=False)
        subprocess.run(
            ["git", "commit", "-m", f"complete {completed_task}: {title}"],
            cwd=SIMULATOR_DIR,
            check=False,
        )

    return {
        "todo_tasks": remaining_tasks,
        "completed_tasks": [completed_task],
        "tests_passed": False,
        "retry_count": 0,
        "tool_call_count": 0,
        "messages": [RemoveMessage(id=m.id) for m in state.get("messages", [])],
    }


def queue_evaluator(state: WorkflowState):
    return "continue" if state.get("todo_tasks") else "done"


def give_up_node(state: WorkflowState):
    print("\n>>> [give_up] starting...")
    task = state["todo_tasks"][0] if state["todo_tasks"] else "unknown"
    step_yaml = state.get("step_plans", {}).get(task, "No step plan found.")
    print(f"\n>>> GIVING UP on '{task}' after {state.get('retry_count', 0)} failed attempts.")
    print(f"\n>>> Step Plan:\n{step_yaml}")
    print(f"\n>>> Last test output:\n{state.get('test_results', 'No output captured.')}")
    diff = subprocess.run(
        ["git", "diff"],
        cwd=SIMULATOR_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"\n>>> Git diff:\n{diff.stdout}")
    return {}


# ==========================================
# 6. GRAPH CONSTRUCTION
# ==========================================
def build_graph(checkpointer, cfg: dict, llm_pool: dict[str, ChatOpenAI]):
    planner = resolve_node("planner", cfg, llm_pool)
    coder = resolve_node("coder", cfg, llm_pool)

    def planner_node(state: WorkflowState):
        print("\n>>> [planner] starting...")
        plan_draft = state.get("plan_draft")
        plan_feedback = state.get("plan_feedback")

        messages = [
            SystemMessage(content=planner["system_prompt"]),
            HumanMessage(content=(
                f"Goal: {state['goal']}\n\n"
                f"Architecture Spec:\n{state.get('architecture_spec', '')}\n\n"
                + (f"Previous plan draft:\n{plan_draft}\n\n" if plan_draft else "")
                + (
                    f"Human feedback:\n{plan_feedback}\n\nRevise the plan accordingly."
                    if plan_feedback
                    else "Produce the initial plan draft."
                )
            )),
        ]

        response = planner["llm"].invoke(messages)
        revised_draft = response.content

        user_response = interrupt(revised_draft)

        _approval_keywords = {"approved", "approve", "yes", "y", "lgtm", "looks good", "go ahead", "ship it", "ok", "okay"}
        if any(kw in user_response.strip().lower() for kw in _approval_keywords):
            step_plans = {}
            todo_tasks = []
            for doc in revised_draft.split("---"):
                doc = doc.strip()
                if not doc:
                    continue
                try:
                    parsed = yaml.safe_load(doc)
                    if parsed and isinstance(parsed, dict) and "id" in parsed:
                        step_id = parsed["id"]
                        step_plans[step_id] = doc
                        todo_tasks.append(step_id)
                except Exception:
                    pass
            return {
                "plan_draft": None,
                "plan_approved": True,
                "plan_feedback": None,
                "todo_tasks": todo_tasks,
                "step_plans": step_plans,
                "retry_count": 0,
                "tool_call_count": 0,
            }
        else:
            return {
                "plan_draft": revised_draft,
                "plan_feedback": user_response,
            }

    def _sanitize_history(messages: list) -> list:
        """Drop any trailing AI message whose tool_calls have no matching ToolMessages.

        This can happen when a checkpoint is saved after the coder node returns
        a tool-call response but before the tool node executes — i.e. exactly
        the state that causes the 400 "insufficient tool messages" error on
        resume.  We truncate back to the last clean point so the LLM can retry.
        """
        # Build a set of tool_call_ids that are already resolved by a ToolMessage.
        resolved: set[str] = set()
        for m in messages:
            if isinstance(m, ToolMessage) and m.tool_call_id:
                resolved.add(m.tool_call_id)

        # Walk backwards and drop any AIMessage with unresolved tool_call_ids.
        clean = list(messages)
        while clean:
            last = clean[-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                pending = [tc["id"] for tc in last.tool_calls if tc["id"] not in resolved]
                if pending:
                    print(
                        f"\n>>> [coder] WARNING: dropping orphaned tool-call message "
                        f"(ids: {pending}) from checkpoint — it was never executed."
                    )
                    clean.pop()
                    continue
            break
        return clean

    def coder_node(state: WorkflowState):
        task = state["todo_tasks"][0] if state.get("todo_tasks") else "unknown"
        print(f"\n>>> [coder] working on: {task}")
        llm_with_tools = coder["llm"].bind_tools(
            [read_rust_file, write_rust_file, run_clippy, run_tests, add_rust_dependency]
        )
        history = _sanitize_history(list(state.get("messages") or []))
        system = SystemMessage(content=_build_system_prompt(state, coder["base_system_prompt"]))

        # When the agent has used up its tool budget, inject a user turn so it
        # wraps up without making further tool calls.  Do this BEFORE invoking
        # the LLM so the LLM sees the instruction rather than having its pending
        # tool calls silently dropped (which corrupts the message history).
        at_limit = state.get("tool_call_count", 0) >= coder["max_tool_calls"]
        if at_limit and history:
            history = [
                *history,
                HumanMessage(
                    content=(
                        "You have reached the tool-call limit for this turn. "
                        "Do NOT call any more tools. Finalise your implementation "
                        "using only what you have already written and read."
                    )
                ),
            ]

        if not history:
            seed = HumanMessage(content="Begin the current task.")
            response = llm_with_tools.invoke([system, seed])
            updates = {"messages": [seed, response]}
            if response.tool_calls:
                updates["tool_call_count"] = state.get("tool_call_count", 0) + 1
            return updates

        response = llm_with_tools.invoke([system, *history])
        updates = {"messages": [response]}
        if response.tool_calls:
            updates["tool_call_count"] = state.get("tool_call_count", 0) + 1
        return updates

    def coder_router(state: WorkflowState):
        last = state["messages"][-1]
        # Always drain pending tool calls first — skipping them leaves unresolved
        # tool_call_ids in the history and causes a 400 from the API on the next turn.
        if last.tool_calls:
            return "tools"
        return "tester"

    workflow = StateGraph(WorkflowState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("coder_tools", ToolNode([read_rust_file, write_rust_file, run_clippy, run_tests, add_rust_dependency]))
    workflow.add_node("tester", tester_node)
    workflow.add_node("queue_manager", queue_manager_node)
    workflow.add_node("give_up", give_up_node)

    workflow.set_entry_point("planner")

    workflow.add_conditional_edges("planner", plan_evaluator, {"planner": "planner", "coder": "coder"})
    workflow.add_conditional_edges("coder", coder_router, {"tools": "coder_tools", "tester": "tester"})
    workflow.add_edge("coder_tools", "coder")
    workflow.add_conditional_edges(
        "tester",
        test_evaluator,
        {"fail": "coder", "pass": "queue_manager", "give_up": "give_up"},
    )
    workflow.add_edge("give_up", END)
    workflow.add_conditional_edges(
        "queue_manager", queue_evaluator, {"continue": "coder", "done": END}
    )

    return workflow.compile(checkpointer=checkpointer)


# ==========================================
# 7. EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARM64 simulator orchestrator")
    parser.add_argument("goal", help="High-level goal for this run")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Discard any existing checkpoint for this goal and start fresh",
    )
    args = parser.parse_args()

    goal = " ".join(args.goal.split())  # normalise whitespace
    thread_id = make_thread_id(goal)
    lg_config = {"configurable": {"thread_id": thread_id}}

    if args.reset:
        reset_thread(thread_id)
        print(f">>> Reset: cleared checkpoint for thread '{thread_id}'")

    cfg, llm_pool = load_config()

    initial_state = {
        "goal": goal,
        "architecture_spec": (
            "Target: ARMv8-A AArch64 (64-bit execution state). "
            "Registers: 31 general-purpose 64-bit registers (X0–X30), SP, PC, PSTATE. "
            "Memory model: flat, byte-addressable. "
            "Instruction encoding: fixed 32-bit little-endian words. "
            "Relevant instructions for MVP: ADD, SUB, MOV (wide immediate), LDR, STR, B, BL, RET, HLT."
        ),
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

    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        existing = checkpointer.get_tuple(lg_config)
        if existing:
            stored_goal = existing.checkpoint.get("channel_values", {}).get("goal")
            if stored_goal and stored_goal != goal:
                sys.exit(
                    f"Error: checkpoint '{thread_id}' was created for a different goal.\n"
                    f"  Stored : {stored_goal}\n"
                    f"  Passed : {goal}\n"
                    f"Pass --reset to discard it and start fresh."
                )

        app = build_graph(checkpointer, cfg, llm_pool)
        stream_input = initial_state if not existing else None

        while True:
            interrupted = False
            interrupt_payload = None
            content_streamed = False
            seen_tool_ids: set[str] = set()

            for mode, data in app.stream(stream_input, lg_config, stream_mode=["updates", "messages"]):
                if mode == "messages":
                    msg_chunk, _metadata = data
                    # Stream LLM content tokens as they arrive
                    content = msg_chunk.content
                    if isinstance(content, str) and content:
                        print(content, end="", flush=True)
                        content_streamed = True
                    # Announce tool calls by name only — suppress argument JSON
                    for tc in getattr(msg_chunk, "tool_call_chunks", []):
                        tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                        if tc_id and name and tc_id not in seen_tool_ids:
                            print(f"\n>>> [tool] {name}(...)", flush=True)
                            seen_tool_ids.add(tc_id)
                elif mode == "updates":
                    if "__interrupt__" in data:
                        interrupted = True
                        interrupt_payload = data["__interrupt__"][0].value
                    elif data:
                        print()  # newline after inline-streamed content

            if not interrupted:
                break

            # If resuming from a checkpoint the plan wasn't re-streamed — show it
            if not content_streamed and interrupt_payload:
                print(interrupt_payload)
            while True:
                user_response = input("\nApprove plan? [yes] or type feedback to revise: ").strip()
                if user_response:
                    break
                print("(no input — please type 'yes' to approve or enter feedback)")
            stream_input = Command(resume=user_response)
