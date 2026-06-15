import operator
import os
import subprocess
import tomllib
import yaml
from pathlib import Path
from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

SIMULATOR_DIR = "../simulator"
MAX_TEST_RETRIES = 5


# ==========================================
# 1. CONFIG
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
# 2. STATE DEFINITION
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
# 3. TOOLS
# ==========================================
@tool
def read_rust_file(filepath: str) -> str:
    """Reads and returns the contents of a file relative to the Rust project root."""
    full_path = os.path.join(SIMULATOR_DIR, filepath)
    with open(full_path, "r") as f:
        return f.read()


@tool
def write_rust_file(filepath: str, content: str) -> str:
    """Writes Rust code to the specified file path relative to the Rust project root."""
    full_path = os.path.join(SIMULATOR_DIR, filepath)
    os.makedirs(
        os.path.dirname(full_path) if os.path.dirname(full_path) else ".", exist_ok=True
    )
    with open(full_path, "w") as f:
        f.write(content)
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
def add_rust_dependency(crate_name: str) -> str:
    """Adds a dependency to the Rust project using cargo add."""
    try:
        subprocess.run(["cargo", "add", crate_name], cwd=SIMULATOR_DIR, check=True)
        return f"Successfully added {crate_name}."
    except Exception as e:
        return f"Failed to add dependency: {e}"


# ==========================================
# 4. NODES (non-LLM — module-level)
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
    print(">>> Running cargo test...")
    try:
        result = subprocess.run(
            ["cargo", "test", "--color=never"],
            cwd=SIMULATOR_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return {
                "test_results": f"Tests passed successfully.\n{result.stdout}",
                "tests_passed": True,
                "retry_count": 0,
            }
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
# 5. GRAPH CONSTRUCTION
# ==========================================
def build_graph(checkpointer, cfg: dict, llm_pool: dict[str, ChatOpenAI]):
    planner = resolve_node("planner", cfg, llm_pool)
    coder = resolve_node("coder", cfg, llm_pool)

    def planner_node(state: WorkflowState):
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

        if user_response.strip().lower() == "approved":
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

    def coder_node(state: WorkflowState):
        llm_with_tools = coder["llm"].bind_tools(
            [read_rust_file, write_rust_file, run_clippy, add_rust_dependency]
        )
        history = state.get("messages") or []
        system = SystemMessage(content=_build_system_prompt(state, coder["base_system_prompt"]))

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
        if state.get("tool_call_count", 0) >= coder["max_tool_calls"]:
            return "tester"
        last = state["messages"][-1]
        return "tools" if last.tool_calls else "tester"

    workflow = StateGraph(WorkflowState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("coder_tools", ToolNode([read_rust_file, write_rust_file, run_clippy, add_rust_dependency]))
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
# 6. EXECUTION
# ==========================================
if __name__ == "__main__":
    cfg, llm_pool = load_config()

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

    config = {"configurable": {"thread_id": "arm64-sim-mvp-run-1"}}

    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        app = build_graph(checkpointer, cfg, llm_pool)

        stream_input = initial_state

        while True:
            interrupted = False
            interrupt_payload = None

            for chunk in app.stream(stream_input, config, stream_mode="updates"):
                if "__interrupt__" in chunk:
                    interrupted = True
                    interrupt_payload = chunk["__interrupt__"][0].value
                else:
                    for node_name, node_state in chunk.items():
                        print(f"\n--- Output from {node_name} ---")
                        if "messages" in node_state and node_state["messages"]:
                            print(node_state["messages"][-1].content)

            if not interrupted:
                break

            print(interrupt_payload)
            user_response = input("Response: ")
            stream_input = Command(resume=user_response)
