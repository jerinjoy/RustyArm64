import operator
import os
import subprocess
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

SIMULATOR_DIR = "../simulator"
MAX_RETRIES = 5

_llm = ChatOpenAI(
    model="deepseek-coder",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)


# ==========================================
# 1. STATE DEFINITION
# ==========================================
class WorkflowState(TypedDict):
    goal: str
    todo_tasks: List[str]
    completed_tasks: Annotated[List[str], operator.add]
    architecture_spec: str
    test_results: str
    messages: Annotated[List[BaseMessage], add_messages]
    tests_passed: bool
    retry_count: int


# ==========================================
# 2. TOOLS
# ==========================================
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
# 3. NODES
# ==========================================
def _build_system_prompt(state: WorkflowState) -> str:
    architecture = state.get("architecture_spec", "No architecture spec provided.")
    current_task = state["todo_tasks"][0] if state["todo_tasks"] else "No tasks left."
    prompt = (
        "You are an expert systems programmer building a Rust-based ARM64 functional simulator.\n"
        "Your coding standards:\n"
        "1. Always write idiomatic, safe Rust.\n"
        "2. Whenever you write or modify a file, you MUST immediately call the `run_clippy` tool.\n"
        "3. If `run_clippy` returns warnings, you must fix them before finishing your turn.\n"
        "4. The Rust project is already initialized. When using the `write_rust_file` tool, "
        "provide paths relative to the Rust project root (e.g., use `src/cpu.rs`, NOT `simulator/src/cpu.rs`).\n\n"
        f"Architecture Spec:\n{architecture}\n\n"
        f"Current Task:\n{current_task}"
    )
    test_results = state.get("test_results", "")
    if test_results and not state.get("tests_passed", False):
        prompt += f"\n\nPrevious test run FAILED — you must fix these errors:\n{test_results}"
    return prompt


def coder_node(state: WorkflowState):
    llm_with_tools = _llm.bind_tools([write_rust_file, run_clippy, add_rust_dependency])
    history = state.get("messages") or []

    # On a fresh task the history is empty — seed it with the task prompt so it's
    # tracked in state and visible in checkpoints / LangSmith traces.
    if not history:
        seed = HumanMessage(content="Begin the current task.")
        return {"messages": [seed, llm_with_tools.invoke([SystemMessage(content=_build_system_prompt(state)), seed])]}

    response = llm_with_tools.invoke([SystemMessage(content=_build_system_prompt(state)), *history])
    return {"messages": [response]}


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
    if state.get("retry_count", 0) >= MAX_RETRIES:
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

    return {
        "todo_tasks": remaining_tasks,
        "completed_tasks": [completed_task],
        "tests_passed": False,
        "retry_count": 0,
        "messages": [RemoveMessage(id=m.id) for m in state.get("messages", [])],
    }


def queue_evaluator(state: WorkflowState):
    return "continue" if state.get("todo_tasks") else "done"


def coder_router(state: WorkflowState):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else "tester"


# ==========================================
# 4. GRAPH CONSTRUCTION
# ==========================================
def build_graph():
    workflow = StateGraph(WorkflowState)

    workflow.add_node("coder", coder_node)
    workflow.add_node("tools", ToolNode([write_rust_file, run_clippy, add_rust_dependency]))
    workflow.add_node("tester", tester_node)
    workflow.add_node("queue_manager", queue_manager_node)

    workflow.set_entry_point("coder")

    workflow.add_conditional_edges("coder", coder_router, {"tools": "tools", "tester": "tester"})
    workflow.add_edge("tools", "coder")
    workflow.add_conditional_edges(
        "tester",
        test_evaluator,
        {"fail": "coder", "pass": "queue_manager", "give_up": END},
    )
    workflow.add_conditional_edges(
        "queue_manager", queue_evaluator, {"continue": "coder", "done": END}
    )

    return workflow.compile(checkpointer=InMemorySaver())


app = build_graph()

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    initial_state = {
        "goal": "Build an ARM64 functional simulator in Rust",
        "todo_tasks": [
            (
                "Implement a Memory struct as a linear byte array [u8; 65536]. "
                "Add a `fetch` method to the Cpu that reads 4 bytes from Memory at the current PC. "
                "If the PC points to invalid memory, return a custom 'InvalidPC' error."
            ),
            (
                "Implement an ELF loader using the `goblin` crate. "
                "The loader must parse the binary, identify the entry point, "
                "load segments into the Memory struct, and set the CPU's PC to the entry point. "
                "Return a custom 'ElfLoadError' if the binary is malformed or invalid."
            ),
        ],
        "completed_tasks": [
            "Create the Cpu struct in src/cpu.rs with 32 general purpose registers."
        ],
        "architecture_spec": "Standard ARMv8 architecture.",
        "test_results": "",
        "messages": [],
        "tests_passed": False,
        "retry_count": 0,
    }

    print("Starting LangGraph execution...")
    for event in app.stream(initial_state, config={"configurable": {"thread_id": "main"}}):
        for node_name, node_state in event.items():
            print(f"\n--- Output from {node_name} ---")
            if "messages" in node_state and node_state["messages"]:
                print(node_state["messages"][-1].content)
