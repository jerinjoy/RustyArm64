import operator
import os
import subprocess
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

SIMULATOR_DIR = "../simulator"


# ==========================================
# 1. STATE DEFINITION
# ==========================================
class WorkflowState(TypedDict):
    goal: str
    todo_tasks: List[str]
    completed_tasks: Annotated[List[str], operator.add]
    architecture_spec: str
    compiler_logs: str
    test_results: str
    messages: Annotated[List[BaseMessage], add_messages]
    tests_passed: bool


# ==========================================
# 2. TOOLS
# ==========================================
@tool
def write_rust_file(filepath: str, content: str) -> str:
    """Writes Rust code to the specified file path relative to the Rust project root."""
    # Ensure we are writing inside the simulator directory
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
        # cwd=SIMULATOR_DIR forces the command to run inside the Rust project
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
CODER_PERSONALITY = """You are an expert systems programmer building a Rust-based ARM64 functional simulator.
Your coding standards:
1. Always write idiomatic, safe Rust.
2. Whenever you write or modify a file, you MUST immediately call the `run_clippy` tool.
3. If `run_clippy` returns warnings, you must fix them before finishing your turn.
4. The Rust project is already initialized. When using the `write_rust_file` tool, provide paths relative to the Rust project root (e.g., use `src/cpu.rs`, NOT `simulator/src/cpu.rs`)."""


def coder_node(state: WorkflowState):
    # Ensure you have export DEEPSEEK_API_KEY="your_key" in your terminal
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set!")

    llm = ChatOpenAI(
        model="deepseek-coder", api_key=api_key, base_url="https://api.deepseek.com/v1"
    )

    llm_with_tools = llm.bind_tools([write_rust_file, run_clippy, add_rust_dependency])

    current_task = state["todo_tasks"][0] if state["todo_tasks"] else "No tasks left."
    architecture = state.get("architecture_spec", "No architecture spec provided.")

    messages = [
        SystemMessage(content=CODER_PERSONALITY),
        HumanMessage(
            content=f"Architecture Spec:\n{architecture}\n\nYour Current Task:\n{current_task}"
        ),
    ]

    # If there are previous messages (like tool outputs), append them
    if state.get("messages"):
        messages.extend(state["messages"])

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def tester_node(state: WorkflowState):
    """Runs the Rust test suite and captures the output."""
    print(">>> Running cargo test...")
    try:
        # Run tests in the simulator directory
        result = subprocess.run(
            [
                "cargo",
                "test",
                "--color=never",
            ],  # Disable color codes for cleaner LLM reading
            cwd=SIMULATOR_DIR,
            capture_output=True,
            text=True,
            check=False,
        )

        # Determine if tests passed based on the return code
        if result.returncode == 0:
            output = f"Tests passed successfully.\n{result.stdout}"
            passed = True
        else:
            output = (
                f"Tests failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
            passed = False

        return {
            "test_results": output,
            "messages": [HumanMessage(content=f"System Test Results:\n{output}")],
            "tests_passed": passed,  # <-- Explicitly set the flag in the state
        }
    except Exception as e:
        error_msg = f"System Error running tests: {e}"
        return {
            "test_results": error_msg,
            "messages": [HumanMessage(content=error_msg)],
        }


def test_evaluator(state: WorkflowState):
    """Checks the boolean flag to route the graph."""
    if state.get("tests_passed", False):
        return "pass"
    else:
        return "fail"


def queue_manager_node(state: WorkflowState):
    """Pops the completed task and prepares the state for the next cycle."""
    todo = state.get("todo_tasks", [])

    if not todo:
        return {}  # Safety catch, shouldn't happen if routing is right

    completed_task = todo[0]
    remaining_tasks = todo[1:]

    print(f"\n>>> Queue Manager: Finished '{completed_task}'")
    print(f">>> Tasks remaining: {len(remaining_tasks)}")

    # We clear out the messages array so the LLM starts with a fresh context
    # for the next task, preventing context window bloat.
    return {
        "todo_tasks": remaining_tasks,
        "completed_tasks": [completed_task],
        "tests_passed": False,  # Reset the flag for the next task
        "messages": [],
    }


def queue_evaluator(state: WorkflowState):
    """Checks if there are tasks remaining in the queue."""
    if len(state.get("todo_tasks", [])) > 0:
        return "continue"
    else:
        return "done"


# ==========================================
# 4. GRAPH CONSTRUCTION
# ==========================================
workflow = StateGraph(WorkflowState)

# Add all nodes
workflow.add_node("coder", coder_node)
workflow.add_node("tools", ToolNode([write_rust_file, run_clippy, add_rust_dependency]))
workflow.add_node("tester", tester_node)  # <-- New node added
workflow.add_node("queue_manager", queue_manager_node)

workflow.set_entry_point("coder")

# 1. Coder Routing:
# If tools are requested, go to 'tools'. If NO tools are requested (text reply), go to 'tester'.
workflow.add_conditional_edges(
    "coder",
    tools_condition,
    {
        "tools": "tools",
        END: "tester",  # <-- Changed from END to "tester"
    },
)

# 2. Tools Routing: Always go back to the coder after writing files/running clippy
workflow.add_edge("tools", "coder")

# 3. Tester Routing:
# Evaluate the test results. Fail -> coder. Pass -> END.
workflow.add_conditional_edges(
    "tester",
    test_evaluator,
    {"fail": "coder", "pass": "queue_manager"},
)

# 3. New Queue Routing: Loop back to coder if tasks remain, otherwise END.
workflow.add_conditional_edges(
    "queue_manager", queue_evaluator, {"continue": "coder", "done": END}
)

app = workflow.compile()

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
        "compiler_logs": "",
        "test_results": "",
        "messages": [],
        "tests_passed": False,
    }

    print("Starting LangGraph execution...")
    # stream() lets you see the outputs as they happen, node by node
    for event in app.stream(initial_state):
        for node_name, node_state in event.items():
            print(f"\n--- Output from {node_name} ---")
            if "messages" in node_state and node_state["messages"]:
                print(node_state["messages"][-1].content)
