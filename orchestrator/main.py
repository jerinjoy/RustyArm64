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
class SimulatorState(TypedDict):
    goal: str
    todo_tasks: List[str]
    completed_tasks: Annotated[List[str], operator.add]
    architecture_spec: str
    compiler_logs: str
    test_results: str
    messages: Annotated[List[BaseMessage], add_messages]


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


# ==========================================
# 3. NODES
# ==========================================
CODER_PERSONALITY = """You are an expert systems programmer building a Rust-based ARM64 functional simulator.
Your coding standards:
1. Always write idiomatic, safe Rust.
2. Structure your code with clear separation between instruction fetch, decode, and execute stages.
3. Whenever you write or modify a file, you MUST immediately call the `run_clippy` tool.
4. If `run_clippy` returns warnings, you must fix them before finishing your turn.
5. The Rust project is already initialized. When using the `write_rust_file` tool, provide paths relative to the Rust project root (e.g., use `src/cpu.rs`, NOT `simulator/src/cpu.rs`)."""


def coder_node(state: SimulatorState):
    # Ensure you have export DEEPSEEK_API_KEY="your_key" in your terminal
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set!")

    llm = ChatOpenAI(
        model="deepseek-coder", api_key=api_key, base_url="https://api.deepseek.com/v1"
    )

    llm_with_tools = llm.bind_tools([write_rust_file, run_clippy])

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


# ==========================================
# 4. GRAPH CONSTRUCTION
# ==========================================
workflow = StateGraph(SimulatorState)

workflow.add_node("coder", coder_node)
workflow.add_node("tools", ToolNode([write_rust_file, run_clippy]))

workflow.set_entry_point("coder")

# Routing logic
workflow.add_conditional_edges("coder", tools_condition, {"tools": "tools", END: END})
workflow.add_edge("tools", "coder")

app = workflow.compile()

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    initial_state = {
        "goal": "Build an ARM64 functional simulator in Rust",
        "todo_tasks": [
            "Create the Cpu struct in src/cpu.rs with 32 general purpose registers."
        ],
        "completed_tasks": [],
        "architecture_spec": "Standard ARMv8 architecture.",
        "compiler_logs": "",
        "test_results": "",
        "messages": [],
    }

    print("Starting LangGraph execution...")
    # stream() lets you see the outputs as they happen, node by node
    for event in app.stream(initial_state):
        for node_name, node_state in event.items():
            print(f"\n--- Output from {node_name} ---")
            if "messages" in node_state:
                print(node_state["messages"][-1].content)
