"""
LangGraph orchestrator for incremental ARM64 Rust simulator development.

Run from the orchestrator/ directory:
    uv run python main.py -s deepseek -k sk-your-key
    uv run python main.py -s openai -k sk-your-key -m gpt-4.1

Service defaults:
    deepseek  → deepseek-v4-pro   (https://api.deepseek.com)
    openai    → gpt-4o            (https://api.openai.com/v1)
    gemini    → gemini-2.5-flash  (requires: uv add langchain-google-genai)

Environment variable fallbacks:
    LLM_SERVICE, OPENAI_API_KEY, LLM_MODEL
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from state import SimulatorState
from nodes import (
    configure_model,
    _model_config,
    architect_node,
    spec_reader_node,
    human_approval_node,
    rust_coder_node,
    test_writer_node,
    cargo_tool_node,
    debugger_node,
)


# ---------------------------------------------------------------------------
# Conditional routing functions
# ---------------------------------------------------------------------------

def route_after_human(state: SimulatorState) -> str:
    """After human_approval: feedback present -> architect, else -> rust_coder."""
    return "architect" if state.get("human_feedback", "") else "rust_coder"


def route_after_cargo(state: SimulatorState) -> str:
    """cargo success + plan remains -> architect;
       cargo success + empty plan   -> END;
       cargo failure                -> debugger."""
    if state.get("cargo_success", False):
        return "architect" if state.get("plan", []) else END
    return "debugger"


def route_after_debugger(state: SimulatorState) -> str:
    """Route to rust_coder or test_writer based on repair_target."""
    return "test_writer" if state.get("repair_target") == "test" else "rust_coder"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    builder = StateGraph(SimulatorState)

    builder.add_node("architect", architect_node)
    builder.add_node("spec_reader", spec_reader_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("rust_coder", rust_coder_node)
    builder.add_node("test_writer", test_writer_node)
    builder.add_node("cargo_tool", cargo_tool_node)
    builder.add_node("debugger", debugger_node)

    builder.add_edge(START, "architect")
    builder.add_edge("architect", "spec_reader")
    builder.add_edge("spec_reader", "human_approval")

    builder.add_conditional_edges(
        "human_approval",
        route_after_human,
        {"architect": "architect", "rust_coder": "rust_coder"},
    )

    builder.add_edge("rust_coder", "test_writer")
    builder.add_edge("test_writer", "cargo_tool")

    builder.add_conditional_edges(
        "cargo_tool",
        route_after_cargo,
        {"architect": "architect", "debugger": "debugger", END: END},
    )

    builder.add_conditional_edges(
        "debugger",
        route_after_debugger,
        {"rust_coder": "rust_coder", "test_writer": "test_writer"},
    )

    return builder


# ---------------------------------------------------------------------------
# Interactive CLI loop
# ---------------------------------------------------------------------------

def _print_banner():
    print("╔══════════════════════════════════════════════╗")
    print("║   ARM64 Simulator – LangGraph Orchestrator   ║")
    print("╚══════════════════════════════════════════════╝")
    print()


def _initial_state() -> dict:
    return {
        "plan": [],
        "current_step": "",
        "codebase": {},
        "spec_context": "",
        "cargo_output": "",
        "cargo_success": False,
        "debugger_feedback": "",
        "repair_target": "",
        "human_feedback": "",
    }


def _human_input(state_values: dict) -> str | None:
    """Display current step/spec and collect user decision.

    Returns:
        "y"     – approved, continue to rust_coder
        <str>   – rejection feedback, route to architect
        None    – should not happen (blank input is re-prompted)
    """
    plan = state_values.get("plan", [])
    current_step = state_values.get("current_step", "(no step)")
    spec_context = state_values.get("spec_context", "")

    print(f"\n{'─' * 50}")
    print(f"Step:       {current_step}")
    print(f"Plan left:  {len(plan)} step(s)")
    if spec_context:
        ctx = spec_context[:400].replace("\n", " ")
        print(f"Spec ctx:   {ctx}…")
    print(f"{'─' * 50}")

    while True:
        raw = input("Approve (y), Reject with feedback (type feedback): ").strip()
        if not raw:
            continue
        if raw.lower() in ("y", "yes"):
            return "y"
        return raw


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LangGraph orchestrator for ARM64 Rust simulator MVP"
    )
    p.add_argument(
        "-s", "--service",
        default=os.environ.get("LLM_SERVICE", "deepseek"),
        choices=["openai", "deepseek", "gemini"],
        help="LLM service to use (default: deepseek, or $LLM_SERVICE)",
    )
    p.add_argument(
        "-k", "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key (or set OPENAI_API_KEY env var)",
    )
    p.add_argument(
        "-m", "--model",
        default=os.environ.get("LLM_MODEL", ""),
        help="Override the default model for the chosen service",
    )
    return p.parse_args(argv)


def run(argv: list[str] | None = None):
    args = _parse_args(argv)

    if not args.api_key:
        print("Error: API key required. Use -k or set OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    model_override = args.model or None

    # ---- Configure the LLM ----
    configure_model(service=args.service, api_key=args.api_key, model=model_override)

    _print_banner()
    print(f"Service: {args.service}  |  Model: {_model_config.resolved_model}")
    print()

    # ---- SQLite checkpointer ----
    db_path = Path(__file__).resolve().parent / "checkpoints.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # ---- Compile with interrupt_before the human gate ----
    builder = build_graph()
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approval"],
    )

    config = {"configurable": {"thread_id": "main"}}
    current_input = _initial_state()

    # ---- Event loop ----
    while True:
        interrupted = False

        try:
            for event in graph.stream(current_input, config):
                if "__interrupt__" in event:
                    interrupted = True
                    break

                for node_name in event:
                    print(f"[{node_name}] ✓", end="", flush=True)

                    if node_name == "cargo_tool":
                        s = graph.get_state(config).values
                        ok = s.get("cargo_success", False)
                        print(f"  {'PASS' if ok else 'FAIL'}")
                        if not ok:
                            tail = s.get("cargo_output", "")[-400:]
                            print(f"    {tail}")
                    elif node_name == "architect":
                        s = graph.get_state(config).values
                        step = s.get("current_step", "")
                        print(f"  → {step}")
                    else:
                        print()

        except Exception as exc:
            print(f"\n[ERROR] {exc}", file=sys.stderr)
            break

        if not interrupted:
            final = graph.get_state(config).values
            ok = final.get("cargo_success", False)
            print(f"\nDone.  cargo_tool {'passed' if ok else 'failed'}.")
            print(f"Final plan: {final.get('plan', [])}")
            break

        # ---- Human-in-the-loop ----
        state_vals = graph.get_state(config).values
        decision = _human_input(state_vals)

        if decision == "y":
            # Approved – resume normally; human_approval routes to rust_coder
            current_input = None
        else:
            # Rejected – inject feedback so human_approval routes to architect
            graph.update_state(config, values={"human_feedback": decision})
            current_input = None

    conn.close()


if __name__ == "__main__":
    run(sys.argv[1:])
