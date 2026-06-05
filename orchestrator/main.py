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
    LLM_SERVICE, OPENAI_API_KEY, LLM_MODEL, ARM64_SPEC_DIR
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
    configure_specs,
    architect_node,
    spec_reader_node,
    human_approval_node,
    rust_coder_node,
    test_writer_node,
    cargo_tool_node,
    debugger_node,
    human_test_approval_node,
    progress_writer_node,
    committer_node,
)

import nodes as _nodes


# ---------------------------------------------------------------------------
# Conditional routing functions
# ---------------------------------------------------------------------------

def route_after_plan_approval(state: SimulatorState) -> str:
    """After human_plan_approval: feedback → architect, note that signals skip → architect,
    else → rust_coder."""
    if state.get("human_feedback", ""):
        return "architect"
    note = state.get("context_note", "").lower()
    if note and any(phrase in note for phrase in (
            "already done", "already exists", "already have", "no need",
            "skip this", "not needed", "unnecessary")):
        return "architect"
    return "rust_coder"


def route_after_cargo(state: SimulatorState) -> str:
    """cargo success + plan remains → human_test_approval;
       cargo success + empty plan   → END;
       cargo failure                → debugger."""
    if state.get("cargo_success", False):
        return "human_test_approval" if state.get("plan", []) else END
    return "debugger"


def route_after_test_approval(state: SimulatorState) -> str:
    """After human_test_approval: feedback → architect, else → progress_writer."""
    return "architect" if state.get("human_feedback", "") else "progress_writer"


def route_after_debugger(state: SimulatorState) -> str:
    """Route to rust_coder or test_writer based on repair_target."""
    return "test_writer" if state.get("repair_target") == "test" else "rust_coder"


def route_after_architect(state: SimulatorState) -> str:
    """Empty plan → END (work complete); otherwise → spec_reader."""
    return END if not state.get("plan") else "spec_reader"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    builder = StateGraph(SimulatorState)

    builder.add_node("architect", architect_node)
    builder.add_node("spec_reader", spec_reader_node)
    builder.add_node("human_plan_approval", human_approval_node)
    builder.add_node("rust_coder", rust_coder_node)
    builder.add_node("test_writer", test_writer_node)
    builder.add_node("cargo_tool", cargo_tool_node)
    builder.add_node("debugger", debugger_node)
    builder.add_node("human_test_approval", human_test_approval_node)
    builder.add_node("progress_writer", progress_writer_node)
    builder.add_node("committer", committer_node)

    builder.add_edge(START, "architect")
    builder.add_conditional_edges(
        "architect",
        route_after_architect,
        {"spec_reader": "spec_reader", END: END},
    )
    builder.add_edge("spec_reader", "human_plan_approval")

    builder.add_conditional_edges(
        "human_plan_approval",
        route_after_plan_approval,
        {"architect": "architect", "rust_coder": "rust_coder"},
    )

    builder.add_edge("rust_coder", "test_writer")
    builder.add_edge("test_writer", "cargo_tool")

    builder.add_conditional_edges(
        "cargo_tool",
        route_after_cargo,
        {"human_test_approval": "human_test_approval", "debugger": "debugger", END: END},
    )

    builder.add_conditional_edges(
        "human_test_approval",
        route_after_test_approval,
        {"architect": "architect", "progress_writer": "progress_writer"},
    )

    builder.add_edge("progress_writer", "committer")
    builder.add_edge("committer", "architect")

    builder.add_conditional_edges(
        "debugger",
        route_after_debugger,
        {"rust_coder": "rust_coder", "test_writer": "test_writer"},
    )

    return builder


# ---------------------------------------------------------------------------
# Cold-start: hydrate state from disk (progress.yaml + simulator/src/)
# ---------------------------------------------------------------------------

def _load_progress_yaml(repo_root: Path) -> dict:
    """Read progress.yaml and return {completed_steps, pending_plan, current_step}.

    Returns empty dict if the file does not exist.
    """
    import yaml

    path = repo_root / "progress.yaml"
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return {
            "completed_steps": data.get("completed_steps", []),
            "plan": data.get("pending_plan", []),
            "current_step": data.get("current_step", ""),
        }
    except Exception as exc:
        print(f"[WARN] Could not parse progress.yaml: {exc}", file=sys.stderr)
        return {}


def _scan_codebase(repo_root: Path) -> dict[str, str]:
    """Read all .rs files under simulator/src/ into a {path: content} dict."""
    src_dir = repo_root / "simulator" / "src"
    if not src_dir.is_dir():
        return {}
    codebase: dict[str, str] = {}
    for rs_file in sorted(src_dir.rglob("*.rs")):
        rel = rs_file.relative_to(src_dir).as_posix()
        codebase[rel] = rs_file.read_text(encoding="utf-8", errors="replace")
    if codebase:
        print(f"[init]  scanned {len(codebase)} Rust source file(s) from {src_dir}")
    return codebase


# ---------------------------------------------------------------------------
# Interactive CLI loop
# ---------------------------------------------------------------------------

def _print_banner():
    print("╔══════════════════════════════════════════════╗")
    print("║   ARM64 Simulator – LangGraph Orchestrator   ║")
    print("╚══════════════════════════════════════════════╝")
    print()


def _initial_state(repo_root: Path, checkpointer: SqliteSaver | None,
                   startup_directive: str = "") -> dict:
    """Return the initial state for a new run.

    If the checkpointer already has saved state (warm resume), return an
    empty dict so the graph resumes from the last checkpoint.  Otherwise
    hydrate from progress.yaml and simulator/src/.  *startup_directive* is
    only applied on cold start.
    """
    if checkpointer is not None:
        config = {"configurable": {"thread_id": "main"}}
        try:
            snapshot = checkpointer.get_tuple(config)
            if snapshot is not None and snapshot.checkpoint:
                parent_ns = getattr(snapshot.checkpoint, "parent_ns", None)
                channel_versions = getattr(snapshot.checkpoint, "channel_versions", {})
                if parent_ns or channel_versions:
                    print("[init]  warm resume from existing checkpoint")
                    return {}
        except (KeyError, AttributeError, IndexError):
            pass

    # Cold start — hydrate from disk
    print("[init]  cold start — hydrating from disk")
    progress = _load_progress_yaml(repo_root)
    codebase = _scan_codebase(repo_root)

    state: dict = {
        "plan": [],
        "current_step": "",
        "completed_steps": [],
        "codebase": {},
        "spec_context": "",
        "cargo_output": "",
        "cargo_success": False,
        "debugger_feedback": "",
        "repair_target": "",
        "human_feedback": "",
        "context_note": "",
        "startup_directive": startup_directive,
    }
    state.update(progress)
    state["codebase"] = codebase

    if state["completed_steps"]:
        print(f"[init]  {len(state['completed_steps'])} completed step(s), "
              f"{len(state['plan'])} pending")
    return state


def _plan_approval_ui(state_values: dict) -> tuple[str, str]:
    """Display current step/spec and collect plan approval decision.

    Returns (decision, context_note).
    decision: "y" = approved, anything else = rejection feedback
    context_note: extra info passed along when using "y: <message>" syntax
    """
    plan = state_values.get("plan", [])
    current_step = state_values.get("current_step", "(no step)")
    spec_context = state_values.get("spec_context", "")
    directive = state_values.get("startup_directive", "")

    print(f"\n{'─' * 50}")
    print(f"Step:       {current_step}")
    print(f"Plan left:  {len(plan)} step(s)")
    if directive:
        print(f"Directive:  {directive}")
    if spec_context:
        ctx = spec_context[:400].replace("\n", " ")
        print(f"Spec ctx:   {ctx}…")
    print(f"{'─' * 50}")

    while True:
        raw = input("y=approve  y:=approve+note  <anything>=reject & revise: ").strip()
        if not raw:
            continue
        if raw.lower() in ("y", "yes"):
            return "y", ""
        if raw.lower().startswith("y:") or raw.lower().startswith("yes:"):
            note = raw.split(":", 1)[1].strip()
            return "y", note
        return raw, ""


def _test_approval_ui(state_values: dict) -> tuple[str, str]:
    """Display cargo test results and collect test approval decision.

    Returns (decision, context_note).
    """
    current_step = state_values.get("current_step", "(no step)")
    cargo_output = state_values.get("cargo_output", "")
    plan = state_values.get("plan", [])

    print(f"\n{'─' * 50}")
    print(f"Step:       {current_step}")
    print(f"Plan left:  {len(plan)} step(s)")
    print(f"Cargo:      PASS")
    if cargo_output:
        tail = cargo_output[-600:]
        print(f"Output:     {tail}")
    print(f"{'─' * 50}")

    while True:
        raw = input("y=approve  y:=approve+note  <anything>=reject & revise: ").strip()
        if not raw:
            continue
        if raw.lower() in ("y", "yes"):
            return "y", ""
        if raw.lower().startswith("y:") or raw.lower().startswith("yes:"):
            note = raw.split(":", 1)[1].strip()
            return "y", note
        return raw, ""


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
    p.add_argument(
        "--spec-dir",
        default=os.environ.get("ARM64_SPEC_DIR", ""),
        help="Root path to ARM64 MRA XML specs (contains ISA_A64/ and SysReg/; "
             "may also be set via ARM64_SPEC_DIR env var)",
    )
    p.add_argument(
        "-p", "--prompt",
        default="",
        help="Optional startup directive for the architect on cold boot only "
             "(e.g. 'focus on integer ALU, skip SIMD')",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Delete the checkpoint DB before starting (forces a fresh cold start)",
    )
    return p.parse_args(argv)


def run(argv: list[str] | None = None):
    args = _parse_args(argv)

    if not args.api_key:
        print("Error: API key required. Use -k or set OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    model_override = args.model or None
    repo_root = Path(__file__).resolve().parent.parent

    # ---- Configure the LLM ----
    configure_model(service=args.service, api_key=args.api_key, model=model_override)

    # ---- Configure spec file paths ----
    if args.spec_dir:
        configure_specs(args.spec_dir)

    _print_banner()
    print(f"Service: {args.service}  |  Model: {_nodes._model_config.resolved_model}")
    print()

    # ---- SQLite checkpointer ----
    db_path = Path(__file__).resolve().parent / "checkpoints.db"
    if args.reset:
        for suffix in ("", "-shm", "-wal"):
            p = Path(str(db_path) + suffix)
            if p.exists():
                p.unlink()
        print("[init]  checkpoint DB reset — forcing cold start")
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # ---- Compile with dual interrupts ----
    builder = build_graph()
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_plan_approval", "human_test_approval"],
    )

    config = {"configurable": {"thread_id": "main"}}
    current_input = _initial_state(repo_root, checkpointer, args.prompt)

    # ---- Event loop ----
    while True:
        interrupted = False
        interrupt_node: str | None = None

        try:
            for event in graph.stream(current_input, config):
                if "__interrupt__" in event:
                    interrupted = True
                    # Extract which node was interrupted
                    for intr in event["__interrupt__"]:
                        if isinstance(intr, dict):
                            interrupt_node = intr.get("id", "")
                        elif hasattr(intr, "id"):
                            interrupt_node = intr.id
                        else:
                            interrupt_node = str(intr)
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
                    elif node_name == "committer":
                        s = graph.get_state(config).values
                        step = s.get("current_step", "")
                        print(f"  → {step}")
                    else:
                        print()

        except Exception as exc:
            print(f"\n[ERROR] {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            break

        if not interrupted:
            final = graph.get_state(config).values
            completed = final.get("completed_steps", [])
            plan = final.get("plan", [])
            print(f"\nDone.  {len(completed)} step(s) completed, "
                  f"{len(plan)} remaining.")
            break

        # ---- Human-in-the-loop ----
        state_vals = graph.get_state(config).values

        if interrupt_node == "human_test_approval":
            decision, note = _test_approval_ui(state_vals)
        else:
            decision, note = _plan_approval_ui(state_vals)

        if decision == "y":
            if note:
                graph.update_state(config, values={"context_note": note})
            current_input = None
        else:
            graph.update_state(config, values={"human_feedback": decision})
            current_input = None

    conn.close()


if __name__ == "__main__":
    run(sys.argv[1:])
