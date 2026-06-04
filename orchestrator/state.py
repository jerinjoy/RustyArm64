from typing import TypedDict, List, Dict, Annotated


def _merge_codebase(a: Dict[str, str], b: Dict[str, str]) -> Dict[str, str]:
    if not isinstance(a, dict):
        a = {}
    if not isinstance(b, dict):
        b = {}
    result = dict(a)
    result.update(b)
    return result


FEEDBACK_KEYS = frozenset({"human_feedback", "debugger_feedback", "repair_target", "cargo_output"})


def clear_feedback(state: dict) -> dict:
    """Return a partial state that resets all feedback/error slots to their empty defaults.

    Called at the top of every node that starts a new iteration so stale feedback from
    a previous iteration cannot leak into the current one.
    """
    return {
        "human_feedback": "",
        "debugger_feedback": "",
        "repair_target": "",
        "cargo_output": "",
    }


class SimulatorState(TypedDict):
    plan: List[str]
    current_step: str
    codebase: Annotated[Dict[str, str], _merge_codebase]
    spec_context: str
    cargo_output: str
    cargo_success: bool
    debugger_feedback: str
    repair_target: str
    human_feedback: str
