"""Iteration accounting helpers for the agent loop."""

from __future__ import annotations

DEFAULT_ITERATION_BILLABLE_TOOLS: list[str] = [
    "suggest_templates",
    "neural_retro",
]


def _tool_call_name(tc):
    """Return the function name of a tool call, robust to dict or object shape."""
    if isinstance(tc, dict):
        fn = tc.get("function") or {}
        return fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
    return getattr(getattr(tc, "function", None), "name", None)


def _is_iteration_billable(tool_calls, billable_set) -> bool:
    """Return True iff this turn should consume one of `max_iterations`.

    New whitelist semantics (was blacklist):
    - A turn with NO tool calls is billable (pure-thinking safety net so
      the loop terminates instead of spinning silently).
    - A turn with at least one call to a tool in `billable_set` is billable.
    - A turn whose tool calls are all OUTSIDE `billable_set` is NOT billable.

    `billable_set` is typically `{"suggest_templates", "neural_retro"}`.
    """
    if not tool_calls:
        return True
    names = {_tool_call_name(tc) for tc in tool_calls}
    return bool(names & set(billable_set))


def _refund_tool_names(tool_calls):
    """Extract tool names for trace logging, robust to dict/object shapes."""
    return [_tool_call_name(tc) for tc in (tool_calls or [])]
