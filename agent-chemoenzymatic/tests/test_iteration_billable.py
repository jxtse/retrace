"""Iteration-billing rule: a turn consumes one iteration iff it includes
at least one call to a tool in the billable whitelist. Pure-thinking
turns (no tool_calls) are ALSO billable as a safety net to prevent
silent infinite loops.
"""

import pytest


def _fake_tool_call(name):
    """Build the minimal dict shape that _tool_call_name reads."""
    return {"function": {"name": name}, "id": "x", "type": "function"}


def test_thinking_only_turn_is_billable():
    from core.iteration import _is_iteration_billable
    assert _is_iteration_billable(
        tool_calls=[],
        billable_set={"suggest_templates", "neural_retro"},
    ) is True


def test_only_billable_tool_is_billable():
    from core.iteration import _is_iteration_billable
    assert _is_iteration_billable(
        tool_calls=[_fake_tool_call("suggest_templates")],
        billable_set={"suggest_templates", "neural_retro"},
    ) is True


def test_only_non_billable_tool_is_not_billable():
    from core.iteration import _is_iteration_billable
    assert _is_iteration_billable(
        tool_calls=[
            _fake_tool_call("score_route"),
            _fake_tool_call("save_best_route"),
            _fake_tool_call("check_buyability"),
        ],
        billable_set={"suggest_templates", "neural_retro"},
    ) is False


def test_mixed_turn_with_one_billable_is_billable():
    from core.iteration import _is_iteration_billable
    assert _is_iteration_billable(
        tool_calls=[
            _fake_tool_call("save_best_route"),
            _fake_tool_call("neural_retro"),
        ],
        billable_set={"suggest_templates", "neural_retro"},
    ) is True


def test_explore_branches_alone_is_not_billable():
    """Per spec §7: explore_branches does NOT count, even though it can
    fan out to many candidates internally."""
    from core.iteration import _is_iteration_billable
    assert _is_iteration_billable(
        tool_calls=[_fake_tool_call("explore_branches")],
        billable_set={"suggest_templates", "neural_retro"},
    ) is False
