"""lean4 (2026-09-16): after saving one route the student declared
`COMPLETE ROUTE FOUND:` at call ~13; the harness (correctly) refused to stop
before the soft threshold, but with advisory nudges disabled it appended the
assistant turn WITHOUT a following user turn, so every later request ended in
two adjacent assistant messages and the model spent 27 more turns re-declaring
completion. Two contracts:

1. Role alternation is structural, never optional: after an ignored
   COMPLETE declaration the loop must still append a user turn.
2. In the lean profile a self-declared COMPLETE after at least one saved
   route and a short grace period must end the run (`exit_reason ==
   'llm_complete'`), instead of burning the remaining budget.
"""
import pytest

from tests.test_lean_profile import make_agent  # noqa: F401  (fixture re-export)

TARGET = "CCOC(=O)c1nc(N2CC[C@H](N)[C@H](OC)C2)sc1C"


def _save_call():
    from tests.test_lean_profile import _tc
    return (None, [_tc("save", "save_best_route", {
        "target_smiles": TARGET, "steps": [f"{TARGET}>>CCO.CC=O"],
        "total_score": 0.0, "label": "candidate",
    })], "tool_calls")


COMPLETE = ("COMPLETE ROUTE FOUND: best route saved.", None, "stop")


def _roles(result):
    return [m["role"] for m in result["messages"] if m["role"] in ("user", "assistant")]


def test_ignored_complete_keeps_role_alternation(make_agent):
    from tests.test_lean_profile import RecordingLLM
    agent = make_agent({
        "max_iterations": 8, "iteration_mode": "plain", "target_prefetch": "off",
        "mini_status_every": 0, "user_nudges_enabled": False,
        "stop_after_distinct_routes": 10, "auto_finish_after_saved": 20,
        "complete_grace_turns": 99,  # disable early stop for this test
    })
    agent.llm = RecordingLLM([COMPLETE] * 8)
    result = agent.run(TARGET)
    roles = _roles(result)
    assert not any(a == b == "assistant" for a, b in zip(roles, roles[1:])), roles
    # every request the model saw must also alternate
    for req in agent.llm.requests:
        r = [m["role"] for m in req["messages"] if m["role"] in ("user", "assistant")]
        assert not any(a == b == "assistant" for a, b in zip(r, r[1:])), r


def test_lean_honours_complete_after_save_and_grace(make_agent):
    from tests.test_lean_profile import RecordingLLM
    agent = make_agent({
        "max_iterations": 40, "iteration_mode": "plain", "target_prefetch": "off",
        "mini_status_every": 0, "user_nudges_enabled": False,
        "stop_after_distinct_routes": 10, "auto_finish_after_saved": 20,
        "complete_grace_turns": 2,
    })
    # save, then declare complete 5 times
    agent.llm = RecordingLLM([_save_call()] + [COMPLETE] * 5)
    result = agent.run(TARGET)
    # 1 save + 2 ignored declarations (grace) + 1 honoured = 4 calls, not 40
    assert agent.llm.call_count == 4, agent.llm.call_count
    assert result.get("bail_reason") in (None, "llm_complete") or result.get("exit_reason") == "llm_complete"


def test_complete_without_any_saved_route_is_never_honoured(make_agent):
    from tests.test_lean_profile import RecordingLLM
    agent = make_agent({
        "max_iterations": 6, "iteration_mode": "plain", "target_prefetch": "off",
        "mini_status_every": 0, "user_nudges_enabled": False,
        "stop_after_distinct_routes": 10, "complete_grace_turns": 1,
    })
    agent.llm = RecordingLLM([COMPLETE] * 6)
    result = agent.run(TARGET)
    assert agent.llm.call_count == 6  # ran to budget; nothing saved → cannot stop early


def test_default_profile_unchanged(make_agent):
    """Full harness default: no grace-based early stop (complete_grace_turns unset)."""
    from tests.test_lean_profile import RecordingLLM
    agent = make_agent({"max_iterations": 5, "iteration_mode": "plain", "target_prefetch": "off",
                        "stop_after_distinct_routes": 10})
    agent.llm = RecordingLLM([_save_call()] + [COMPLETE] * 4)
    agent.run(TARGET)
    assert agent.llm.call_count == 5
