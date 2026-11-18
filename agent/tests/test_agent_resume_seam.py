"""Public continuation seam on agent.run() (audit #4, human-in-the-loop).

run(target, resume_from=..., feedback=..., extra_rounds=...) must:
  * NOT clear a pre-populated scratchpad — it rehydrates prior routes/todos.
  * Append the chemist feedback as a user message (rendered from the
    human_feedback template) to the rehydrated prior conversation.
  * Drive _run_with_native_tools with initial_messages, the right
    iteration_offset (= prior iterations), and the extra_rounds budget.

We build a real agent via build_agent (so prompts/stores are wired), then
monkeypatch _run_with_native_tools to capture its kwargs instead of running
the full LLM loop, and stub the postprocess to pass the result through.
"""
import tempfile

import pytest

from core import build_agent
from tools.scratchpad import RouteEntry


def _config(tmpdir):
    return {
        "llm": {"base_url": "http://localhost:0/v1", "model": "fake", "max_tokens": 100},
        "agent": {
            "max_depth": 12,
            "max_iterations": 50,
            "iteration_billable_tools": ["suggest_templates", "neural_retro"],
            "stop_after_distinct_routes": 3,
            "tool_timeout_s": 5,
            "wallclock_s": 60,
            "auto_finish_after_saved": None,
            "continue_extra_rounds": 9,
        },
        "scratchpad": {"max_routes": 10},
        "prompts": {
            "core": "prompts/core.md",
            "sections_dir": "prompts/sections",
            "messages_dir": "prompts/messages",
        },
        "result_store": {
            "dir": tmpdir + "/_session_{session}",
            "summary_max_chars": 800,
            "keep_recent_full_per_tool": 4,
            "retention_days": 0,
        },
        "postprocess": {"enabled": False},
        "neural_retro": {"enabled": False},
        "verifier": {"enabled": False},
        "templates": {"dataset_dir": None},
        "stocks": {},
        "precedent_recall": {"enabled": False},
    }


@pytest.fixture
def agent(monkeypatch):
    import core.agent as agent_mod
    monkeypatch.setattr(
        agent_mod.RetrosynthesisAgent, "_prefetch_disconnections",
        lambda self, msgs, tgt: None,
    )
    with tempfile.TemporaryDirectory() as td:
        ag = build_agent(_config(td))
        yield ag


def _capture_native(ag, monkeypatch):
    """Replace _run_with_native_tools with a capturing stub.

    Returns a dict that gets filled with the call kwargs. The stub returns a
    minimal main-path-shaped result so run()'s tail (postprocess + todo
    attach) doesn't blow up.
    """
    captured = {}

    def fake_native(self, target_smiles, *, initial_messages=None,
                    max_iterations=None, iteration_offset=0):
        captured["target_smiles"] = target_smiles
        captured["initial_messages"] = initial_messages
        captured["max_iterations"] = max_iterations
        captured["iteration_offset"] = iteration_offset
        return {
            "success": True,
            "route_text": "COMPLETE ROUTE FOUND: x",
            "iterations": iteration_offset + 1,
            "time": 0.0,
            "messages": initial_messages or [],
            "ranked": [],
            "best_route": None,
            "score_best_route": None,
            "all_routes": [],
            "bail_reason": None,
        }

    import core.agent as agent_mod
    monkeypatch.setattr(agent_mod.RetrosynthesisAgent, "_run_with_native_tools", fake_native)
    return captured


def test_resume_does_not_clear_scratchpad(agent, monkeypatch):
    captured = _capture_native(agent, monkeypatch)

    prior = RouteEntry(
        label="PRIOR", total_score=0.6, steps=["CCCCO>>CCCC.OCC"],
    ).to_dict()
    resume_from = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "find route"},
            {"role": "assistant", "content": "COMPLETE ROUTE FOUND: prior"},
        ],
        "all_routes": [prior],
        "todos": [{"id": "t1", "content": "refine step 2", "status": "pending"}],
        "iterations": 7,
    }

    agent.run("CCCCO", resume_from=resume_from, feedback="make step 2 greener",
              extra_rounds=9)

    # Scratchpad was rehydrated, NOT cleared.
    labels = {r.label for r in agent.scratchpad_store.list()}
    assert "PRIOR" in labels
    # Todos were rehydrated too.
    assert any(t.id == "t1" for t in agent.todo_store.read())


def test_resume_appends_feedback_and_offsets(agent, monkeypatch):
    captured = _capture_native(agent, monkeypatch)

    resume_from = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "find route"},
        ],
        "all_routes": [],
        "todos": [],
        "iterations": 12,
    }

    agent.run("CCO", resume_from=resume_from, feedback="use a milder oxidant",
              extra_rounds=9)

    msgs = captured["initial_messages"]
    # Prior conversation preserved (deep-copied) + the feedback appended last.
    assert msgs[0]["content"] == "sys"
    assert msgs[-1]["role"] == "user"
    assert "use a milder oxidant" in msgs[-1]["content"]
    # Offset = prior iteration count; budget = extra_rounds.
    assert captured["iteration_offset"] == 12
    assert captured["max_iterations"] == 9
    assert captured["target_smiles"] == "CCO"


def test_default_run_clears_scratchpad(agent, monkeypatch):
    _capture_native(agent, monkeypatch)
    # Pre-seed a route; a fresh (non-resume) run must clear it.
    agent.scratchpad_store.save(
        RouteEntry(label="STALE", total_score=0.5, steps=["CCO>>CC=O"])
    )
    agent.run("CCO")
    assert "STALE" not in {r.label for r in agent.scratchpad_store.list()}
