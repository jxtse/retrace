"""score_route surfaces route grounding (informational, audit #2 option A).

The full execute() must report a `route_grounding` block and, when the route
is ungrounded, prepend an "ungrounded route" warning — WITHOUT folding it into
any pillar score (grounding is orthogonal). save_best_route is the hard gate;
this is the pre-save signal the LLM sees.
"""
import json

import pytest

from tools.verifier.score_route import RouteVerifierTool as ScoreRouteTool


def _make_tool(monkeypatch):
    tool = ScoreRouteTool.__new__(ScoreRouteTool)
    tool.weights = {"feasibility": 1.0, "cost": 0.0, "greenness": 0.0}
    tool.pillar_parallel = False
    # Stub all three pillars so execute() runs without rater/cost/green backends.
    monkeypatch.setattr(
        tool, "_score_feasibility",
        lambda steps, notes: {"score": 0.8, "summary": {"score": 0.8}},
    )
    monkeypatch.setattr(
        tool, "_score_cost",
        lambda smi, steps, notes: {"score": 0.5, "summary": {"score": 0.5}},
    )
    monkeypatch.setattr(
        tool, "_score_greenness",
        lambda steps, notes: {"score": 0.5, "summary": {"score": 0.5}},
    )
    return tool


def _steps(*pairs):
    return [{"product": p, "reactants": list(r)} for p, r in pairs]


def test_grounded_route_reports_ok(monkeypatch):
    tool = _make_tool(monkeypatch)
    out = json.loads(tool.execute(
        smiles="CCCCO",
        route_steps=_steps(("CCCCO", ["CCCC", "OCC"]), ("CCCC", ["CCCCBr"])),
    ))
    assert out["route_grounding"]["ok"] is True
    assert out["route_grounding"]["root_matches_target"] is True
    # grounding does not touch the feasibility pillar
    assert out["feasibility_score"] == pytest.approx(0.8)
    assert "warnings" not in out or not any(
        "ungrounded" in w for w in out["warnings"]
    )


def test_ungrounded_route_warns(monkeypatch):
    tool = _make_tool(monkeypatch)
    out = json.loads(tool.execute(
        smiles="CCCCO",
        route_steps=_steps(("CCCC", ["CCCCBr"])),  # never produces target
    ))
    assert out["route_grounding"]["ok"] is False
    assert out["route_grounding"]["root_matches_target"] is False
    assert out["warnings"][0].startswith("ungrounded route")
    # score is untouched — grounding is informational here
    assert out["feasibility_score"] == pytest.approx(0.8)
