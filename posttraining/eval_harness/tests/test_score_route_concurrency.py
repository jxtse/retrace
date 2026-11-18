"""P2 regression: score_route pillar + per-step feasibility concurrency.

The three pillars are disjoint and the per-step feasibility backend (the
reaction_rater client) is thread-safe, so score_route now (a) runs the 3
pillars concurrently and (b) fans per-step feasibility scoring across a small
worker pool. Concurrency must be a pure speedup: identical scores, identical
per-step ORDER, and deterministic notes/warnings ordering regardless of which
thread finishes first.
"""
from __future__ import annotations

import time

from tools.verifier import score_route as sr_mod
from tools.verifier.score_route import RouteVerifierTool as ScoreRouteTool


class _SlowSkewRater:
    """Rater whose latency DECREASES with step index, so later steps finish
    first under threads — exposes any order-dependence in reassembly. Returns
    a distinct confidence per step so we can assert correct index mapping."""

    def __init__(self, n):
        self._n = n

    def rate(self, reactants, product):
        # product is "C" + "C"*i + "O", so the C count recovers the step index.
        i = product.count("C") - 1
        time.sleep(0.005 * (self._n - i))
        return {"confidence": round(0.1 + 0.01 * i, 4), "feasible": False}


def _make_tool(monkeypatch, pillar_parallel, workers, rater):
    tool = ScoreRouteTool.__new__(ScoreRouteTool)
    tool.weights = {"feasibility": 1.0, "cost": 0.0, "greenness": 0.0}
    tool.carbon_fabrication_penalty = 0.0
    tool._judge = None
    tool.judge_cfg = None
    tool._rater = rater
    tool.pillar_parallel = pillar_parallel
    tool.feasibility_step_workers = workers
    monkeypatch.setattr(sr_mod, "warning_for_step", lambda r, p: None)
    return tool


def _steps(n):
    # Each step has a unique product index so order can be verified.
    return [{"reactants": [f"C{'C'*i}=O"], "product": f"C{'C'*i}O"} for i in range(n)]


def test_per_step_order_preserved_under_out_of_order_completion(monkeypatch):
    n = 6
    # Per-step concurrency only (skip pillar pool to isolate this axis).
    tool = _make_tool(monkeypatch, pillar_parallel=False, workers=4,
                      rater=_SlowSkewRater(n))
    out = tool._score_feasibility(_steps(n), notes=[])
    # per_step must be in submission (index) order, not completion order.
    assert [ps["step"] for ps in out["per_step"]] == list(range(n))


def test_parallel_matches_serial(monkeypatch):
    n = 5
    steps = _steps(n)

    # Serial reference.
    serial = _make_tool(monkeypatch, pillar_parallel=False, workers=1,
                        rater=_SlowSkewRater(n))
    ref = serial._score_feasibility(steps, notes=[])

    # Parallel per-step.
    par = _make_tool(monkeypatch, pillar_parallel=False, workers=4,
                     rater=_SlowSkewRater(n))
    got = par._score_feasibility(steps, notes=[])

    assert got["score"] == ref["score"]
    assert [ps["step"] for ps in got["per_step"]] == [ps["step"] for ps in ref["per_step"]]
    # Warnings reassembled in index order; balanced + no stereo flag here, so
    # both surfaces are empty — compare them to lock determinism regardless.
    assert got["summary"]["warnings"] == ref["summary"]["warnings"]


def test_pillar_parallel_merges_notes_deterministically(monkeypatch):
    n = 3
    tool = _make_tool(monkeypatch, pillar_parallel=True, workers=2,
                      rater=_SlowSkewRater(n))

    # Cost/greenness each push a note; assert feasibility→cost→greenness order.
    monkeypatch.setattr(
        tool, "_score_cost",
        lambda smi, steps, notes: (notes.append("COST_NOTE"),
                                   {"score": 0.5, "summary": {"score": 0.5}})[1],
    )
    monkeypatch.setattr(
        tool, "_score_greenness",
        lambda steps, notes: (notes.append("GREEN_NOTE"),
                              {"score": 0.5, "summary": {"score": 0.5}})[1],
    )
    import json
    out = json.loads(tool.execute(
        smiles="CCO",
        route_steps=_steps(n),
    ))
    # The pillars ran concurrently but notes must merge in fixed pillar order,
    # so the LAST note (the one surfaced to the LLM) is always greenness's.
    assert out["note"].startswith("GREEN_NOTE")
    # Scores still computed.
    assert "feasibility_score" in out
    assert out["n_steps"] == n
