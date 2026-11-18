"""Tests for the rater-only feasibility path in score_route.

The discriminative reaction_rater is the SOLE per-step feasibility signal —
the ReactionT5v2-forward round-trip was removed. These tests inject a fake
rater into a bypass-constructed RouteVerifierTool and assert:

  (a) confident rater       -> step == rater confidence, rater dict recorded
  (b) rater returns None    -> _score_feasibility RAISES RuntimeError
                               (no fallback scorer, must not silently score 0)
"""
import pytest

from tools.verifier import score_route as sr_mod
from tools.verifier.score_route import RouteVerifierTool as ScoreRouteTool


class _FakeRater:
    """Returns a fixed confidence (or None) regardless of input."""

    def __init__(self, conf):
        self._conf = conf
        self.calls = 0

    def rate(self, reactants, product):
        self.calls += 1
        if self._conf is None:
            return None
        return {"confidence": self._conf, "feasible": self._conf >= 0.5}


def _make_tool(monkeypatch, rater):
    tool = ScoreRouteTool.__new__(ScoreRouteTool)
    tool.weights = {"feasibility": 1.0, "cost": 0.0, "greenness": 0.0}
    tool.carbon_fabrication_penalty = 0.0
    tool._judge = None
    tool.judge_cfg = None
    tool._rater = rater
    monkeypatch.setattr(
        tool, "_score_cost",
        lambda smi, steps, notes: {"score": 0.5, "summary": {"score": 0.5}},
    )
    monkeypatch.setattr(
        tool, "_score_greenness",
        lambda steps, notes: {"score": 0.5, "summary": {"score": 0.5}},
    )
    monkeypatch.setattr(sr_mod, "warning_for_step", lambda reactants, product: None)
    return tool


def _feasibility(tool):
    notes = []
    return tool._score_feasibility(
        [{"reactants": ["CC=O"], "product": "CCO"}], notes
    )


def test_confident_rater_scores_to_rater(monkeypatch):
    rater = _FakeRater(0.8)
    tool = _make_tool(monkeypatch, rater)

    feas = _feasibility(tool)

    assert rater.calls == 1
    step = feas["per_step"][0]
    assert step["rater"]["confidence"] == pytest.approx(0.8)
    assert step["rater"]["feasible"] is True
    assert "round_trip" not in step
    # single step, no judge -> route score == that step's score == rater conf
    assert feas["score"] == pytest.approx(0.8)


def test_low_rater_passes_through(monkeypatch):
    # No T5 rescue any more: a low rater score IS the step score (judge is off).
    rater = _FakeRater(0.2)
    tool = _make_tool(monkeypatch, rater)

    feas = _feasibility(tool)

    assert rater.calls == 1
    assert feas["score"] == pytest.approx(0.2)


def test_rater_unavailable_raises(monkeypatch):
    # Endpoint down -> rate() returns None -> the verifier must fail loudly,
    # NOT silently score the step 0 (a dead endpoint must not masquerade as an
    # infeasible route).
    rater = _FakeRater(None)
    tool = _make_tool(monkeypatch, rater)

    with pytest.raises(RuntimeError, match="reaction_rater unavailable"):
        _feasibility(tool)


# ── Dispersion penalty (feasibility_dispersion_lambda) ────────────────────────


class _SeqRater:
    """Returns confidences from a list, one per call, in order."""

    def __init__(self, confs):
        self._confs = list(confs)
        self.calls = 0

    def rate(self, reactants, product):
        c = self._confs[self.calls]
        self.calls += 1
        return {"confidence": c, "feasible": c >= 0.5}


def _multistep_feas(tool, confs):
    # One reaction per confidence; distinct C->C->C... carbons so the carbon
    # penalty never fires (we test dispersion in isolation).
    rater = _SeqRater(confs)
    tool._rater = rater
    steps = [{"reactants": ["CC=O"], "product": "CCO"} for _ in confs]
    # _score_feasibility fans steps out across a thread pool; force order by
    # scoring serially so _SeqRater's positional confidences line up.
    tool.feasibility_step_workers = 1
    return tool._score_feasibility(steps, [])


def test_dispersion_docks_toward_worst_step(monkeypatch):
    # mean(0.9,0.9,0.3)=0.7, min=0.3; lambda=0.3 -> 0.7 - 0.3*(0.7-0.3) = 0.58
    tool = _make_tool(monkeypatch, rater=None)
    tool.feasibility_dispersion_lambda = 0.3
    feas = _multistep_feas(tool, [0.9, 0.9, 0.3])
    assert feas["summary"]["raw_mean"] == pytest.approx(0.7)
    assert feas["summary"]["worst_step"] == pytest.approx(0.3)
    assert feas["score"] == pytest.approx(0.58)


def test_dispersion_lambda_zero_is_pure_mean(monkeypatch):
    tool = _make_tool(monkeypatch, rater=None)
    tool.feasibility_dispersion_lambda = 0.0
    feas = _multistep_feas(tool, [0.9, 0.9, 0.3])
    assert feas["score"] == pytest.approx(0.7)
    assert "worst_step" not in feas["summary"]


def test_dispersion_noop_when_all_steps_equal(monkeypatch):
    # min == mean -> penalty contributes zero regardless of lambda.
    tool = _make_tool(monkeypatch, rater=None)
    tool.feasibility_dispersion_lambda = 0.5
    feas = _multistep_feas(tool, [0.6, 0.6, 0.6])
    assert feas["score"] == pytest.approx(0.6)


def test_dispersion_noop_for_single_step(monkeypatch):
    # A 1-step route has min == mean by construction; score unchanged.
    tool = _make_tool(monkeypatch, rater=None)
    tool.feasibility_dispersion_lambda = 0.5
    feas = _multistep_feas(tool, [0.42])
    assert feas["score"] == pytest.approx(0.42)
