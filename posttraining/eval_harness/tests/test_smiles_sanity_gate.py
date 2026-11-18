"""Tests for the per-molecule SMILES sanitization gate in feasibility scoring.

The gate sits in front of the reaction_rater: a step whose reactant or product
cannot be parsed/sanitized by RDKit is a BROKEN ROUTE, scored 0 with the rater
skipped — distinct from a rater OUTAGE, which raises. These tests pin both the
standalone helper and its integration into _score_one_feasibility_step.
"""
import pytest

from tools.verifier import score_route as sr_mod
from tools.verifier.score_route import RouteVerifierTool as ScoreRouteTool
from tools.verifier.smiles_sanity import check_step_smiles, first_invalid_smiles


# ── helper unit tests ──────────────────────────────────────────────────────

def test_valid_smiles_pass():
    assert first_invalid_smiles(["CC=O", "c1ccccc1", "CCO"]) is None
    ok, bad = check_step_smiles(["CC=O"], "CCO")
    assert ok is True and bad is None


def test_unparseable_smiles_flagged():
    # Pentavalent carbon — RDKit rejects on sanitization.
    bad = first_invalid_smiles(["CC=O", "C(C)(C)(C)(C)C"])
    assert bad == "C(C)(C)(C)(C)C"


def test_garbage_string_flagged():
    ok, bad = check_step_smiles(["NOT_A_SMILES"], "CCO")
    assert ok is False
    assert bad == "NOT_A_SMILES"


def test_empty_product_flagged():
    ok, bad = check_step_smiles(["CCO"], "")
    assert ok is False
    assert bad == ""


# ── integration: the gate inside _score_one_feasibility_step ────────────────

class _RaterSpy:
    """Records whether the rater was called; should be SKIPPED for bad SMILES."""

    def __init__(self):
        self.calls = 0

    def rate(self, reactants, product):
        self.calls += 1
        return {"confidence": 0.9, "feasible": True}


def _make_tool(rater):
    tool = ScoreRouteTool.__new__(ScoreRouteTool)
    tool.weights = {"feasibility": 1.0, "cost": 0.0, "greenness": 0.0}
    tool.carbon_fabrication_penalty = 0.0
    tool._judge = None
    tool.judge_cfg = None
    tool._rater = rater
    return tool


def test_invalid_product_scores_zero_and_skips_rater(monkeypatch):
    monkeypatch.setattr(sr_mod, "warning_for_step", lambda r, p: None)
    rater = _RaterSpy()
    tool = _make_tool(rater)

    entry, score, warns = tool._score_one_feasibility_step(
        0, {"reactants": ["CC=O"], "product": "C(C)(C)(C)(C)C"}
    )

    assert score == 0.0
    assert rater.calls == 0, "rater must be skipped for an unparseable step"
    assert entry["error"] == "invalid SMILES"
    assert "invalid SMILES" in warns[0]


def test_valid_step_still_reaches_rater(monkeypatch):
    monkeypatch.setattr(sr_mod, "warning_for_step", lambda r, p: None)
    rater = _RaterSpy()
    tool = _make_tool(rater)

    entry, score, warns = tool._score_one_feasibility_step(
        0, {"reactants": ["CC=O"], "product": "CCO"}
    )

    assert rater.calls == 1
    assert score == pytest.approx(0.9)
    assert "rater" in entry
