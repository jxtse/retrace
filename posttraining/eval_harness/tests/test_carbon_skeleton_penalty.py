"""Tests for the carbon-skeleton fabrication penalty in feasibility scoring.

This is NOT atom/mass conservation (that was removed — organic reactions are
written unbalanced). It is the one balance signal that IS diagnostic: a POSITIVE
``carbon_delta`` (product carries more skeletal carbons than every precursor
combined) means a carbon was fabricated, and the feasibility pillar docks the
step ``carbon_fabrication_penalty`` per fabricated carbon. A NEGATIVE delta (a
fragment leaves) is benign and must NOT be penalised.
"""
import pytest

from tools.verifier import score_route as sr_mod
from tools.verifier.carbon_skeleton import carbon_delta, forms_new_ring
from tools.verifier.score_route import RouteVerifierTool as ScoreRouteTool


# ── helper unit tests ──────────────────────────────────────────────────────

def test_carbon_delta_conserved():
    # ethanol (2C) from acetaldehyde (2C) — carbon conserved.
    assert carbon_delta(["CC=O"], "CCO") == 0


def test_carbon_delta_fabricated_positive():
    # product benzene ring (6C) from a 2-carbon precursor + water only:
    # +4 carbons conjured from nowhere.
    assert carbon_delta(["CC=O", "O"], "c1ccccc1") == 4


def test_carbon_delta_fragment_leaves_negative():
    # precursors carry 4C, product keeps 2C — a 2-carbon fragment departs.
    assert carbon_delta(["CCCC"], "CC") == -2


def test_forms_new_ring_detects_ring_formation():
    # two acyclic precursors → a cyclopentane ring is formed.
    assert forms_new_ring(["C=CC=CC", "C=CC"], "C1CCCC1") is True
    # acyclic coupling, no ring formed.
    assert forms_new_ring(["CCCCBr", "CCO"], "CCCC") is False
    # a ring already present in the precursor and carried through is not "new".
    assert forms_new_ring(["C1CCCCC1O"], "C1CCCCC1") is False


def test_forms_new_ring_known_blind_spot_cyclic_plus_cyclic():
    # DOCUMENTED LIMITATION (RETRACE §2): the real idx0 fake Diels–Alder fuses
    # two already-cyclic precursors (cyclopentadiene + maleic anhydride) into a
    # bicyclic adduct without raising the combined ring-atom total, so the
    # ring-atom metric does NOT flag it. This asserts that known blind spot so
    # nobody "fixes" the metric by loosening it (every looser ring metric drops
    # below ~52% precision — see the error analysis).
    reactants = ["O=C1C=C(CCl)C(=O)O1", "C1=CCC=C1"]
    product = "O=C1OC(=O)[C@@]2(CCl)C=CC[C@H]12"
    assert carbon_delta(reactants, product) < 0          # carbon IS lost
    assert forms_new_ring(reactants, product) is False   # but not flagged (blind spot)


# ── integration: the penalty inside _score_one_feasibility_step ─────────────

class _StubRater:
    """Always-confident rater so the penalty is the only thing moving the score."""

    def __init__(self):
        self.calls = 0

    def rate(self, reactants, product):
        self.calls += 1
        return {"confidence": 0.9, "feasible": True}


def _make_tool(monkeypatch, penalty):
    tool = ScoreRouteTool.__new__(ScoreRouteTool)
    tool.weights = {"feasibility": 1.0, "cost": 0.0, "greenness": 0.0}
    tool.carbon_fabrication_penalty = penalty
    tool._judge = None
    tool.judge_cfg = None
    tool._rater = _StubRater()
    monkeypatch.setattr(sr_mod, "warning_for_step", lambda r, p: None)
    return tool


def test_positive_cdelta_docks_score(monkeypatch):
    # +4 fabricated carbons × 0.2 = 0.8 penalty, capped at the 0.9 step score.
    tool = _make_tool(monkeypatch, penalty=0.2)
    entry, score, warns = tool._score_one_feasibility_step(
        0, {"reactants": ["CC=O", "O"], "product": "c1ccccc1"}
    )
    assert score == pytest.approx(0.9 - 0.8)
    assert entry["carbon_fabrication_penalty"] == pytest.approx(0.8)
    assert any("fabricated carbon" in w for w in warns)


def test_penalty_capped_at_step_score(monkeypatch):
    # A huge fabricated-carbon count can never drive the score below 0.
    tool = _make_tool(monkeypatch, penalty=0.5)
    _, score, _ = tool._score_one_feasibility_step(
        0, {"reactants": ["C"], "product": "c1ccc2ccccc2c1"}  # naphthalene, 10C
    )
    assert score == pytest.approx(0.0)


def test_negative_cdelta_not_penalised(monkeypatch):
    # A leaving fragment (negative delta) must keep the full rater score.
    tool = _make_tool(monkeypatch, penalty=0.2)
    entry, score, warns = tool._score_one_feasibility_step(
        0, {"reactants": ["CCCC"], "product": "CC"}
    )
    assert score == pytest.approx(0.9)
    assert "carbon_fabrication_penalty" not in entry
    assert not any("fabricated carbon" in w for w in warns)


def test_penalty_disabled_reverts_to_informational(monkeypatch):
    # carbon_fabrication_penalty = 0.0 → positive delta no longer docks.
    tool = _make_tool(monkeypatch, penalty=0.0)
    entry, score, _ = tool._score_one_feasibility_step(
        0, {"reactants": ["CC=O", "O"], "product": "c1ccccc1"}
    )
    assert score == pytest.approx(0.9)
    assert "carbon_fabrication_penalty" not in entry


# ── informational ring-carbon warning (negative delta, NEVER docks) ─────────
# A bimolecular ring-forming step that loses skeletal carbon is a possible
# wrong-cycloaddition product (RETRACE_ERROR_ANALYSIS §2). Surfaced as a
# warning only — a pure structural rule is ~51% precise, so it must not alter
# the score.

def test_ring_carbon_warning_fires_but_does_not_dock(monkeypatch):
    # 5C + 3C precursors → cyclopentane (5C): ring formed, 3 carbons vanish,
    # two independent precursors → suspect cycloaddition. Score must stay 0.9.
    tool = _make_tool(monkeypatch, penalty=0.2)
    entry, score, warns = tool._score_one_feasibility_step(
        0, {"reactants": ["C=CC=CC", "C=CC"], "product": "C1CCCC1"}
    )
    assert score == pytest.approx(0.9)  # informational only — never docked
    assert entry["ring_carbon_warning"]["carbon_delta"] == -3
    assert entry["ring_carbon_warning"]["n_precursors"] == 2
    assert any("possible wrong cycloaddition" in w for w in warns)
    assert "carbon_fabrication_penalty" not in entry


def test_ring_carbon_warning_skips_unimolecular_ring_closure(monkeypatch):
    # A unimolecular lactonisation loses a fragment (−3) and forms a ring, but
    # one precursor → a legitimate intramolecular closure, NOT a cycloaddition.
    # Must not warn and must not dock.
    tool = _make_tool(monkeypatch, penalty=0.2)
    entry, score, warns = tool._score_one_feasibility_step(
        0, {"reactants": ["OCCCCCC(=O)OCC"], "product": "O=C1CCCCO1"}
    )
    assert score == pytest.approx(0.9)
    assert "ring_carbon_warning" not in entry
    assert not any("cycloaddition" in w for w in warns)


def test_ring_carbon_warning_skips_carbon_conserved_cycloaddition(monkeypatch):
    # A real bimolecular Diels–Alder conserves carbon (delta 0) → no warning.
    tool = _make_tool(monkeypatch, penalty=0.2)
    entry, score, warns = tool._score_one_feasibility_step(
        0, {"reactants": ["C=CC=C", "C=C"], "product": "C1CCC=CC1"}
    )
    assert score == pytest.approx(0.9)
    assert "ring_carbon_warning" not in entry
    assert not any("cycloaddition" in w for w in warns)


def test_ring_carbon_warning_skips_acyclic_negative_delta(monkeypatch):
    # Bimolecular, negative delta, but NO ring formed → no warning.
    tool = _make_tool(monkeypatch, penalty=0.2)
    entry, score, _ = tool._score_one_feasibility_step(
        0, {"reactants": ["CCCCBr", "CCO"], "product": "CCCC"}
    )
    assert score == pytest.approx(0.9)
    assert "ring_carbon_warning" not in entry
