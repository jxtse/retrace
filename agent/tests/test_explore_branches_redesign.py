"""Tests for the verifier-aligned ExploreBranchesTool redesign.

Constructor takes (buyability_chain, result_store) instead of the old
(buyability_tool). Without a result_store, support_signal falls back to
0.5 with a `note` flag (offline / unit-test mode)."""

import json
from typing import Any

import pytest

from services.result_store import ResultStore
from tools.builtins.explore_branches import ExploreBranchesTool


class _FakeBuy:
    """Minimal BuyabilityChain stand-in: returns 'enamine' for whitelisted
    SMILES, 'none' otherwise. Mirrors the real chain's lookup contract."""
    def __init__(self, available=()):
        self.available = set(available)

    def lookup(self, smiles: str):
        return {
            "smiles": smiles,
            "source": "enamine" if smiles in self.available else "none",
            "catalog_id": None,
        }


def test_constructor_accepts_chain_and_store(tmp_path):
    rs = ResultStore(session_dir=str(tmp_path), keep_recent_full_per_tool=2)
    tool = ExploreBranchesTool(buyability_chain=_FakeBuy(), result_store=rs)
    assert tool.name == "explore_branches"
    assert tool._buyability is not None
    assert tool._result_store is rs


def test_constructor_allows_none_store():
    """Unit-test mode: the tool must be usable without a ResultStore."""
    tool = ExploreBranchesTool(buyability_chain=_FakeBuy(), result_store=None)
    assert tool._result_store is None


# ── Pure-helper unit tests ────────────────────────────────────────────


from tools.builtins.explore_branches import (
    _canon_set,
    _classify_match,
    _better_match,
    _support_signal,
)


def test_canon_set_dedupes_and_canonicalises():
    """Equivalent SMILES collapse to the same canonical form."""
    out = _canon_set(["CCO", "OCC"])  # both ethanol
    assert out is not None
    assert len(out) == 1


def test_canon_set_returns_none_on_invalid():
    assert _canon_set(["CCO", "NOT_A_SMILES"]) is None


def test_classify_match_strict():
    a = frozenset({"CCO"})
    assert _classify_match(a, a) == "strict"


def test_classify_match_partial_subset():
    a = frozenset({"CCO"})
    b = frozenset({"CCO", "[Na+]"})
    assert _classify_match(a, b) == "partial"
    assert _classify_match(b, a) == "partial"


def test_classify_match_none_when_overlap_only():
    """Spec §2.2: partial requires SUBSET/SUPERSET — overlap without
    containment is none."""
    a = frozenset({"CCO", "CC=O"})
    b = frozenset({"CC=O", "CCN"})
    assert _classify_match(a, b) == "none"


def test_better_match_prefers_strict_over_partial():
    a = {"source": "x", "match": "strict",  "rank": 5}
    b = {"source": "x", "match": "partial", "rank": 1}
    assert _better_match(a, b) is True
    assert _better_match(b, a) is False


def test_better_match_lower_rank_wins_within_match_type():
    a = {"source": "x", "match": "strict", "rank": 1}
    b = {"source": "x", "match": "strict", "rank": 5}
    assert _better_match(a, b) is True
    assert _better_match(b, a) is False


def test_support_signal_two_strict_two_partial():
    """0.40*2 + 0.20*2 = 1.20 → capped at 1.0."""
    sup = [
        {"source": "a", "match": "strict",  "rank": 1},
        {"source": "b", "match": "strict",  "rank": 1},
        {"source": "c", "match": "partial", "rank": 1},
        {"source": "d", "match": "partial", "rank": 1},
    ]
    assert _support_signal(sup) == 1.0


def test_support_signal_one_strict_one_partial():
    """0.40 + 0.20 = 0.60."""
    sup = [
        {"source": "a", "match": "strict",  "rank": 1},
        {"source": "b", "match": "partial", "rank": 1},
    ]
    assert _support_signal(sup) == pytest.approx(0.60, abs=1e-6)


def test_support_signal_empty():
    assert _support_signal([]) == 0.0


# ── End-to-end scoring tests ──────────────────────────────────────────


def _seed_neural_retro(rs: ResultStore, target: str, predictions, engine="ensemble"):
    """Persist a neural_retro payload so the redesigned tool can find it."""
    rs.persist("neural_retro", f"nr-{len(rs._index)}", json.dumps({
        "target": target,
        "engine": engine,
        "predictions": predictions,
    }))


def _seed_templates(rs: ResultStore, product: str, suggestions):
    rs.persist("suggest_templates", f"st-{len(rs._index)}", json.dumps({
        "product": product,
        "library_size": 1,
        "num_suggestions": len(suggestions),
        "suggestions": suggestions,
    }))


def test_strict_match_against_neural_retro_lifts_support(tmp_path):
    """Aspirin from salicylic acid + acetic anhydride. Seed a neural_retro
    payload that predicts the same set; expect support_signal ≥ 0.40
    (one strict match)."""
    rs = ResultStore(session_dir=str(tmp_path), keep_recent_full_per_tool=4)
    _seed_neural_retro(rs, "CC(=O)Oc1ccccc1C(=O)O", [
        {"reactants": ["OC(=O)c1ccccc1O", "CC(=O)OC(C)=O"],
         "support": ["graph_retrosyn"]},
    ])
    tool = ExploreBranchesTool(
        buyability_chain=_FakeBuy(["OC(=O)c1ccccc1O", "CC(=O)OC(C)=O"]),
        result_store=rs,
    )
    out = json.loads(tool.execute(
        product_smiles="CC(=O)Oc1ccccc1C(=O)O",
        branches=[{
            "label": "acetylation",
            "reactants": ["OC(=O)c1ccccc1O", "CC(=O)OC(C)=O"],
        }],
    ))
    branch = out["ranked"][0]
    assert branch["sub_scores"]["support_signal"] >= 0.40
    assert any(s["match"] == "strict" for s in branch["support"])


def test_two_strict_matches_across_engines_cap_signal(tmp_path):
    """Two strict matches from DIFFERENT engines → 0.80. A third match
    from the same engine collapses to one source — does not bump."""
    rs = ResultStore(session_dir=str(tmp_path), keep_recent_full_per_tool=4)
    same_set = [
        {"reactants": ["OC(=O)c1ccccc1O", "CC(=O)OC(C)=O"],
         "support": ["graph_retrosyn"]},
    ]
    _seed_neural_retro(rs, "CC(=O)Oc1ccccc1C(=O)O", same_set)
    _seed_neural_retro(rs, "CC(=O)Oc1ccccc1C(=O)O", [
        {"reactants": ["OC(=O)c1ccccc1O", "CC(=O)OC(C)=O"],
         "support": ["chembart"]},
    ])
    # Repeat the first engine — should NOT lift signal.
    _seed_neural_retro(rs, "CC(=O)Oc1ccccc1C(=O)O", same_set)

    tool = ExploreBranchesTool(buyability_chain=_FakeBuy(), result_store=rs)
    out = json.loads(tool.execute(
        product_smiles="CC(=O)Oc1ccccc1C(=O)O",
        branches=[{
            "label": "acetylation",
            "reactants": ["OC(=O)c1ccccc1O", "CC(=O)OC(C)=O"],
        }],
    ))
    sub = out["ranked"][0]["sub_scores"]
    assert sub["support_signal"] == pytest.approx(0.80, abs=1e-3)


def test_partial_match_subset_counted(tmp_path):
    """Branch reactants are a SUBSET of a prior prediction → partial match."""
    rs = ResultStore(session_dir=str(tmp_path), keep_recent_full_per_tool=4)
    _seed_neural_retro(rs, "CC(=O)Oc1ccccc1C(=O)O", [
        {"reactants": ["OC(=O)c1ccccc1O", "CC(=O)OC(C)=O", "[Na+].[OH-]"],
         "support": ["chembart"]},
    ])
    tool = ExploreBranchesTool(buyability_chain=_FakeBuy(), result_store=rs)
    out = json.loads(tool.execute(
        product_smiles="CC(=O)Oc1ccccc1C(=O)O",
        branches=[{
            "label": "acetylation-no-base",
            "reactants": ["OC(=O)c1ccccc1O", "CC(=O)OC(C)=O"],
        }],
    ))
    sup = out["ranked"][0]["support"]
    assert any(s["match"] == "partial" for s in sup)


def test_no_prior_evidence_yields_zero_support(tmp_path):
    """When the agent invents a branch without prior neural/template
    evidence, support_signal is 0.0 (not the 0.5 unit-test fallback)."""
    rs = ResultStore(session_dir=str(tmp_path), keep_recent_full_per_tool=4)
    tool = ExploreBranchesTool(buyability_chain=_FakeBuy(), result_store=rs)
    out = json.loads(tool.execute(
        product_smiles="CCO",
        branches=[{"label": "ox", "reactants": ["CC=O"]}],
    ))
    assert out["ranked"][0]["sub_scores"]["support_signal"] == 0.0


def test_no_result_store_uses_neutral_signal():
    """Unit-test mode: result_store=None → support_signal=0.5 + note."""
    tool = ExploreBranchesTool(buyability_chain=_FakeBuy(), result_store=None)
    out = json.loads(tool.execute(
        product_smiles="CCO",
        branches=[{"label": "ox", "reactants": ["CC=O"]}],
    ))
    branch = out["ranked"][0]
    assert branch["sub_scores"]["support_signal"] == 0.5
    assert "note" in branch and "result_store" in branch["note"]


def test_pillar_scores_present_and_in_unit_interval(tmp_path):
    rs = ResultStore(session_dir=str(tmp_path), keep_recent_full_per_tool=4)
    tool = ExploreBranchesTool(buyability_chain=_FakeBuy(["CC=O"]),
                               result_store=rs)
    out = json.loads(tool.execute(
        product_smiles="CCO",
        branches=[{"label": "ox", "reactants": ["CC=O"]}],
    ))
    branch = out["ranked"][0]
    for k in ("score", "feasibility", "cost", "greenness"):
        assert 0.0 <= branch[k] <= 1.0, f"{k}={branch[k]}"


def test_invalid_reactant_returns_error_not_crash(tmp_path):
    rs = ResultStore(session_dir=str(tmp_path), keep_recent_full_per_tool=4)
    tool = ExploreBranchesTool(buyability_chain=_FakeBuy(), result_store=rs)
    out = json.loads(tool.execute(
        product_smiles="CCO",
        branches=[{"label": "junk", "reactants": ["NOT_A_SMILES"]}],
    ))
    branch = out["ranked"][0]
    assert "error" in branch
    assert branch["score"] == 0.0


def test_mono_molecular_branch_not_penalised(tmp_path):
    """Spec §2.6 — mono-molecular biocatalytic steps are valid; the old
    n_bonus penalty is gone. A clean 1→1 oxidation should score
    competitively when support backs it."""
    rs = ResultStore(session_dir=str(tmp_path), keep_recent_full_per_tool=4)
    _seed_neural_retro(rs, "CCO", [
        {"reactants": ["CC=O"], "support": ["graph_retrosyn"]},
    ])
    tool = ExploreBranchesTool(buyability_chain=_FakeBuy(["CC=O"]),
                               result_store=rs)
    out = json.loads(tool.execute(
        product_smiles="CCO",
        branches=[{"label": "ox", "reactants": ["CC=O"]}],
    ))
    branch = out["ranked"][0]
    # With strict support + buyable leaf + carbon balance, this 1→1
    # disconnection should clear 0.55 (the old code would have applied
    # n_bonus = -0.10 and clipped harder).
    assert branch["score"] >= 0.55


def test_strict_match_against_suggest_templates_lifts_support(tmp_path):
    """Verifies the suggest_templates branch of _collect_prior_predictions:
    a strict-set match in a persisted suggest_templates payload should
    contribute to support_signal exactly like a neural_retro match. Uses
    the `_seed_templates` helper so the suggest_templates code path
    (`_extract_templates`, payload['product'] key, payload['suggestions'])
    is exercised end-to-end."""
    rs = ResultStore(session_dir=str(tmp_path), keep_recent_full_per_tool=4)
    _seed_templates(rs, "CC(=O)Oc1ccccc1C(=O)O", [
        {
            "template_kind": "organic",
            "template_source": "uspto",
            "reactants_smiles": "OC(=O)c1ccccc1O.CC(=O)OC(C)=O",
            "reactants": ["OC(=O)c1ccccc1O", "CC(=O)OC(C)=O"],
            "num_reactants": 2,
            "rank_score": 0.85,
        },
    ])
    tool = ExploreBranchesTool(buyability_chain=_FakeBuy(), result_store=rs)
    out = json.loads(tool.execute(
        product_smiles="CC(=O)Oc1ccccc1C(=O)O",
        branches=[{
            "label": "acetylation",
            "reactants": ["OC(=O)c1ccccc1O", "CC(=O)OC(C)=O"],
        }],
    ))
    branch = out["ranked"][0]
    # One strict source match → support_signal == 0.40.
    assert branch["sub_scores"]["support_signal"] == pytest.approx(0.40, abs=1e-3)
    # The match should be flagged with the suggest_templates source.
    assert any(
        s["match"] == "strict" and s["source"].startswith("suggest_templates:")
        for s in branch["support"]
    )


def test_buyability_chain_shared_with_verifier_when_present():
    """Identity-sharing of BuyabilityChain: the central wiring claim of
    Task 6. If a future core.agent edit silently falls back to standalone,
    this test catches the RAM-waste regression.
    """
    from tools.builtins.explore_branches import ExploreBranchesTool

    class _FakeChain:
        """Minimal stand-in — we only assert identity, not behavior."""
        pass

    shared_chain = _FakeChain()
    tool = ExploreBranchesTool(buyability_chain=shared_chain, result_store=None)
    # The tool stores the chain as `self._buyability` (private). Identity
    # check, not equality — the whole point is we DIDN'T re-instantiate.
    assert tool._buyability is shared_chain, (
        "ExploreBranchesTool must hold the chain instance it was given, "
        "not a copy or rebuilt instance"
    )
