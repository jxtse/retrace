"""lean5 (2026-09-16): the student scored a route containing `OCl=CCl`
(Cl with valence 3, RDKit rejects it). The vendored ASKCOS
`get_n_conditions(..., return_scores=True)` catches every exception and
returns `[[]]` — a single value — so the adapter's `combos, scores = ...`
raised ValueError and score_route returned a whole-tool
{"error": "not enough values to unpack"} instead of a per-step verdict.

Contracts:
1. ConditionPredictor.predict_top1 must translate the vendored failure
   sentinel into a clean ConditionPredictionError, never a bare ValueError.
2. score_route's greenness pillar must record a per-step error and score
   that step 0.0 instead of failing the whole tool call (the model already
   gets a per-step feasibility failure for invalid SMILES; greenness should
   behave the same way).
"""
import pytest

from tools.verifier import condition_prediction as cp


class _StubModel:
    def __init__(self, mode):
        self.mode = mode

    def get_n_conditions(self, rxn, n=10, singleSlvt=True, with_smiles=True, return_scores=False):
        if self.mode == "sentinel":
            return [[]]                     # vendored failure sentinel, ignores return_scores
        if self.mode == "empty_pair":
            return ([], [])
        if self.mode == "ok":
            return ([[25.0, "water", "", ""]], [0.9])
        raise AssertionError(self.mode)


def _predictor(mode):
    p = cp.ConditionPredictor.__new__(cp.ConditionPredictor)
    p._model = _StubModel(mode)
    p._ensure_loaded = lambda: p._model
    # predict_top1 is lru_cached per instance method in the original; bypass cache
    p._predict_cached = lambda rxn: cp.ConditionPredictor._predict_uncached(p, rxn)
    return p


def test_sentinel_return_raises_clean_error():
    p = _predictor("sentinel")
    with pytest.raises(cp.ConditionPredictionError):
        p.predict_top1("OCl=CCl.CC>>CCCl")


def test_empty_pair_raises_clean_error():
    p = _predictor("empty_pair")
    with pytest.raises(cp.ConditionPredictionError):
        p.predict_top1("C>>C")


def test_ok_path_unchanged():
    p = _predictor("ok")
    out = p.predict_top1("CC(=O)O.OCC>>CC(=O)OCC")
    assert out["temperature"] == 25.0 and out["solvent"] == "water" and out["score"] == 0.9


def test_greenness_pillar_isolates_failed_step():
    """A step whose condition prediction fails scores 0 and is reported, the
    other steps and the pillar total are still computed."""
    from tools.verifier.score_route import RouteVerifierTool

    tool = RouteVerifierTool.__new__(RouteVerifierTool)

    class _Cond:
        def predict_top1(self, rxn):
            if "OCl=CCl" in rxn:
                raise cp.ConditionPredictionError("bad smiles")
            return {"temperature": 25.0, "solvent": "water", "reagent": "", "catalyst": "", "score": 0.9}

    tool.condition = _Cond()
    steps = [
        {"product": "CCCl", "reactants": ["OCl=CCl", "CC"]},
        {"product": "CC(=O)OCC", "reactants": ["CC(=O)O", "OCC"]},
    ]
    notes = []
    out = tool._score_greenness(steps, notes)
    per_step = out.get("per_step") or out.get("steps") or []
    assert any("error" in s for s in per_step), out
    assert any("condition prediction failed" in n for n in notes), notes
    assert 0.0 <= float(out["score"]) < 1.0
