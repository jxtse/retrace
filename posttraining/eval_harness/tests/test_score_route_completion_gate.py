import json
from pathlib import Path


def test_incomplete_route_is_capped_and_surfaced():
    """A high-quality partial route cannot look like a completed high score."""
    src = Path("tools/verifier/score_route.py").read_text()
    ns = {}
    # Avoid constructing heavyweight model-backed dependencies: execute() only
    # needs these fields/methods, so instantiate without __init__ and stub the
    # three pillars deterministically.
    from tools.verifier.score_route import RouteVerifierTool
    tool = object.__new__(RouteVerifierTool)
    tool.weights = {"feasibility": 0.70, "cost": 0.15, "greenness": 0.15}
    tool.incomplete_route_score_cap = 0.6
    tool.pillar_parallel = False
    tool.result_store = None
    tool._score_feasibility = lambda steps, notes: {
        "score": 0.99, "summary": {"score": 0.99}, "per_step": []
    }
    tool._score_cost = lambda smiles, steps, notes: {
        "score": 0.80,
        "summary": {"score": 0.80},
        "all_leaves_buyable": False,
        "buyable_leaf_fraction": 0.5,
        "non_buyable_leaves": ["CC_COMPLEX_CORE"],
        "leaves": [],
    }
    tool._score_greenness = lambda steps, notes: {
        "score": 0.90, "summary": {"score": 0.90}
    }
    out = json.loads(tool.execute(
        smiles="CCO", route_steps=[{"product": "CCO", "reactants": ["CC", "O"]}]
    ))
    assert out["route_complete"] is False
    assert out["total_score"] == 0.6
    assert out["uncapped_quality_score"] > 0.9
    assert out["buyable_leaf_fraction"] == 0.5
    assert out["non_buyable_leaves"] == ["CC_COMPLEX_CORE"]
    assert any("incomplete route" in x for x in out["warnings"])


def test_complete_route_keeps_uncapped_score():
    from tools.verifier.score_route import RouteVerifierTool
    tool = object.__new__(RouteVerifierTool)
    tool.weights = {"feasibility": 0.70, "cost": 0.15, "greenness": 0.15}
    tool.incomplete_route_score_cap = 0.6
    tool.pillar_parallel = False
    tool.result_store = None
    tool._score_feasibility = lambda steps, notes: {
        "score": 0.99, "summary": {"score": 0.99}, "per_step": []
    }
    tool._score_cost = lambda smiles, steps, notes: {
        "score": 0.80, "summary": {"score": 0.80},
        "all_leaves_buyable": True, "buyable_leaf_fraction": 1.0,
        "non_buyable_leaves": [], "leaves": [],
    }
    tool._score_greenness = lambda steps, notes: {
        "score": 0.90, "summary": {"score": 0.90}
    }
    out = json.loads(tool.execute(
        smiles="CCO", route_steps=[{"product": "CCO", "reactants": ["CC", "O"]}]
    ))
    assert out["route_complete"] is True
    assert out["total_score"] == out["uncapped_quality_score"]
    assert out["total_score"] > 0.9
