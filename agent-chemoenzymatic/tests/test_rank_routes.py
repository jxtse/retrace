"""rank_routes returns every saved route, sorted by total_score, with
sub-score breakdown surfaced.
"""
import json

import pytest

from tools.scratchpad import RouteEntry, ScratchpadStore
from tools.builtins.rank_routes import RankRoutesTool


def _populate(store):
    store.save(RouteEntry(
        label="route_A",
        total_score=0.62,
        steps=["c1ccccc1Cl >> c1ccccc1.Cl"],
        sub_scores={"feasibility": 0.55, "cost": 0.7, "greenness": 0.7},
        starting_materials=["c1ccccc1"],
        notes="open-ring strategy",
    ))
    store.save(RouteEntry(
        label="route_B",
        total_score=0.81,
        steps=["CCO >> CC=O", "CC=O >> CC"],
        sub_scores={"feasibility": 0.85, "cost": 0.78, "greenness": 0.78},
        starting_materials=["CC"],
        notes="textbook reduction",
    ))
    store.save(RouteEntry(
        label="route_C",
        total_score=0.74,
        steps=["c1ccc(O)cc1 >> c1ccccc1"],
        sub_scores={"feasibility": 0.7, "cost": 0.8, "greenness": 0.75},
        starting_materials=["c1ccccc1"],
        notes=None,
    ))


def test_rank_routes_orders_by_total_score_descending():
    store = ScratchpadStore(max_routes=10)
    _populate(store)

    tool = RankRoutesTool(scratchpad_store=store)
    out = json.loads(tool.execute())

    labels = [r["label"] for r in out["routes"]]
    assert labels == ["route_B", "route_C", "route_A"]
    assert out["routes"][0]["rank"] == 1
    assert out["routes"][0]["total_score"] == pytest.approx(0.81)
    # Sub-scores surfaced.
    assert "feasibility" in out["routes"][0]
    assert "cost" in out["routes"][0]
    assert "greenness" in out["routes"][0]
    assert out["schema_version"] == 1


def test_rank_routes_empty_store():
    store = ScratchpadStore(max_routes=10)
    tool = RankRoutesTool(scratchpad_store=store)
    out = json.loads(tool.execute())
    assert out["routes"] == []
    assert out["schema_version"] == 1
