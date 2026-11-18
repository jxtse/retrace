"""Projection stub for explore_branches must surface the new pillar
scores (feasibility / cost / greenness) and a support-count rollup, while
keeping the payload under the 600-char default budget."""

import json

from services.result_store import _proj_explore_branches


def test_proj_keeps_pillar_scores_per_branch():
    payload = {
        "product": "CCO",
        "ranked": [
            {
                "name": "ox",
                "reactants": ["CC=O"],
                "score": 0.81,
                "feasibility": 0.84,
                "cost": 0.78,
                "greenness": 0.71,
                "support": [
                    {"source": "neural_retro:graph_retrosyn",
                     "match": "strict", "rank": 1},
                    {"source": "suggest_templates:enzymatic",
                     "match": "partial", "rank": 4},
                ],
            }
        ],
    }
    out = _proj_explore_branches(payload)
    branch = out["ranked"][0]
    assert branch["score"] == 0.81
    assert branch["feasibility"] == 0.84
    assert branch["cost"] == 0.78
    assert branch["greenness"] == 0.71
    assert branch["reactants"] == ["CC=O"]


def test_proj_includes_support_counts():
    payload = {
        "product": "CCO",
        "ranked": [{
            "name": "ox", "reactants": ["CC=O"],
            "score": 0.5, "feasibility": 0.5, "cost": 0.5, "greenness": 0.5,
            "support": [
                {"source": "a", "match": "strict",  "rank": 1},
                {"source": "b", "match": "strict",  "rank": 1},
                {"source": "c", "match": "partial", "rank": 1},
            ],
        }],
    }
    out = _proj_explore_branches(payload)
    branch = out["ranked"][0]
    assert branch["support_count_strict"] == 2
    assert branch["support_count_partial"] == 1


def test_proj_handles_missing_keys_gracefully():
    """Old-shape payloads (no support / pillar fields) must not crash the
    projector — they just project as zeros / empty lists."""
    payload = {"ranked": [{"name": "x", "reactants": ["CCO"], "score": 0.4}]}
    out = _proj_explore_branches(payload)
    branch = out["ranked"][0]
    assert branch["score"] == 0.4
    assert branch["feasibility"] == 0
    assert branch["support_count_strict"] == 0
    assert branch["support_count_partial"] == 0


def test_proj_handles_non_list_support():
    """If support is a string or None, projector treats it as empty."""
    payload = {
        "ranked": [{
            "name": "x", "reactants": ["CCO"], "score": 0.4,
            "support": "garbage",
        }],
    }
    out = _proj_explore_branches(payload)
    assert out["ranked"][0]["support_count_strict"] == 0
    assert out["ranked"][0]["support_count_partial"] == 0


def test_proj_handles_non_dict_support_items():
    """Items inside support that aren't dicts must not crash the projector."""
    payload = {
        "ranked": [{
            "name": "x", "reactants": ["CCO"], "score": 0.4,
            "support": ["strict", None, {"match": "strict", "rank": 1}],
        }],
    }
    out = _proj_explore_branches(payload)
    # Only the well-formed dict counts.
    assert out["ranked"][0]["support_count_strict"] == 1
    assert out["ranked"][0]["support_count_partial"] == 0


def test_proj_skips_non_dict_ranked_items():
    """Items inside ranked that aren't dicts must be silently skipped."""
    payload = {
        "ranked": [
            "junk",
            None,
            {"name": "x", "reactants": ["CCO"], "score": 0.4},
        ],
    }
    out = _proj_explore_branches(payload)
    # Only the well-formed dict survives.
    assert len(out["ranked"]) == 1
    assert out["ranked"][0]["score"] == 0.4
