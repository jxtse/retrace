"""Smoke test for scripts/trace_route.py — the offline traceability CLI.

After the `_detail` persist seam (2026-06-20), a chemist can join a run's saved
routes back to (a) the per-step score detail of score_route and (b) the raw
per-engine predictions of neural_retro. trace_route.py is the reader. These
tests build a minimal synthetic session dir (transcript.json + results/) and
assert the join path + formatters surface the detail, and that a pre-upgrade
run (no `_detail`) degrades gracefully rather than crashing.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import trace_route as tr  # noqa: E402


def _session(tmp, *, with_detail):
    os.makedirs(os.path.join(tmp, "results"), exist_ok=True)

    # one saved route, joined to a score_route result via save_best_route.
    transcript = {
        "scratchpad": {
            "routes": [
                {"label": "acetylation", "total_score": 0.81,
                 "sub_scores": {"feasibility": 0.9}, "num_steps": 1,
                 "steps": [{"product": "CC(=O)Oc1ccccc1C(=O)O"}]},
            ],
        },
        "events": [
            {"event": "tool_call", "name": "save_best_route",
             "args": {"label": "acetylation",
                      "score_route_result_id": "call_sr1"}},
        ],
    }
    with open(os.path.join(tmp, "transcript.json"), "w") as fh:
        json.dump(transcript, fh)

    sr_content = {"total_score": 0.81, "feasibility_score": 0.9}
    if with_detail:
        sr_content["_detail"] = {
            "weights": {"feasibility": 0.7, "cost": 0.15, "greenness": 0.15},
            "feasibility": {"score": 0.9, "per_step": [
                {"step": 0, "rater": {"confidence": 0.9}, "final_score": 0.9}]},
            "cost": {"score": 0.6, "step_efficiency": 0.8, "leaf_economics": 0.5,
                     "sc_decrease_score": 0.4,
                     "leaves": [{"smiles": "CC(=O)Cl", "leaf_score": 0.5,
                                 "price_per_g": None, "vendor": "enamine"}]},
            "greenness": {"score": 0.83, "per_step": [
                {"step": 0, "atom_economy": 0.83,
                 "solvent": {"name": "water", "class": "green"}}]},
        }
    with open(os.path.join(tmp, "results", "call_sr1.json"), "w") as fh:
        json.dump({"tool": "score_route", "tool_call_id": "call_sr1",
                   "content": sr_content}, fh)

    # a neural_retro result with per-engine raw top-K under _detail.
    nr_content = {
        "target": "CC(=O)Oc1ccccc1C(=O)O",
        "predictions": [{"reactants": "CC(=O)OC(C)=O.O=C(O)c1ccccc1O",
                         "score": 1.0, "support": ["chembart", "rxngraphormer"]}],
    }
    if with_detail:
        nr_content["_detail"] = {
            "per_engine_predictions": {
                "chembart": [{"reactants": "CC(=O)OC(C)=O.O=C(O)c1ccccc1O",
                              "score": 0.97}],
                "rxngraphormer": [{"reactants": "CC(=O)OC(C)=O.O=C(O)c1ccccc1O",
                                   "score": 1.0}],
            },
            "per_engine_counts": {"chembart": 50, "rxngraphormer": 76},
            "engine_errors": {},
        }
    with open(os.path.join(tmp, "results", "call_nr1.json"), "w") as fh:
        json.dump({"tool": "neural_retro", "tool_call_id": "call_nr1",
                   "content": nr_content}, fh)
    return tmp


def test_trace_routes_joins_and_surfaces_detail():
    with tempfile.TemporaryDirectory() as tmp:
        _session(tmp, with_detail=True)
        traced = tr.trace_routes(tmp, None)
        assert len(traced) == 1
        t = traced[0]
        assert t["label"] == "acetylation"
        assert t["score_route_result_id"] == "call_sr1"
        # the join recovered the on-disk _detail.
        assert t["detail"]["feasibility"]["per_step"][0]["final_score"] == 0.9
        # formatters render without error and carry the key facts.
        feas = "\n".join(tr._fmt_feasibility(t["detail"]))
        assert "rater-only" in feas and "final=0.9" in feas
        cost = "\n".join(tr._fmt_cost(t["detail"]))
        assert "enamine" in cost
        green = "\n".join(tr._fmt_greenness(t["detail"]))
        assert "green" in green


def test_trace_route_filter():
    with tempfile.TemporaryDirectory() as tmp:
        _session(tmp, with_detail=True)
        assert len(tr.trace_routes(tmp, "acetyl")) == 1
        assert len(tr.trace_routes(tmp, "no-such-route")) == 0


def test_per_engine_predictions_reads_raw_topk():
    with tempfile.TemporaryDirectory() as tmp:
        _session(tmp, with_detail=True)
        engines = tr.per_engine_predictions(tmp)
        assert len(engines) == 1
        pep = engines[0]["per_engine_predictions"]
        assert set(pep.keys()) == {"chembart", "rxngraphormer"}
        # each engine's OWN raw score for the shared precursor set is preserved,
        # not the max-collapsed 1.0 the LLM saw.
        assert pep["chembart"][0]["score"] == 0.97
        assert pep["rxngraphormer"][0]["score"] == 1.0


def test_pre_upgrade_run_degrades_gracefully():
    # A run from before the _detail seam: results have no _detail. The join must
    # still resolve result ids and simply report detail=None, not crash.
    with tempfile.TemporaryDirectory() as tmp:
        _session(tmp, with_detail=False)
        traced = tr.trace_routes(tmp, None)
        assert traced[0]["score_route_result_id"] == "call_sr1"
        assert traced[0]["detail"] is None
        assert tr.per_engine_predictions(tmp) == []
