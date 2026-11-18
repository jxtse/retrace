"""Audit-only `_detail` traceability seam (2026-06-20).

After a run a chemist must be able to trace, per route:
  * WHICH neural engine produced WHICH precursor set (and that engine's own
    raw score/template, before the ensemble max-collapsed it), and
  * the per-step scoring detail of score_route — each step's final
    feasibility score, whether a judge ran (rater-only vs rater+judge), and
    the cost / greenness per-step breakdown.

None of this used to survive: score_route and neural_retro computed rich
per-step structures but returned only scalar aggregates, so nothing reached
the result store. The fix is a `_detail` convention at the persist boundary:
a tool attaches a heavy block under the reserved `_detail` key;
ResultStore.persist writes it to DISK in full (recoverable via load_result)
but STRIPS it from the string that re-enters the message stream, so it never
costs context tokens — not even in the first-write recency window, and not in
a demoted projection stub.

These tests pin all three layers:
  1. the persist convention (disk full / message-stream slim / load full),
  2. score_route's `_detail` (per-step feasibility final_score + backend
     attribution + cost leaves + greenness per_step),
  3. neural_retro's `_detail` (per_engine_predictions raw top-K).
"""
import json
import os
import tempfile

from services.result_store import ResultStore, _proj_score_route, _proj_neural_retro


# ── 1. The persist convention ────────────────────────────────────────────────


def _store(tmp):
    return ResultStore(
        session_dir=tmp, summary_max_chars=800, keep_recent_full_per_tool=4,
    )


def _payload_with_detail():
    return {
        "total_score": 0.81,
        "feasibility_score": 0.75,
        "n_steps": 2,
        "route_fp": "abc123",
        "_detail": {"feasibility": {"per_step": [{"step": 0, "final_score": 0.72}]}},
    }


def test_persist_strips_detail_from_returned_content():
    # The string that re-enters the message stream must be slim.
    with tempfile.TemporaryDirectory() as tmp:
        rs = _store(tmp)
        returned = rs.persist("score_route", "tc1", json.dumps(_payload_with_detail()))
        rdict = json.loads(returned)
        assert "_detail" not in rdict
        # The slim scalars and the injected _result_id survive.
        assert rdict["total_score"] == 0.81
        assert rdict["_result_id"] == "tc1"


def test_persist_keeps_detail_on_disk():
    # The on-disk file retains the full _detail for offline audit.
    with tempfile.TemporaryDirectory() as tmp:
        rs = _store(tmp)
        rs.persist("score_route", "tc1", json.dumps(_payload_with_detail()))
        blob = json.load(open(os.path.join(tmp, "results", "tc1.json")))
        content = blob["content"]
        assert "_detail" in content
        assert content["_detail"]["feasibility"]["per_step"][0]["final_score"] == 0.72


def test_load_result_returns_detail():
    # load_result reads disk, so it recovers the full detail.
    with tempfile.TemporaryDirectory() as tmp:
        rs = _store(tmp)
        rs.persist("score_route", "tc1", json.dumps(_payload_with_detail()))
        loaded = rs.load("tc1")
        assert "_detail" in loaded["content"]


def test_demoted_stub_omits_detail():
    # Push past the recency window so tc1 demotes, then its projection stub
    # must still be slim (the allowlist projector never copies _detail).
    with tempfile.TemporaryDirectory() as tmp:
        rs = ResultStore(session_dir=tmp, summary_max_chars=800,
                         keep_recent_full_per_tool=1)
        rs.persist("score_route", "tc1", json.dumps(_payload_with_detail()))
        rs.persist("score_route", "tc2", json.dumps(_payload_with_detail()))
        stub = rs.stub_for("tc1")
        assert "_detail" not in json.loads(stub)


def test_payload_without_detail_unchanged():
    # A tool that attaches no _detail behaves exactly as before.
    with tempfile.TemporaryDirectory() as tmp:
        rs = _store(tmp)
        returned = rs.persist(
            "score_route", "tc1",
            json.dumps({"total_score": 0.5, "n_steps": 1}),
        )
        rdict = json.loads(returned)
        assert rdict["total_score"] == 0.5
        blob = json.load(open(os.path.join(tmp, "results", "tc1.json")))
        assert blob["content"]["total_score"] == 0.5


# ── 2. Projectors never leak _detail ─────────────────────────────────────────


def test_proj_score_route_omits_detail():
    payload = {"total_score": 0.8, "feasibility_score": 0.7, "_detail": {"x": 1}}
    out = _proj_score_route(payload)
    assert "_detail" not in out
    assert out["total_score"] == 0.8


def test_proj_neural_retro_omits_detail():
    payload = {
        "engine": "ensemble",
        "predictions": [{"reactants": "CCO", "support": ["chembart"], "score": 0.9}],
        "_detail": {"per_engine_predictions": {"chembart": [{"reactants": "CCO"}]}},
    }
    out = _proj_neural_retro(payload)
    assert "_detail" not in out
    # support attribution still surfaces to the LLM in the stub.
    assert out["predictions"][0]["support"] == ["chembart"]


# ── 3. score_route assembles a _detail block that the slim payload omits ──────


def test_score_route_execute_emits_detail():
    """execute() must (a) keep the returned string slim — no _detail — and
    (b) carry the full per-pillar detail under _detail so persist can disk it.

    We bypass-construct the tool and stub all three pillars with canned
    per-step structures; this isolates the `out`/`_detail` assembly from the
    rater / cost / greenness backends.
    """
    from tools.verifier.score_route import RouteVerifierTool

    tool = RouteVerifierTool.__new__(RouteVerifierTool)
    tool.weights = {"feasibility": 0.7, "cost": 0.15, "greenness": 0.15}
    tool.pillar_parallel = False

    feas = {
        "score": 0.75,
        "summary": {"score": 0.75, "warnings": []},
        "per_step": [
            {"step": 0, "rater": {"confidence": 0.6, "feasible": True},
             "final_score": 0.72,
             "step_judge": {"score": 0.8, "verdict": "plausible",
                            "lifted_from": 0.6, "lifted_to": 0.72}},
            {"step": 1, "rater": {"confidence": 0.9, "feasible": True},
             "final_score": 0.9},
        ],
    }
    cost = {
        "score": 0.62, "step_efficiency": 0.7, "leaf_economics": 0.5,
        "sc_decrease_score": 0.4,
        "leaves": [{"smiles": "CCO", "leaf_score": 1.0, "price_per_g": 0.1,
                    "vendor": "Sigma"}],
        "summary": {"score": 0.62, "warnings": []},
    }
    green = {
        "score": 0.9,
        "summary": {"score": 0.9, "warnings": []},
        "per_step": [{"step": 0, "atom_economy": 0.8,
                      "solvent": {"name": "water", "class": "green"}}],
    }
    tool._score_feasibility = lambda steps, notes: feas
    tool._score_cost = lambda smi, steps, notes: cost
    tool._score_greenness = lambda steps, notes: green

    route = [
        {"reactants": ["CC=O"], "product": "CCO"},
        {"reactants": ["CCCl"], "product": "CCCC"},
    ]
    out = json.loads(tool.execute(smiles="CCO", route_steps=route))

    # The LLM-facing payload is slim — no _detail in the returned string.
    # (persist would strip it anyway; this proves execute also routes the
    # detail there, not into the scalar surface.)
    assert "_detail" in out  # present in the FULL returned dict (pre-persist)
    d = out["_detail"]
    # Feasibility per-step carries the post-judge final_score + backend tag.
    fs = d["feasibility"]["per_step"]
    assert fs[0]["final_score"] == 0.72
    assert "step_judge" in fs[0]          # rater+judge backend
    assert "step_judge" not in fs[1]      # rater-only backend
    # Cost leaves carry vendor / price for provenance.
    assert d["cost"]["leaves"][0]["vendor"] == "Sigma"
    # Greenness per-step carries atom economy + solvent class.
    assert d["greenness"]["per_step"][0]["solvent"]["class"] == "green"
    # Pillar weights recorded so a re-derivation is possible offline.
    assert d["weights"]["feasibility"] == 0.7

    # And the scalar pillar surface is still exactly what the LLM acts on.
    assert out["feasibility_score"] == 0.75
    assert out["cost_score"] == 0.62
    assert out["greenness_score"] == 0.9


# ── 4. neural_retro ensemble carries per-engine raw predictions ───────────────


def test_neural_retro_ensemble_emits_per_engine_detail(monkeypatch):
    """_run_ensemble must attach per-engine raw top-K under _detail, so post-
    run one can see exactly what each engine produced (and its own score for a
    precursor set) before dedup max-collapsed the support into one entry."""
    import tools.neural_retro as nr
    from tools.neural_retro import NeuralRetroTool
    from tests.conftest import FakeLLMClient

    class _EngineA:
        def predict(self, target_smiles, topk, detail_level, **kwargs):
            return {"target": target_smiles, "engine": "engA", "predictions": [
                {"reactants": "CCO.CC(=O)Cl", "score": 0.9, "template": "tA"},
                {"reactants": "CCBr", "score": 0.3},
            ]}

    class _EngineB:
        def predict(self, target_smiles, topk, detail_level, **kwargs):
            # Votes for the SAME first precursor set as A but with a LOWER
            # score — dedup keeps max(0.9, 0.5); the raw 0.5 only survives in
            # _detail.
            return {"target": target_smiles, "engine": "engB", "predictions": [
                {"reactants": "CCO.CC(=O)Cl", "score": 0.5},
            ]}

    tool = NeuralRetroTool(engines=[], llm_client=FakeLLMClient(),
                           sanity_enabled=False, engine_modes={})
    tool._engines = {"engA": _EngineA(), "engB": _EngineB()}
    # Pass-through bucket/pick so we exercise the real dedup + envelope.
    monkeypatch.setattr(nr, "bucket_by_classifier",
                        lambda preds, target_smiles, n_buckets: {0: preds})
    monkeypatch.setattr(
        nr, "pick_per_bucket",
        lambda buckets, k_per_bucket, return_topk, distance_threshold=0.65,
               return_meta=False: (
            (list(buckets[0])[:return_topk],
             {"n_clusters": 1, "n_input": len(buckets[0]),
              "n_returned": min(len(buckets[0]), return_topk)})
            if return_meta else list(buckets[0])[:return_topk]
        ),
    )

    out = tool._run_ensemble(target_smiles="CC(=O)OCC", per_engine_topk=10,
                             return_topk=12, detail_level="brief", model=None)

    assert "_detail" in out
    pep = out["_detail"]["per_engine_predictions"]
    # Each engine's OWN raw top-K is preserved separately.
    assert set(pep.keys()) == {"engA", "engB"}
    assert len(pep["engA"]) == 2
    a_scores = {r["reactants"]: r["score"] for r in pep["engA"]}
    assert a_scores["CCO.CC(=O)Cl"] == 0.9
    # engB's lower raw score for the shared precursor set survives in _detail,
    # even though the merged survivor keeps the max (0.9).
    assert pep["engB"][0]["score"] == 0.5
    # The merged survivor still records BOTH engines as support.
    shared = next(p for p in out["predictions"]
                  if (p.get("reactants") or "").count(".") == 1
                  and "CC(=O)Cl" in p.get("reactants", ""))
    assert set(shared["support"]) == {"engA", "engB"}
    assert shared["score"] == 0.9  # max-collapsed
