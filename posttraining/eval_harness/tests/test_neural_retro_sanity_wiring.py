"""Wiring test: NeuralRetroTool threads candidates through
`tools._precursor_sanity.filter_precursors` and surfaces a
`dropped_by_sanity` report on its result envelope.

We do the LEAST invasive thing: monkeypatch the module-level
`filter_precursors` symbol and the bucket/pick helpers that
`_run_ensemble` calls after the filter, then drive `_run_ensemble`
directly with a stubbed engine that returns a fixed prediction list.
"""
from __future__ import annotations

import tools.neural_retro as nr
from tools.neural_retro import NeuralRetroTool
from tests.conftest import FakeLLMClient

# All four sanity-wiring tests pre-date the HTTP migration and care about
# the precursor sanity filter, not transport. They register no engines
# (`engines=[]`), so no engine_modes are needed; an empty dict avoids the
# constructor's mode-validation loop (http_only is now the only valid mode).
_LOCAL_ONLY_MODES: dict = {}


class _StubEngine:
    def predict(self, target_smiles, topk, detail_level, **kwargs):
        return {
            "target": target_smiles,
            "engine": "stub",
            "predictions": [
                {"reactants": "CN", "score": 0.4},
                {"reactants": "c1ccccc1Br.CN", "score": 0.7},
            ],
        }


def test_constructor_threads_sanity_kwargs():
    fake = FakeLLMClient()
    tool = NeuralRetroTool(
        engines=[],
        llm_client=fake,
        sanity_enabled=False,
        sanity_batch_size=7,
        sanity_max_dropped_logged=3,
        engine_modes=_LOCAL_ONLY_MODES,
    )
    assert tool._sanity_enabled is False
    assert tool._sanity_batch_size == 7
    assert tool._sanity_max_logged == 3
    # The adapter wraps the raw LLM client.
    assert tool._llm_client is not None
    assert tool._llm_client._llm is fake
    assert tool._sanity_cache == {}


def test_run_ensemble_invokes_filter_and_reports_dropped(monkeypatch):
    fake = FakeLLMClient()
    tool = NeuralRetroTool(
        engines=[],
        llm_client=fake,
        engine_modes=_LOCAL_ONLY_MODES,
    )
    # Stub a single engine into the constructed tool.
    tool._engines = {"stub": _StubEngine()}

    # Capture filter call and drop the first candidate.
    calls = {}

    def fake_filter(*, target_smiles, candidates, llm_client, batch_size, cache):
        calls["target"] = target_smiles
        calls["n_candidates"] = len(candidates)
        calls["llm_client"] = llm_client
        calls["batch_size"] = batch_size
        calls["cache"] = cache
        kept = candidates[1:]
        dropped = candidates[:1]
        reasons = ["leaving group only"]
        return kept, dropped, reasons

    monkeypatch.setattr(nr, "filter_precursors", fake_filter, raising=False)

    # Make bucketing / pick a no-op pass-through so we can focus on wiring.
    monkeypatch.setattr(
        nr, "bucket_by_classifier",
        lambda preds, target_smiles, n_buckets: {0: preds},
    )
    # Signature mirrors the real `pick_per_bucket(..., return_meta=True)`
    # path that `_run_ensemble` now uses (May 2026 funnel-fix).
    monkeypatch.setattr(
        nr, "pick_per_bucket",
        lambda buckets, k_per_bucket, return_topk, distance_threshold=0.65,
               return_meta=False: (
            (
                list(buckets[0])[:return_topk],
                {"n_clusters": 1, "n_input": len(buckets[0]),
                 "n_returned": min(len(buckets[0]), return_topk)},
            )
            if return_meta
            else list(buckets[0])[:return_topk]
        ),
    )

    out = tool._run_ensemble(
        target_smiles="c1ccc(NC)cc1",
        per_engine_topk=10,
        return_topk=12,
        detail_level="brief",
        model=None,
    )

    assert calls["target"] == "c1ccc(NC)cc1"
    assert calls["n_candidates"] >= 1
    assert calls["llm_client"] is not None
    assert calls["batch_size"] == 20
    assert calls["cache"] is tool._sanity_cache

    assert "dropped_by_sanity" in out
    assert out["dropped_by_sanity"]["n_dropped"] == 1
    assert out["dropped_by_sanity"]["examples"][0]["reason"] == "leaving group only"
    # The kept candidate survived and made it to predictions
    # (note: dedup_predictions canonicalizes the SMILES).
    assert any("CN" in (p.get("reactants") or "") and "c1ccccc1" in (p.get("reactants") or "")
               for p in out["predictions"])


def test_run_ensemble_skips_filter_when_disabled(monkeypatch):
    tool = NeuralRetroTool(
        engines=[],
        llm_client=FakeLLMClient(),
        sanity_enabled=False,
        engine_modes=_LOCAL_ONLY_MODES,
    )
    tool._engines = {"stub": _StubEngine()}

    def boom(**kwargs):
        raise AssertionError("filter must not run when disabled")

    monkeypatch.setattr(nr, "filter_precursors", boom, raising=False)
    monkeypatch.setattr(
        nr, "bucket_by_classifier",
        lambda preds, target_smiles, n_buckets: {0: preds},
    )
    monkeypatch.setattr(
        nr, "pick_per_bucket",
        lambda buckets, k_per_bucket, return_topk, distance_threshold=0.65,
               return_meta=False: (
            (
                list(buckets[0])[:return_topk],
                {"n_clusters": 1, "n_input": len(buckets[0]),
                 "n_returned": min(len(buckets[0]), return_topk)},
            )
            if return_meta
            else list(buckets[0])[:return_topk]
        ),
    )

    out = tool._run_ensemble(
        target_smiles="c1ccc(NC)cc1",
        per_engine_topk=10,
        return_topk=12,
        detail_level="brief",
        model=None,
    )
    assert out["dropped_by_sanity"]["n_dropped"] == 0


def test_run_single_threads_dropped_by_sanity(monkeypatch):
    tool = NeuralRetroTool(
        engines=[],
        llm_client=FakeLLMClient(),
        engine_modes=_LOCAL_ONLY_MODES,
    )
    tool._engines = {"stub": _StubEngine()}

    def fake_filter(*, target_smiles, candidates, llm_client, batch_size, cache):
        return candidates[1:], candidates[:1], ["nope"]

    monkeypatch.setattr(nr, "filter_precursors", fake_filter, raising=False)

    out = tool._run_single("stub", "c1ccc(NC)cc1", topk=10, detail_level="brief", model=None)
    assert out["dropped_by_sanity"]["n_dropped"] == 1
    assert len(out["predictions"]) == 1
