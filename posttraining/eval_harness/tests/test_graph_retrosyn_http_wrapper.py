"""Tests for `_GraphRetrosynHTTPEngine` — pure HTTP.

The in-process DMPNNFP engine (~1.26 GiB CUDA, torch + PyG) was removed
2026-06-08; the engine is now a thin client over the shared retroplanner
`/single_step` service. Mirrors `test_rxngraphormer_http_wrapper.py`
modulo two differences:
- HTTP model name is "GraphFP (USPTO-Full)", not "RxNGormer".
- `predict` takes no `model=` kwarg (there is no checkpoint selector).
"""
from unittest.mock import MagicMock

import pytest

from tools._retroplanner_http import RetroplannerHTTPError
from tools.neural_retro import _GraphRetrosynHTTPEngine


def _client_returning(entries):
    c = MagicMock()
    c.predict.return_value = entries
    return c


def _client_raising():
    c = MagicMock()
    c.predict.side_effect = RetroplannerHTTPError("simulated empty result")
    return c


def test_http_success_returns_envelope():
    client = _client_returning(
        [{"reactants": "A.B", "score": 0.50, "template": "t"}],
    )
    eng = _GraphRetrosynHTTPEngine(http_client=client)
    out = eng.predict("CC(=O)Oc1ccccc1C(=O)O", topk=5, detail_level="brief")
    assert out["target"] == "CC(=O)Oc1ccccc1C(=O)O"
    assert out["engine"] == "graph_retrosyn"
    assert out["predictions"][0]["reactants"] == "A.B"
    client.predict.assert_called_once_with(
        smiles="CC(=O)Oc1ccccc1C(=O)O",
        model_name="GraphFP (USPTO-Full)",
        topk=5,
    )
    # No envelope-level fallback key — would leak to the LLM.
    assert "_fallback" not in out


def test_http_failure_propagates():
    """No local fallback exists; an HTTP error surfaces to the caller
    (the ensemble fan-out isolates it per-engine)."""
    client = _client_raising()
    eng = _GraphRetrosynHTTPEngine(http_client=client)
    with pytest.raises(RetroplannerHTTPError):
        eng.predict("X", topk=1, detail_level="brief")


def test_brief_detail_strips_template():
    client = _client_returning(
        [{"reactants": "A", "score": 0.5, "template": "t"}],
    )
    eng = _GraphRetrosynHTTPEngine(http_client=client)
    out = eng.predict("X", topk=1, detail_level="brief")
    assert "template" not in out["predictions"][0]


def test_full_detail_keeps_template():
    client = _client_returning(
        [{"reactants": "A", "score": 0.5, "template": "t"}],
    )
    eng = _GraphRetrosynHTTPEngine(http_client=client)
    out = eng.predict("X", topk=1, detail_level="full")
    assert out["predictions"][0]["template"] == "t"


def test_invalid_engine_mode_raises():
    with pytest.raises(ValueError, match="engine_mode"):
        _GraphRetrosynHTTPEngine(
            http_client=MagicMock(),
            engine_mode="bogus",
        )
    # The retired modes must also fail loudly.
    with pytest.raises(ValueError, match="engine_mode"):
        _GraphRetrosynHTTPEngine(
            http_client=MagicMock(),
            engine_mode="local_only",
        )
