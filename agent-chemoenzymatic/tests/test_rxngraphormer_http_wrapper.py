"""Tests for `_RXNGraphormerHTTPEngine` — pure HTTP.

The local `.venv-rxngraphormer/` subprocess fallback was removed
2026-06-08; the engine is now a thin client over the shared
retroplanner `/single_step` service. Mocks `RetroplannerHTTPClient`
so no network is needed.
"""
import logging
from unittest.mock import MagicMock

import pytest

from tools._retroplanner_http import RetroplannerHTTPError
from tools.neural_retro import _RXNGraphormerHTTPEngine


def _http_client_returning(entries):
    c = MagicMock()
    c.predict.return_value = entries
    return c


def _http_client_raising():
    c = MagicMock()
    c.predict.side_effect = RetroplannerHTTPError("simulated empty result")
    return c


# ---------- success path ----------


def test_http_success_returns_envelope():
    client = _http_client_returning(
        [{"reactants": "OCc1ccccc1", "score": 0.83, "template": "[RxN]"}],
    )
    eng = _RXNGraphormerHTTPEngine(http_client=client)
    out = eng.predict("c1ccccc1", topk=10, detail_level="brief")
    assert out["target"] == "c1ccccc1"
    assert out["engine"] == "rxngraphormer"
    assert out["predictions"][0]["reactants"] == "OCc1ccccc1"
    client.predict.assert_called_once_with(
        smiles="c1ccccc1", model_name="RxNGormer (USPTO-Full)", topk=10,
    )
    # No envelope-level fallback key — would leak to the LLM.
    assert "_fallback" not in out


def test_http_failure_propagates():
    """No local fallback exists; an HTTP error surfaces to the caller
    (the ensemble fan-out isolates it per-engine)."""
    client = _http_client_raising()
    eng = _RXNGraphormerHTTPEngine(http_client=client)
    with pytest.raises(RetroplannerHTTPError):
        eng.predict("X", topk=1, detail_level="brief")


# ---------- detail level ----------


def test_brief_detail_strips_template():
    client = _http_client_returning(
        [{"reactants": "X", "score": 0.5, "template": "t"}],
    )
    eng = _RXNGraphormerHTTPEngine(http_client=client)
    out = eng.predict("X", topk=1, detail_level="brief")
    assert "template" not in out["predictions"][0]


# ---------- model parameter handling ----------


def test_explicit_model_logs_info_but_calls_http(caplog):
    client = _http_client_returning(
        [{"reactants": "X", "score": 0.5}],
    )
    eng = _RXNGraphormerHTTPEngine(http_client=client)
    with caplog.at_level(logging.INFO, logger="tools.neural_retro"):
        eng.predict("X", topk=1, detail_level="brief", model="USPTO_50k")
    client.predict.assert_called_once()
    assert any(
        "ignores the model parameter" in r.message for r in caplog.records
    ), "expected info log about HTTP ignoring model="


# ---------- engine_mode guard ----------


def test_invalid_engine_mode_raises():
    with pytest.raises(ValueError, match="engine_mode"):
        _RXNGraphormerHTTPEngine(
            http_client=MagicMock(),
            engine_mode="bogus",
        )
    # The retired modes must also fail loudly.
    with pytest.raises(ValueError, match="engine_mode"):
        _RXNGraphormerHTTPEngine(
            http_client=MagicMock(),
            engine_mode="http_with_local_fallback",
        )
