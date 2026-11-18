"""Verify _PistachioRingbreakerEngine has been migrated to use the
shared RetroplannerHTTPClient while preserving its per-canonical-SMILES
result cache and HTTP-only failure mode.
"""
from unittest.mock import MagicMock

import pytest

from tools._retroplanner_http import RetroplannerHTTPError
from tools.neural_retro import _PistachioRingbreakerEngine


def _client_returning(entries):
    c = MagicMock()
    c.predict.return_value = entries
    return c


def test_uses_shared_client_when_provided():
    client = _client_returning(
        [{"reactants": "A.B", "score": 0.8, "template": "t"}],
    )
    eng = _PistachioRingbreakerEngine(
        api_url="http://ignored", topk=5, http_client=client,
    )
    out = eng.predict("c1ccccc1", topk=5, detail_level="brief")
    client.predict.assert_called_once_with(
        smiles="c1ccccc1",
        model_name="Pistachio Ringbreaker",
        topk=5,
    )
    assert out["engine"] == "pistachio_ringbreaker"
    assert out["predictions"][0]["reactants"] == "A.B"


def test_per_canonical_smiles_cache_hits_avoid_second_call():
    client = _client_returning(
        [{"reactants": "A.B", "score": 0.8}],
    )
    eng = _PistachioRingbreakerEngine(
        api_url="http://ignored", topk=5, http_client=client,
    )
    eng.predict("c1ccccc1", topk=5, detail_level="brief")
    eng.predict("C1=CC=CC=C1", topk=5, detail_level="brief")
    assert client.predict.call_count == 1, (
        "second call (same canonical SMILES) should hit cache"
    )


def test_no_fallback_on_http_error():
    """HTTP-only engine: error must propagate, no local engine exists."""
    client = MagicMock()
    client.predict.side_effect = RetroplannerHTTPError("boom")
    eng = _PistachioRingbreakerEngine(
        api_url="http://ignored", topk=5, http_client=client,
    )
    with pytest.raises(RetroplannerHTTPError):
        eng.predict("X", topk=5, detail_level="brief")


def test_full_detail_includes_template():
    client = _client_returning(
        [{"reactants": "A.B", "score": 0.8, "template": "t"}],
    )
    eng = _PistachioRingbreakerEngine(
        api_url="http://ignored", topk=5, http_client=client,
    )
    out = eng.predict("X", topk=5, detail_level="full")
    assert out["predictions"][0].get("template") == "t"


def test_brief_detail_strips_template():
    client = _client_returning(
        [{"reactants": "A.B", "score": 0.8, "template": "t"}],
    )
    eng = _PistachioRingbreakerEngine(
        api_url="http://ignored", topk=5, http_client=client,
    )
    out = eng.predict("X", topk=5, detail_level="brief")
    assert "template" not in out["predictions"][0]
