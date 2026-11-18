"""Verify _ChemBartEngine has been migrated off the standalone HTML
scraper onto the shared RetroplannerHTTPClient (host-d-primary), while
preserving its per-canonical-SMILES result cache.
"""
from unittest.mock import MagicMock

import pytest

from tools._retroplanner_http import RetroplannerHTTPError
from tools.neural_retro import _ChemBartEngine


def _client_returning(entries):
    c = MagicMock()
    c.predict.return_value = entries
    return c


def test_uses_shared_client_with_primary_model_name():
    client = _client_returning(
        [{"reactants": "O=C(Cl)c1ccccc1", "score": 0.7}],
    )
    eng = _ChemBartEngine(
        api_url="http://ignored", topk=5, http_client=client,
    )
    out = eng.predict("O=C(c1ccccc1)c1ccccc1", topk=5, detail_level="brief")
    client.predict.assert_called_once_with(
        smiles="O=C(c1ccccc1)c1ccccc1",
        model_name="ChemBart (USPTO-Full)",
        topk=5,
    )
    assert out["engine"] == "chembart"
    assert out["predictions"][0]["reactants"] == "O=C(Cl)c1ccccc1"


def test_per_canonical_smiles_cache_hits_avoid_second_call():
    client = _client_returning([{"reactants": "A.B", "score": 0.8}])
    eng = _ChemBartEngine(
        api_url="http://ignored", topk=5, http_client=client,
    )
    eng.predict("c1ccccc1", topk=5, detail_level="brief")
    eng.predict("C1=CC=CC=C1", topk=5, detail_level="brief")
    assert client.predict.call_count == 1, (
        "second call (same canonical SMILES) should hit cache"
    )


def test_error_propagates():
    client = MagicMock()
    client.predict.side_effect = RetroplannerHTTPError("boom")
    eng = _ChemBartEngine(
        api_url="http://ignored", topk=5, http_client=client,
    )
    with pytest.raises(RetroplannerHTTPError):
        eng.predict("X", topk=5, detail_level="brief")


def test_full_detail_tags_chembart_template():
    client = _client_returning([{"reactants": "A.B", "score": 0.8}])
    eng = _ChemBartEngine(
        api_url="http://ignored", topk=5, http_client=client,
    )
    out = eng.predict("X", topk=5, detail_level="full")
    assert out["predictions"][0].get("template") == "[ChemBart]"


def test_brief_detail_strips_template():
    client = _client_returning(
        [{"reactants": "A.B", "score": 0.8, "template": "t"}],
    )
    eng = _ChemBartEngine(
        api_url="http://ignored", topk=5, http_client=client,
    )
    out = eng.predict("X", topk=5, detail_level="brief")
    assert "template" not in out["predictions"][0]


def test_builds_own_client_when_none_provided():
    """No shared client → engine builds one from api_url (no network)."""
    eng = _ChemBartEngine(
        api_url="http://host-d.invalid/retroplanner/api/single_step", topk=5,
    )
    assert eng._http is not None
    assert "host-d.invalid" in eng._http.urls[0]


def test_requires_url_or_client():
    with pytest.raises(ValueError):
        _ChemBartEngine(api_url=None, topk=5)
