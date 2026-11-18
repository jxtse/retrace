"""Verify NeuralRetroTool wires engine_modes correctly.

The in-process DMPNNFP / RXNGraphormer engines (and their torch/PyG/
modelscope deps + the `.venv-rxngraphormer` subprocess) were removed
2026-06-08. Every neural engine now reaches its model through the shared
retroplanner `/single_step` HTTP service, so `http_only` is the only
accepted mode and the two engines are always the HTTP wrappers. These
tests assert the dispatch type and the config-time guards (invalid mode,
HTTP requested but no URL) without any network.
"""
import pytest

from tools.neural_retro import (
    NeuralRetroTool,
    _GraphRetrosynHTTPEngine,
    _RXNGraphormerHTTPEngine,
)


def _make_tool(**overrides):
    """Construct NeuralRetroTool with safe defaults for every kwarg."""
    kw = dict(
        template_relevance_url=None,
        chembart_url=None,
        pistachio_ringbreaker_url="http://example.invalid/single_step",
        topk=5,
        per_engine_topk=5,
        return_topk=5,
        n_buckets=2,
        cluster_distance=0.65,
        k_per_cluster=1,
        engines=["graph_retrosyn", "rxngraphormer", "pistachio_ringbreaker"],
        llm_client=None,
        sanity_enabled=False,
        retroplanner_http_url="http://example.invalid/single_step",
        retroplanner_http_timeout_s=5,
        retroplanner_http_retries=0,
        engine_modes={
            "rxngraphormer": "http_only",
            "graph_retrosyn": "http_only",
            "pistachio_ringbreaker": "http_only",
        },
    )
    kw.update(overrides)
    return NeuralRetroTool(**kw)


def test_http_only_uses_http_wrappers():
    tool = _make_tool()
    assert isinstance(
        tool._engines["rxngraphormer"], _RXNGraphormerHTTPEngine,
    )
    assert isinstance(
        tool._engines["graph_retrosyn"], _GraphRetrosynHTTPEngine,
    )


def test_missing_engine_modes_defaults_to_http_only():
    """Omitting engine_modes still routes to the HTTP wrappers."""
    tool = _make_tool(engine_modes=None)
    assert isinstance(
        tool._engines["rxngraphormer"], _RXNGraphormerHTTPEngine,
    )
    assert isinstance(
        tool._engines["graph_retrosyn"], _GraphRetrosynHTTPEngine,
    )


def test_chembart_retry_budget_is_wired_to_shared_http_client():
    tool = _make_tool(
        engines=["chembart"],
        retroplanner_chembart_max_attempts=2,
        retroplanner_chembart_deadline_s=130,
        retroplanner_chembart_retry_min_budget_s=75,
        retroplanner_retry_jitter_s=0.25,
    )
    client = tool._engines["chembart"]._http
    assert client._chembart_max_attempts == 2
    assert client._chembart_deadline_s == 130
    assert client._chembart_retry_min_budget_s == 75
    assert client._retry_jitter_s == 0.25


def test_invalid_engine_mode_raises():
    # `http_with_local_fallback` / `local_only` are gone; only http_only
    # validates, so a stale overlay carrying the old mode fails loudly.
    with pytest.raises(ValueError, match="engine_mode"):
        _make_tool(engine_modes={"rxngraphormer": "http_with_local_fallback"})
    with pytest.raises(ValueError, match="engine_mode"):
        _make_tool(engine_modes={"rxngraphormer": "nonsense"})


def test_http_mode_without_url_raises():
    """Misconfiguration: engine_mode requests HTTP but no URL provided."""
    with pytest.raises(ValueError, match="retroplanner_http_url"):
        _make_tool(retroplanner_http_url=None)
