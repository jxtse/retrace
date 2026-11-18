"""Verify neural_retro defaults are present in the built-in config."""

from core.config import load_config


def test_retroplanner_http_defaults_present():
    cfg = load_config(thinking="fast")
    nr = cfg["neural_retro"]
    assert nr["retroplanner_http_url"].startswith("http://")
    assert nr["retroplanner_http_urls"] == [
        "http://retro-primary.example.org:8001/retroplanner/api/single_step",
        "http://retro-fallback.example.org:8001/retroplanner/api/single_step",
    ]
    # host-d (primary) weighted over host-a (fallback).
    assert nr["retroplanner_http_weights"] == [8, 1]
    # Primary host is host-d.
    assert "retro-primary.example.org" in nr["retroplanner_http_url"]
    assert "/single_step" in nr["retroplanner_http_url"]
    assert nr["retroplanner_http_timeout_s"] == 120
    assert nr["retroplanner_http_retries"] == 3


def test_engine_modes_defaults():
    # The in-process DMPNNFP / RXNGraphormer engines were removed
    # 2026-06-08; every neural engine now routes through the shared
    # retroplanner HTTP service, so http_only is the only mode.
    cfg = load_config(thinking="fast")
    modes = cfg["neural_retro"]["engine_modes"]
    assert modes["rxngraphormer"] == "http_only"
    assert modes["graph_retrosyn"] == "http_only"
    assert modes["pistachio_ringbreaker"] == "http_only"
