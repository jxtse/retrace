"""Deployment settings must resolve without opening a network connection."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from core.config import load_config

HARNESS = Path(__file__).resolve().parents[1]


def test_copyable_example():
    for args, expected in [(["--help"], "check-config"),
                           (["check-config", "--config", "../deployment.example.yaml"], "llm.base_url")]:
        result = subprocess.run([sys.executable, "main.py", *args], cwd=HARNESS,
                                capture_output=True, text=True, env={**os.environ, "RETROPLANNER_HTTP_URL": ""})
        assert result.returncode == (0 if args == ["--help"] else 2)
        assert expected in result.stdout + result.stderr


def test_paths_and_precedence(tmp_path, monkeypatch):
    from core.deployment import stock_path, retroplanner_urls, validate_deployment

    monkeypatch.chdir(tmp_path)
    cfg_file = tmp_path / "settings.yaml"
    cfg_file.write_text(yaml.safe_dump({
        "llm": {"base_url": "https://coordinator.invalid/v1", "model": "fixture"},
        "neural_retro": {"retroplanner_http_urls": ["https://shared.invalid"]},
        "verifier": {"feasibility": {"rater": {"http_url": ["https://rater.invalid"]}}},
        "stocks": {"BioNav-stock": str(tmp_path / "stock.csv"), "Zinc_Fix-stock": "assets/zinc.csv"},
    }))
    cfg = load_config(cfg_file)
    assert stock_path(cfg["base_dir"], cfg["stocks"]["BioNav-stock"]) == tmp_path / "stock.csv"
    assert stock_path(cfg["base_dir"], cfg["stocks"]["Zinc_Fix-stock"]) == HARNESS / "assets/zinc.csv"
    monkeypatch.delenv("RETROPLANNER_HTTP_URLS", raising=False)
    monkeypatch.setenv("RETROPLANNER_HTTP_URL", "https://override.invalid")
    assert retroplanner_urls(cfg["neural_retro"]) == "https://override.invalid"
    validate_deployment(cfg)


def test_validate_selected_only(monkeypatch):
    from core.deployment import validate_deployment

    cfg = load_config(cli_overrides={"llm": {"base_url": "http://fixture.invalid", "model": "fixture"},
                                     "verifier": {"enabled": False},
                                     "neural_retro": {"engines": ["template_relevance"]}})
    with pytest.raises(ValueError, match="template_relevance_url"):
        validate_deployment(cfg)
    cfg["neural_retro"]["enabled"] = False
    validate_deployment(cfg)
    cfg["verifier"]["enabled"] = True
    monkeypatch.setenv("REACTION_RATER_URL", "http://fixture.invalid:secret-value")
    with pytest.raises(ValueError, match="verifier.feasibility.rater.http_url") as error:
        validate_deployment(cfg)
    assert "secret-value" not in str(error.value)


def test_factory_routes_selected_engines_and_stock(tmp_path, monkeypatch):
    from unittest.mock import patch
    from core import build_agent

    stock = tmp_path / "catalog.csv"
    stock.write_text("smiles\nCC\n")
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump({
        "llm": {"base_url": "https://main.invalid", "model": "fixture"},
        "postprocess": {"enabled": False},
        "verifier": {"enabled": False},
        "stocks": {"BioNav-stock": str(stock)},
        "result_store": {"dir": str(tmp_path / "runs"), "retention_days": 0},
        "neural_retro": {"engines": ["graph_retrosyn", "chembart"],
                         "retroplanner_http_url": "https://yaml.invalid",
                         "chembart_url": "https://dedicated.invalid"},
    }))
    monkeypatch.setenv("RETROPLANNER_HTTP_URLS", "https://first.invalid,https://second.invalid")
    monkeypatch.setenv("RETROPLANNER_HTTP_URL", "https://ignored.invalid")
    monkeypatch.setenv("CHEMBART_API_URL", "https://engine.invalid")

    with patch("tools.check_buyability.CheckBuyabilityTool._load_stock_cached", return_value=()) as load:
        agent = build_agent(load_config(path))
    load.assert_called_once_with("BioNav-stock", str(stock))
    engines = agent.tools.get("neural_retro")._engines
    assert engines["graph_retrosyn"]._http.urls == ("https://first.invalid", "https://second.invalid")
    assert engines["chembart"]._http.urls == ("https://engine.invalid",)


def test_credentials_scoped_to_provider(monkeypatch):
    from core.llm_client import LLMClient

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    for url, authorized in [("https://openrouter.ai/api/v1", True),
                            ("https://fixture.invalid", False)]:
        client = LLMClient(base_url=url, model="fixture")
        assert ("Authorization" in client.session.headers) is authorized
        assert not client.fallback_enabled


@pytest.mark.parametrize("rater", ["unexpected", {"http_url": ["https://ok.invalid", "missing-scheme"]}])
def test_nested_rater_diagnostic(rater, monkeypatch):
    from core.deployment import validate_deployment

    monkeypatch.delenv("REACTION_RATER_URL", raising=False)
    cfg = load_config(cli_overrides={"verifier": {"feasibility": {"rater": rater}}})
    with pytest.raises(ValueError, match="verifier.feasibility.rater"):
        validate_deployment(cfg)


def test_unused_shared_url_for_reaxys(monkeypatch):
    from core.deployment import validate_deployment

    monkeypatch.setenv("RETROPLANNER_HTTP_URL", "stale-invalid-value")
    cfg = load_config(cli_overrides={
        "llm": {"base_url": "https://main.invalid", "model": "fixture"},
        "verifier": {"enabled": False},
        "neural_retro": {"engines": ["template_relevance"], "template_relevance_url": "https://reaxys.invalid"},
    })
    validate_deployment(cfg)


def test_string_false_cannot_enable_fallback():
    with pytest.raises(ValueError, match="llm.fallback.enabled"):
        load_config(cli_overrides={"llm": {"fallback": {"enabled": "false"}}})
