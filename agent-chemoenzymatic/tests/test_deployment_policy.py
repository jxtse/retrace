import pytest
import yaml

from core.config import load_config


@pytest.mark.parametrize("session_kind", ["primary", "reset", "fallback", "stream"])
def test_llm_request_redirect_policy(session_kind, monkeypatch):
    from unittest.mock import Mock
    import requests
    from core.llm_client import LLMClient
    from services.streaming import stream_chat_completion

    monkeypatch.setenv("LLM_API_KEY", "fixture")
    monkeypatch.setenv("LLM_FALLBACK_API_KEY", "fixture")
    response = Mock(status_code=200)
    response.iter_lines.return_value = ["data: [DONE]"]
    send = Mock(return_value=response)
    monkeypatch.setattr(requests.Session, "send", send)
    base = "https://fixture.invalid/v1"
    if session_kind == "stream":
        assert list(stream_chat_completion(base, "fixture", []))[-1]["type"] == "done"
    else:
        client = LLMClient(base_url=base, model="fixture", fallback_enabled=True,
                           fallback_base_url=base)
        if session_kind == "reset":
            client._reset_session()
        session = client._get_fallback_session() if session_kind == "fallback" else client.session
        assert session.post(base) is response
    assert send.call_args.kwargs["allow_redirects"] is False


def test_stream_rejects_3xx_before_reading_events(monkeypatch):
    from unittest.mock import Mock
    import requests
    from services.streaming import stream_chat_completion

    response = requests.Response()
    response.status_code = 307
    response._content = b""
    response._content_consumed = True
    response.iter_lines = Mock(return_value=[])
    transport = Mock(return_value=response)
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", transport)
    events = list(stream_chat_completion("https://fixture.invalid/v1", "fixture", []))
    assert events == [{"type": "error", "error": "LLM endpoint returned HTTP 307; configure the final endpoint URL."}]
    transport.assert_called_once()
    response.iter_lines.assert_not_called()


@pytest.mark.parametrize("value", ['"false"', "0", "null"])
def test_chemprice_requires_yaml_boolean(tmp_path, value):
    path = tmp_path / "config.yaml"
    path.write_text(f"verifier:\n  cost:\n    chemprice:\n      enabled: {value}\n")
    with pytest.raises(ValueError, match=r"verifier\.cost\.chemprice\.enabled"):
        load_config(path)


def test_check_config_reports_invalid_chemprice(tmp_path, capsys):
    import main

    path = tmp_path / "config.yaml"
    path.write_text('verifier: {cost: {chemprice: {enabled: "false"}}}')
    args = main.build_parser().parse_args(["check-config", "--config", str(path)])
    with pytest.raises(SystemExit) as error:
        args.func(args)
    assert error.value.code == 2
    assert "verifier.cost.chemprice.enabled" in capsys.readouterr().err


def test_runtime_preflight_checks_chemprice_and_accepts_omission():
    from core.deployment import validate_deployment

    config = load_config(cli_overrides={
        "llm": {"base_url": "https://fixture.invalid", "model": "fixture"},
        "neural_retro": {"enabled": False},
        "verifier": {"feasibility": {"rater": {"http_url": "https://fixture.invalid"}}},
    })
    pricing = config["verifier"]["cost"]["chemprice"]
    assert pricing.pop("enabled") is True
    validate_deployment(config)
    pricing["enabled"] = "false"
    with pytest.raises(ValueError, match=r"verifier\.cost\.chemprice\.enabled"):
        validate_deployment(config)


def test_chemprice_false_reaches_factory(tmp_path):
    from unittest.mock import patch
    from core import build_agent

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "llm": {"base_url": "https://fixture.invalid", "model": "fixture"},
        "neural_retro": {"enabled": False},
        "verifier": {"feasibility": {"rater": {"http_url": "https://fixture.invalid"}},
                     "cost": {"chemprice": {"enabled": False}}},
        "result_store": {"dir": str(tmp_path / "runs")},
    }))
    with patch("tools.verifier.RouteVerifierTool") as verifier:
        build_agent(load_config(path))
    assert verifier.call_args.kwargs["chemprice_enabled"] is False


@pytest.mark.parametrize("explicit", [True, False])
def test_distributed_config_selection(tmp_path, monkeypatch, explicit):
    import shlex
    from core import evaluation

    monkeypatch.setattr(evaluation, "REPO_ROOT", tmp_path)
    path = tmp_path / "selected config.yaml"
    path.write_text(yaml.safe_dump({
        "llm": {"base_url": "https://fixture.invalid", "model": "fixture"},
        "verifier": {"enabled": False}, "neural_retro": {"enabled": False},
    }))
    monkeypatch.setenv("AGENT_CONFIG", "missing.yaml" if explicit else path.name)
    monkeypatch.setenv("AGENT_MODEL", "fixture-env-model")
    kwargs = dict(suite_name="retro190", indices=[0], output_dir=tmp_path / "plan",
                  hosts="worker", full_cap=True, max_workers="worker=1", dry_run=True,
                  config_path=path.name if explicit else None)
    summary = evaluation.distribute_benchmark(**kwargs)
    remote_args = shlex.split(shlex.split(summary["commands"][0]["cmd"])[-1])
    assert remote_args[remote_args.index("--config") + 1] == str(path)
    assert remote_args[remote_args.index("--model") + 1] == "fixture-env-model"
    path.write_text("unknown_option: true")
    with pytest.raises(ValueError, match="unknown_option"):
        evaluation.distribute_benchmark(**kwargs)
