import shlex

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


def worker_script_tokens(ssh_argv, host):
    # OpenSSH joins the arguments after the destination before the remote shell parses them.
    remote_args = ssh_argv[ssh_argv.index(host) + 1:]
    shell, flag, script = shlex.split(" ".join(remote_args))
    assert (shell, flag) == ("bash", "-lc")
    assert len(remote_args) == 1
    return shlex.split(script)


def test_distributed_dispatch_argv(tmp_path, monkeypatch, isolated_deployment):
    from subprocess import CompletedProcess
    from unittest.mock import Mock
    from core import evaluation

    run = Mock(return_value=CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(evaluation.subprocess, "run", run)
    output_dir = tmp_path / "worker results"
    summary = evaluation.distribute_benchmark(
        suite_name="retro190", indices=[0, 2], hosts="worker", full_cap=True,
        max_workers="worker=2", output_dir=output_dir, thinking="thorough",
    )

    run.assert_called_once()
    argv, = run.call_args.args
    assert isinstance(argv, list)
    assert not run.call_args.kwargs.get("shell", False)
    assert argv[:-1] == ["ssh", "-n", "-f", "-o", "BatchMode=yes", "-o",
                         "ConnectTimeout=15", "--", "worker"]
    assert shlex.split(summary["commands"][0]["cmd"]) == argv
    tokens = worker_script_tokens(argv, "worker")
    assert tokens[:3] == ["cd", str(evaluation.REPO_ROOT), "&&"]
    assert tokens[tokens.index("nohup") + 1:tokens.index(">")] == [
        ".venv/bin/python", "main.py", "evaluate", "retro190", "--local",
        "--indices", "0,2", "--workers", "2", "--gpu-ids", "0",
        "--output-dir", str(output_dir), "--thinking", "thorough", "--skip-existing",
    ]


@pytest.mark.parametrize("explicit", [True, False])
def test_distributed_config_selection(tmp_path, monkeypatch, explicit):
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
    remote_args = worker_script_tokens(shlex.split(summary["commands"][0]["cmd"]), "worker")
    assert remote_args[remote_args.index("--config") + 1] == str(path)
    assert remote_args[remote_args.index("--model") + 1] == "fixture-env-model"
    path.write_text("unknown_option: true")
    with pytest.raises(ValueError, match="unknown_option"):
        evaluation.distribute_benchmark(**kwargs)


@pytest.fixture
def isolated_deployment(monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr("core.evaluation.subprocess.run",
                        Mock(side_effect=AssertionError("unexpected process launch")))
    monkeypatch.setenv("_AGENT_DOTENV_LOADED", "1")
    for name in ("AGENT_CONFIG", "AGENT_MODEL", "AGENT_THINKING", "REACTION_RATER_URL",
                 "LLM_API_KEY", "LLM_FALLBACK_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("config,env_model,cli_model,expected", [
    (True, None, None, "yaml-model"),
    (True, "env-model", None, "env-model"),
    (True, "env-model", "cli-model", "cli-model"),
    (False, None, None, ""),
])
def test_pinned_worker_plan(tmp_path, monkeypatch, isolated_deployment,
                            config, env_model, cli_model, expected):
    import json
    import main
    from core.config import load_benchmark_config

    parser = main.build_parser()
    args = ["evaluate", "retro190", "--indices", "0,1", "--hosts", "one,two",
            "--max-workers", "one=1,two=1", "--full-cap", "--dry-run",
            "--output-dir", str(tmp_path)]
    if config:
        path = tmp_path / "worker settings.yaml"
        path.write_text("llm: {model: yaml-model, base_url: https://fixture.invalid}\n"
                        "neural_retro: {enabled: false}\n")
        monkeypatch.setenv("AGENT_CONFIG", str(path))
    if env_model:
        monkeypatch.setenv("AGENT_MODEL", env_model)
    if cli_model:
        args.extend(["--model", cli_model])

    parsed = parser.parse_args(args)
    assert parsed.func(parsed) == 0
    plan = json.loads((tmp_path / "distribute_plan.json").read_text())
    assert plan["model"] == expected
    assert len(plan["commands"]) == 2
    for launch in plan["commands"]:
        assert plan["assignments"][launch["host"]]["model"] == expected
        tokens = worker_script_tokens(shlex.split(launch["cmd"]), launch["host"])
        worker = parser.parse_args(tokens[tokens.index("main.py") + 1:tokens.index(">")])
        assert worker.local
        assert worker.model == (expected or None)
        assert worker.config == (str(path) if config else None)
    if config:
        with pytest.raises(ValueError, match="verifier.feasibility.rater.http_url"):
            load_benchmark_config()


@pytest.mark.parametrize("contents,override,error", [
    (None, None, FileNotFoundError),
    ("llm: [", None, ValueError),
    ("[]", "cli-model", ValueError),
    ("llm: {model: '   '}", None, ValueError),
])
def test_coordinator_rejects_invalid_selection(tmp_path, isolated_deployment,
                                               contents, override, error):
    from core.evaluation import distribute_benchmark

    path = tmp_path / "config.yaml"
    if contents is not None:
        path.write_text(contents)
    with pytest.raises(error):
        distribute_benchmark(suite_name="retro190", indices=[0], hosts="one",
                             max_workers="one=1", full_cap=True, dry_run=True,
                             output_dir=tmp_path, config_path=path, model=override)
    assert not (tmp_path / "distribute_plan.json").exists()


@pytest.mark.parametrize("host,variable", [("openrouter.ai", "OPENROUTER_API_KEY"),
                                          ("api.openai.com", "OPENAI_API_KEY")])
@pytest.mark.parametrize("template,public", [
    ("https://{}", True), ("https://{}:443", True), ("https://{}:8443", False),
    ("https://fixture.invalid", False), ("https://fixture@{}", False), ("http://{}", False),
])
def test_provider_origin(host, variable, template, public, isolated_deployment, monkeypatch):
    from core.llm_client import LLMClient
    from services.streaming import _auth_headers

    monkeypatch.setenv(variable, "public-fixture")
    url = template.format(host) + "/v1"
    client = LLMClient(base_url=url, model="fixture", fallback_enabled=True, fallback_base_url=url)
    header = "Bearer public-fixture" if public else None
    assert client.session.headers.get("Authorization") == header
    assert _auth_headers(url).get("Authorization") == header
    monkeypatch.setenv("LLM_API_KEY", "primary-fixture")
    assert _auth_headers(url)["Authorization"] == "Bearer primary-fixture"
    monkeypatch.setenv("LLM_FALLBACK_API_KEY", "fallback-fixture")
    expected = header if public and host == "openrouter.ai" else "Bearer fallback-fixture"
    assert client._get_fallback_session().headers["Authorization"] == expected
