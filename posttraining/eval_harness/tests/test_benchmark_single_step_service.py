from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import pytest

import scripts.benchmark_single_step_service as bench
from scripts.benchmark_single_step_service import (
    BenchmarkConfigError,
    PersistentHTTPTransport,
    RequestTimedOut,
    RequestSpec,
    Scenario,
    ScenarioRateLimiter,
    TransportResponse,
    build_parser,
    build_request_specs,
    build_scenarios,
    load_smiles,
    main,
    normalize_url,
    payload_key,
    perform_request,
    percentile,
    response_body_hashes,
    run,
    summarize_records,
    validate_response,
)


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.models: list[str] = []


class _FakeTransport:
    """Deterministic transport that exercises the harness without sockets."""

    def __init__(self, delay_s: float = 0.0) -> None:
        self.state = _State()
        self.delay_s = delay_s

    def post_json(
        self,
        _url: str,
        payload: Mapping[str, Any],
        timeout_s: float,
        max_response_bytes: int,
    ) -> TransportResponse:
        del max_response_bytes
        smiles = str(payload["smiles"])
        model = str(payload["savedOptions"]["oneStepModel"][0])
        with self.state.lock:
            self.state.calls += 1
            self.state.active += 1
            self.state.max_active = max(self.state.max_active, self.state.active)
            self.state.models.append(model)
        started = time.monotonic()
        try:
            if smiles == "TIMEOUT":
                time.sleep(min(timeout_s, 0.01))
                raise RequestTimedOut("simulated timeout", timeout_s * 1000.0)
            if self.delay_s:
                time.sleep(self.delay_s)
            if smiles == "SERVERERR":
                status = 503
                body: object = {"status": "error", "message": "busy"}
            elif smiles == "INVALID":
                status = 200
                body = {"status": "success", "one_step_results": {"reactants": "bad"}}
            else:
                status = 200
                body = {
                    "status": "success",
                    "message": "Data received",
                    "one_step_results": {
                        "reactants": ["C.C"],
                        "scores": [0.9],
                        "model_full_name": [model],
                    },
                }
            encoded = json.dumps(body).encode("utf-8")
            return TransportResponse(
                status=status,
                body=encoded,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                response_bytes=len(encoded),
                body_truncated=False,
                content_type="application/json",
            )
        finally:
            with self.state.lock:
                self.state.active -= 1


FAKE_URL = "http://benchmark.invalid/retroplanner/api/single_step"


def _scenario(
    url: str,
    *,
    models: tuple[str, ...] = ("model-a",),
    key_mode: str = "same",
    concurrency: int = 1,
    requests_per_model: int = 1,
    kind: str = "same-model",
) -> Scenario:
    return Scenario(
        scenario_id="s001",
        url=url,
        url_index=1,
        kind=kind,
        models=models,
        key_mode=key_mode,
        concurrency_per_model=concurrency,
        requests_per_model=requests_per_model,
        topk=10,
    )


def test_normalize_url_accepts_base_and_exact_endpoint() -> None:
    assert normalize_url("http://host-d:8001/retroplanner/") == (
        "http://host-d:8001/retroplanner/api/single_step"
    )
    assert normalize_url("https://example.test") == (
        "https://example.test/retroplanner/api/single_step"
    )
    exact = "http://host-a:8001/retroplanner/api/single_step"
    assert normalize_url(exact) == exact
    with pytest.raises(BenchmarkConfigError):
        normalize_url("ftp://example.test/retroplanner/")
    with pytest.raises(BenchmarkConfigError):
        normalize_url("http://user:secret@example.test/retroplanner/")


def test_response_body_hashes_canonicalize_json_key_order() -> None:
    raw_a, canonical_a = response_body_hashes(b'{"b":2,"a":1}')
    raw_b, canonical_b = response_body_hashes(b'{"a":1,"b":2}')
    assert raw_a != raw_b
    assert canonical_a == canonical_b
    assert response_body_hashes(b"not-json")[1] is None


def test_load_smiles_ignores_comments_blanks_bom_and_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "smiles.txt"
    path.write_text("\ufeff# comment\nCCO\n\nCCN\nCCO\n", encoding="utf-8")
    assert load_smiles(path) == ["CCO", "CCN"]


def test_build_scenarios_models_effective_concurrency() -> None:
    models = ["m1", "m2", "m3", "m4"]
    scenarios = build_scenarios(
        ["http://example.test/api"],
        models,
        ["same-model", "four-model"],
        ["same"],
        [4],
        requests_per_model=8,
        topk=5,
    )
    assert len(scenarios) == 5
    assert [scenario.effective_concurrency for scenario in scenarios[:4]] == [4] * 4
    four_model = scenarios[-1]
    assert four_model.models == tuple(models)
    assert four_model.effective_concurrency == 16
    assert four_model.measured_requests == 32


def test_four_model_scenario_requires_four_models() -> None:
    with pytest.raises(BenchmarkConfigError, match="requires at least four"):
        build_scenarios(
            ["http://example.test/api"],
            ["m1", "m2", "m3"],
            ["four-model"],
            ["same"],
            [1],
            requests_per_model=1,
            topk=10,
        )


def test_request_specs_same_and_different_cache_keys() -> None:
    same = _scenario(
        "http://example.test/api", requests_per_model=3, key_mode="same"
    )
    same_specs = build_request_specs(same, ["CCO", "CCN", "CCC"])
    same_keys = [payload_key({"s": spec.smiles, "m": spec.model}) for spec in same_specs]
    assert len(set(same_keys)) == 1

    different = _scenario(
        "http://example.test/api", requests_per_model=3, key_mode="different"
    )
    different_specs = build_request_specs(different, ["CCO", "CCN", "CCC"])
    different_keys = [
        payload_key({"s": spec.smiles, "m": spec.model}) for spec in different_specs
    ]
    assert len(set(different_keys)) == 3
    with pytest.raises(BenchmarkConfigError, match="needs 3 distinct"):
        build_request_specs(different, ["CCO", "CCN"])


def test_different_key_warmup_uses_non_measured_smiles() -> None:
    scenario = _scenario(
        "http://example.test/api", requests_per_model=2, key_mode="different"
    )
    measured = build_request_specs(scenario, ["A", "B", "WARM"])
    warmup = build_request_specs(
        scenario, ["A", "B", "WARM"], phase="warmup", warmup_per_model=2
    )
    assert {spec.smiles for spec in measured} == {"A", "B"}
    assert {spec.smiles for spec in warmup} == {"WARM"}


def _response(body: object, status: int = 200) -> TransportResponse:
    encoded = json.dumps(body).encode("utf-8")
    return TransportResponse(
        status=status,
        body=encoded,
        elapsed_ms=1.0,
        response_bytes=len(encoded),
        body_truncated=False,
        content_type="application/json",
    )


def test_persistent_transport_serializes_documented_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_body = json.dumps(
        {
            "status": "success",
            "one_step_results": {"reactants": [], "scores": []},
        }
    ).encode("utf-8")

    class DummyResponse:
        status = 200

        def read(self, _limit: int) -> bytes:
            return response_body

        def getheader(self, name: str) -> str | None:
            return "application/json" if name == "Content-Type" else None

    class DummyConnection:
        sock = None

        def __init__(self) -> None:
            self.timeout = 0.0
            self.request_args: tuple[object, ...] | None = None
            self.request_kwargs: dict[str, object] | None = None

        def request(self, *args: object, **kwargs: object) -> None:
            self.request_args = args
            self.request_kwargs = kwargs

        def getresponse(self) -> DummyResponse:
            return DummyResponse()

        def close(self) -> None:
            return

    connection = DummyConnection()
    transport = PersistentHTTPTransport()
    monkeypatch.setattr(transport, "_new_connection", lambda _origin, _timeout: connection)
    payload = {
        "smiles": "CCO",
        "savedOptions": {"topk": 7, "oneStepModel": ["ChemBart (USPTO-Full)"]},
    }
    response = transport.post_json(
        FAKE_URL, payload, timeout_s=3.0, max_response_bytes=1024
    )
    assert response.status == 200
    assert response.body == response_body
    assert connection.request_args is not None
    assert connection.request_args[:2] == (
        "POST",
        "/retroplanner/api/single_step",
    )
    assert connection.request_kwargs is not None
    assert json.loads(connection.request_kwargs["body"]) == payload


def test_response_validation_accepts_empty_results_and_rejects_bad_schema() -> None:
    valid, error, count = validate_response(
        _response(
            {
                "status": "success",
                "one_step_results": {"reactants": [], "scores": []},
            }
        )
    )
    assert (valid, error, count) == (True, None, 0)

    valid, error, count = validate_response(
        _response(
            {
                "status": "success",
                "one_step_results": {"reactants": ["C"], "scores": []},
            }
        )
    )
    assert valid is False
    assert "lengths differ" in str(error)
    assert count is None


def test_percentiles_and_summary_counts() -> None:
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    records = [
        {"outcome": "success", "latency_ms": 10.0, "response_valid": True},
        {"outcome": "success", "latency_ms": 30.0, "response_valid": True},
        {"outcome": "timeout", "latency_ms": 50.0, "response_valid": None},
        {"outcome": "http_5xx", "latency_ms": 20.0, "response_valid": False},
        {"outcome": "invalid_response", "latency_ms": 15.0, "response_valid": False},
    ]
    summary = summarize_records(records, duration_s=2.0)
    assert summary["attempted"] == 5
    assert summary["success"] == 2
    assert summary["timeout"] == 1
    assert summary["http_5xx"] == 1
    assert summary["invalid_response"] == 1
    assert summary["throughput_rps"] == 2.5
    assert summary["successful_throughput_rps"] == 1.0
    assert summary["latency_ms"]["p50"] == 20.0
    assert summary["response_validation"] == {
        "checked": 4,
        "valid": 2,
        "invalid": 2,
        "valid_rate": 0.5,
    }


@pytest.mark.parametrize(
    ("smiles", "timeout_s", "expected"),
    [
        ("CCO", 1.0, "success"),
        ("SERVERERR", 1.0, "http_5xx"),
        ("INVALID", 1.0, "invalid_response"),
        ("TIMEOUT", 0.03, "timeout"),
    ],
)
def test_request_outcome_classification_with_transport(
    smiles: str, timeout_s: float, expected: str
) -> None:
    scenario = _scenario(FAKE_URL)
    record = perform_request(
        scenario,
        RequestSpec(0, "model-a", smiles, "measure"),
        timeout_s=timeout_s,
        max_response_bytes=1024 * 1024,
        transport=_FakeTransport(),
        rate_limiter=ScenarioRateLimiter(0),
    )
    assert record["outcome"] == expected
    assert record["success"] is (expected == "success")


def test_main_is_dry_run_by_default_and_writes_nothing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _FakeTransport()
    monkeypatch.setattr(bench, "PersistentHTTPTransport", lambda: transport)
    exit_code = main(
        [
            "--url",
            FAKE_URL,
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "dry",
        ]
    )
    assert exit_code == 0
    assert transport.state.calls == 0
    assert list(tmp_path.iterdir()) == []
    output = capsys.readouterr()
    assert '"mode": "dry-run"' in output.out
    assert "DRY RUN ONLY" in output.err


def test_high_load_requires_second_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = _FakeTransport()
    monkeypatch.setattr(bench, "PersistentHTTPTransport", lambda: transport)
    exit_code = main(
        [
            "--url",
            FAKE_URL,
            "--concurrency",
            "8",
            "--requests-per-model",
            "8",
            "--execute",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "blocked",
        ]
    )
    assert exit_code == 2
    assert transport.state.calls == 0
    assert not (tmp_path / "blocked.jsonl").exists()


def test_execute_writes_jsonl_summary_and_per_model_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = _FakeTransport(delay_s=0.01)
    monkeypatch.setattr(bench, "PersistentHTTPTransport", lambda: transport)
    exit_code = main(
        [
            "--url",
            FAKE_URL,
            "--model",
            "model-a",
            "--requests-per-model",
            "2",
            "--max-rps",
            "4",
            "--scenario-cooldown",
            "0",
            "--execute",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "smoke",
        ]
    )
    assert exit_code == 0
    assert transport.state.calls == 2
    detail_lines = (tmp_path / "smoke.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(detail_lines) == 2
    assert all(json.loads(line)["outcome"] == "success" for line in detail_lines)
    summary = json.loads(
        (tmp_path / "smoke.summary.json").read_text(encoding="utf-8")
    )
    assert summary["totals"]["success"] == 2
    assert summary["totals"]["timeout"] == 0
    assert summary["totals"]["http_5xx"] == 0
    assert summary["totals"]["latency_ms"]["p95"] is not None
    assert summary["by_model"]["model-a"]["success"] == 2
    assert summary["by_url"][FAKE_URL]["success"] == 2
    assert summary["scenarios"][0]["metrics"]["throughput_rps"] > 0


def test_four_model_concurrency_reaches_server_in_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = ["m1", "m2", "m3", "m4"]
    transport = _FakeTransport(delay_s=0.08)
    monkeypatch.setattr(bench, "PersistentHTTPTransport", lambda: transport)
    argv = [
        "--url",
        FAKE_URL,
        "--scenario",
        "four-model",
        "--concurrency",
        "1",
        "--requests-per-model",
        "1",
        "--max-rps",
        "0",
        "--scenario-cooldown",
        "0",
        "--execute",
        "--confirm-high-load",
        "--output-dir",
        str(tmp_path),
        "--run-name",
        "four",
    ]
    for model in models:
        argv.extend(["--model", model])
    exit_code = main(argv)
    assert exit_code == 0
    assert transport.state.calls == 4
    assert transport.state.max_active >= 3
    assert set(transport.state.models) == set(models)
    summary = json.loads(
        (tmp_path / "four.summary.json").read_text(encoding="utf-8")
    )
    scenario = summary["scenarios"][0]
    assert scenario["effective_concurrency"] == 4
    assert scenario["metrics"]["attempted"] == 4
    assert set(scenario["by_model"]) == set(models)


def test_run_rejects_existing_output_without_overwrite(tmp_path: Path) -> None:
    (tmp_path / "existing.jsonl").write_text("owned by user\n", encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(
        [
            "--url",
            "http://127.0.0.1:1/retroplanner/",
            "--execute",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "existing",
        ]
    )
    with pytest.raises(BenchmarkConfigError, match="output exists"):
        run(args)
    assert (tmp_path / "existing.jsonl").read_text(encoding="utf-8") == "owned by user\n"
