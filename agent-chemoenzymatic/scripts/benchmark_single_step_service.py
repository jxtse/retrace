#!/usr/bin/env python3
"""Safe benchmark harness for the retroplanner single-step HTTP service.

The harness is deliberately dry-run by default.  A low-load run requires
``--execute``; configurations that can create more than a small amount of
load additionally require ``--confirm-high-load``.  Every measured request is
written to JSONL and an aggregate JSON summary is produced after the run.

Examples
--------
Inspect the low-load default plan (does not send HTTP requests)::

    python3 scripts/benchmark_single_step_service.py \
      --url "$RETROPLANNER_SERVICE_URL"

Run two requests against one model, capped at one request/second::

    python3 scripts/benchmark_single_step_service.py \
      --url "$RETROPLANNER_SERVICE_URL" \
      --model "ChemBart (USPTO-Full)" --execute

Measure same-model and four-model fan-out.  Here ``--concurrency 4`` means
four in-flight requests for a same-model scenario and 4 models x 4 requests
= 16 in-flight requests for a four-model scenario::

    python3 scripts/benchmark_single_step_service.py \
      --url "$RETROPLANNER_SERVICE_URL" \
      --model "ChemBart (USPTO-Full)" \
      --model "DMPNNFP (USPTO-Full)" \
      --model "RxNGormer (USPTO-Full)" \
      --model "Pistachio Ringbreaker" \
      --scenario same-model --scenario four-model \
      --key-mode both --concurrency 1,2,4 \
      --requests-per-model 8 --warmup 1 --max-rps 16 \
      --execute --confirm-high-load

``--topk`` is intentionally a single value applied uniformly to every model
in a run.  That makes mixed-model service-capacity comparisons controlled,
but it is not a faithful model-quality baseline when the production tools use
different top-k values.  Run separate single-model benchmarks with their
actual top-k values for that comparison.

The implementation uses only the Python standard library, including a
thread-local persistent ``http.client`` connection for each benchmark worker.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import http.client
import json
import math
import os
import re
import socket
import ssl
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, TextIO
from urllib.parse import SplitResult, urlsplit, urlunsplit


ALLOWED_CONCURRENCIES = (1, 2, 4, 8, 16)
DEFAULT_MODEL = "ChemBart (USPTO-Full)"
DEFAULT_SMILES = (
    "CCO",
    "CC(=O)O",
    "CCN",
    "c1ccccc1",
    "CCOC(=O)C",
    "CC(C)O",
    "O=C(c1ccccc1)c1ccccc1",
    "CCOC(=O)c1ccccc1",
    "CC(C)(C)OC(=O)N",
    "COc1ccccc1",
    "CC1=CC=CC=C1",
    "O=C(O)c1ccccc1",
    "NCCc1ccccc1",
    "CC(C)C(=O)O",
    "CCOC(=O)N",
    "CC(C)(C)c1ccccc1",
)
SCENARIO_KINDS = ("same-model", "four-model")
KEY_MODES = ("same", "different")
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class BenchmarkConfigError(ValueError):
    """Raised for an unsafe or inconsistent benchmark configuration."""


class RequestTimedOut(RuntimeError):
    """A request exceeded its connect/read timeout."""

    def __init__(self, message: str, elapsed_ms: float):
        super().__init__(message)
        self.elapsed_ms = elapsed_ms


class RequestNetworkError(RuntimeError):
    """A request failed before a complete HTTP response was read."""

    def __init__(self, message: str, elapsed_ms: float):
        super().__init__(message)
        self.elapsed_ms = elapsed_ms


@dataclasses.dataclass(frozen=True)
class Scenario:
    scenario_id: str
    url: str
    url_index: int
    kind: str
    models: tuple[str, ...]
    key_mode: str
    concurrency_per_model: int
    requests_per_model: int
    topk: int

    @property
    def effective_concurrency(self) -> int:
        return self.concurrency_per_model * len(self.models)

    @property
    def measured_requests(self) -> int:
        return self.requests_per_model * len(self.models)

    def as_plan(self, warmup_per_model: int) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "url": display_url(self.url),
            "kind": self.kind,
            "models": list(self.models),
            "key_mode": self.key_mode,
            "concurrency_per_model": self.concurrency_per_model,
            "effective_concurrency": self.effective_concurrency,
            "measured_requests": self.measured_requests,
            "warmup_requests": warmup_per_model * len(self.models),
            "topk": self.topk,
        }


@dataclasses.dataclass(frozen=True)
class RequestSpec:
    request_index: int
    model: str
    smiles: str
    phase: str


@dataclasses.dataclass(frozen=True)
class TransportResponse:
    status: int
    body: bytes
    elapsed_ms: float
    response_bytes: int
    body_truncated: bool
    content_type: str | None
    x_cache: str | None = None
    server_timing: str | None = None


class JSONTransport(Protocol):
    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        timeout_s: float,
        max_response_bytes: int,
    ) -> TransportResponse:
        """Send one JSON request and return the complete HTTP response."""


class ScenarioRateLimiter:
    """Thread-safe, evenly spaced request-start limiter for one scenario."""

    def __init__(self, max_rps: float):
        self._interval = 0.0 if max_rps == 0 else 1.0 / max_rps
        self._next_start = 0.0
        self._lock = threading.Lock()

    def wait(self) -> float:
        """Wait for a slot and return the number of milliseconds waited."""
        if self._interval == 0:
            return 0.0
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_start)
            self._next_start = slot + self._interval
        delay = max(0.0, slot - time.monotonic())
        if delay:
            time.sleep(delay)
        return delay * 1000.0


class PersistentHTTPTransport:
    """One persistent HTTP(S) connection per worker thread and origin."""

    def __init__(self) -> None:
        self._local = threading.local()
        self._ssl_context = ssl.create_default_context()

    def _connections(self) -> dict[tuple[str, str, int], http.client.HTTPConnection]:
        connections = getattr(self._local, "connections", None)
        if connections is None:
            connections = {}
            self._local.connections = connections
        return connections

    @staticmethod
    def _origin(parts: SplitResult) -> tuple[str, str, int]:
        assert parts.hostname is not None
        port = parts.port or (443 if parts.scheme == "https" else 80)
        return parts.scheme, parts.hostname, port

    def _new_connection(
        self, origin: tuple[str, str, int], timeout_s: float
    ) -> http.client.HTTPConnection:
        scheme, host, port = origin
        if scheme == "https":
            return http.client.HTTPSConnection(
                host, port, timeout=timeout_s, context=self._ssl_context
            )
        return http.client.HTTPConnection(host, port, timeout=timeout_s)

    def _drop_connection(self, origin: tuple[str, str, int]) -> None:
        connection = self._connections().pop(origin, None)
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        timeout_s: float,
        max_response_bytes: int,
    ) -> TransportResponse:
        parts = urlsplit(url)
        origin = self._origin(parts)
        connections = self._connections()
        connection = connections.get(origin)
        if connection is None:
            connection = self._new_connection(origin, timeout_s)
            connections[origin] = connection
        connection.timeout = timeout_s
        if connection.sock is not None:
            connection.sock.settimeout(timeout_s)

        target = parts.path or "/"
        if parts.query:
            target += "?" + parts.query
        encoded = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "single-step-service-benchmark/1.0",
        }

        started = time.monotonic()
        try:
            connection.request("POST", target, body=encoded, headers=headers)
            response = connection.getresponse()
            body = response.read(max_response_bytes + 1)
            body_truncated = len(body) > max_response_bytes
            if body_truncated:
                body = body[:max_response_bytes]
                # The unread tail makes this connection unsafe to reuse.
                self._drop_connection(origin)
            elapsed_ms = (time.monotonic() - started) * 1000.0
            return TransportResponse(
                status=response.status,
                body=body,
                elapsed_ms=elapsed_ms,
                response_bytes=len(body),
                body_truncated=body_truncated,
                content_type=response.getheader("Content-Type"),
                x_cache=response.getheader("X-Cache"),
                server_timing=response.getheader("Server-Timing"),
            )
        except (socket.timeout, TimeoutError) as exc:
            elapsed_ms = (time.monotonic() - started) * 1000.0
            self._drop_connection(origin)
            raise RequestTimedOut(str(exc) or "request timed out", elapsed_ms) from exc
        except (OSError, http.client.HTTPException) as exc:
            elapsed_ms = (time.monotonic() - started) * 1000.0
            self._drop_connection(origin)
            raise RequestNetworkError(
                f"{type(exc).__name__}: {exc}", elapsed_ms
            ) from exc


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_url(raw_url: str) -> str:
    """Validate a URL and expand a retroplanner base URL to single_step."""
    raw_url = raw_url.strip()
    if any(ord(character) < 32 for character in raw_url):
        raise BenchmarkConfigError("URL contains a control character")
    parts = urlsplit(raw_url)
    if parts.scheme not in {"http", "https"}:
        raise BenchmarkConfigError(f"URL must use http or https: {raw_url!r}")
    if not parts.hostname:
        raise BenchmarkConfigError(f"URL has no hostname: {raw_url!r}")
    if parts.username is not None or parts.password is not None:
        raise BenchmarkConfigError("credentials in --url are not supported")
    try:
        parts.port
    except ValueError as exc:
        raise BenchmarkConfigError(f"URL has an invalid port: {raw_url!r}") from exc
    if parts.fragment:
        raise BenchmarkConfigError("URL fragments are not sent to HTTP services")

    path = parts.path.rstrip("/")
    if not path:
        path = "/retroplanner/api/single_step"
    elif path.endswith("/retroplanner"):
        path += "/api/single_step"
    normalized = urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))
    return normalized


def display_url(url: str) -> str:
    """Return a log-safe URL with query parameters removed."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _dedupe(items: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[Any] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _split_repeated(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    return [piece.strip() for value in values for piece in value.split(",") if piece.strip()]


def load_smiles(path: Path | None) -> list[str]:
    """Load unique SMILES, one per line; blank and ``#`` lines are ignored."""
    if path is None:
        return list(DEFAULT_SMILES)
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise BenchmarkConfigError(f"cannot read SMILES file {path}: {exc}") from exc
    smiles = _dedupe(
        line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")
    )
    if not smiles:
        raise BenchmarkConfigError(f"SMILES file contains no usable lines: {path}")
    return smiles


def build_scenarios(
    urls: Sequence[str],
    models: Sequence[str],
    kinds: Sequence[str],
    key_modes: Sequence[str],
    concurrencies: Sequence[int],
    requests_per_model: int,
    topk: int,
) -> list[Scenario]:
    scenarios: list[Scenario] = []
    ordinal = 1
    for url_index, url in enumerate(urls, start=1):
        for kind in kinds:
            if kind == "same-model":
                model_groups = [(model,) for model in models]
            elif kind == "four-model":
                if len(models) < 4:
                    raise BenchmarkConfigError(
                        "four-model scenario requires at least four --model values"
                    )
                model_groups = [tuple(models[:4])]
            else:
                raise BenchmarkConfigError(f"unknown scenario kind: {kind}")
            for model_group in model_groups:
                for key_mode in key_modes:
                    for concurrency in concurrencies:
                        scenarios.append(
                            Scenario(
                                scenario_id=f"s{ordinal:03d}",
                                url=url,
                                url_index=url_index,
                                kind=kind,
                                models=tuple(model_group),
                                key_mode=key_mode,
                                concurrency_per_model=concurrency,
                                requests_per_model=requests_per_model,
                                topk=topk,
                            )
                        )
                        ordinal += 1
    return scenarios


def build_request_specs(
    scenario: Scenario,
    smiles: Sequence[str],
    *,
    phase: str = "measure",
    warmup_per_model: int = 0,
) -> list[RequestSpec]:
    """Build interleaved model requests for one scenario."""
    count = scenario.requests_per_model if phase == "measure" else warmup_per_model
    if count <= 0:
        return []
    if phase not in {"measure", "warmup"}:
        raise BenchmarkConfigError(f"unknown request phase: {phase}")

    if scenario.key_mode == "different" and phase == "measure":
        if len(smiles) < count:
            raise BenchmarkConfigError(
                f"{scenario.scenario_id} needs {count} distinct SMILES, got {len(smiles)}"
            )
        selected = list(smiles[:count])
    elif scenario.key_mode == "different":
        # Keep warmup off the measured miss keys when another SMILES exists.
        start = min(scenario.requests_per_model, len(smiles) - 1)
        pool = list(smiles[start:]) or [smiles[-1]]
        selected = [pool[i % len(pool)] for i in range(count)]
    else:
        selected = [smiles[0]] * count

    specs: list[RequestSpec] = []
    request_index = 0
    for round_index in range(count):
        for model in scenario.models:
            specs.append(
                RequestSpec(
                    request_index=request_index,
                    model=model,
                    smiles=selected[round_index],
                    phase=phase,
                )
            )
            request_index += 1
    return specs


def make_payload(spec: RequestSpec, topk: int) -> dict[str, Any]:
    return {
        "smiles": spec.smiles,
        "savedOptions": {"topk": topk, "oneStepModel": [spec.model]},
    }


def payload_key(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def response_body_hashes(body: bytes) -> tuple[str, str | None]:
    """Return raw and canonical-JSON SHA256 without retaining response content."""

    raw_hash = hashlib.sha256(body).hexdigest()
    try:
        parsed = json.loads(body.decode("utf-8"))
        canonical = json.dumps(
            parsed,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return raw_hash, None
    return raw_hash, hashlib.sha256(canonical).hexdigest()


def validate_response(
    response: TransportResponse,
) -> tuple[bool, str | None, int | None]:
    """Validate the documented success schema without rejecting empty results."""
    if response.body_truncated:
        return False, "response exceeds --max-response-bytes", None
    try:
        body = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"invalid JSON: {exc}", None
    if not isinstance(body, dict):
        return False, "JSON response is not an object", None
    if body.get("status") != "success":
        return False, f"body status is {body.get('status')!r}, not 'success'", None
    results = body.get("one_step_results")
    if not isinstance(results, dict):
        return False, "one_step_results is not an object", None
    reactants = results.get("reactants")
    scores = results.get("scores")
    if not isinstance(reactants, list):
        return False, "one_step_results.reactants is not a list", None
    if not isinstance(scores, list):
        return False, "one_step_results.scores is not a list", None
    if len(reactants) != len(scores):
        return False, "reactants and scores lengths differ", None
    if not all(isinstance(value, str) for value in reactants):
        return False, "reactants contains a non-string value", None
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in scores):
        return False, "scores contains a non-finite/non-numeric value", None
    return True, None, len(reactants)


def parse_server_timing(value: str | None) -> dict[str, float]:
    """Parse the deployment patch's ``name;dur=milliseconds`` metrics."""
    parsed: dict[str, float] = {}
    if not value:
        return parsed
    for item in value.split(","):
        fields = [field.strip() for field in item.split(";")]
        name = fields[0]
        if not name:
            continue
        for field in fields[1:]:
            key, separator, raw = field.partition("=")
            if separator and key.strip().lower() == "dur":
                try:
                    duration = float(raw.strip().strip('"'))
                except ValueError:
                    continue
                if math.isfinite(duration) and duration >= 0:
                    parsed[name] = duration
                break
    return parsed


def perform_request(
    scenario: Scenario,
    spec: RequestSpec,
    *,
    timeout_s: float,
    max_response_bytes: int,
    transport: JSONTransport,
    rate_limiter: ScenarioRateLimiter,
    submitted_monotonic: float | None = None,
) -> dict[str, Any]:
    rate_wait_ms = rate_limiter.wait()
    started_monotonic = time.monotonic()
    queue_wait_ms = 0.0
    if submitted_monotonic is not None:
        queue_wait_ms = max(0.0, (started_monotonic - submitted_monotonic) * 1000.0)
    started_at = utc_now()
    payload = make_payload(spec, scenario.topk)
    base: dict[str, Any] = {
        "schema_version": 1,
        "scenario_id": scenario.scenario_id,
        "phase": spec.phase,
        "request_index": spec.request_index,
        "started_at": started_at,
        "url": display_url(scenario.url),
        "scenario_kind": scenario.kind,
        "key_mode": scenario.key_mode,
        "model": spec.model,
        "smiles": spec.smiles,
        "payload_key_sha256": payload_key(payload),
        "topk": scenario.topk,
        "concurrency_per_model": scenario.concurrency_per_model,
        "effective_concurrency": scenario.effective_concurrency,
        "queue_wait_ms": round(queue_wait_ms, 3),
        "rate_limit_wait_ms": round(rate_wait_ms, 3),
    }
    try:
        response = transport.post_json(
            scenario.url,
            payload,
            timeout_s=timeout_s,
            max_response_bytes=max_response_bytes,
        )
    except RequestTimedOut as exc:
        return {
            **base,
            "outcome": "timeout",
            "success": False,
            "http_status": None,
            "latency_ms": round(exc.elapsed_ms, 3),
            "response_bytes": 0,
            "response_body_sha256": None,
            "response_canonical_json_sha256": None,
            "response_valid": None,
            "validation_error": None,
            "result_count": None,
            "x_cache": None,
            "server_timing": None,
            "server_timing_ms": {},
            "error": str(exc)[:500],
        }
    except RequestNetworkError as exc:
        return {
            **base,
            "outcome": "network_error",
            "success": False,
            "http_status": None,
            "latency_ms": round(exc.elapsed_ms, 3),
            "response_bytes": 0,
            "response_body_sha256": None,
            "response_canonical_json_sha256": None,
            "response_valid": None,
            "validation_error": None,
            "result_count": None,
            "x_cache": None,
            "server_timing": None,
            "server_timing_ms": {},
            "error": str(exc)[:500],
        }

    valid, validation_error, result_count = validate_response(response)
    raw_body_sha256, canonical_body_sha256 = response_body_hashes(response.body)
    if 500 <= response.status <= 599:
        outcome = "http_5xx"
    elif 400 <= response.status <= 499:
        outcome = "http_4xx"
    elif not 200 <= response.status <= 299:
        outcome = "http_other"
    elif not valid:
        outcome = "invalid_response"
    else:
        outcome = "success"
    return {
        **base,
        "outcome": outcome,
        "success": outcome == "success",
        "http_status": response.status,
        "latency_ms": round(response.elapsed_ms, 3),
        "response_bytes": response.response_bytes,
        "response_truncated": response.body_truncated,
        "response_body_sha256": raw_body_sha256,
        "response_canonical_json_sha256": canonical_body_sha256,
        "content_type": response.content_type,
        "x_cache": response.x_cache.lower() if response.x_cache else None,
        "server_timing": response.server_timing,
        "server_timing_ms": parse_server_timing(response.server_timing),
        "response_valid": valid if 200 <= response.status <= 299 else False,
        "validation_error": validation_error,
        "result_count": result_count,
        "error": None,
    }


def percentile(values: Sequence[float], percent: float) -> float | None:
    """Linear-interpolated percentile (equivalent to NumPy's default)."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (percent / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def summarize_records(
    records: Sequence[Mapping[str, Any]], duration_s: float
) -> dict[str, Any]:
    """Summarize measured records.  Warmup records must be filtered by caller."""
    counts = collections.Counter(str(record["outcome"]) for record in records)
    success_latencies = [
        float(record["latency_ms"])
        for record in records
        if record.get("outcome") == "success"
    ]
    checked = [record for record in records if record.get("response_valid") is not None]
    valid_count = sum(record.get("response_valid") is True for record in checked)
    request_keys = {
        str(record["payload_key_sha256"])
        for record in records
        if record.get("payload_key_sha256")
    }
    cache_statuses = collections.Counter(
        str(record.get("x_cache") or "missing").lower() for record in records
    )
    timing_groups: dict[str, list[float]] = collections.defaultdict(list)
    for record in records:
        timings = record.get("server_timing_ms")
        if isinstance(timings, Mapping):
            for name, value in timings.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    timing_groups[str(name)].append(float(value))
    attempted = len(records)
    duration_s = max(0.0, float(duration_s))
    throughput = attempted / duration_s if duration_s else 0.0
    successful_throughput = counts["success"] / duration_s if duration_s else 0.0
    latency = {
        "population": "validated_success",
        "count": len(success_latencies),
        "min": _round_or_none(min(success_latencies) if success_latencies else None),
        "mean": _round_or_none(
            sum(success_latencies) / len(success_latencies) if success_latencies else None
        ),
        "p50": _round_or_none(percentile(success_latencies, 50)),
        "p95": _round_or_none(percentile(success_latencies, 95)),
        "p99": _round_or_none(percentile(success_latencies, 99)),
        "max": _round_or_none(max(success_latencies) if success_latencies else None),
    }
    return {
        "attempted": attempted,
        "success": counts["success"],
        "timeout": counts["timeout"],
        "http_5xx": counts["http_5xx"],
        "http_4xx": counts["http_4xx"],
        "http_other": counts["http_other"],
        "invalid_response": counts["invalid_response"],
        "network_error": counts["network_error"],
        "success_rate": round(counts["success"] / attempted, 6) if attempted else 0.0,
        "duration_s": round(duration_s, 6),
        "throughput_rps": round(throughput, 6),
        "successful_throughput_rps": round(successful_throughput, 6),
        "latency_ms": latency,
        "cache_keys": {
            "unique": len(request_keys),
            "reuse_ratio": (
                round(1.0 - len(request_keys) / attempted, 6)
                if attempted and request_keys
                else None
            ),
        },
        "server_observability": {
            "x_cache_counts": dict(sorted(cache_statuses.items())),
            "server_timing_present": sum(
                bool(record.get("server_timing")) for record in records
            ),
            "server_timing_ms": {
                name: {
                    "count": len(values),
                    "p50": _round_or_none(percentile(values, 50)),
                    "p95": _round_or_none(percentile(values, 95)),
                    "p99": _round_or_none(percentile(values, 99)),
                    "max": _round_or_none(max(values)),
                }
                for name, values in sorted(timing_groups.items())
            },
        },
        "response_validation": {
            "checked": len(checked),
            "valid": valid_count,
            "invalid": len(checked) - valid_count,
            "valid_rate": round(valid_count / len(checked), 6) if checked else None,
        },
    }


def _by_field(
    records: Sequence[Mapping[str, Any]],
    field: str,
    active_durations: Mapping[str, float],
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for record in records:
        groups[str(record[field])].append(record)
    return {
        key: summarize_records(group, active_durations.get(key, 0.0))
        for key, group in sorted(groups.items())
    }


def write_jsonl_line(handle: TextIO, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def execute_scenario(
    scenario: Scenario,
    smiles: Sequence[str],
    *,
    warmup_per_model: int,
    timeout_s: float,
    max_rps: float,
    max_response_bytes: int,
    transport: JSONTransport,
    details_handle: TextIO,
) -> tuple[list[dict[str, Any]], float, dict[str, int]]:
    warmup_specs = build_request_specs(
        scenario, smiles, phase="warmup", warmup_per_model=warmup_per_model
    )
    warmup_counts: collections.Counter[str] = collections.Counter()
    warmup_limiter = ScenarioRateLimiter(max_rps)
    # Warmup is intentionally serial: it loads/warms models without becoming
    # an unreported burst before the measured scenario.
    for spec in warmup_specs:
        record = perform_request(
            scenario,
            spec,
            timeout_s=timeout_s,
            max_response_bytes=max_response_bytes,
            transport=transport,
            rate_limiter=warmup_limiter,
        )
        warmup_counts[record["outcome"]] += 1
        write_jsonl_line(details_handle, record)

    specs = build_request_specs(scenario, smiles, phase="measure")
    limiter = ScenarioRateLimiter(max_rps)
    records: list[dict[str, Any]] = []
    measured_started = time.monotonic()
    submitted = measured_started
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=scenario.effective_concurrency,
        thread_name_prefix="single-step-bench",
    ) as executor:
        futures = [
            executor.submit(
                perform_request,
                scenario,
                spec,
                timeout_s=timeout_s,
                max_response_bytes=max_response_bytes,
                transport=transport,
                rate_limiter=limiter,
                submitted_monotonic=submitted,
            )
            for spec in specs
        ]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            write_jsonl_line(details_handle, record)
    duration_s = time.monotonic() - measured_started
    records.sort(key=lambda record: int(record["request_index"]))
    return records, duration_s, dict(warmup_counts)


def _high_load_reasons(
    scenarios: Sequence[Scenario], max_rps: float, total_requests: int, topk: int
) -> list[str]:
    reasons: list[str] = []
    peak = max((scenario.effective_concurrency for scenario in scenarios), default=0)
    largest = max((scenario.measured_requests for scenario in scenarios), default=0)
    if peak > 4:
        reasons.append(f"effective concurrency reaches {peak} (>4)")
    if max_rps == 0:
        reasons.append("rate limit is disabled")
    elif max_rps > 4:
        reasons.append(f"per-scenario rate limit is {max_rps:g} rps (>4)")
    if largest > 16:
        reasons.append(f"a scenario sends {largest} measured requests (>16)")
    if total_requests > 64:
        reasons.append(f"the full plan sends {total_requests} requests (>64)")
    if topk > 50:
        reasons.append(f"topk is {topk} (>50)")
    return reasons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run-by-default benchmark for retroplanner single_step HTTP service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        action="append",
        required=True,
        help="Endpoint or /retroplanner/ base URL; repeat for host-d/host-a",
    )
    parser.add_argument(
        "--model",
        "--models",
        action="append",
        help="Model display name; repeat (or comma-separate) for multiple models",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        help="same-model and/or four-model; repeat or comma-separate",
    )
    parser.add_argument(
        "--smiles-file",
        type=Path,
        help="UTF-8 text file containing one SMILES per line",
    )
    parser.add_argument(
        "--key-mode",
        choices=("same", "different", "both"),
        default="same",
        help="Repeat one cache key, use distinct SMILES keys, or run both",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help=(
            "savedOptions.topk applied uniformly to every model; use separate runs "
            "for model-specific production top-k values"
        ),
    )
    parser.add_argument(
        "--concurrency",
        action="append",
        help="Per-model concurrency from 1,2,4,8,16; repeat/comma-separate",
    )
    parser.add_argument(
        "--requests-per-model",
        type=int,
        default=2,
        help="Measured requests sent to each model in each scenario",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Serial warmup requests per model before each measured scenario",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="Per-request connect/read timeout"
    )
    parser.add_argument(
        "--max-rps",
        type=float,
        default=1.0,
        help="Request starts/second for each scenario; 0 disables the limiter",
    )
    parser.add_argument(
        "--scenario-cooldown",
        type=float,
        default=1.0,
        help="Pause between measured scenarios",
    )
    parser.add_argument(
        "--max-response-bytes",
        type=int,
        default=10 * 1024 * 1024,
        help="Maximum response bytes retained and parsed per request",
    )
    parser.add_argument(
        "--max-requests-per-scenario",
        type=int,
        default=64,
        help="Hard safety cap including measured and warmup requests",
    )
    parser.add_argument(
        "--max-total-requests",
        type=int,
        default=512,
        help="Hard safety cap for the complete plan including warmups",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_results")
    )
    parser.add_argument(
        "--run-name",
        help="Output basename (safe timestamp is used when omitted)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow replacing output files"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send HTTP requests; omission always means dry-run",
    )
    parser.add_argument(
        "--confirm-high-load",
        action="store_true",
        help="Second confirmation required for concurrency/rate/count above safe defaults",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit 2 if any measured request is not a validated success",
    )
    return parser


def _validate_and_prepare(
    args: argparse.Namespace,
) -> tuple[list[str], list[str], list[str], list[str], list[int], list[str], list[Scenario], list[str], int]:
    urls = _dedupe(normalize_url(value) for value in args.url)
    models = _dedupe(_split_repeated(args.model) or [DEFAULT_MODEL])
    kinds = _dedupe(_split_repeated(args.scenario) or ["same-model"])
    unknown_kinds = [kind for kind in kinds if kind not in SCENARIO_KINDS]
    if unknown_kinds:
        raise BenchmarkConfigError(f"unknown --scenario value(s): {unknown_kinds}")
    raw_concurrencies = _split_repeated(args.concurrency) or ["1"]
    try:
        concurrencies = _dedupe(int(value) for value in raw_concurrencies)
    except ValueError as exc:
        raise BenchmarkConfigError("--concurrency values must be integers") from exc
    invalid_concurrency = [
        value for value in concurrencies if value not in ALLOWED_CONCURRENCIES
    ]
    if invalid_concurrency:
        raise BenchmarkConfigError(
            f"--concurrency must be one of {ALLOWED_CONCURRENCIES}; got {invalid_concurrency}"
        )
    key_modes = list(KEY_MODES) if args.key_mode == "both" else [args.key_mode]
    if not 1 <= args.topk <= 1000:
        raise BenchmarkConfigError("--topk must be between 1 and 1000")
    if args.requests_per_model <= 0:
        raise BenchmarkConfigError("--requests-per-model must be positive")
    if args.requests_per_model < max(concurrencies):
        raise BenchmarkConfigError(
            "--requests-per-model must be at least the largest --concurrency "
            "so every requested concurrency can actually be reached"
        )
    if args.warmup < 0:
        raise BenchmarkConfigError("--warmup cannot be negative")
    if args.timeout <= 0:
        raise BenchmarkConfigError("--timeout must be positive")
    if args.max_rps < 0:
        raise BenchmarkConfigError("--max-rps cannot be negative")
    if args.scenario_cooldown < 0:
        raise BenchmarkConfigError("--scenario-cooldown cannot be negative")
    if args.max_response_bytes <= 0:
        raise BenchmarkConfigError("--max-response-bytes must be positive")
    if args.max_requests_per_scenario <= 0 or args.max_total_requests <= 0:
        raise BenchmarkConfigError("request safety caps must be positive")

    smiles = load_smiles(args.smiles_file)
    scenarios = build_scenarios(
        urls,
        models,
        kinds,
        key_modes,
        concurrencies,
        args.requests_per_model,
        args.topk,
    )
    distinct_needed = args.requests_per_model + (
        1 if args.warmup and "different" in key_modes else 0
    )
    if "different" in key_modes and len(smiles) < distinct_needed:
        raise BenchmarkConfigError(
            "different key mode needs at least --requests-per-model unique SMILES"
            + (" plus one separate warmup SMILES" if args.warmup else "")
        )
    per_scenario_counts = [
        scenario.measured_requests + args.warmup * len(scenario.models)
        for scenario in scenarios
    ]
    too_large = [
        scenario.scenario_id
        for scenario, count in zip(scenarios, per_scenario_counts)
        if count > args.max_requests_per_scenario
    ]
    if too_large:
        raise BenchmarkConfigError(
            "scenario request cap exceeded by "
            + ", ".join(too_large)
            + "; raise --max-requests-per-scenario intentionally"
        )
    total_requests = sum(per_scenario_counts)
    if total_requests > args.max_total_requests:
        raise BenchmarkConfigError(
            f"plan has {total_requests} requests, above --max-total-requests "
            f"{args.max_total_requests}"
        )
    high_load_reasons = _high_load_reasons(
        scenarios, args.max_rps, total_requests, args.topk
    )
    return (
        urls,
        models,
        kinds,
        key_modes,
        concurrencies,
        smiles,
        scenarios,
        high_load_reasons,
        total_requests,
    )


def _default_run_name() -> str:
    return dt.datetime.now().strftime("single_step_%Y%m%d_%H%M%S")


def _make_plan(
    args: argparse.Namespace,
    *,
    urls: Sequence[str],
    models: Sequence[str],
    kinds: Sequence[str],
    key_modes: Sequence[str],
    concurrencies: Sequence[int],
    scenarios: Sequence[Scenario],
    high_load_reasons: Sequence[str],
    total_requests: int,
) -> dict[str, Any]:
    return {
        "mode": "execute" if args.execute else "dry-run",
        "http_requests_will_be_sent": bool(args.execute),
        "urls": [display_url(url) for url in urls],
        "models": list(models),
        "scenario_kinds": list(kinds),
        "key_modes": list(key_modes),
        "concurrencies_per_model": list(concurrencies),
        "topk": args.topk,
        "topk_policy": "uniform_across_models",
        "requests_per_model": args.requests_per_model,
        "warmup_per_model_per_scenario": args.warmup,
        "timeout_s": args.timeout,
        "max_rps_per_scenario": args.max_rps,
        "scenario_cooldown_s": args.scenario_cooldown,
        "scenario_count": len(scenarios),
        "total_requests_including_warmup": total_requests,
        "high_load_confirmation_required": bool(high_load_reasons),
        "high_load_reasons": list(high_load_reasons),
        "scenarios": [scenario.as_plan(args.warmup) for scenario in scenarios],
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path | None, Path | None]:
    (
        urls,
        models,
        kinds,
        key_modes,
        concurrencies,
        smiles,
        scenarios,
        high_load_reasons,
        total_requests,
    ) = _validate_and_prepare(args)
    plan = _make_plan(
        args,
        urls=urls,
        models=models,
        kinds=kinds,
        key_modes=key_modes,
        concurrencies=concurrencies,
        scenarios=scenarios,
        high_load_reasons=high_load_reasons,
        total_requests=total_requests,
    )
    if not args.execute:
        return plan, None, None
    if high_load_reasons and not args.confirm_high_load:
        raise BenchmarkConfigError(
            "high-load plan requires --confirm-high-load: "
            + "; ".join(high_load_reasons)
        )

    run_name = args.run_name or _default_run_name()
    if not RUN_NAME_RE.fullmatch(run_name):
        raise BenchmarkConfigError(
            "--run-name may contain only letters, digits, dot, underscore, and dash"
        )
    output_dir: Path = args.output_dir
    details_path = output_dir / f"{run_name}.jsonl"
    summary_path = output_dir / f"{run_name}.summary.json"
    temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    if not args.overwrite and (
        details_path.exists() or summary_path.exists() or temporary_path.exists()
    ):
        raise BenchmarkConfigError(
            f"output exists for run {run_name!r}; use --overwrite or a new --run-name"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    scenario_summaries: list[dict[str, Any]] = []
    model_active_durations: collections.Counter[str] = collections.Counter()
    key_mode_active_durations: collections.Counter[str] = collections.Counter()
    url_active_durations: collections.Counter[str] = collections.Counter()
    transport = PersistentHTTPTransport()
    benchmark_started_at = utc_now()
    benchmark_started = time.monotonic()
    open_mode = "w" if args.overwrite else "x"
    with details_path.open(open_mode, encoding="utf-8") as details_handle:
        for index, scenario in enumerate(scenarios):
            records, duration_s, warmup_counts = execute_scenario(
                scenario,
                smiles,
                warmup_per_model=args.warmup,
                timeout_s=args.timeout,
                max_rps=args.max_rps,
                max_response_bytes=args.max_response_bytes,
                transport=transport,
                details_handle=details_handle,
            )
            all_records.extend(records)
            for model in scenario.models:
                model_active_durations[model] += duration_s
            key_mode_active_durations[scenario.key_mode] += duration_s
            url_active_durations[display_url(scenario.url)] += duration_s
            scenario_summary = {
                **scenario.as_plan(args.warmup),
                "warmup_outcomes": warmup_counts,
                "metrics": summarize_records(records, duration_s),
                "by_model": {
                    model: summarize_records(
                        [record for record in records if record["model"] == model],
                        duration_s,
                    )
                    for model in scenario.models
                },
            }
            scenario_summaries.append(scenario_summary)
            print(
                json.dumps(
                    {
                        "scenario_id": scenario.scenario_id,
                        "success": scenario_summary["metrics"]["success"],
                        "attempted": scenario_summary["metrics"]["attempted"],
                        "throughput_rps": scenario_summary["metrics"]["throughput_rps"],
                        "p95_ms": scenario_summary["metrics"]["latency_ms"]["p95"],
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )
            if index + 1 < len(scenarios) and args.scenario_cooldown:
                time.sleep(args.scenario_cooldown)

    benchmark_wall_duration = time.monotonic() - benchmark_started
    aggregate_measured_duration = sum(
        scenario["metrics"]["duration_s"] for scenario in scenario_summaries
    )
    totals = summarize_records(all_records, aggregate_measured_duration)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_started_at": benchmark_started_at,
        "benchmark_finished_at": utc_now(),
        "benchmark_wall_duration_s": round(benchmark_wall_duration, 6),
        "details_jsonl": str(details_path.resolve()),
        "plan": plan,
        "totals": totals,
        "by_model": _by_field(all_records, "model", model_active_durations),
        "by_url": _by_field(all_records, "url", url_active_durations),
        "by_key_mode": _by_field(
            all_records, "key_mode", key_mode_active_durations
        ),
        "scenarios": scenario_summaries,
    }
    with temporary_path.open(open_mode, encoding="utf-8") as temporary_handle:
        temporary_handle.write(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    os.replace(temporary_path, summary_path)
    return summary, details_path, summary_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result, details_path, summary_path = run(args)
    except BenchmarkConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not args.execute:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        print(
            "DRY RUN ONLY: add --execute to send low-load HTTP requests; "
            "high-load plans also require --confirm-high-load.",
            file=sys.stderr,
        )
        return 0

    print(
        json.dumps(
            {
                "details_jsonl": str(details_path),
                "summary_json": str(summary_path),
                "totals": result["totals"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_error and result["totals"]["success"] != result["totals"]["attempted"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
