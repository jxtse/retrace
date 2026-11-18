"""Shared HTTP client for the retroplanner `/single_step` endpoint.

Used by `_RXNGraphormerHTTPEngine`, `_GraphRetrosynHTTPEngine`, and
`_PistachioRingbreakerEngine` (post-migration) so the wire contract,
retry/backoff policy, and score normalisation live in exactly one
place.

The endpoint accepts::

    POST <url>
    Content-Type: application/json
    {"smiles": "<SMILES>",
     "savedOptions": {"topk": <int>, "oneStepModel": ["<model name>"]}}

and returns a body of the form::

    {"status": "success",
     "one_step_results": {
         "reactants": [<dot-joined SMILES>, ...],
         "scores":    [<float>, ...],
         "template":  [<str>, ...],
         ...
     }}

Different `oneStepModel` choices return scores on incompatible scales
(RxNGormer is log-space negative; GraphFP / Pistachio Ringbreaker are
already in [0, 1]). This module normalises the scores per call so
downstream `dedup_predictions(... max(score) ...)` and
`cluster_candidates(...)` ranking stay comparable across engines.
"""
from __future__ import annotations

import logging
import math
import random
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Sequence

import requests

logger = logging.getLogger(__name__)

_MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    # host-d (now the primary host) exposes the DMPNNFP model as
    # `DMPNNFP (USPTO-Full)`; the legacy host-a deployment exposed the same
    # model as `GraphFP (USPTO-Full)`. host-d is tried first now that it is
    # the primary endpoint, so list the host-d name first to avoid a wasted
    # 4xx round-trip on every call; `GraphFP (USPTO-Full)` is retained as
    # the host-a-fallback alias. Confirmed against host-d's
    # `Available 'oneStepModel' options:` error payload 2026-05-27, which
    # lists exactly `DMPNNFP (USPTO-Full)`.
    "GraphFP (USPTO-Full)": (
        "DMPNNFP (USPTO-Full)",
        "GraphFP (USPTO-Full)",
    ),
    # ChemBART: host-d exposes the model as `ChemBart (USPTO-Full)`; the
    # legacy host-a deployment exposes the same model as `ChemBart (Full)`.
    # host-d name first (primary host), host-a name as the failover alias —
    # so a first-hop to host-d uses the canonical name and a retry that
    # lands on host-a automatically switches to the name host-a accepts.
    "ChemBart (USPTO-Full)": (
        "ChemBart (USPTO-Full)",
        "ChemBart (Full)",
    ),
}

_CHEMBART_MODEL_NAMES = frozenset({
    "ChemBart (USPTO-Full)",
    "ChemBart (Full)",
})


class RetroplannerHTTPError(RuntimeError):
    """Raised by `RetroplannerHTTPClient.predict` for *any* terminal
    failure: connection / timeout exhaustion, HTTP 4xx/5xx, JSON parse
    failure, non-success body, or empty `one_step_results`.

    Wrappers catch this single type to decide whether to fall back to a
    local engine.
    """


def _normalise_scores(raw_scores: Sequence[float]) -> List[float]:
    """Per-call score normalisation to [0, 1].

    - All-negative (RxNGormer log-space): exp + max-normalise.
    - All in [0, 1] (GraphFP / Ringbreaker): pass through.
    - Mixed-sign or > 1: min-max normalise; degenerate (single element
      / all equal) returns 0.5 to avoid division by zero.
    - Empty input: empty output.
    """
    if not raw_scores:
        return []
    if max(raw_scores) <= 0:
        exp_scores = [math.exp(s) for s in raw_scores]
        m = max(exp_scores) or 1.0
        return [s / m for s in exp_scores]
    if min(raw_scores) >= 0 and max(raw_scores) <= 1:
        return [float(s) for s in raw_scores]
    lo, hi = min(raw_scores), max(raw_scores)
    if hi > lo:
        return [(s - lo) / (hi - lo) for s in raw_scores]
    return [0.5 for _ in raw_scores]


class RetroplannerHTTPClient:
    """Thin POST + retry + timeout wrapper over `/single_step`.

    Parameters
    ----------
    url
        Full POST URL or list of POST URLs (e.g.
        ``http://retro-primary.example.org:8001/retroplanner/api/single_step``).
        When multiple URLs are supplied, first attempts are spread by a
        thread-safe weighted round-robin (see ``weights``) and transient
        retries advance to the next endpoint in the list.
    weights
        Optional per-URL integer weights, parallel to ``url``. A host
        with weight N receives ~N× the first-hop share of a host with
        weight 1, so ``[8, 1]`` sends ~8/9 of first attempts to the
        primary (host-d) and ~1/9 to the fallback (host-a) — host-a keeps
        sharing load under high concurrency rather than sitting idle.
        Selection uses smooth weighted round-robin (Nginx SWRR), so the
        stream is interleaved rather than bursty. Defaults to equal
        weights, which reproduces plain round-robin (0, 1, 0, 1, ...).
        Either failure still advances to the next endpoint, so the
        weighting is a load-distribution knob, not a single-point-of-
        failure.
    timeout_s
        Per-request timeout (default 120 s — RxNGormer typically takes
        2–3 s but the endpoint can stall on cold-start).
    retries
        Number of retries on `ConnectionError` / `Timeout` / 5xx
        (default 3 → up to 4 attempts total).
    backoff_base
        Sleep base in seconds between retries (default 1.0). Sleep
        sequence is `[base, base*2, base*4, ...]`. Pass 0 in tests to
        skip waiting.
    chembart_max_attempts
        Maximum total HTTP attempts (including the first) for ChemBART.
        This is deliberately separate from ``retries``: a capacity 503 from
        the two-replica ChemBART pool must not turn four Agent workers into a
        4-attempt retry herd, while the lighter models retain their existing
        retry policy. Default 2.
    chembart_deadline_s
        Total wall-clock budget for one logical ChemBART prediction,
        including every attempt and backoff. Each request timeout is clamped
        to the remaining budget. Default 130 s, leaving a small margin above
        host-d's 110 s pool deadline without outliving the Agent's 300 s default
        tool budget.
    chembart_retry_min_budget_s
        Minimum request budget that must remain *after* retry backoff before a
        second ChemBART POST may start. Production default 75 s matches the
        deployed replicas' minimum safe execution budget. This distinguishes a
        fast capacity 503 (retryable) from a 503 returned after roughly the
        whole 110 s server queue deadline (terminal; retrying would create
        ghost compute that the client cannot wait for).
    retry_jitter_s
        Maximum small positive jitter added to ChemBART retry sleeps. The
        server's ``Retry-After`` remains a floor. Default 0.25 s.
    """

    def __init__(
        self,
        url: str | Sequence[str],
        timeout_s: float = 120.0,
        retries: int = 3,
        backoff_base: float = 1.0,
        weights: Optional[Sequence[int]] = None,
        chembart_max_attempts: int = 2,
        chembart_deadline_s: float = 130.0,
        chembart_retry_min_budget_s: float = 75.0,
        retry_jitter_s: float = 0.25,
    ):
        if isinstance(url, str):
            urls = [url]
            raw_weights = list(weights) if weights is not None else None
        else:
            urls = list(url)
            raw_weights = list(weights) if weights is not None else None
        # Deduplicate URLs while keeping the first occurrence's position,
        # and carry each surviving URL's weight along with it.
        deduped: "OrderedDict[str, int]" = OrderedDict()
        for i, u in enumerate(urls):
            u = (u or "").strip()
            if not u or u in deduped:
                continue
            w = 1
            if raw_weights is not None and i < len(raw_weights):
                try:
                    w = int(raw_weights[i])
                except (TypeError, ValueError):
                    w = 1
            deduped[u] = max(1, w)
        if not deduped:
            raise ValueError("RetroplannerHTTPClient requires at least one URL")
        self._urls = tuple(deduped.keys())
        self._weights = tuple(deduped.values())
        self._rr_lock = threading.Lock()
        # Smooth-weighted-round-robin running counters (Nginx SWRR). With
        # all weights equal this yields the same 0, 1, 0, 1, ... sequence
        # as the previous plain round-robin.
        self._swrr_current = [0] * len(self._urls)
        self._timeout = float(timeout_s)
        self._retries = int(retries)
        self._backoff_base = float(backoff_base)
        self._chembart_max_attempts = int(chembart_max_attempts)
        self._chembart_deadline_s = float(chembart_deadline_s)
        self._chembart_retry_min_budget_s = float(chembart_retry_min_budget_s)
        self._retry_jitter_s = float(retry_jitter_s)
        if self._chembart_max_attempts < 1:
            raise ValueError("chembart_max_attempts must be at least 1")
        if not math.isfinite(self._chembart_deadline_s) or self._chembart_deadline_s <= 0:
            raise ValueError("chembart_deadline_s must be finite and positive")
        if (
            not math.isfinite(self._chembart_retry_min_budget_s)
            or self._chembart_retry_min_budget_s < 0
            or self._chembart_retry_min_budget_s > self._chembart_deadline_s
        ):
            raise ValueError(
                "chembart_retry_min_budget_s must be finite, non-negative, "
                "and no larger than chembart_deadline_s"
            )
        if not math.isfinite(self._retry_jitter_s) or self._retry_jitter_s < 0:
            raise ValueError("retry_jitter_s must be finite and non-negative")
        # Connection-pooled session: previously every `predict()` call
        # spun up a fresh TCP connection (and TLS handshake on https
        # endpoints), which on a small ensemble target was adding 50–
        # 200 ms per engine call. requests.Session is thread-safe for
        # the use we make of it (one POST per call, no shared cookies)
        # and the underlying urllib3 connection pool keeps the pipe
        # warm. The pool size is bumped past the default 10 because
        # the agent regularly fans 8+ engines out concurrently from
        # explore_branches / the ensemble fan-out.
        self._session = requests.Session()
        from requests.adapters import HTTPAdapter
        adapter = HTTPAdapter(
            pool_connections=16,
            pool_maxsize=32,
            max_retries=0,  # we own retries; don't double-retry inside urllib3
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    @property
    def urls(self) -> tuple[str, ...]:
        """Configured endpoints, exposed for tests and diagnostics."""
        return self._urls

    @property
    def weights(self) -> tuple[int, ...]:
        """Per-URL first-hop weights, parallel to ``urls``."""
        return self._weights

    def _next_start_index(self) -> int:
        """Pick the next first-hop endpoint via smooth weighted round-robin.

        Nginx's SWRR algorithm: each call adds every endpoint's weight to
        its running counter, selects the endpoint with the highest
        counter, then subtracts the total weight from the winner. The
        resulting stream is interleaved (not bursty) and, over any window
        of `sum(weights)` calls, hands each endpoint exactly its weight's
        share of first hops. With all weights equal this reduces to plain
        0, 1, 0, 1, ... round-robin.
        """
        with self._rr_lock:
            total = 0
            best = 0
            best_val = None
            for i, w in enumerate(self._weights):
                self._swrr_current[i] += w
                total += w
                if best_val is None or self._swrr_current[i] > best_val:
                    best_val = self._swrr_current[i]
                    best = i
            self._swrr_current[best] -= total
            return best

    @staticmethod
    def _model_names(model_name: str) -> tuple[str, ...]:
        return _MODEL_ALIASES.get(model_name, (model_name,))

    @staticmethod
    def _retry_after_seconds(response: Any) -> Optional[float]:
        """Parse an RFC 7231 Retry-After value into non-negative seconds.

        host-d currently emits the delta-seconds form (``Retry-After: 1``),
        but accepting an HTTP date keeps the client correct for proxies and
        future service implementations. Invalid/negative values fall back to
        the normal exponential policy.
        """
        headers = getattr(response, "headers", None) or {}
        raw = headers.get("Retry-After")
        if raw is None:
            return None
        value = str(raw).strip()
        if not value:
            return None
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            try:
                when = parsedate_to_datetime(value)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                seconds = (when - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                return None
        if not math.isfinite(seconds) or seconds < 0:
            return None
        return seconds

    @staticmethod
    def _parse_body(body: dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(body, dict):
            raise RetroplannerHTTPError("JSON response is not an object")
        if body.get("status") != "success":
            raise RetroplannerHTTPError(
                f"non-success status: {body.get('message', '?')!r}"
            )

        osr = body.get("one_step_results") or {}
        reactants = osr.get("reactants") or []
        if not reactants:
            raise RetroplannerHTTPError("empty one_step_results")

        scores = osr.get("scores") or []
        templates = osr.get("template") or []
        try:
            raw_scores_f = [float(s) for s in scores]
        except (TypeError, ValueError) as e:
            raise RetroplannerHTTPError(f"non-numeric scores: {e}") from e
        norm_scores = _normalise_scores(raw_scores_f)

        entries: List[Dict[str, Any]] = []
        for i, smi in enumerate(reactants):
            if not smi:
                continue
            sc = norm_scores[i] if i < len(norm_scores) else 0.0
            entry: Dict[str, Any] = {
                "reactants": smi,
                "score": float(sc),
            }
            if i < len(templates) and templates[i]:
                entry["template"] = str(templates[i])
            entries.append(entry)
        if not entries:
            raise RetroplannerHTTPError("empty one_step_results")
        return entries

    def predict(
        self, smiles: str, model_name: str, topk: int,
    ) -> List[Dict[str, Any]]:
        """POST a single-target prediction and return parsed entries.

        Returns a list of `{"reactants": str, "score": float [, "template": str]}`
        sorted in the order the endpoint returned. Raises
        `RetroplannerHTTPError` on any terminal failure.
        """
        last_err: Optional[Exception] = None
        start_idx = self._next_start_index()
        model_names = self._model_names(model_name)
        is_chembart = model_name in _CHEMBART_MODEL_NAMES
        max_attempts = (
            self._chembart_max_attempts if is_chembart else self._retries + 1
        )
        deadline = (
            time.monotonic() + self._chembart_deadline_s
            if is_chembart else None
        )
        http_attempts = 0
        for attempt in range(max_attempts):
            url = self._urls[(start_idx + attempt) % len(self._urls)]
            retry_endpoint = False
            retry_after_s: Optional[float] = None
            for model_attempt, candidate_model in enumerate(model_names):
                payload = {
                    "smiles": smiles,
                    "savedOptions": {
                        "topk": int(topk),
                        "oneStepModel": [candidate_model],
                    },
                }
                request_timeout = self._timeout
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RetroplannerHTTPError(
                            "ChemBART request deadline exhausted before "
                            f"attempt {attempt + 1}/{max_attempts}: {last_err}"
                        )
                    request_timeout = min(request_timeout, remaining)
                if is_chembart:
                    if http_attempts >= max_attempts:
                        raise RetroplannerHTTPError(
                            "ChemBART HTTP attempt budget exhausted after "
                            f"{http_attempts} attempts: {last_err}"
                        )
                    http_attempts += 1
                try:
                    resp = self._session.post(
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=request_timeout,
                    )
                except requests.ReadTimeout as e:
                    # A read timeout is ambiguous: the POST may already be
                    # running on the GPU. Retrying the only endpoint can create
                    # a ghost duplicate after the Agent has abandoned the
                    # original call. With a distinct endpoint we retain the
                    # historical failover behaviour.
                    last_err = e
                    if len(self._urls) == 1:
                        raise RetroplannerHTTPError(
                            f"read timeout from {url}; single endpoint not retried"
                        ) from e
                    retry_endpoint = True
                    break
                except (requests.ConnectionError, requests.Timeout) as e:
                    last_err = e
                    retry_endpoint = True
                    break

                # Classify HTTP status before reading the body.
                try:
                    resp.raise_for_status()
                except requests.HTTPError as e:
                    status = getattr(resp, "status_code", 0)
                    last_err = e
                    if 500 <= status < 600:
                        retry_after_s = self._retry_after_seconds(resp)
                        retry_endpoint = True
                        break
                    if model_attempt + 1 < len(model_names):
                        logger.info(
                            "retroplanner model name %r failed on %s (HTTP %s); "
                            "trying alias %r",
                            candidate_model, url, status, model_names[model_attempt + 1],
                        )
                        continue
                    raise RetroplannerHTTPError(
                        f"HTTP {status} from {url}: {e}"
                    ) from e

                try:
                    body = resp.json()
                except ValueError as e:
                    raise RetroplannerHTTPError(f"JSON parse failure: {e}") from e

                try:
                    entries = self._parse_body(body)
                except RetroplannerHTTPError as e:
                    last_err = e
                    if model_attempt + 1 < len(model_names):
                        logger.info(
                            "retroplanner model name %r returned no usable "
                            "predictions on %s (%s); trying alias %r",
                            candidate_model, url, e, model_names[model_attempt + 1],
                        )
                        continue
                    raise

                if candidate_model != model_name:
                    logger.info(
                        "retroplanner used model alias %r for requested model %r on %s",
                        candidate_model, model_name, url,
                    )
                return entries

            if retry_endpoint:
                can_retry = attempt + 1 < max_attempts and (
                    not is_chembart or http_attempts < max_attempts
                )
                if can_retry:
                    sleep_s = self._backoff_base * (2 ** attempt)
                    if retry_after_s is not None:
                        sleep_s = max(sleep_s, retry_after_s)
                    # Positive-only jitter keeps Retry-After a strict floor.
                    # Restrict it to ChemBART so other models preserve their
                    # exact historical exponential schedule.
                    if is_chembart and self._retry_jitter_s:
                        sleep_s += random.uniform(0.0, self._retry_jitter_s)
                    if deadline is not None:
                        remaining = deadline - time.monotonic()
                        required = sleep_s + self._chembart_retry_min_budget_s
                        if required > remaining:
                            raise RetroplannerHTTPError(
                                "ChemBART retry skipped: insufficient remaining "
                                "budget after backoff "
                                f"(sleep={sleep_s:.3f}s, "
                                f"minimum_request={self._chembart_retry_min_budget_s:.3f}s, "
                                f"remaining={max(0.0, remaining):.3f}s): {last_err}"
                            )
                    logger.warning(
                        "retroplanner POST attempt %d/%d to %s failed (%s); "
                        "backing off %.3fs%s",
                        attempt + 1, max_attempts, url, last_err, sleep_s,
                        (
                            f" (Retry-After={retry_after_s:g}s)"
                            if retry_after_s is not None else ""
                        ),
                    )
                    time.sleep(sleep_s)
                    continue
                raise RetroplannerHTTPError(
                    "request failed after "
                    f"{http_attempts if is_chembart else attempt + 1} attempts: "
                    f"{last_err}"
                )

        # Defensive — every branch above either returns or raises.
        raise RetroplannerHTTPError(
            f"unreachable retry loop exit: {last_err}"
        )
