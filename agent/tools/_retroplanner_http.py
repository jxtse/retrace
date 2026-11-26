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
import threading
import time
from collections import OrderedDict
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
        Full POST URL or ordered list of POST URLs.
        When multiple URLs are supplied, first attempts are spread by a
        thread-safe weighted round-robin (see ``weights``) and transient
        retries advance to the next endpoint in the list.
    weights
        Optional per-URL integer weights, parallel to ``url``. A host
        with weight N receives ~N× the first-hop share of a host with
        weight 1, so ``[8, 1]`` sends ~8/9 of first attempts to the
        first endpoint and ~1/9 to the second endpoint, which keeps
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
    """

    def __init__(
        self,
        url: str | Sequence[str],
        timeout_s: float = 120.0,
        retries: int = 3,
        backoff_base: float = 1.0,
        weights: Optional[Sequence[int]] = None,
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
        for attempt in range(self._retries + 1):
            url = self._urls[(start_idx + attempt) % len(self._urls)]
            retry_endpoint = False
            for model_attempt, candidate_model in enumerate(model_names):
                payload = {
                    "smiles": smiles,
                    "savedOptions": {
                        "topk": int(topk),
                        "oneStepModel": [candidate_model],
                    },
                }
                try:
                    resp = self._session.post(
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=self._timeout,
                    )
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
                if attempt < self._retries:
                    sleep_s = self._backoff_base * (2 ** attempt)
                    logger.warning(
                        "retroplanner POST attempt %d/%d to %s failed (%s); "
                        "backing off %.1fs",
                        attempt + 1, self._retries + 1, url, last_err, sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                raise RetroplannerHTTPError(
                    f"request failed after {attempt + 1} attempts: {last_err}"
                )

        # Defensive — every branch above either returns or raises.
        raise RetroplannerHTTPError(
            f"unreachable retry loop exit: {last_err}"
        )
