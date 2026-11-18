"""HTTP client for the retroplanner ``reaction_rater`` endpoint.

The rater is a *discriminative* reaction-feasibility classifier and the sole
per-step feasibility signal in ``score_route`` (the generative
ReactionT5v2-forward round-trip was removed 2026-06-09). It takes a complete
reaction ``r1.r2>>product`` and returns a single feasibility verdict +
confidence, covering the enzymatic / cascade space a USPTO-trained forward
model is blind to.

Server contract (see docs/api_guide.md, reaction_rater)::

    POST {url}
    {"reaction": "r1.r2>>product"}
    -> {"results": {"reaction_is_feasible": bool, "confidence": float},
        "status": "success"}

Production currently routes to host-d (retro-primary.example.org); the retired host-a host
must not remain in active endpoint lists. The client is deliberately fail-soft
at the TRANSPORT layer:
``rate()`` returns ``None`` on *any* terminal failure rather than raising, so
the caller decides the policy. The feasibility pillar (``score_route``) treats
a ``None`` as fatal and raises — there is no fallback scorer since the T5
round-trip was removed — but other callers may choose to degrade gracefully.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

import requests

logger = logging.getLogger(__name__)


class ReactionRaterClient:
    """Thin POST + retry wrapper over the ``reaction_rater`` endpoint.

    Parameters
    ----------
    url
        Full POST URL or list of URLs. Retries advance to the next URL in
        the list (round-robin within one ``rate`` call).
    timeout_s
        Per-request timeout (default 30 s).
    retries
        Retries on connection / timeout / 5xx (default 2 → 3 attempts).
    backoff_base
        Sleep base between retries; ``[base, base*2, ...]``. Pass 0 in tests.
    """

    def __init__(
        self,
        url: str | Sequence[str],
        timeout_s: float = 30.0,
        retries: int = 2,
        backoff_base: float = 0.5,
    ):
        urls = [url] if isinstance(url, str) else list(url)
        deduped: list[str] = []
        for u in urls:
            u = (u or "").strip()
            if u and u not in deduped:
                deduped.append(u)
        if not deduped:
            raise ValueError("ReactionRaterClient requires at least one URL")
        self._urls = tuple(deduped)
        self._timeout = float(timeout_s)
        self._retries = int(retries)
        self._backoff_base = float(backoff_base)
        self._rr_lock = threading.Lock()
        self._rr_idx = 0
        self._session = requests.Session()
        from requests.adapters import HTTPAdapter
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=16, max_retries=0)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    @property
    def urls(self) -> tuple[str, ...]:
        return self._urls

    def _next_start_index(self) -> int:
        with self._rr_lock:
            i = self._rr_idx
            self._rr_idx = (self._rr_idx + 1) % len(self._urls)
            return i

    @staticmethod
    def _build_reaction(reactants: List[str], product: str) -> str:
        return ".".join(r for r in reactants if r) + ">>" + product

    @staticmethod
    def _parse_body(body: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(body, dict) or body.get("status") != "success":
            return None
        results = body.get("results")
        if not isinstance(results, dict) or "confidence" not in results:
            return None
        try:
            conf = float(results["confidence"])
        except (TypeError, ValueError):
            return None
        return {
            "confidence": max(0.0, min(1.0, conf)),
            "feasible": bool(results.get("reaction_is_feasible", conf >= 0.5)),
        }

    def rate(self, reactants: List[str], product: str) -> Optional[Dict[str, Any]]:
        """Score one reaction. Returns ``{"confidence", "feasible"}`` or
        ``None`` on any terminal failure (the caller decides what a ``None``
        means — ``score_route`` treats it as fatal and raises).
        """
        if not reactants or not product:
            return None
        payload = {"reaction": self._build_reaction(reactants, product)}
        start_idx = self._next_start_index()
        last_err: Optional[Exception] = None
        for attempt in range(self._retries + 1):
            url = self._urls[(start_idx + attempt) % len(self._urls)]
            try:
                resp = self._session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self._timeout,
                )
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
            else:
                status = getattr(resp, "status_code", 0)
                if 500 <= status < 600:
                    last_err = requests.HTTPError(f"HTTP {status}")
                elif status >= 400:
                    # 4xx is a request-shape bug — won't fix on retry.
                    logger.warning(
                        "reaction_rater HTTP %s from %s; treating as unavailable",
                        status, url,
                    )
                    return None
                else:
                    try:
                        parsed = self._parse_body(resp.json())
                    except ValueError as e:
                        logger.warning("reaction_rater JSON parse failed (%s); unavailable", e)
                        return None
                    if parsed is None:
                        logger.warning("reaction_rater returned unusable body; unavailable")
                    return parsed
            if attempt < self._retries:
                time.sleep(self._backoff_base * (2 ** attempt))
        logger.warning(
            "reaction_rater unavailable after %d attempts (%s); returning None",
            self._retries + 1, last_err,
        )
        return None
