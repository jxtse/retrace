"""Unified neural one-step retrosynthesis tool.

A single tool with an ``engine`` parameter; every engine now reaches its
model through the shared retroplanner ``/single_step`` HTTP service (the
in-process DMPNNFP GNN and the ``.venv-rxngraphormer`` transformer
subprocess were removed 2026-06-08):

  - ``graph_retrosyn``         : GraphFP / DMPNNFP, USPTO templates (HTTP)
  - ``rxngraphormer``          : RxNGormer transformer, USPTO-Full (HTTP)
  - ``template_relevance``     : configured Reaxys HTTP API
  - ``chembart``               : ChemBART HTTP service
  - ``pistachio_ringbreaker``  : retroplanner HTTP API, ring-opening specialist
                                 (excluded from the default ensemble — call
                                 explicitly when the target has rings worth
                                 opening)

Output schema is unified across engines::

    {"target": <smiles>,
     "engine": <name>,
     "predictions": [{"reactants": <smiles_with_dots>, "score": <float>,
                      "template": <smarts?>}, ...]}
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

import requests

from tools.base import Tool
from tools._ensemble_utils import (
    bucket_by_classifier,
    dedup_predictions,
    pick_per_bucket,
)
from tools._precursor_sanity import filter_precursors
from tools._retroplanner_http import RetroplannerHTTPClient, RetroplannerHTTPError


class _ChatStringAdapter:
    """Adapt agent's `LLMClient.chat()` (returns dict) to the
    `chat(messages) -> str` interface that the precursor sanity
    filter expects. Tests can pass a plain object whose `.chat()`
    already returns a string; the adapter handles both shapes.
    """

    def __init__(self, llm: Any):
        self._llm = llm

    def chat(self, messages, **kwargs) -> str:
        # Prefer the stateless `chat_with_override` (use_responses_state=False)
        # so the sanity judge never inherits or mutates the main client's
        # session / GPT-5.5 Responses conversation state. Concurrent
        # neural_retro calls in the same turn would otherwise race on the
        # shared session swap. Fall back to `.chat()` for test stubs that
        # only implement the simple interface.
        override = getattr(self._llm, "chat_with_override", None)
        if callable(override):
            result = override(messages, use_responses_state=False)
        else:
            result = self._llm.chat(messages, **kwargs)
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            try:
                content = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                return ""
            return content or ""
        return str(result) if result is not None else ""

logger = logging.getLogger(__name__)


# Per-engine routing mode. The local DMPNNFP / RXNGraphormer engines (and
# their torch/PyG/modelscope deps + the .venv-rxngraphormer subprocess) were
# removed 2026-06-08 — every neural engine now goes through the shared
# retroplanner `/single_step` HTTP service, so `http_only` is the only mode.
# Kept as a tuple (not a bare string) so a stale
# `engine_modes: {graph_retrosyn: http_with_local_fallback}` overlay fails
# loudly at config time instead of silently routing nowhere.
_ENGINE_MODES = ("http_only",)


# ---------------------------------------------------------------------------
# Engine implementations
# ---------------------------------------------------------------------------


class _GraphRetrosynHTTPEngine:
    """GraphFP (USPTO-Full / DMPNNFP) via the shared retroplanner
    ``/single_step`` HTTP service.

    The in-process DMPNNFP engine (~1.26 GiB CUDA, torch + PyG) was
    removed 2026-06-08; this is now a thin HTTP client. `engine_mode`
    is retained for config symmetry but ``http_only`` is the only
    accepted value.
    """

    _MODEL_NAME = "GraphFP (USPTO-Full)"
    _VALID_MODES = _ENGINE_MODES

    def __init__(
        self,
        http_client,
        engine_mode: str = "http_only",
    ):
        if engine_mode not in self._VALID_MODES:
            raise ValueError(
                f"engine_mode={engine_mode!r} not in {self._VALID_MODES}"
            )
        self._http = http_client
        self._mode = engine_mode

    def predict(
        self, target_smiles: str, topk: int, detail_level: str,
    ) -> Dict[str, Any]:
        entries = self._http.predict(
            smiles=target_smiles,
            model_name=self._MODEL_NAME,
            topk=topk,
        )
        if detail_level == "brief":
            entries = [
                {k: v for k, v in d.items() if k != "template"}
                for d in entries
            ]
        return {
            "target": target_smiles,
            "engine": "graph_retrosyn",
            "predictions": entries[:topk],
        }


class _RXNGraphormerHTTPEngine:
    """RxNGormer (USPTO-Full) via the shared retroplanner
    ``/single_step`` HTTP service.

    The local ``.venv-rxngraphormer`` subprocess engine (opennmt-py
    2.3.0 + torch 2.1, modelscope weight fetch) was removed 2026-06-08;
    this is now a thin HTTP client. `engine_mode` is retained for config
    symmetry but ``http_only`` is the only accepted value, and the HTTP
    endpoint always serves USPTO-Full (the `model=` checkpoint selector
    is no longer honourable, so it is logged and ignored).

    Parameters
    ----------
    http_client
        Pre-built `RetroplannerHTTPClient`. Tests inject a mock.
    engine_mode
        Only ``"http_only"``. See ``core.config`` `neural_retro.engine_modes`.
    """

    # The retroplanner `/single_step` endpoint distinguishes
    # ``"RxNGormer"`` (a stub that always returns empty
    # ``one_step_results``) from ``"RxNGormer (USPTO-Full)"`` (the
    # actual full-reaction model). We pin the full model so HTTP calls
    # return real predictions.
    _MODEL_NAME = "RxNGormer (USPTO-Full)"
    _VALID_MODES = _ENGINE_MODES

    def __init__(
        self,
        http_client: "RetroplannerHTTPClient",
        engine_mode: str = "http_only",
    ):
        if engine_mode not in self._VALID_MODES:
            raise ValueError(
                f"engine_mode={engine_mode!r} not in {self._VALID_MODES}"
            )
        self._http = http_client
        self._mode = engine_mode

    def predict(
        self,
        target_smiles: str,
        topk: int,
        detail_level: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        if model is not None:
            logger.info(
                "rxngraphormer: HTTP endpoint serves USPTO-Full only and "
                "ignores the model parameter (requested=%r)",
                model,
            )

        entries = self._http.predict(
            smiles=target_smiles,
            model_name=self._MODEL_NAME,
            topk=topk,
        )

        # Strip 'template' when caller asked for brief.
        if detail_level == "brief":
            entries = [
                {k: v for k, v in d.items() if k != "template"}
                for d in entries
            ]
        return {
            "target": target_smiles,
            "engine": "rxngraphormer",
            "predictions": entries[:topk],
        }


class _TemplateRelevanceEngine:
    """Reaxys template-relevance HTTP API at the configured URL."""

    def __init__(self, api_url: Optional[str], topk: int,
                 retries: int = 3, backoff_base: float = 1.0):
        if not api_url:
            raise ValueError("neural_retro.template_relevance_url is required")
        self._api_url = api_url
        self._topk = topk
        self._retries = int(retries)
        self._backoff_base = float(backoff_base)
        # Pooled session — same rationale as RetroplannerHTTPClient: avoid
        # per-call TCP/TLS handshakes when the ensemble fans out concurrently.
        self._session = requests.Session()
        from requests.adapters import HTTPAdapter
        self._session.mount(
            "http://",
            HTTPAdapter(pool_connections=8, pool_maxsize=16, max_retries=0),
        )
        self._session.mount(
            "https://",
            HTTPAdapter(pool_connections=8, pool_maxsize=16, max_retries=0),
        )

    def predict(
        self, target_smiles: str, topk: int, detail_level: str
    ) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        data = json.dumps({"smiles": [target_smiles], "max_num_templates": topk})
        # Retry transient connection / 5xx failures before propagating.
        # Without this a single TorchServe restart on the inference box
        # would surface as a hard tool error in the middle of an ensemble
        # call, even though the engine recovers within ~1-2s.
        last_err: Optional[Exception] = None
        for attempt in range(self._retries + 1):
            try:
                resp = self._session.post(
                    self._api_url, headers=headers, data=data, timeout=60,
                )
                resp.raise_for_status()
                break
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                if attempt < self._retries:
                    sleep_s = self._backoff_base * (2 ** attempt)
                    logger.warning(
                        "template_relevance attempt %d/%d failed (%s); backing off %.1fs",
                        attempt + 1, self._retries + 1, e, sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                raise
            except requests.HTTPError as e:
                status = getattr(resp, "status_code", 0)
                if 500 <= status < 600 and attempt < self._retries:
                    last_err = e
                    sleep_s = self._backoff_base * (2 ** attempt)
                    logger.warning(
                        "template_relevance attempt %d/%d returned 5xx (%s); backing off %.1fs",
                        attempt + 1, self._retries + 1, e, sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                raise
        else:
            # Defensive — the for-else triggers only if every attempt raised
            # and we somehow fell out without re-raising.
            raise RuntimeError(
                f"template_relevance: exhausted retries: {last_err}"
            )
        result = resp.json()[0]

        entries: List[Dict[str, Any]] = []
        for i in range(len(result.get("reactants", []))):
            entry = {
                "reactants": result["reactants"][i],
                "score": round(result["scores"][i], 6),
            }
            if detail_level == "full":
                tmpl = result["templates"][i]
                entry["template"] = (
                    tmpl.get("reaction_smarts", "")
                    if isinstance(tmpl, dict)
                    else str(tmpl)
                )
            entries.append(entry)
        return {
            "target": target_smiles,
            "engine": "template_relevance",
            "predictions": entries[:topk],
        }


class _ChemBartEngine:
    """ChemBART via the shared retroplanner ``/single_step`` endpoint.

    Both dedicated and shared clients use the same JSON prediction contract.

    Parameters
    ----------
    api_url
        Backwards-compatible single_step URL, used only to build a
        fallback `RetroplannerHTTPClient` when ``http_client`` is None.
    topk
        Default top-K (caller usually overrides per-call).
    http_client
        Pre-built shared client. When provided, ``api_url`` is unused.
    timeout_s
        Per-request timeout for the fallback-built client only.
    """

    _MODEL_NAME = "ChemBart (USPTO-Full)"

    def __init__(
        self,
        api_url: Optional[str],
        topk: int,
        http_client: Optional[RetroplannerHTTPClient] = None,
        timeout_s: float = 120.0,
    ):
        self._topk = topk
        if http_client is not None:
            self._http = http_client
        else:
            if not api_url:
                raise ValueError(
                    "_ChemBartEngine: provide http_client or api_url"
                )
            self._http = RetroplannerHTTPClient(
                url=api_url, timeout_s=timeout_s, retries=3,
            )
        # Per-canonical-SMILES result cache (bounded LRU at 64 entries).
        # The agent often re-asks the same SMILES across iterations.
        self._result_cache: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()

    def predict(
        self, target_smiles: str, topk: int, detail_level: str
    ) -> Dict[str, Any]:
        try:
            from rdkit import Chem
            _m = Chem.MolFromSmiles(target_smiles)
            _ckey = Chem.MolToSmiles(_m) if _m is not None else target_smiles
        except Exception:
            _ckey = target_smiles
        cache_key = (_ckey, int(topk), detail_level)
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            self._result_cache.move_to_end(cache_key)
            return {
                "target": cached["target"],
                "engine": cached["engine"],
                "predictions": [dict(p) for p in cached["predictions"]],
            }

        entries = self._http.predict(
            smiles=target_smiles,
            model_name=self._MODEL_NAME,
            topk=topk,
        )
        if detail_level == "full":
            for d in entries:
                d.setdefault("template", "[ChemBart]")
        else:
            entries = [
                {k: v for k, v in d.items() if k != "template"}
                for d in entries
            ]
        out = {
            "target": target_smiles,
            "engine": "chembart",
            "predictions": entries[:topk],
        }
        if len(self._result_cache) > 64:
            self._result_cache.popitem(last=False)
        self._result_cache[cache_key] = {
            "target": out["target"],
            "engine": out["engine"],
            "predictions": [dict(p) for p in out["predictions"]],
        }
        return out


class _PistachioRingbreakerEngine:
    """Pistachio Ringbreaker via the shared retroplanner HTTP client.

    Specialist template-relevance model trained for ring-opening /
    ring-modifying disconnections. HTTP-only — no local equivalent
    exists. Participates in the default ensemble fan-out alongside the
    general-purpose engines (was previously excluded; the LLM never
    reached for the standalone alias on ring-rich targets, so the
    disconnections it would have suggested were simply never seen).

    Preserves the per-canonical-SMILES result cache from the previous
    in-engine implementation (ringbreaker round-trips are 10–60 s and
    the agent often re-asks the same SMILES across iterations).

    Parameters
    ----------
    api_url
        Backwards-compatible URL kept so existing callers don't break.
        Used only as the URL to build a fallback `RetroplannerHTTPClient`
        when ``http_client`` is None (e.g. in older config that hasn't
        been migrated to ``retroplanner_http_url`` yet).
    topk
        Default top-K (caller usually overrides per-call).
    http_client
        Pre-built shared client. When provided, ``api_url`` is unused.
    timeout_s
        Per-request timeout for the fallback-built client only.
    """

    def __init__(
        self,
        api_url: Optional[str],
        topk: int,
        http_client: Optional[RetroplannerHTTPClient] = None,
        timeout_s: float = 120.0,
    ):
        self._topk = topk
        if http_client is not None:
            self._http = http_client
        else:
            if not api_url:
                raise ValueError(
                    "_PistachioRingbreakerEngine: provide http_client or api_url"
                )
            self._http = RetroplannerHTTPClient(
                url=api_url, timeout_s=timeout_s, retries=3,
            )
        # Per-canonical-SMILES result cache (bounded FIFO at 64 entries).
        self._result_cache: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()

    def predict(
        self, target_smiles: str, topk: int, detail_level: str,
    ) -> Dict[str, Any]:
        try:
            from rdkit import Chem
            _m = Chem.MolFromSmiles(target_smiles)
            _ckey = Chem.MolToSmiles(_m) if _m is not None else target_smiles
        except Exception:
            _ckey = target_smiles
        cache_key = (_ckey, int(topk), detail_level)
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            self._result_cache.move_to_end(cache_key)
            return {
                "target": cached["target"],
                "engine": cached["engine"],
                "predictions": [dict(p) for p in cached["predictions"]],
            }

        entries = self._http.predict(
            smiles=target_smiles,
            model_name="Pistachio Ringbreaker",
            topk=topk,
        )
        if detail_level == "brief":
            entries = [
                {k: v for k, v in d.items() if k != "template"}
                for d in entries
            ]
        out = {
            "target": target_smiles,
            "engine": "pistachio_ringbreaker",
            "predictions": entries[:topk],
        }
        if len(self._result_cache) > 64:
            self._result_cache.popitem(last=False)
        self._result_cache[cache_key] = {
            "target": out["target"],
            "engine": out["engine"],
            "predictions": [dict(p) for p in out["predictions"]],
        }
        return out


# ---------------------------------------------------------------------------
# Unified tool
# ---------------------------------------------------------------------------


class NeuralRetroTool(Tool):
    """Unified neural one-step retrosynthesis tool (multi-engine ensemble).

    Default behavior is **ensemble**: every configured engine is fanned
    out concurrently at ``per_engine_topk`` candidates each, results are
    canonicalized + deduplicated across engines, bucketed by reaction
    class (rxnfp 10-class clustering when available), and the top
    ``return_topk`` diverse precursor sets are returned to the LLM. Each
    surviving entry carries a ``support`` field listing the engines that
    proposed it — cross-engine agreement is the cheap confidence signal.

    Pass ``engine='<name>'`` explicitly to bypass the ensemble and get a
    single engine's raw top-K (back-compat with the older API).
    """

    name = "neural_retro"
    description = (
        "Neural one-step retrosynthesis. **Default = ensemble**: fans out "
        "across every configured engine (graph_retrosyn, rxngraphormer, "
        "pistachio_ringbreaker for ring-opening, template_relevance, chembart), "
        "dedupes, clusters, and returns the top diverse precursor sets. Each "
        "result carries `support: [engines]` — entries with 2+ engine support "
        "are high-confidence (PREFER these). Ring-opening picks now show up "
        "in the default ensemble (look for `support: [\"pistachio_ringbreaker\"]`); "
        "no separate ringbreaker call is needed on ring-rich targets. "
        "Cost: ~30–60s/call, **counts toward the iteration budget**. "
        "Pairs with `suggest_templates` (run both per non-leaf)."
    )
    tier = 3
    # Hook for engines that should be registered but skipped during the
    # default ensemble fan-out. Empty by design today — pistachio_ringbreaker
    # used to live here (the rationale was "specialist, slow HTTP, pollutes
    # acyclic targets") but in practice the LLM never reached for the
    # standalone alias on ring-rich hard_mol cases (idx 01/06/07/08, all
    # tagged "无法正确开环"), so the disconnections it would have suggested
    # were simply never seen. Per-engine error isolation in `_run_ensemble`
    # already handles a slow/empty ringbreaker response gracefully, and the
    # cross-engine dedupe + clustering absorbs its 1–2 typical hits without
    # flooding the candidate pool.
    _ENSEMBLE_EXCLUDED: frozenset[str] = frozenset()
    parameters = {
        "type": "object",
        "properties": {
            "target_smiles": {
                "type": "string",
                "description": "SMILES of the target molecule to decompose",
            },
            "engine": {
                "type": "string",
                "enum": [
                    "ensemble",
                    "graph_retrosyn",
                    "rxngraphormer",
                    "template_relevance",
                    "chembart",
                    "pistachio_ringbreaker",
                ],
                "default": "ensemble",
                "description": (
                    "'ensemble' (default) fans out across all configured "
                    "engines (including pistachio_ringbreaker) and returns "
                    "deduplicated, bucketed top-K. Pass an explicit engine "
                    "name to restrict to one engine — useful when you want a "
                    "cheap ringbreaker-only call (saves the cost of running "
                    "graph_retrosyn + rxngraphormer) or to debug why a "
                    "particular engine missed a candidate."
                ),
            },
            "per_engine_topk": {
                "type": "integer",
                "default": 50,
                "description": (
                    "Ensemble only: candidates each engine emits before "
                    "dedup + bucketing. Default 50 matches ChemBART's "
                    "recommended sample size for typical targets."
                ),
            },
            "return_topk": {
                "type": "integer",
                "default": 12,
                "description": (
                    "Ensemble only: total diverse candidates returned to "
                    "the LLM after bucketing."
                ),
            },
            "topk": {
                "type": "integer",
                "default": 10,
                "description": (
                    "Single-engine only: top-K candidates from the chosen engine."
                ),
            },
            "detail_level": {
                "type": "string",
                "enum": ["brief", "full"],
                "default": "brief",
                "description": "'brief': reactants + score. 'full' adds template SMARTS where supported.",
            },
        },
        "required": ["target_smiles"],
    }

    def __init__(
        self,
        template_relevance_url: Optional[str] = None,
        chembart_url: Optional[str] = None,
        pistachio_ringbreaker_url: Optional[str] = None,
        topk: int = 10,
        per_engine_topk: int = 50,
        per_engine_caps: Optional[Dict[str, int]] = None,
        return_topk: int = 15,
        n_buckets: int = 10,
        cluster_distance: float = 0.65,
        k_per_cluster: int = 2,
        engines: Optional[List[str]] = None,
        llm_client: Any = None,
        sanity_enabled: bool = True,
        sanity_batch_size: int = 20,
        sanity_max_dropped_logged: int = 10,
        sanity_max_calls_per_run: int = 100,
        # NEW: shared retroplanner HTTP endpoint + per-engine routing.
        retroplanner_http_url: Optional[str | Sequence[str]] = None,
        retroplanner_http_weights: Optional[Sequence[int]] = None,
        retroplanner_http_timeout_s: float = 120.0,
        retroplanner_http_retries: int = 3,
        retroplanner_chembart_max_attempts: int = 2,
        retroplanner_chembart_deadline_s: float = 130.0,
        retroplanner_chembart_retry_min_budget_s: float = 75.0,
        retroplanner_retry_jitter_s: float = 0.25,
        engine_modes: Optional[Dict[str, str]] = None,
    ):
        # Precursor sanity filter wiring (see tools/_precursor_sanity.py).
        self._llm_client = _ChatStringAdapter(llm_client) if llm_client is not None else None
        self._sanity_enabled = bool(sanity_enabled)
        self._sanity_batch_size = int(sanity_batch_size)
        self._sanity_max_logged = int(sanity_max_dropped_logged)
        # Hard cap on cumulative sanity-filter LLM calls per process. The
        # cache is the first line of defence (skips re-judging the same
        # candidate batch), but a runaway agent that explores many
        # distinct precursor sets could still rack up hundreds of judge
        # calls. Past this we fail-open: the filter returns every
        # candidate untouched, with a warning. Reset only by restarting
        # the process; this is intentionally a safety net, not a routine
        # gate.
        self._sanity_max_calls = int(sanity_max_calls_per_run)
        self._sanity_call_count = 0
        self._sanity_cache: dict = {}
        self._topk = topk
        self._per_engine_topk = per_engine_topk
        # Per-engine override of `per_engine_topk` for ensemble fan-out.
        # Motivating evidence (probe 2026-05-15 against the deployed
        # retroplanner HTTP service):
        #   * chembart returns HTTP 500 for any topk>50 — server-side hard
        #     limit, do not raise above 50.
        #   * rxngraphormer has an internal threshold: 2 candidates at
        #     topk=50 vs 105 at topk=200; we need ≥200 to unlock the model's
        #     real diversity.
        #   * graph_retrosyn partially honours topk: 38 at 50 vs 85 at 200.
        # Engines absent from this dict fall back to `per_engine_topk`.
        self._per_engine_caps: Dict[str, int] = {
            k: int(v) for k, v in (per_engine_caps or {}).items()
        }
        self._return_topk = return_topk
        self._n_buckets = n_buckets
        # Morgan/Butina knobs (May 2026 tuning — see tools/_ensemble_utils.py
        # cluster_candidates() docstring for the rationale on the new
        # 0.65 / k_per_cluster=2 defaults).
        self._cluster_distance = float(cluster_distance)
        self._k_per_cluster = int(k_per_cluster)
        self._engines: Dict[str, Any] = {}
        # `is None` (not falsy) so an explicit `engines=[]` registers no
        # engines instead of silently falling through to the default set
        # — callers that inject a stub engine afterwards rely on this, and
        # the default set now requires a retroplanner HTTP URL.
        if engines is None:
            engines = [
                "graph_retrosyn", "rxngraphormer", "template_relevance", "chembart",
            ]

        # Validate engine_modes early so mistakes surface at config time,
        # not when the LLM happens to call the engine.
        _VALID_MODES = set(_ENGINE_MODES)
        engine_modes = dict(engine_modes or {})
        for k, v in engine_modes.items():
            if v not in _VALID_MODES:
                raise ValueError(
                    f"engine_mode for {k!r} must be one of {sorted(_VALID_MODES)}, got {v!r}"
                )
        # Every neural engine now goes through the shared retroplanner
        # `/single_step` HTTP service; the in-process DMPNNFP / RXNGraphormer
        # engines were removed 2026-06-08. `http_only` is the only mode.
        engine_modes.setdefault("rxngraphormer", "http_only")
        engine_modes.setdefault("graph_retrosyn", "http_only")
        engine_modes.setdefault("pistachio_ringbreaker", "http_only")

        def client(url, weights=None):
            return RetroplannerHTTPClient(
                url=url,
                timeout_s=retroplanner_http_timeout_s,
                retries=retroplanner_http_retries,
                weights=weights,
                chembart_max_attempts=retroplanner_chembart_max_attempts,
                chembart_deadline_s=retroplanner_chembart_deadline_s,
                chembart_retry_min_budget_s=retroplanner_chembart_retry_min_budget_s,
                retry_jitter_s=retroplanner_retry_jitter_s,
            )

        http_client = client(retroplanner_http_url, retroplanner_http_weights) if retroplanner_http_url else None

        # graph_retrosyn (HTTP-only)
        if "graph_retrosyn" in engines:
            mode = engine_modes["graph_retrosyn"]
            if http_client is None:
                raise ValueError(
                    "graph_retrosyn engine_mode requires HTTP but "
                    "retroplanner_http_url is not set"
                )
            self._engines["graph_retrosyn"] = _GraphRetrosynHTTPEngine(
                http_client=http_client,
                engine_mode=mode,
            )

        # rxngraphormer (HTTP-only)
        if "rxngraphormer" in engines:
            mode = engine_modes["rxngraphormer"]
            if http_client is None:
                raise ValueError(
                    "rxngraphormer engine_mode requires HTTP but "
                    "retroplanner_http_url is not set"
                )
            self._engines["rxngraphormer"] = _RXNGraphormerHTTPEngine(
                http_client=http_client,
                engine_mode=mode,
            )

        if "template_relevance" in engines:
            self._engines["template_relevance"] = _TemplateRelevanceEngine(
                api_url=template_relevance_url,
                topk=topk,
            )
        if "chembart" in engines and (
            http_client is not None or chembart_url
        ):
            self._engines["chembart"] = _ChemBartEngine(
                api_url=chembart_url,
                topk=topk,
                http_client=client(chembart_url) if chembart_url else http_client,
            )
        if "pistachio_ringbreaker" in engines and (
            http_client is not None or pistachio_ringbreaker_url
        ):
            # Pistachio is HTTP-only (no local equivalent). The class
            # doesn't take engine_mode, so silently accepting other modes
            # would mask a misconfiguration. Reject anything else early.
            mode = engine_modes["pistachio_ringbreaker"]
            if mode != "http_only":
                raise ValueError(
                    f"engine_mode for 'pistachio_ringbreaker' must be 'http_only' "
                    f"(no local engine exists), got {mode!r}"
                )
            self._engines["pistachio_ringbreaker"] = _PistachioRingbreakerEngine(
                api_url=pistachio_ringbreaker_url,
                topk=topk,
                http_client=client(pistachio_ringbreaker_url) if pistachio_ringbreaker_url else http_client,
            )

    @property
    def available_engines(self) -> List[str]:
        return list(self._engines.keys())

    # ------------------------------------------------------------------
    # Single-engine path (back-compat) and ensemble path
    # ------------------------------------------------------------------

    def _apply_sanity_filter(
        self, target_smiles: str, candidates: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run the LLM precursor sanity judge if enabled. Fail-open."""
        if (
            not self._sanity_enabled
            or self._llm_client is None
            or not candidates
        ):
            return candidates, {"n_dropped": 0, "examples": []}
        if self._sanity_call_count >= self._sanity_max_calls:
            logger.warning(
                "precursor_sanity: reached per-process call cap (%d); "
                "fail-open and keeping all %d candidates for target %s",
                self._sanity_max_calls, len(candidates), target_smiles,
            )
            return candidates, {
                "n_dropped": 0,
                "examples": [],
                "skipped_reason": f"sanity_max_calls_per_run={self._sanity_max_calls} reached",
            }
        self._sanity_call_count += 1
        try:
            kept, dropped, reasons = filter_precursors(
                target_smiles=target_smiles,
                candidates=candidates,
                llm_client=self._llm_client,
                batch_size=self._sanity_batch_size,
                cache=self._sanity_cache,
            )
        except Exception as e:
            logger.warning("precursor_sanity: filter raised %s; keeping all", e)
            return candidates, {"n_dropped": 0, "examples": []}
        if dropped:
            logger.info(
                "precursor_sanity dropped %d/%d candidates for target %s",
                len(dropped), len(dropped) + len(kept), target_smiles,
            )
            report = {
                "n_dropped": len(dropped),
                "examples": [
                    {"reactants": c.get("reactants"), "reason": r}
                    for c, r in list(zip(dropped, reasons))[: self._sanity_max_logged]
                ],
            }
        else:
            report = {"n_dropped": 0, "examples": []}
        return kept, report

    def _run_single(
        self,
        engine: str,
        target_smiles: str,
        topk: int,
        detail_level: str,
        model: Optional[str],
        _apply_sanity: bool = True,
    ) -> Dict[str, Any]:
        eng = self._engines.get(engine)
        if eng is None:
            return {
                "error": f"Engine '{engine}' is not configured.",
                "available": self.available_engines,
                "target": target_smiles,
                "engine": engine,
                "predictions": [],
            }
        if engine == "rxngraphormer":
            out = eng.predict(target_smiles, topk, detail_level, model=model)
        else:
            out = eng.predict(target_smiles, topk, detail_level)
        if _apply_sanity:
            preds = out.get("predictions") or []
            kept, report = self._apply_sanity_filter(target_smiles, preds)
            out["predictions"] = kept
            out["dropped_by_sanity"] = report
        return out

    def _run_ensemble(
        self,
        target_smiles: str,
        per_engine_topk: int,
        return_topk: int,
        detail_level: str,
        model: Optional[str],
    ) -> Dict[str, Any]:
        if not self._engines:
            return {
                "error": "No engines configured.",
                "target": target_smiles,
                "engine": "ensemble",
                "predictions": [],
            }

        # _ENSEMBLE_EXCLUDED is currently empty — every registered engine
        # (including pistachio_ringbreaker) participates in the fan-out.
        # The set is kept as a hook in case a future engine needs to be
        # registered-but-skipped here.
        ensemble_engines = [
            n for n in self._engines if n not in self._ENSEMBLE_EXCLUDED
        ]
        if not ensemble_engines:
            return {
                "error": (
                    "No general-purpose engines configured for ensemble; "
                    "only specialists registered: "
                    f"{sorted(self._engines)}. Call with explicit "
                    "engine='<name>'."
                ),
                "target": target_smiles,
                "engine": "ensemble",
                "predictions": [],
            }

        engine_results: Dict[str, List[Dict[str, Any]]] = {}
        engine_errors: Dict[str, str] = {}
        per_engine_counts: Dict[str, int] = {}
        per_engine_ms: Dict[str, int] = {}

        def _run(name: str) -> tuple[str, Dict[str, Any], int]:
            t0 = time.monotonic()
            # Per-engine cap overrides the ensemble default. ChemBart's HTTP
            # service hard-fails at topk>50; RxNGormer/GraphFP need topk≥200
            # before the model unlocks more than a handful of candidates.
            eng_topk = self._per_engine_caps.get(name, per_engine_topk)
            try:
                out = self._run_single(
                    name, target_smiles, eng_topk, detail_level, model,
                    _apply_sanity=False,
                )
                return name, out, int((time.monotonic() - t0) * 1000)
            except Exception as e:
                logger.warning("ensemble: engine=%s failed: %s", name, e)
                return name, {"error": str(e), "predictions": []}, int((time.monotonic() - t0) * 1000)

        with ThreadPoolExecutor(max_workers=max(1, len(ensemble_engines))) as ex:
            futs = {ex.submit(_run, n): n for n in ensemble_engines}
            for f in as_completed(futs):
                name, out, ms = f.result()
                per_engine_ms[name] = ms
                if "error" in out and not out.get("predictions"):
                    engine_errors[name] = out["error"]
                    engine_results[name] = []
                else:
                    preds = out.get("predictions") or []
                    engine_results[name] = preds
                per_engine_counts[name] = len(engine_results[name])

        merged = dedup_predictions(engine_results)
        n_unique_after_dedup = len(merged)
        merged, sanity_report = self._apply_sanity_filter(target_smiles, merged)
        if not merged:
            # Distinguish "every engine ERRORED" (fleet/HTTP outage) from
            # "engines ran but produced 0 candidates" (a genuine dead end /
            # likely-leaf). Both yield predictions:[], but only the latter is a
            # real result. If we returned a bare predictions:[] for a total
            # engine outage, two things break:
            #   • the agent's _is_error_payload() (which keys on a TOP-LEVEL
            #     `error`) returns False, so the empty result gets MEMOIZED into
            #     the short-circuit cache and served for the rest of the run —
            #     a transient outage permanently poisons that disconnection.
            #   • the LLM reads predictions:[] as "this node is a leaf / dead
            #     end" and may declare a non-leaf molecule a buyable leaf,
            #     truncating a viable route.
            # So when zero engines produced a non-error response, emit a
            # top-level `error` so the result is treated as a failure: not
            # cached, and surfaced to the LLM as an engine outage rather than a
            # confirmed leaf.
            all_engines_errored = bool(ensemble_engines) and len(engine_errors) == len(
                ensemble_engines
            )
            payload = {
                "target": target_smiles,
                "engine": "ensemble",
                "per_engine_counts": per_engine_counts,
                "per_engine_ms": per_engine_ms,
                "engine_errors": engine_errors,
                "predictions": [],
                "n_unique_after_dedup": n_unique_after_dedup,
                "n_clusters": 0,
                "dropped_by_sanity": sanity_report,
            }
            if all_engines_errored:
                payload["error"] = (
                    "all ensemble engines failed (no engine produced a usable "
                    "response): " + "; ".join(
                        f"{n}: {e}" for n, e in engine_errors.items()
                    )
                )
                payload["note"] = (
                    "engine outage — NOT a confirmed dead end / leaf; retry later"
                )
            else:
                payload["note"] = "all engines returned 0 valid candidates"
            return payload

        bucketed = bucket_by_classifier(
            merged, target_smiles=target_smiles, n_buckets=self._n_buckets,
        )
        survivors, cluster_meta = pick_per_bucket(
            bucketed,
            k_per_bucket=self._k_per_cluster,
            return_topk=return_topk,
            distance_threshold=self._cluster_distance,
            return_meta=True,
        )

        # Trim per-entry templates list to keep result envelope small.
        # Also strip internal dedup state (`bucket`, `cluster_size`) — these
        # are Morgan2/Butina internals the LLM cannot act on. Rename
        # `cluster_size` to `diversity_group_size` only when detail_level=full.
        for s in survivors:
            tmpls = s.get("templates") or []
            if detail_level != "full":
                s.pop("templates", None)
                s.pop("bucket", None)
                s.pop("cluster_size", None)
            else:
                s["templates"] = tmpls[:3]
                # Rename internal-sounding key on full output too
                if "cluster_size" in s:
                    s["diversity_group_size"] = s.pop("cluster_size")
                s.pop("bucket", None)

        envelope: Dict[str, Any] = {
            "target": target_smiles,
            "engine": "ensemble",
            "per_engine_counts": per_engine_counts,
            "per_engine_ms": per_engine_ms,
            "n_unique_after_dedup": n_unique_after_dedup,
            "n_clusters": cluster_meta.get("n_clusters"),
            "predictions": survivors,
            "dropped_by_sanity": sanity_report,
        }
        if engine_errors:
            envelope["engine_errors"] = engine_errors

        # ── Audit-only per-engine raw predictions (traceability) ─────────────
        # `predictions` above is the post-dedup/-cluster survivor set; each
        # survivor keeps a `support` list naming the engines that voted for it,
        # but the per-engine raw top-K (and each engine's OWN score/template for
        # a given precursor set, before max-collapse during dedup) is otherwise
        # lost. We attach it under the reserved `_detail` key so
        # ResultStore.persist writes it to disk — recoverable post-run via
        # load_result / offline audit ("exactly what did engine X produce, with
        # what score, before the ensemble merged it?") — WITHOUT bloating
        # context: persist strips `_detail` from the returned string and the
        # projection stub omits it.
        per_engine_predictions: Dict[str, List[Dict[str, Any]]] = {}
        for name, preds in engine_results.items():
            rows: List[Dict[str, Any]] = []
            for p in preds or []:
                row = {
                    "reactants": p.get("reactants"),
                    "score": p.get("score"),
                }
                tmpl = p.get("template")
                if tmpl:
                    row["template"] = tmpl
                rows.append(row)
            per_engine_predictions[name] = rows
        envelope["_detail"] = {
            "per_engine_predictions": per_engine_predictions,
            "per_engine_counts": dict(per_engine_counts),
            "per_engine_ms": dict(per_engine_ms),
            "engine_errors": dict(engine_errors),
        }
        return envelope

    def execute(
        self,
        target_smiles: str,
        engine: str = "ensemble",
        topk: Optional[int] = None,
        per_engine_topk: Optional[int] = None,
        return_topk: Optional[int] = None,
        model: Optional[str] = None,
        detail_level: str = "brief",
        **kwargs,
    ) -> str:
        try:
            if engine == "ensemble":
                out = self._run_ensemble(
                    target_smiles=target_smiles,
                    per_engine_topk=per_engine_topk or self._per_engine_topk,
                    return_topk=return_topk or self._return_topk,
                    detail_level=detail_level,
                    model=model,
                )
            else:
                out = self._run_single(
                    engine,
                    target_smiles,
                    topk or self._topk,
                    detail_level,
                    model,
                )
            return json.dumps(out)
        except Exception as e:
            logger.exception("neural_retro engine=%s failed", engine)
            return json.dumps({
                "error": str(e),
                "engine": engine,
                "target_smiles": target_smiles,
            })
