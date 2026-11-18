"""Factory for assembling the retrosynthesis agent and tool registry."""

from __future__ import annotations

import logging
import os
import time
import hashlib
from typing import Optional

from core.agent import RetrosynthesisAgent
from core.llm_client import LLMClient
from core.substrate_client import make_llm_client
from core.prompt_loader import PromptLoader
from services.autocompact import CompactionConfig
from services.result_store import ResultStore
from tools.base import ToolRegistry

logger = logging.getLogger(__name__)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _prompt_artifact(path: str) -> dict:
    with open(path, "rb") as fh:
        data = fh.read()
    return {
        "path": path,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _split_url_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.replace(",", " ").split() if part.strip()]


def _retroplanner_urls(neural_cfg: dict) -> str | list[str] | None:
    env_urls = _split_url_list(os.environ.get("RETROPLANNER_HTTP_URLS"))
    if env_urls:
        return env_urls
    env_url = os.environ.get("RETROPLANNER_HTTP_URL")
    if env_url:
        env_url_parts = _split_url_list(env_url)
        return env_url_parts if len(env_url_parts) > 1 else env_url
    cfg_urls = neural_cfg.get("retroplanner_http_urls") or []
    if cfg_urls:
        return cfg_urls
    return neural_cfg.get("retroplanner_http_url")


def _retroplanner_weights(neural_cfg: dict) -> list[int] | None:
    """Per-URL first-hop weights — only when URLs come from config.

    If either env override (RETROPLANNER_HTTP_URLS / RETROPLANNER_HTTP_URL)
    is set, the configured URL list is bypassed, so the configured weights
    would be misaligned with the actual endpoints; fall back to equal
    weights (None) in that case.
    """
    if os.environ.get("RETROPLANNER_HTTP_URLS") or os.environ.get(
        "RETROPLANNER_HTTP_URL"
    ):
        return None
    if not (neural_cfg.get("retroplanner_http_urls") or []):
        return None
    weights = neural_cfg.get("retroplanner_http_weights")
    if not weights:
        return None
    try:
        return [int(w) for w in weights]
    except (TypeError, ValueError):
        return None


def _build_sanity_judge_client(neural_cfg: dict, main_llm: LLMClient) -> LLMClient:
    """Dedicated LLM client for the precursor sanity judge (tools/neural_retro.py).

    The judge is a trivial fail-open yes/no on whether a candidate precursor
    set is obviously broken. It fans out ~6 concurrent threads per neural_retro
    turn, and under a large multi-worker run every worker reaches the model
    through the SAME shared SSH tunnel (localhost:4141). If the judge inherits
    the main client's retry budget (max_retries=10, 120s cap) a single
    transient tunnel reset becomes a self-sustaining thundering herd:
    ~6 threads x 10 retries = ~60 concurrent attempts saturate the tunnel so
    none complete, every episode hits 10/10, and ~600s of backoff per call
    fails open anyway (observed JMC case006: 50/51 episodes exhausted, 0
    verdicts, dominant per-case wall cost — see the sanity-judge-retry-herd
    memory).

    Unlike the earlier conditional version, this ALWAYS returns a dedicated
    client so the hard retry cap + cheap reasoning tier apply unconditionally —
    crucially, even when the main agent model is already gpt-5.5 on the shared
    tunnel (the large-scale setup), where reusing main_llm would reintroduce
    the herd. The endpoint defaults to the main client's base_url/model so no
    extra config is needed; a ``precursor_sanity_filter.judge_llm`` block (or
    SANITY_JUDGE_* env) can point it elsewhere / pin a faster model.

        neural_retro:
          precursor_sanity_filter:
            judge_llm:
              base_url: http://localhost:4141/v1   # default: main client's
              model: gpt-5.5                        # default: main client's
              reasoning_effort: low                 # default: low (fast tier)
              max_retries: 2
    """
    sanity_cfg = neural_cfg.get("precursor_sanity_filter", {}) or {}
    judge_cfg = sanity_cfg.get("judge_llm", {}) or {}
    base_url = os.environ.get("SANITY_JUDGE_BASE_URL") or judge_cfg.get("base_url")
    model = os.environ.get("SANITY_JUDGE_MODEL") or judge_cfg.get("model")
    effort = os.environ.get("SANITY_JUDGE_EFFORT") or judge_cfg.get(
        "reasoning_effort", "low"
    )
    client = make_llm_client(
        base_url=base_url or main_llm.base_url,
        model=model or main_llm.model,
        temperature=float(judge_cfg.get("temperature", 0.0)),
        max_tokens=int(judge_cfg.get("max_tokens", 8000)),
        # Pinned tier: trivial classification, so default to LOW reasoning
        # effort for speed regardless of the agent's (possibly xhigh) setting.
        reasoning_effort=str(effort),
        # Stateless side-call; never inherit the main GPT-5.5 Responses
        # conversation (it also fans out concurrently and would race on it).
        responses_state="manual",
        read_timeout=float(judge_cfg.get("timeout_s", 30.0)),
        # The judge FAILS OPEN, so it must never burn the main client's 10x /
        # 120s-backoff budget. Cap hard so a call either succeeds in ~7s or
        # fails open in a few. Tunable via judge_llm.{max_retries,retry_delay_s}.
        max_retries=int(judge_cfg.get("max_retries", 2)),
        retry_delay=float(judge_cfg.get("retry_delay_s", 1.0)),
        # No OpenRouter fallback for an advisory side-call — a fail-open judge
        # has no business spending the cross-cloud fallback budget.
        fallback_enabled=False,
    )
    logger.info(
        "precursor_sanity judge -> dedicated capped client: model=%s base_url=%s "
        "effort=%s max_retries=%d",
        client.model, client.base_url, client.reasoning_effort, client.max_retries,
    )
    return client


def build_agent(config: dict | None = None) -> RetrosynthesisAgent:
    """Build a RetrosynthesisAgent from config."""
    if config is None:
        from core.config import load_config
        config = load_config()

    # LLM
    llm_cfg = config.get("llm", {}) if isinstance(config.get("llm"), dict) else {}
    fallback_cfg = (
        llm_cfg.get("fallback", {}) if isinstance(llm_cfg.get("fallback"), dict) else {}
    )
    llm = make_llm_client(
        base_url=config.get("llm_base_url", llm_cfg.get("base_url", "http://localhost:4141/v1")),
        model=config.get("llm_model", llm_cfg.get("model", "gpt-5.5")),
        temperature=config.get("temperature", llm_cfg.get("temperature", 0.2)),
        max_tokens=config.get("max_tokens", llm_cfg.get("max_tokens", 300000)),
        max_retries=int(config.get("llm_max_retries", llm_cfg.get("max_retries", 6))),
        retry_delay=float(config.get("llm_retry_delay_s", llm_cfg.get("retry_delay_s", 2.0))),
        connect_timeout=float(config.get("llm_connect_timeout_s", llm_cfg.get("connect_timeout_s", 10.0))),
        read_timeout=float(config.get("llm_read_timeout_s", llm_cfg.get("read_timeout_s", 300.0))),
        fallback_enabled=config.get("llm_fallback_enabled", fallback_cfg.get("enabled", False)),
        fallback_base_url=config.get(
            "llm_fallback_base_url",
            fallback_cfg.get("base_url", "https://openrouter.ai/api/v1"),
        ),
        fallback_model=config.get(
            "llm_fallback_model",
            fallback_cfg.get("model", "anthropic/claude-opus-4.7"),
        ),
        reasoning_effort=config.get("llm_reasoning_effort", llm_cfg.get("reasoning_effort", "medium")),
        responses_state=config.get("llm_responses_state", llm_cfg.get("responses_state", "manual")),
        context_window=config.get("llm_context_window", llm_cfg.get("context_window")),
    )

    # Tool registry
    registry = ToolRegistry()

    # ── Plan-state tracking (TodoWrite / TodoRead) ──
    from tools.todo import register_todo_tools
    todo_store = register_todo_tools(registry)

    # ── Result store: filesystem-as-memory for heavy tool results ──
    rs_cfg = config.get("result_store", {}) or {}
    base_dir = config.get("base_dir", REPO_ROOT)
    rs_dir_tmpl = rs_cfg.get("dir", "runs/_session_{session}/results_root")
    # Session id: prefer AGENT_SESSION_ID (set by batch runners with
    # `<runid>_case<idx>_<ts>`), else fall back to a fresh timestamp.
    # Avoid the literal "_session_default" name — it caused 4 of the 11
    # hard_mol cases to overwrite each other's transcripts.
    env_session = os.environ.get("AGENT_SESSION_ID")
    if env_session and env_session not in ("default", "_session_default"):
        session_id = env_session
    else:
        # Use full pid + 6 hex chars of os.urandom so two agents started
        # within the same wall-second on the same box can't land on the
        # same session_id. The previous `os.getpid() % 100000` modulo
        # collided when CI fanned out 200+ parallel agents.
        import secrets
        session_id = "auto-{ts}-{pid}-{tok}".format(
            ts=time.strftime("%Y%m%d-%H%M%S"),
            pid=os.getpid(),
            tok=secrets.token_hex(3),
        )
    rs_dir = rs_dir_tmpl.replace("{session}", session_id)
    if not os.path.isabs(rs_dir):
        rs_dir = os.path.join(base_dir, rs_dir)
    result_store = ResultStore(
        session_dir=rs_dir,
        summary_max_chars=int(rs_cfg.get("summary_max_chars", 600)),
        keep_recent_full_per_tool=int(rs_cfg.get("keep_recent_full_per_tool", 4)),
    )
    # Best-effort TTL cleanup of old session dirs. Disabled when
    # `result_store.retention_days` is unset / 0. Runs once per
    # build_agent() so a long-running batch doesn't repeatedly scan
    # the runs/ tree.
    retention_days = rs_cfg.get("retention_days")
    if retention_days:
        try:
            runs_root = os.path.dirname(rs_dir)
            removed = ResultStore.cleanup_old_sessions(
                runs_dir=runs_root, retention_days=int(retention_days),
            )
            if removed:
                logger.info(
                    "Result store TTL cleanup: removed %d session dirs older than %d days",
                    removed, int(retention_days),
                )
        except Exception:
            logger.exception("Result store TTL cleanup failed (non-fatal)")

    # ── JIT prompt-section loader + result loader ──
    from tools.read_doc import ReadDocTool
    from tools.load_result import LoadResultTool
    prompts_cfg = config.get("prompts", {}) or {}
    sections_dir = prompts_cfg.get("sections_dir", "prompts/sections")
    if not os.path.isabs(sections_dir):
        sections_dir = os.path.join(base_dir, sections_dir)
    registry.register(ReadDocTool(base_dir=sections_dir))
    registry.register(LoadResultTool(result_store))

    # ── Long-horizon memory: scratchpad + backtrack tools ──
    from tools.scratchpad import register_scratchpad_tools
    scratchpad_cfg = config.get("scratchpad", {}) or {}
    scratchpad_store = register_scratchpad_tools(
        registry,
        max_routes=scratchpad_cfg.get("max_routes"),
        # Forwarded so SaveBestRouteTool can validate the LLM-supplied
        # `total_score` against the persisted score_route payload pointed at
        # by `score_route_result_id`. Without this the verification path is
        # dormant and the LLM can claim arbitrary scores.
        result_store=result_store,
    )

    # ── Template library: propagate configured dataset_dir ──
    base_dir = config.get("base_dir", REPO_ROOT)
    templates_cfg = config.get("templates", {}) or {}
    dataset_dir = templates_cfg.get("dataset_dir")
    if dataset_dir:
        abs_dir = dataset_dir if os.path.isabs(dataset_dir) else os.path.join(base_dir, dataset_dir)
        try:
            from tools.builtins.template_library import set_dataset_dir
            set_dataset_dir(abs_dir)
            logger.info("Template dataset_dir=%s", abs_dir)
        except Exception as e:
            logger.warning("Could not set template dataset_dir=%s: %s", abs_dir, e)

    # ── Precedent recall (Adaptive RAG): build PrecedentDB if configured ──
    from services.precedent_db import maybe_load as _maybe_load_precedent_db
    precedent_db = _maybe_load_precedent_db(
        config.get("precedent_recall"), base_dir
    )

    # ── Tier 1+2: Built-in tools (pure RDKit) ──
    from tools.builtins import register_all_builtins
    register_all_builtins(registry, config=config, precedent_db=precedent_db)

    # ── Tier 3: External tools (buyability — still direct tools) ──
    # Vendored building-block CSVs live under vendor/legacy_retro_planner/
    # (used to live in ../retro_planner/, now moved out to ../baselines/).
    vendor_legacy_dir = os.path.join(base_dir, "vendor", "legacy_retro_planner")

    # Check buyability — stays as a direct tool (frequent atomic calls)
    stock_files = {}
    stocks_config = config.get("stocks", {
        "BioNav-stock": "building_block_dataset/bionav_all_building_block.csv",
    })
    for name, rel_path in stocks_config.items():
        full_path = os.path.join(vendor_legacy_dir, rel_path)
        if os.path.exists(full_path):
            stock_files[name] = full_path
        else:
            logger.warning("Stock file not found: %s", full_path)

    if stock_files:
        from tools.check_buyability import CheckBuyabilityTool
        buyability_tool = CheckBuyabilityTool(stock_files)
        registry.register(buyability_tool)
    else:
        logger.warning("No stock files loaded!")
        buyability_tool = None

    # ExploreBranchesTool is registered AFTER RouteVerifierTool below so
    # we can reuse the verifier's BuyabilityChain instance (Enamine
    # pickle is large; loading it twice wastes RAM).

    # ── Multi-dimensional route verifier (default `score_route`) ──
    verifier_cfg = config.get("verifier", {}) or {}
    if verifier_cfg.get("enabled", True):
        from tools.verifier import RouteVerifierTool

        def _abs(p: Optional[str]) -> Optional[str]:
            if not p:
                return None
            return p if os.path.isabs(p) else os.path.join(base_dir, p)

        cost_cfg = verifier_cfg.get("cost", {}) or {}
        green_cfg = verifier_cfg.get("greenness", {}) or {}
        feas_cfg = verifier_cfg.get("feasibility", {}) or {}
        judge_cfg_raw = feas_cfg.get("judge", {}) or {}

        cache_dir_tmpl = verifier_cfg.get("cache_dir", "runs/_session_default")
        # Resolve {session} placeholder lazily (build_agent runs once per process).
        # Prefer the per-worker ResultStore session_dir over AGENT_SESSION_ID:
        # batch drivers (run_hard_mol_batch / FDA_small_mol/run_benchmark) don't
        # set the env var, so all workers would otherwise collapse onto
        # `runs/_session_default/price_cache.db`. sqlite + NFS does not tolerate
        # concurrent writers — routing the cache into the per-worker session dir
        # keeps each price_cache.db single-writer.
        env_session = os.environ.get("AGENT_SESSION_ID")
        if env_session and env_session not in ("default", "_session_default"):
            cache_dir = cache_dir_tmpl.replace("{session}", env_session)
            cache_dir = _abs(cache_dir)
        else:
            # Reuse the ResultStore's already-unique per-worker dir.
            cache_dir = result_store.session_dir

        # Build a JudgeConfig from YAML; defaults pull from the agent's main
        # LLM block so the judge runs against the same model unless overridden.
        from tools.verifier.llm_judge import JudgeConfig
        judge_config = JudgeConfig(
            enabled=bool(judge_cfg_raw.get("enabled", True)),
            base_url=judge_cfg_raw.get("base_url", config.get("llm_base_url")),
            model=judge_cfg_raw.get("model", config.get("llm_model")),
            temperature=float(judge_cfg_raw.get("temperature", 0.0)),
            max_tokens=int(judge_cfg_raw.get("max_tokens", 16000)),
            trigger_feasibility=float(judge_cfg_raw.get("trigger_feasibility", 0.4)),
            judge_cap=float(judge_cfg_raw.get("judge_cap", 0.7)),
            step_judge_cap=float(judge_cfg_raw.get("step_judge_cap", 0.85)),
            timeout_s=float(judge_cfg_raw.get("timeout_s", 120.0)),
            # Pinned tier — does NOT inherit `llm_reasoning_effort`, so judge
            # cost stays predictable regardless of main-agent tier overrides.
            reasoning_effort=str(judge_cfg_raw.get("reasoning_effort", "medium")),
            uphold_gate=bool(judge_cfg_raw.get("uphold_gate", False)),
        )

        # The feasibility judge is pinned (base_url/model in JudgeConfig) so the
        # SAME judge grades every benchmarked main agent and feasibility scores
        # stay comparable across model conditions (see the
        # judge-pinned-opus48-fairness rationale). That pin only takes effect if
        # the judge runs on its OWN client: WholeRouteJudge._get_llm() honours
        # cfg.base_url/cfg.model only when no client is injected. Injecting the
        # MAIN agent client (the previous behaviour) silently defeated the pin —
        # the judge ran on whatever model drove the agent (gpt-5.5 grading its
        # own routes, Qwen grading Qwen). So: build a dedicated judge client
        # whenever the YAML explicitly pins a judge base_url/model that differs
        # from the main client; otherwise reuse the main client (no extra
        # connection for the common "judge == main model" case). Mirrors
        # `_build_sanity_judge_client`.
        judge_wants_distinct = bool(
            judge_cfg_raw.get("base_url") or judge_cfg_raw.get("model")
        ) and (
            (judge_config.base_url or "") != (llm.base_url or "")
            or (judge_config.model or "") != (llm.model or "")
        )
        if judge_wants_distinct:
            verifier_judge_client = make_llm_client(
                base_url=judge_config.base_url or llm.base_url,
                model=judge_config.model or llm.model,
                temperature=judge_config.temperature,
                max_tokens=judge_config.max_tokens,
                read_timeout=judge_config.timeout_s,
                reasoning_effort=judge_config.reasoning_effort,
                # Stateless advisory side-call; never inherit the main agent's
                # Responses conversation (it runs concurrently under the pillar
                # pool and would race on the shared native-state).
                responses_state="manual",
                # Fail-open advisory judge — don't burn the main client's large
                # retry/fallback budget.
                max_retries=int(judge_cfg_raw.get("max_retries", 2)),
                fallback_enabled=False,
            )
            logger.info(
                "feasibility judge -> dedicated pinned client: model=%s base_url=%s "
                "effort=%s (independent of main agent model=%s)",
                verifier_judge_client.model, verifier_judge_client.base_url,
                verifier_judge_client.reasoning_effort, llm.model,
            )
        else:
            verifier_judge_client = llm

        registry.register(RouteVerifierTool(
            weights=verifier_cfg.get("weights"),
            cache_dir=cache_dir,
            rater_http_url=(feas_cfg.get("rater") or {}).get("http_url"),
            rater_timeout_s=float((feas_cfg.get("rater") or {}).get("timeout_s", 30.0)),
            rater_retries=int((feas_cfg.get("rater") or {}).get("retries", 2)),
            enamine_pkl=_abs(cost_cfg.get("enamine_index")),
            buyability_db=_abs(cost_cfg.get("buyability_db")),
            legacy_csv_dir=_abs(cost_cfg.get("legacy_csvs")),
            condition_model_dir=_abs(green_cfg.get("condition_model_dir")),
            carbon_fabrication_penalty=float(verifier_cfg.get("carbon_fabrication_penalty", 0.2)),
            feasibility_dispersion_lambda=float(verifier_cfg.get("feasibility_dispersion_lambda", 0.3)),
            chemprice_enabled=bool((cost_cfg.get("chemprice") or {}).get("enabled", True)),
            chemprice_env_file=_abs((cost_cfg.get("chemprice") or {}).get("env_file")),
            chemprice_verifier_repo=_abs((cost_cfg.get("chemprice") or {}).get("verifier_repo")),
            chemprice_timeout_s=float((cost_cfg.get("chemprice") or {}).get("timeout_s", 30.0)),
            chemprice_cache_ttl_days=int((cost_cfg.get("chemprice") or {}).get("cache_ttl_days", 30)),
            judge_config=judge_config,
            llm_client=verifier_judge_client,
            result_store=result_store,
            pillar_parallel=bool(verifier_cfg.get("pillar_parallel", True)),
            feasibility_step_workers=int(verifier_cfg.get("feasibility_step_workers", 4)),
            stock_buyability_tool=buyability_tool,
            incomplete_route_score_cap=float(
                verifier_cfg.get("incomplete_route_score_cap", 0.6)
            ),
        ))
        logger.info("Registered RouteVerifierTool (3-pillar; cache_dir=%s)", cache_dir)

    # ── ExploreBranchesTool wiring (share verifier's BuyabilityChain) ──
    try:
        from tools.verifier.buyability import BuyabilityChain
        from tools.builtins.explore_branches import ExploreBranchesTool

        # _abs() is defined inside the verifier block at L1596; re-define
        # here so this block works regardless of whether the verifier
        # branch ran. (Identical body to the inner one.)
        def _abs_eb(p):
            if not p:
                return None
            return p if os.path.isabs(p) else os.path.join(base_dir, p)

        verifier_instance = registry.get("score_route")
        if verifier_instance is not None and hasattr(verifier_instance, "buyability"):
            buyability_chain = verifier_instance.buyability
            chain_source = "verifier"
        else:
            cost_cfg_local = (config.get("verifier", {}) or {}).get("cost", {}) or {}
            buyability_chain = BuyabilityChain(
                enamine_pkl=_abs_eb(cost_cfg_local.get("enamine_index")),
                buyability_db=_abs_eb(cost_cfg_local.get("buyability_db")),
                legacy_csv_dir=_abs_eb(cost_cfg_local.get("legacy_csvs")),
            )
            chain_source = "standalone"
        registry.register(ExploreBranchesTool(
            buyability_chain=buyability_chain,
            result_store=result_store,
        ))
        logger.info(
            "Registered ExploreBranchesTool (verifier-aligned, "
            "chain=%s, result_store=%s)",
            chain_source, "yes" if result_store else "no",
        )
    except Exception:
        logger.exception("Could not init explore_branches")

    # ── Neural single-step retrosynthesis (multi-engine ensemble) ──
    neural_cfg = config.get("neural_retro", {}) or {}
    if neural_cfg.get("enabled", False):
        try:
            from tools.neural_retro import NeuralRetroTool
            registry.register(NeuralRetroTool(
                template_relevance_url=neural_cfg.get("template_relevance_url"),
                chembart_url=os.environ.get("CHEMBART_API_URL") or neural_cfg.get("chembart_url"),  # env > config; no hardcoded fallback (engine self-disables if missing)
                pistachio_ringbreaker_url=(
                    os.environ.get("RINGBREAKER_API_URL")
                    or neural_cfg.get("pistachio_ringbreaker_url")
                ),  # env > config; no hardcoded fallback (engine self-disables if missing)
                topk=neural_cfg.get("topk", 10),
                per_engine_topk=neural_cfg.get("per_engine_topk", 50),
                per_engine_caps=neural_cfg.get("per_engine_caps") or {},
                return_topk=neural_cfg.get("return_topk", 15),
                n_buckets=neural_cfg.get("n_buckets", 10),
                cluster_distance=neural_cfg.get("cluster_distance", 0.65),
                k_per_cluster=neural_cfg.get("k_per_cluster", 2),
                engines=neural_cfg.get(
                    "engines",
                    ["graph_retrosyn", "rxngraphormer", "chembart", "pistachio_ringbreaker"],
                ),
                # Dedicated capped+cheap client — NOT the main `llm`. The
                # judge fails open + fans out concurrently over the shared
                # tunnel; reusing main_llm reintroduces the retry herd. See
                # _build_sanity_judge_client.
                llm_client=_build_sanity_judge_client(neural_cfg, llm),
                sanity_enabled=neural_cfg.get("precursor_sanity_filter", {}).get("enabled", True),
                sanity_batch_size=neural_cfg.get("precursor_sanity_filter", {}).get("batch_size", 20),
                sanity_max_dropped_logged=neural_cfg.get("precursor_sanity_filter", {}).get("max_dropped_logged", 10),
                sanity_max_calls_per_run=neural_cfg.get("precursor_sanity_filter", {}).get("max_calls_per_run", 100),
                # Shared retroplanner HTTP endpoint + per-engine routing.
                retroplanner_http_url=_retroplanner_urls(neural_cfg),
                retroplanner_http_weights=_retroplanner_weights(neural_cfg),
                retroplanner_http_timeout_s=neural_cfg.get(
                    "retroplanner_http_timeout_s", 120,
                ),
                retroplanner_http_retries=neural_cfg.get(
                    "retroplanner_http_retries", 3,
                ),
                retroplanner_chembart_max_attempts=neural_cfg.get(
                    "retroplanner_chembart_max_attempts", 2,
                ),
                retroplanner_chembart_deadline_s=neural_cfg.get(
                    "retroplanner_chembart_deadline_s", 130,
                ),
                retroplanner_chembart_retry_min_budget_s=neural_cfg.get(
                    "retroplanner_chembart_retry_min_budget_s", 75,
                ),
                retroplanner_retry_jitter_s=neural_cfg.get(
                    "retroplanner_retry_jitter_s", 0.25,
                ),
                engine_modes=neural_cfg.get("engine_modes") or {},
            ))
            logger.info("Registered tool: neural_retro")
        except Exception:
            logger.exception("Could not create NeuralRetroTool")

    logger.info("Registered tools: %s", registry.tool_names)

    compaction_cfg = config.get("compaction", {}) or {}
    compaction_config = CompactionConfig(
        max_tokens=compaction_cfg.get("max_tokens", 180_000),
        auto_compact_threshold=compaction_cfg.get("auto_compact_threshold", 0.85),
        micro_compact_threshold=compaction_cfg.get("micro_compact_threshold", 0.60),
        keep_recent_tail=compaction_cfg.get("keep_recent_tail", 28),
        keep_head_messages=compaction_cfg.get("keep_head_messages", 2),
        summary_max_tokens=compaction_cfg.get("summary_max_tokens", 3500),
        allow_heuristic_summary=compaction_cfg.get("allow_heuristic_summary", True),
    )
    # ── Mandatory modular system prompt (prompts/core.md) ──
    core_path = prompts_cfg.get("core")
    if not core_path:
        raise ValueError(
            "config: prompts.core is required (path to system prompt file)."
        )
    if not os.path.isabs(core_path):
        core_path = os.path.join(base_dir, core_path)
    try:
        with open(core_path) as f:
            system_prompt_text = f.read()
        logger.info(
            "Loaded modular system prompt from %s (%d chars)",
            core_path, len(system_prompt_text),
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"prompts.core file not found: {core_path}"
        ) from e

    reaction_ordering_path = os.path.join(sections_dir, "core_reaction_ordering.md")
    try:
        with open(reaction_ordering_path) as f:
            reaction_ordering_block = f.read().rstrip("\n")
        logger.info(
            "Loaded reaction-ordering prompt section from %s (%d chars)",
            reaction_ordering_path, len(reaction_ordering_block),
        )
    except FileNotFoundError:
        logger.warning(
            "Reaction-ordering prompt section missing at %s; prompt placeholder "
            "will render empty.",
            reaction_ordering_path,
        )
        reaction_ordering_block = ""

    prompt_artifacts = {
        "core": _prompt_artifact(core_path),
        "reaction_ordering": (
            _prompt_artifact(reaction_ordering_path)
            if os.path.exists(reaction_ordering_path)
            else {"path": reaction_ordering_path, "missing": True}
        ),
        "sections_dir": sections_dir,
    }

    # ── PromptLoader for messages/* templates (used by Tasks 4-7) ──
    messages_dir = prompts_cfg.get("messages_dir", "prompts/messages")
    if not os.path.isabs(messages_dir):
        messages_dir = os.path.join(base_dir, messages_dir)
    prompts_loader = PromptLoader(base_dir=messages_dir)
    logger.info("Initialised PromptLoader at %s", messages_dir)
    prompt_artifacts["messages_dir"] = messages_dir

    post_cfg = config.get("postprocess", {}) or {}
    reviewer_client = None
    if bool(post_cfg.get("enabled", False)):
        fallback_cfg = (
            post_cfg.get("fallback", {})
            if isinstance(post_cfg.get("fallback"), dict)
            else {}
        )
        reviewer_client = make_llm_client(
            base_url=post_cfg.get("base_url") or llm.base_url,
            model=post_cfg.get("model", "claude-opus-4-8"),
            temperature=float(post_cfg.get("temperature", 0.0)),
            max_tokens=int(post_cfg.get("max_tokens", 32000)),
            reasoning_effort=str(post_cfg.get("reasoning_effort", "xhigh")),
            responses_state=str(post_cfg.get("responses_state", "manual")),
            read_timeout=float(post_cfg.get("timeout_s", 300.0)),
            max_retries=int(post_cfg.get("max_retries", 2)),
            fallback_enabled=bool(fallback_cfg.get("enabled", False)),
            fallback_base_url=fallback_cfg.get(
                "base_url", "https://openrouter.ai/api/v1"
            ),
            fallback_model=fallback_cfg.get(
                "model", "anthropic/claude-opus-4.7"
            ),
        )
        logger.info(
            "reaction-ordering reviewer -> model=%s base_url=%s effort=%s",
            reviewer_client.model,
            reviewer_client.base_url,
            reviewer_client.reasoning_effort,
        )

    ag_cfg = config.get("agent", {}) if isinstance(config.get("agent"), dict) else {}
    stop_after_distinct_routes = int(ag_cfg.get("stop_after_distinct_routes", 10))
    agent = RetrosynthesisAgent(
        llm_client=llm,
        tool_registry=registry,
        max_depth=ag_cfg.get("max_depth", config.get("max_depth", 12)),
        max_iterations=ag_cfg.get("max_iterations", config.get("max_iterations", 50)),
        compaction_config=compaction_config,
        compaction_enabled=compaction_cfg.get("enabled", True),
        todo_store=todo_store,
        scratchpad_store=scratchpad_store,
        stop_after_distinct_routes=stop_after_distinct_routes,
        result_store=result_store,
        system_prompt=system_prompt_text,
        iteration_billable_tools=ag_cfg.get("iteration_billable_tools"),
        prompts=prompts_loader,
        auto_finish_after_saved=ag_cfg.get("auto_finish_after_saved", 20),
        reaction_ordering_block=reaction_ordering_block,
        reviewer_client=reviewer_client,
        postprocess_config=post_cfg,
        effective_config=config,
        prompt_artifacts=prompt_artifacts,
        precedent_prefetch_enabled=(
            precedent_db is not None
            and bool(
                ((config.get("precedent_recall") or {}).get("prefetch") or {})
                .get("enabled", False)
            )
        ),
        precedent_prefetch_k=int(
            ((config.get("precedent_recall") or {}).get("prefetch") or {})
            .get("k", 3)
        ),
        precedent_prefetch_source=str(
            ((config.get("precedent_recall") or {}).get("prefetch") or {})
            .get("source", "mixed")
        ),
        precedent_prefetch_min_similarity=float(
            ((config.get("precedent_recall") or {}).get("thresholds") or {})
            .get("medium", 0.4)
        ),
        tool_timeout_s=ag_cfg.get("tool_timeout_s"),
        wallclock_s=ag_cfg.get("wallclock_s", 5400.0),
    )
    return agent
