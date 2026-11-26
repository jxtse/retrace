"""Built-in configuration presets for the retrosynthesis agent."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml

from core.deployment import validate_chemprice
from services.config_schema import validate_config

REPO_ROOT = Path(__file__).resolve().parents[1]
ThinkingLevel = Literal["fast", "standard", "thorough"]


BASE_CONFIG: dict[str, Any] = {'llm': {'base_url': None,
         'model': None,
         'temperature': 0.2,
         'max_tokens': 300000,
         'reasoning_effort': 'medium',
         'responses_state': 'manual',
         'fallback': {'enabled': False,
                      'base_url': 'https://openrouter.ai/api/v1',
                      'model': 'anthropic/claude-opus-4.7'}},
 'agent': {'max_depth': 12,
           'max_iterations': 50,
           'stop_after_distinct_routes': 10,
           'auto_finish_after_saved': 20,
           'continue_extra_rounds': 15,
           'iteration_billable_tools': ['suggest_templates', 'neural_retro'],
           'tool_timeout_s': 300,
           'wallclock_s': 14400},
 'scratchpad': {'max_routes': 80},
 'prompts': {'core': 'prompts/core.md',
             'sections_dir': 'prompts/sections',
             'messages_dir': 'prompts/messages'},
 'postprocess': {'enabled': True,
                 'base_url': None,
                 'model': 'claude-opus-4-8',
                 'temperature': 0.0,
                 'max_tokens': 32000,
                 'reasoning_effort': 'xhigh',
                 'responses_state': 'manual',
                 'max_review_rounds': 1,
                 'timeout_s': 300.0,
                 'max_retries': 2,
                 'fallback': {'enabled': False,
                              'base_url': 'https://openrouter.ai/api/v1',
                              'model': 'anthropic/claude-opus-4.7'}},
 'result_store': {'dir': 'runs/_session_{session}',
                  'summary_max_chars': 2000,
                  'keep_recent_full_per_tool': 4,
                  'retention_days': 14},
 'templates': {'dataset_dir': None},
 'suggest_templates': {'enzymatic_max_in_any': 3},
 'enzymatic_retro': {'enabled': True},
 'stocks': {'BioNav-stock': 'building_block_dataset/bionav_all_building_block.csv',
            'Zinc_Fix-stock': 'building_block_dataset/zinc_stock_2021_10_3_canonical_smiles_total_10312151_add_8546.csv'},
 'neural_retro': {'enabled': True,
                  'topk': 10,
                  'per_engine_topk': 50,
                  'per_engine_caps': {'chembart': 50,
                                      'rxngraphormer': 200,
                                      'graph_retrosyn': 200,
                                      'pistachio_ringbreaker': 20},
                  'return_topk': 15,
                  'n_buckets': 10,
                  'cluster_distance': 0.4,
                  'k_per_cluster': 2,
                  'engines': ['graph_retrosyn',
                              'rxngraphormer',
                              'chembart',
                              'pistachio_ringbreaker'],
                  'chembart_url': None,
                  'pistachio_ringbreaker_url': None,
                  'retroplanner_http_url': None,
                  'retroplanner_http_urls': [],
                  'retroplanner_http_weights': [],
                  'retroplanner_http_timeout_s': 120,
                  'retroplanner_http_retries': 3,
                  # ChemBART-specific retry budget. host-d already gates actual
                  # cold inference at two replicas after its cache lookup, so
                  # the Agent must not add a local semaphore (it would block
                  # warm hits and would not span worker processes). Instead,
                  # cap one logical call at two attempts / 130 s and honour
                  # the service's Retry-After response without a retry herd.
                  'retroplanner_chembart_max_attempts': 2,
                  'retroplanner_chembart_deadline_s': 130,
                  'retroplanner_chembart_retry_min_budget_s': 75,
                  'retroplanner_retry_jitter_s': 0.25,
                  'engine_modes': {'rxngraphormer': 'http_only',
                                   'graph_retrosyn': 'http_only',
                                   'pistachio_ringbreaker': 'http_only'},
                  # Historical evaluations ran with precursor filtering disabled.
                  'precursor_sanity_filter': {
                                              'enabled': False,
                                              'batch_size': 20,
                                              'max_dropped_logged': 10,
                                              'max_calls_per_run': 100,
                                              'judge_llm': {'model': 'gpt-4.1',
                                                            'reasoning_effort': 'low',
                                                            'max_retries': 2,
                                                            'retry_delay_s': 1.0,
                                                            'timeout_s': 6.0}}},
 'precedent_recall': {'enabled': False,
                      'db_root': 'precedent-db',
                      'routes_file': 'routes.jsonl',
                      'fps_file': 'target_fps.npy',
                      'source_masks_file': 'source_masks.npz',
                      'defaults': {'k': 3,
                                   'min_similarity': 0.2,
                                   'mmr_lambda': 0.5,
                                   'top_n_for_mmr': 20},
                      'thresholds': {'high': 0.7, 'medium': 0.4, 'low': 0.2},
                      'prefetch': {'enabled': True, 'source': 'mixed', 'k': 3},
                      'multi_source': {'enzyme_friendly_force_one': True}},
 'evaluation': {'dataset': 'dataset/retro-190_dataset/routes_possible_test_hard.pkl',
                'output': 'eval_results.json'},
 'verifier': {'enabled': True,
              'weights': {'feasibility': 0.7, 'cost': 0.15, 'greenness': 0.15},
              # The 3 pillars (feasibility/cost/greenness) are disjoint +
              # thread-safe, so score them concurrently; within feasibility,
              # fan per-step rater calls out across this many workers
              # (preserving index order). Env overrides:
              # SCORE_ROUTE_PILLAR_PARALLEL, SCORE_ROUTE_FEAS_WORKERS.
              'pillar_parallel': True,
              'feasibility_step_workers': 4,
              # Home for the cost pillar's price_cache.db; {session} is
              # resolved per-worker so each sqlite file stays single-writer
              # (NFS does not tolerate concurrent sqlite writers).
              'cache_dir': 'runs/_session_{session}',
              # Carbon-skeleton fabrication penalty (NOT mass balance). A
              # positive carbon_delta (product has more skeletal carbons than
              # every precursor combined = carbon fabricated) docks feasibility
              # by this much per fabricated carbon; the LLM judge can still lift
              # a docked step. Set 0.0 to disable.
              'carbon_fabrication_penalty': 0.2,
              # Feasibility dispersion penalty. The pillar stays mean-based but
              # is docked toward its worst surviving step:
              #   feas = mean - lambda * (mean - min).
              # lambda=0 = pure mean (old behaviour); lambda=1 = worst-step.
              # Calibrated to 0.3 on the 3304-route RETRACE chemist audit
              # (within-target rank-1 optimum). Applied AFTER per-step judge
              # lift so only judge-surviving low steps are penalised.
              'feasibility_dispersion_lambda': 0.3,
              'feasibility': {'judge': {'enabled': True,
                                        'base_url': None,
                                        'model': 'claude-opus-4-8',
                                        'temperature': 0.0,
                                        # opus-4.8 medium reasoning spends the
                                        # output budget on thinking before it
                                        # writes the one-line JSON verdict; at
                                        # 2000 it hits the ceiling mid-thought
                                        # and returns content=[] (the guard
                                        # turns that ghost into a 502). The
                                        # ceiling is not a per-call cost floor —
                                        # a real verdict still spends only
                                        # thinking + ~50 JSON tokens — so the
                                        # headroom buys a finished reply, not a
                                        # bigger bill.
                                        'max_tokens': 32000,
                                        'trigger_feasibility': 0.4,
                                        'judge_cap': 0.85,
                                        'step_judge_cap': 0.85,
                                        'timeout_s': 120.0},
                              'rater': {'http_url': None,
                                        'timeout_s': 30.0,
                                        'retries': 3}},
              'cost': {'enamine_index': 'dataset/enamine_index.pkl',
                       'buyability_db': 'dataset/buyability/source_mols.db',
                       'chemprice': {'enabled': True,
                                     'env_file': None,
                                     'verifier_repo': None,
                                     'timeout_s': 18,
                                     'cache_ttl_days': 30}},
              'greenness': {'condition_model_dir': 'tools/condition_predictor'}}}


THINKING_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "llm": {"reasoning_effort": "medium"},
        "agent": {
            "max_iterations": 50,
            "stop_after_distinct_routes": 10,
            "auto_finish_after_saved": 20,
            "wallclock_s": 14400,
        },
        "scratchpad": {"max_routes": 80},
    },
    "standard": {
        "llm": {"reasoning_effort": "high"},
        "agent": {
            "max_iterations": 100,
            # Relaxed 15 -> 10: a capable model self-declares COMPLETE around
            # distinct=6, so a threshold of 15 only bought a long tail.
            "stop_after_distinct_routes": 10,
            "auto_finish_after_saved": 30,
            "wallclock_s": 21600,
        },
        "scratchpad": {"max_routes": 120},
    },
    "thorough": {
        "llm": {"reasoning_effort": "xhigh"},
        "agent": {
            "max_iterations": 200,
            "stop_after_distinct_routes": 20,
            "auto_finish_after_saved": 40,
            "wallclock_s": 36000,
        },
        "scratchpad": {"max_routes": 200},
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(base.get(key), dict)
        ):
            _deep_merge(base[key], value)
        elif value is None and isinstance(base.get(key), dict):
            # Empty YAML block (e.g. `agent:` with no children parses to None).
            # Treat as "no override" rather than erasing the inherited preset dict.
            continue
        else:
            base[key] = copy.deepcopy(value)
    return base


def _read_override(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    try:
        with open(cfg_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError:
        raise ValueError("invalid YAML syntax in config file") from None
    if not isinstance(raw, dict):
        raise ValueError(f"config override must be a mapping: {cfg_path}")
    return raw


def _project_for_runtime(config: dict[str, Any]) -> dict[str, Any]:
    """Return nested config plus the flat keys consumed by older call sites."""
    out = copy.deepcopy(config)
    llm = out.get("llm", {}) or {}
    fallback = llm.get("fallback", {}) or {}
    agent = out.get("agent", {}) or {}

    out["llm_base_url"] = llm.get("base_url")
    out["llm_model"] = llm.get("model")
    out["temperature"] = llm.get("temperature", 0.2)
    out["max_tokens"] = llm.get("max_tokens", 300000)
    out["llm_fallback_enabled"] = bool(fallback.get("enabled", False))
    out["llm_fallback_base_url"] = fallback.get("base_url", "https://openrouter.ai/api/v1")
    out["llm_fallback_model"] = fallback.get("model", "anthropic/claude-opus-4.7")
    out["llm_reasoning_effort"] = llm.get("reasoning_effort", "medium")
    out["llm_responses_state"] = llm.get("responses_state", "manual")
    out["llm_context_window"] = llm.get("context_window")
    out["postprocess"] = out.get("postprocess", {}) or {}
    out["max_depth"] = agent.get("max_depth", 12)
    out["max_iterations"] = agent.get("max_iterations", 50)
    out["base_dir"] = str(REPO_ROOT)
    return out


def load_config(
    config_path: str | os.PathLike[str] | None = None,
    *,
    thinking: str = "fast",
    cli_overrides: dict[str, Any] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Load the built-in config, apply a thinking preset, then optional overrides."""
    if thinking not in THINKING_PRESETS:
        choices = ", ".join(sorted(THINKING_PRESETS))
        raise ValueError(f"unknown thinking preset {thinking!r}; expected one of: {choices}")

    config = copy.deepcopy(BASE_CONFIG)
    _deep_merge(config, THINKING_PRESETS[thinking])
    _deep_merge(config, _read_override(config_path or os.getenv("AGENT_CONFIG")))
    if cli_overrides:
        _deep_merge(config, cli_overrides)

    validate_config(config, strict=strict)
    if isinstance(config.get("verifier"), dict):
        validate_chemprice(config["verifier"])
    projected = _project_for_runtime(config)
    projected["thinking"] = thinking
    return projected


def available_thinking_levels() -> tuple[str, ...]:
    return tuple(THINKING_PRESETS.keys())


def load_benchmark_config() -> dict[str, Any]:
    """Use the worker environment for both driver preflight and per-case runs."""
    from core.deployment import validate_deployment
    from core.llm_client import _load_dotenv_once

    _load_dotenv_once()
    model = os.getenv("AGENT_MODEL")
    config = load_config(
        thinking=os.getenv("AGENT_THINKING", "fast"),
        cli_overrides={"llm": {"model": model}} if model else None,
    )
    validate_deployment(config)
    return config
