"""Pydantic v2 models for built-in agent config and override files.

Replaces the frozenset allow-list at services/config_schema.py with a
proper schema so typos surface as ValidationError at startup (the
original motivation — see the docstring on services/config_schema.py).

Strict-by-default: every block has `extra="forbid"` so a typo in a
well-modelled block surfaces immediately. Blocks consumed as opaque
dicts by other modules (verifier, compaction, stocks, precedent_recall)
are declared as `dict[str, Any]` so the root extra="forbid" check still
catches genuinely-unknown top-level keys without forcing us to mirror
every nested knob here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool


_STRICT = ConfigDict(extra="forbid")


# ─── llm: ────────────────────────────────────────────────────────────


class LLMFallbackConfig(BaseModel):
    model_config = _STRICT
    # Disabled by default to match BASE_CONFIG / _project_for_runtime /
    # agent_factory (see "chore: disable LLM fallback by default"). An
    # override that supplies an `llm:` block but omits `fallback` must not
    # silently re-enable it.
    enabled: StrictBool = False
    base_url: str | None = "https://openrouter.ai/api/v1"
    model: str = "anthropic/claude-opus-4.7"


class LLMConfig(BaseModel):
    model_config = _STRICT
    base_url: str | None = None
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    max_retries: int = 6
    retry_delay_s: float = 2.0
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 300.0
    # Reasoning tier for the main agent.
    # - gpt-5.5*  : sent as reasoning.effort to /v1/responses
    # - claude-*  : ignored (use model-id suffix instead)
    # - other     : ignored
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "medium"
    responses_state: Literal["auto", "manual"] = "auto"
    # Total model context window (input+output share it). Consumed by the
    # chat-completions empty-recovery path to grow the output budget within
    # real headroom when a reasoning model (Qwen3.x via vllm) truncates
    # mid-thought (finish_reason=length). None/omitted disables that path
    # (gpt-5.5 / Claude don't exhibit it and route elsewhere).
    context_window: int | None = None
    fallback: LLMFallbackConfig = Field(default_factory=LLMFallbackConfig)


# ─── agent: ──────────────────────────────────────────────────────────


class AgentBlockConfig(BaseModel):
    model_config = _STRICT
    max_depth: int = 12
    max_iterations: int = 50
    stop_after_distinct_routes: int = 10
    auto_finish_after_saved: int | None = 20
    continue_extra_rounds: int = 15
    iteration_billable_tools: list[str] = Field(
        default_factory=lambda: [
            "suggest_templates",
            "neural_retro",
        ]
    )
    tool_timeout_s: float = 300
    wallclock_s: float | None = 14400


# ─── scratchpad / prompts / result_store / templates ────────────────


class ScratchpadConfig(BaseModel):
    model_config = _STRICT
    max_routes: int = 25


class PromptsConfig(BaseModel):
    model_config = _STRICT
    core: str = "prompts/core.md"
    sections_dir: str = "prompts/sections"
    messages_dir: str = "prompts/messages"


class PostprocessConfig(BaseModel):
    model_config = _STRICT
    enabled: StrictBool = True
    base_url: str | None = None
    model: str = "claude-opus-4-8"
    temperature: float = 0.0
    max_tokens: int = 32000
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "xhigh"
    responses_state: Literal["auto", "manual"] = "manual"
    max_review_rounds: int = 1
    timeout_s: float = 300.0
    max_retries: int = 2
    fallback: LLMFallbackConfig = Field(default_factory=LLMFallbackConfig)


class ResultStoreConfig(BaseModel):
    model_config = _STRICT
    dir: str = "runs/_session_{session}"
    summary_max_chars: int = 2000
    keep_recent_full_per_tool: int = 4
    retention_days: int = 0


class TemplatesConfig(BaseModel):
    model_config = _STRICT
    dataset_dir: str | None = None


class SuggestTemplatesConfig(BaseModel):
    model_config = _STRICT
    enzymatic_max_in_any: int = 3


# ─── neural_retro: ──────────────────────────────────────────────────


class NeuralRetroConfig(BaseModel):
    model_config = _STRICT
    enabled: StrictBool = True
    topk: int = 10
    per_engine_topk: int = 50
    per_engine_caps: dict[str, int] = Field(default_factory=dict)
    return_topk: int = 15
    n_buckets: int = 10
    cluster_distance: float = 0.4
    k_per_cluster: int = 2
    engines: list[str] = Field(default_factory=list)
    chembart_url: str | None = None
    pistachio_ringbreaker_url: str | None = None
    retroplanner_http_url: str | list[str] | None = None
    retroplanner_http_urls: list[str] = Field(default_factory=list)
    retroplanner_http_weights: list[int] = Field(default_factory=list)
    retroplanner_http_timeout_s: float = 120
    retroplanner_http_retries: int = 3
    engine_modes: dict[str, str] = Field(default_factory=dict)
    template_relevance_url: str | None = None
    precursor_sanity_filter: dict[str, Any] = Field(default_factory=dict)


# ─── evaluation ─────────────────────────────────────────────────────


class EvaluationConfig(BaseModel):
    model_config = _STRICT
    dataset: str | None = None
    output: str = "eval_results.json"

# ─── Root ───────────────────────────────────────────────────────────


class AgentConfig(BaseModel):
    """Root model. `extra="forbid"` to surface typo'd top-level keys.

    We deliberately don't model `verifier`, `compaction`, `stocks`, or
    `precedent_recall` fully — those are consumed as opaque dicts by
    downstream modules, and a partial schema here would create drift
    between schema and runtime. They are declared as `dict[str, Any]`
    so the root extra="forbid" check still catches unknown top-level
    keys without forcing a nested rewrite.
    """

    model_config = ConfigDict(extra="forbid")
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentBlockConfig = Field(default_factory=AgentBlockConfig)
    scratchpad: ScratchpadConfig = Field(default_factory=ScratchpadConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    result_store: ResultStoreConfig = Field(default_factory=ResultStoreConfig)
    templates: TemplatesConfig = Field(default_factory=TemplatesConfig)
    suggest_templates: SuggestTemplatesConfig = Field(
        default_factory=SuggestTemplatesConfig
    )
    stocks: dict[str, str] = Field(default_factory=dict)
    neural_retro: NeuralRetroConfig = Field(default_factory=NeuralRetroConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    verifier: dict[str, Any] = Field(default_factory=dict)
    compaction: dict[str, Any] = Field(default_factory=dict)
    precedent_recall: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "LLMFallbackConfig",
    "LLMConfig",
    "AgentBlockConfig",
    "ScratchpadConfig",
    "PromptsConfig",
    "PostprocessConfig",
    "ResultStoreConfig",
    "TemplatesConfig",
    "SuggestTemplatesConfig",
    "NeuralRetroConfig",
    "EvaluationConfig",
    "AgentConfig",
]
