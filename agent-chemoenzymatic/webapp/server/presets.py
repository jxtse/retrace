"""Web UI model-family overlay for built-in reasoning-budget presets.

The frontend picks a model family plus one of the core thinking presets.
Budget values are sourced from ``core.config.THINKING_PRESETS`` so the
webapp and CLI stay in lockstep.

For Claude the reasoning tier is encoded in the model-id suffix
(``-high`` / ``-xhigh``); the 4141 proxy ignores ``reasoning_effort``
on the Anthropic route. For gpt-5.5 the suffix mechanism doesn't exist
and ``reasoning_effort`` is what actually flips the tier on the
Responses route. We pair them in lockstep so swapping family preserves
intent.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from core.config import THINKING_PRESETS

ModelFamily = Literal["claude", "gpt5_5"]
Preset = Literal["fast", "standard", "thorough"]


class _PresetSpec(TypedDict):
    model: str
    reasoning_effort: str
    max_iterations: int
    stop_after_distinct_routes: int
    auto_finish_after_saved: int
    wallclock_s: int
    scratchpad_max_routes: int


def _spec(model: str, preset: Preset) -> _PresetSpec:
    budget = THINKING_PRESETS[preset]
    return {
        "model": model,
        "reasoning_effort": budget["llm"]["reasoning_effort"],
        "max_iterations": budget["agent"]["max_iterations"],
        "stop_after_distinct_routes": budget["agent"]["stop_after_distinct_routes"],
        "auto_finish_after_saved": budget["agent"]["auto_finish_after_saved"],
        "wallclock_s": budget["agent"]["wallclock_s"],
        "scratchpad_max_routes": budget["scratchpad"]["max_routes"],
    }


PRESETS: dict[tuple[str, str], _PresetSpec] = {
    ("claude", "fast"): _spec("claude-opus-4.7", "fast"),
    ("claude", "standard"): _spec("claude-opus-4.7-high", "standard"),
    ("claude", "thorough"): _spec("claude-opus-4.7-xhigh", "thorough"),
    ("gpt5_5", "fast"): _spec("gpt-5.5", "fast"),
    ("gpt5_5", "standard"): _spec("gpt-5.5", "standard"),
    ("gpt5_5", "thorough"): _spec("gpt-5.5", "thorough"),
}


def list_presets() -> list[dict]:
    """JSON-friendly view of the preset table for /api/presets."""
    out = []
    for (family, preset), spec in PRESETS.items():
        out.append({
            "family": family,
            "preset": preset,
            "model": spec["model"],
            "reasoning_effort": spec["reasoning_effort"],
            "max_iterations": spec["max_iterations"],
            "stop_after_distinct_routes": spec["stop_after_distinct_routes"],
            "auto_finish_after_saved": spec["auto_finish_after_saved"],
            "wallclock_s": spec["wallclock_s"],
            "scratchpad_max_routes": spec["scratchpad_max_routes"],
        })
    return out


def apply(config: dict, family: str, preset: str) -> dict:
    """Mutate ``config`` (the dict produced by ``core.config.load_config``) to
    use the chosen preset. Returns the same dict for convenience.

    Raises ValueError on an unknown family/preset combination — callers
    are expected to surface this as a 400.
    """
    key = (family, preset)
    if key not in PRESETS:
        raise ValueError(f"unknown preset: family={family!r} preset={preset!r}")
    spec = PRESETS[key]
    config["llm_model"] = spec["model"]
    config["llm_reasoning_effort"] = spec["reasoning_effort"]
    config["max_iterations"] = spec["max_iterations"]
    agent_block = config.setdefault("agent", {}) or {}
    agent_block["max_iterations"] = spec["max_iterations"]
    agent_block["stop_after_distinct_routes"] = spec["stop_after_distinct_routes"]
    agent_block["auto_finish_after_saved"] = spec["auto_finish_after_saved"]
    agent_block["wallclock_s"] = spec["wallclock_s"]
    config["agent"] = agent_block
    scratchpad_block = config.setdefault("scratchpad", {}) or {}
    scratchpad_block["max_routes"] = spec["scratchpad_max_routes"]
    config["scratchpad"] = scratchpad_block
    llm_block = config.setdefault("llm", {}) if isinstance(config.get("llm"), dict) else {}
    if isinstance(llm_block, dict):
        llm_block["model"] = spec["model"]
        llm_block["reasoning_effort"] = spec["reasoning_effort"]
        config["llm"] = llm_block
    return config
