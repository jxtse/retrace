"""Tests for the config schema validator.

The validator catches typos in built-in config overlays before any heavy
weights load, instead of silently falling back to defaults.
"""
from __future__ import annotations

import pytest

from services.config_schema import ConfigValidationError, validate_config


# ---- happy path ----------------------------------------------------


def test_real_config_validates_clean():
    """The built-in base config must pass strict validation."""
    from core.config import BASE_CONFIG

    n = validate_config(BASE_CONFIG, strict=True)
    assert n == 0


# ---- typo detection ------------------------------------------------


def test_trailing_underscore_typo_caught_with_hint():
    """The original bug: `max_iterations_:` instead of `max_iterations:`.
    Validator should name the suspected correct spelling in its message."""
    raw = {"agent": {"max_iterations_": 99}}
    with pytest.raises(ConfigValidationError) as exc:
        validate_config(raw, strict=True)
    assert "max_iterations" in str(exc.value)
    # The hint format intentionally mentions "underscore" so a grep on
    # logs surfaces these for cleanup.
    assert "underscore" in str(exc.value).lower()


def test_unknown_top_level_block_raises_in_strict_mode():
    raw = {"agent": {}, "totally_invented_block": {"k": 1}}
    with pytest.raises(ConfigValidationError, match="totally_invented_block"):
        validate_config(raw, strict=True)


def test_unknown_subkey_raises_in_strict_mode():
    raw = {"neural_retro": {"enabled": True, "topk": 10, "made_up_knob": 7}}
    with pytest.raises(ConfigValidationError, match="made_up_knob"):
        validate_config(raw, strict=True)


def test_llm_accepts_responses_state_and_minimal_reasoning():
    raw = {"llm": {"model": "gpt-5.5", "reasoning_effort": "minimal", "responses_state": "manual"}}
    n = validate_config(raw, strict=True)
    assert n == 0


# ---- lenient mode --------------------------------------------------


def test_lenient_mode_returns_count_not_raises():
    raw = {
        "agent": {"max_iterations_": 1, "another_typo": 2},  # 2 issues
        "totally_invented_block": {},                         # 1 more
    }
    n = validate_config(raw, strict=False)
    assert n >= 3


def test_validates_when_blocks_are_none():
    """Empty / missing blocks are not typos — they're just defaults."""
    raw = {"agent": None, "neural_retro": None}
    n = validate_config(raw, strict=True)
    assert n == 0


def test_root_must_be_mapping():
    with pytest.raises(ConfigValidationError, match="mapping"):
        validate_config([1, 2, 3], strict=True)
