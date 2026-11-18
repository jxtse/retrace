"""Regression tests for built-in config presets and build_agent plumbing."""

from __future__ import annotations

import textwrap
from unittest.mock import patch

import pytest

from core.config import load_config
from services.config_schema import ConfigValidationError


def _minimal_config(**agent_overrides):
    """Build a minimal config dict that build_agent can ingest offline."""
    return {
        "llm": {"base_url": "http://localhost:0", "model": "test-model"},
        "agent": {
            "max_depth": 12,
            "max_iterations": 77,
            "stop_after_distinct_routes": 10,
            "iteration_billable_tools": ["suggest_templates", "neural_retro"],
            "auto_finish_after_saved": 5,
            **agent_overrides,
        },
        "prompts": {
            "core": "prompts/core.md",
            "sections_dir": "prompts/sections",
            "messages_dir": "prompts/messages",
        },
        "scratchpad": {"max_routes": 25},
        "result_store": {"dir": "/tmp/test_session", "retention_days": 0},
        "postprocess": {"enabled": False},
        "verifier": {"enabled": False},
        "neural_retro": {"enabled": False},
        "templates": {"dataset_dir": None},
        "stocks": {},
        "precedent_recall": {"enabled": False},
    }


def test_build_agent_reads_nested_max_depth():
    from core import build_agent

    cfg = _minimal_config(max_depth=12)
    with patch("core.agent_factory.LLMClient"), patch("tools.builtins.register_all_builtins"):
        agent = build_agent(cfg)
    assert agent.max_depth == 12


def test_build_agent_reads_nested_max_iterations():
    from core import build_agent

    cfg = _minimal_config(max_iterations=77)
    with patch("core.agent_factory.LLMClient"), patch("tools.builtins.register_all_builtins"):
        agent = build_agent(cfg)
    assert agent.max_iterations == 77


def test_build_agent_can_disable_compaction():
    from core import build_agent

    cfg = _minimal_config()
    cfg["compaction"] = {"enabled": False}
    with patch("core.agent_factory.LLMClient"), patch("tools.builtins.register_all_builtins"):
        agent = build_agent(cfg)
    assert agent.compaction_enabled is False


@pytest.mark.parametrize(
    ("thinking", "reasoning", "max_iterations", "stop_after", "auto_finish", "wallclock", "max_routes"),
    [
        ("fast", "medium", 50, 10, 20, 14400, 80),
        ("standard", "high", 100, 10, 30, 21600, 120),
        ("thorough", "xhigh", 200, 20, 40, 36000, 200),
    ],
)
def test_thinking_presets_project_exact_values(
    thinking,
    reasoning,
    max_iterations,
    stop_after,
    auto_finish,
    wallclock,
    max_routes,
):
    cfg = load_config(thinking=thinking)

    assert cfg["thinking"] == thinking
    assert cfg["llm"]["reasoning_effort"] == reasoning
    assert cfg["llm_reasoning_effort"] == reasoning
    assert cfg["agent"]["max_iterations"] == max_iterations
    assert cfg["max_iterations"] == max_iterations
    assert cfg["agent"]["stop_after_distinct_routes"] == stop_after
    assert cfg["agent"]["auto_finish_after_saved"] == auto_finish
    assert cfg["agent"]["wallclock_s"] == wallclock
    assert cfg["scratchpad"]["max_routes"] == max_routes


def test_override_can_replace_valid_keys(tmp_path):
    cfg_path = tmp_path / "override.yaml"
    cfg_path.write_text(textwrap.dedent(
        """
        llm:
          model: test-model-override
        agent:
          max_iterations: 12
        scratchpad:
          max_routes: 7
        """
    ).strip())

    cfg = load_config(str(cfg_path), thinking="standard")

    assert cfg["llm"]["model"] == "test-model-override"
    assert cfg["llm_model"] == "test-model-override"
    assert cfg["agent"]["max_iterations"] == 12
    assert cfg["scratchpad"]["max_routes"] == 7


@pytest.mark.parametrize(
    "legacy_yaml",
    [
        "scoring:\n  enabled: false\n",
        "agent:\n  early_exit_score: 0.8\n",
        "agent:\n  min_distinct_routes: 3\n",
        "agent:\n  min_billable_iterations: 10\n",
        "agent:\n  target_route_score: 0.75\n",
        "agent:\n  forced_refinement_rounds: 2\n",
        "agent:\n  iteration_exempt_tools: [check_buyability]\n",
        "one_step:\n  enabled: true\n",
        "rxngraphormer:\n  enabled: true\n",
    ],
)
def test_override_rejects_legacy_keys(tmp_path, legacy_yaml):
    cfg_path = tmp_path / "legacy.yaml"
    cfg_path.write_text(legacy_yaml)

    with pytest.raises(ConfigValidationError):
        load_config(str(cfg_path))


def test_load_config_passes_through_precedent_recall_block(tmp_path):
    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(textwrap.dedent(
        """
        precedent_recall:
          enabled: true
          db_root: "/some/where/precedent-db"
          routes_file: "routes.jsonl"
          fps_file: "target_fps.npy"
          source_masks_file: "source_masks.npz"
          defaults:
            k: 3
            min_similarity: 0.2
          prefetch:
            enabled: true
            source: "mixed"
            k: 3
        """
    ).strip())

    cfg = load_config(str(cfg_path))

    pr = cfg["precedent_recall"]
    assert pr["enabled"] is True
    assert pr["db_root"] == "/some/where/precedent-db"
    assert pr["prefetch"]["enabled"] is True
    assert pr["defaults"]["k"] == 3


def test_load_config_projects_responses_state(tmp_path):
    cfg_path = tmp_path / "responses_state.yaml"
    cfg_path.write_text(textwrap.dedent(
        """
        llm:
          model: "gpt-5.5"
          reasoning_effort: "minimal"
          responses_state: "manual"
        """
    ).strip())

    cfg = load_config(str(cfg_path))

    assert cfg["llm_reasoning_effort"] == "minimal"
    assert cfg["llm_responses_state"] == "manual"


def test_load_config_projects_postprocess_defaults():
    cfg = load_config(thinking="fast")

    pp = cfg["postprocess"]
    assert pp["enabled"] is True
    assert pp["model"] == "claude-opus-4-8"
    assert pp["reasoning_effort"] == "xhigh"
    assert pp["responses_state"] == "manual"
    assert pp["fallback"]["enabled"] is False
