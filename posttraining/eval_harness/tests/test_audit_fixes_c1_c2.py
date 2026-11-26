"""Regression tests for the post-substrate-migration audit fixes C1 and C2.

C1 — feasibility-judge fairness pin.
  The 0.70-weight feasibility judge is pinned (verifier.feasibility.judge
  base_url/model, default :3000 + claude-opus-4-8) so the SAME judge grades
  every benchmarked main agent and feasibility scores stay comparable across
  model conditions. Previously `agent_factory` injected the MAIN agent client
  into the verifier, and `WholeRouteJudge._get_llm()` early-returns an injected
  client without consulting `cfg.base_url`/`cfg.model` — so the pin was dead
  config and the judge ran on the main model (gpt-5.5 grading gpt-5.5). The fix
  builds a dedicated pinned judge client whenever the YAML pins a judge
  model/base_url distinct from the main client.

C2 — ensemble all-engines-down must be an error, not an empty leaf.
  When every ensemble engine errors, `_run_ensemble` returned predictions:[]
  with no top-level `error` — byte-identical to a genuine no-disconnection
  result, so `_is_error_payload` returned False, the empty result was memoized
  into the short-circuit cache, and the LLM read it as a confirmed leaf. The fix
  emits a top-level `error` when (and only when) every engine errored.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def patched_verifier_externals(monkeypatch):
    """Stub the verifier's external-weight dependencies so the C1 tests build a
    real agent in a CLEAN checkout (CI / fresh clone).

    `RouteVerifierTool.__init__` eagerly constructs a `ConditionPredictor`, which
    hard-requires the un-versioned `tools/condition_predictor/` weights and raises
    FileNotFoundError when they're absent. These tests exercise only the
    feasibility-JUDGE client routing (the C1 fix in agent_factory), so the
    greenness condition model is irrelevant — replace it with a no-op stub. The
    judge-client construction we assert on runs before/independent of this.
    """
    import tools.verifier.score_route as sr

    class _StubConditionPredictor:
        def __init__(self, *a, **k):
            pass

        def predict_top1(self, *a, **k):  # pragma: no cover - not exercised here
            return {"temperature": 25.0, "solvent": "", "reagent": "", "score": 0.0}

    monkeypatch.setenv("REACTION_RATER_URL", "https://rater.invalid")
    monkeypatch.setenv("RETROPLANNER_HTTP_URL", "https://planner.invalid")
    monkeypatch.setattr(sr, "ConditionPredictor", _StubConditionPredictor)
    return monkeypatch


# ── C1 ──────────────────────────────────────────────────────────────

def _build_agent_with_judge_pin(main_model: str, judge_model: str,
                                 judge_base_url: str):
    from core.config import load_config
    from core.agent_factory import build_agent

    cfg = load_config(thinking="standard", cli_overrides={"llm": {"base_url": "https://main.invalid"}})
    cfg["llm_model"] = main_model
    # Pin the feasibility judge distinct from the main model.
    cfg.setdefault("verifier", {}).setdefault("feasibility", {})["judge"] = {
        "enabled": True,
        "base_url": judge_base_url,
        "model": judge_model,
        "reasoning_effort": "medium",
    }
    return build_agent(cfg)


def _judge_client(agent):
    verifier = agent.tools.get("score_route")
    assert verifier is not None, "score_route tool not registered"
    judge = getattr(verifier, "_judge", None)
    assert judge is not None, "WholeRouteJudge not constructed"
    return judge._get_llm()


def test_feasibility_judge_uses_pinned_model_not_main(patched_verifier_externals):
    # Main agent is gpt-5.5; judge pinned to claude-opus-4-8 on substrate.
    agent = _build_agent_with_judge_pin(
        main_model="gpt-5.5",
        judge_model="claude-opus-4-8",
        judge_base_url="http://localhost:3000/v1",
    )
    judge_llm = _judge_client(agent)
    # The pin must win: the judge runs on opus-4-8, NOT the main gpt-5.5.
    assert judge_llm.model == "claude-opus-4-8", judge_llm.model
    assert judge_llm.model != agent.llm.model
    assert judge_llm.base_url == "http://localhost:3000/v1"
    # It must be a DISTINCT client, never the main agent client.
    assert judge_llm is not agent.llm


def test_feasibility_judge_reuses_main_when_not_pinned_distinct(patched_verifier_externals):
    # When the judge model/base_url equal the main client's, no separate client
    # is needed — reuse the main client (no wasted connection).
    from core.config import load_config
    from core.agent_factory import build_agent

    cfg = load_config(thinking="standard", cli_overrides={"llm": {"base_url": "https://main.invalid"}})
    cfg["llm_model"] = "gpt-5.5"
    # Judge pinned to the SAME endpoint+model as main → reuse.
    base = cfg.get("llm_base_url")
    cfg.setdefault("verifier", {}).setdefault("feasibility", {})["judge"] = {
        "enabled": True,
        "base_url": base,
        "model": "gpt-5.5",
    }
    agent = build_agent(cfg)
    judge_llm = _judge_client(agent)
    assert judge_llm is agent.llm  # reused, not a second client


# ── C2 ──────────────────────────────────────────────────────────────

def _is_error_payload(content: str) -> bool:
    # Mirrors core.agent.RetrosynthesisAgent._is_error_payload.
    if not content:
        return True
    try:
        parsed = json.loads(content)
    except Exception:
        return False
    return isinstance(parsed, dict) and "error" in parsed


def _bare_ensemble_tool(monkeypatch):
    """A NeuralRetroTool with just the attributes _run_ensemble touches and two
    fake engines, so we can drive the all-fail vs zero-candidate paths without
    any HTTP / model setup."""
    from tools.neural_retro import NeuralRetroTool

    tool = NeuralRetroTool.__new__(NeuralRetroTool)
    tool._engines = {"graph_retrosyn": object(), "chembart": object()}
    tool._per_engine_caps = {}
    tool._per_engine_topk = 10
    tool._return_topk = 10
    tool._n_buckets = 4

    def _no_sanity(target, merged):
        return merged, {"n_dropped": 0}

    monkeypatch.setattr(tool, "_apply_sanity_filter", _no_sanity, raising=False)
    return tool


def test_ensemble_all_engines_fail_emits_top_level_error(monkeypatch):
    tool = _bare_ensemble_tool(monkeypatch)

    def _boom(name, *a, **k):
        raise RuntimeError(f"HTTP 500 from {name}")

    monkeypatch.setattr(tool, "_run_single", _boom, raising=False)

    out = tool._run_ensemble(
        target_smiles="CCO", per_engine_topk=10, return_topk=10,
        detail_level="brief", model=None,
    )
    # All engines errored → MUST carry a top-level error so it is not cached
    # and not read as a confirmed leaf.
    assert "error" in out, out
    assert out["predictions"] == []
    assert _is_error_payload(json.dumps(out)) is True


def test_ensemble_zero_candidates_is_not_an_error(monkeypatch):
    tool = _bare_ensemble_tool(monkeypatch)

    # Engines RUN fine but return 0 predictions → genuine dead end, NOT an error.
    def _empty(name, *a, **k):
        return {"predictions": []}

    monkeypatch.setattr(tool, "_run_single", _empty, raising=False)

    out = tool._run_ensemble(
        target_smiles="CCO", per_engine_topk=10, return_topk=10,
        detail_level="brief", model=None,
    )
    # No error key → treated as a real (empty) result, may be cached.
    assert "error" not in out, out
    assert out["predictions"] == []
    assert _is_error_payload(json.dumps(out)) is False
