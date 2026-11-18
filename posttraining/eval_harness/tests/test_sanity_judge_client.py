"""Regression test for the precursor-sanity-judge retry-herd fix.

The judge (tools/neural_retro.py precursor sanity filter) fails open and
fans out ~6 concurrent threads per neural_retro turn over a single shared
SSH tunnel. If it inherits the MAIN agent client's retry budget
(max_retries=10, 120s cap) a transient tunnel reset becomes a thundering
herd: ~60 concurrent attempts saturate the tunnel, 0 verdicts return, and
each call burns ~600s of backoff before failing open anyway (JMC case006).

`core.agent_factory._build_sanity_judge_client` must therefore ALWAYS hand
the tool a DEDICATED client with a hard retry cap and a cheap reasoning
tier — even when the main model is already gpt-5.5 on the shared tunnel,
where reusing the main client would reintroduce the herd.
"""
from __future__ import annotations

from core.agent_factory import _build_sanity_judge_client
from core.llm_client import LLMClient


def _main_client() -> LLMClient:
    # Mirrors the large-scale setup: main agent already IS gpt-5.5 on the
    # shared localhost tunnel, with the dangerous (correct-for-main) budget.
    return LLMClient(
        base_url="http://localhost:4141/v1",
        model="gpt-5.5",
        max_retries=10,
        retry_delay=2.0,
        read_timeout=300.0,
        reasoning_effort="xhigh",
    )


def test_judge_is_a_distinct_capped_client():
    main = _main_client()
    judge = _build_sanity_judge_client({}, main)

    # Always a NEW client, never the main one (the whole point: capping the
    # herd must apply even when main is already gpt-5.5 on the shared tunnel).
    assert judge is not main

    # Hard retry cap — the herd-prevention invariant.
    assert judge.max_retries <= 3, judge.max_retries
    assert judge.retry_delay <= 1.5
    assert judge.read_timeout <= 60.0

    # Cheap pinned tier regardless of the agent's (xhigh) effort.
    assert judge.reasoning_effort == "low"

    # Stateless side-call; must not inherit the main Responses conversation.
    assert judge.responses_state == "manual"

    # A fail-open advisory judge has no business on the cross-cloud fallback.
    assert judge.fallback_enabled is False


def test_judge_defaults_endpoint_to_main_client():
    main = _main_client()
    judge = _build_sanity_judge_client({}, main)
    # No judge_llm block configured → reuse the main endpoint/model so the
    # cap applies with zero extra config.
    assert judge.base_url == main.base_url
    assert judge.model == main.model


def test_judge_llm_block_overrides(monkeypatch):
    monkeypatch.delenv("SANITY_JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("SANITY_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("SANITY_JUDGE_EFFORT", raising=False)
    main = _main_client()
    neural_cfg = {
        "precursor_sanity_filter": {
            "judge_llm": {
                "base_url": "http://elsewhere:9000/v1",
                "model": "gpt-5.5-mini",
                "reasoning_effort": "medium",
                "max_retries": 1,
                "retry_delay_s": 0.5,
                "timeout_s": 20.0,
            }
        }
    }
    judge = _build_sanity_judge_client(neural_cfg, main)
    assert judge.base_url == "http://elsewhere:9000/v1"
    assert judge.model == "gpt-5.5-mini"
    assert judge.reasoning_effort == "medium"
    assert judge.max_retries == 1
    assert judge.retry_delay == 0.5
    assert judge.read_timeout == 20.0


def test_env_overrides_take_precedence(monkeypatch):
    monkeypatch.setenv("SANITY_JUDGE_BASE_URL", "http://env-host:1234/v1")
    monkeypatch.setenv("SANITY_JUDGE_MODEL", "env-model")
    monkeypatch.setenv("SANITY_JUDGE_EFFORT", "minimal")
    main = _main_client()
    # config block present but env must win.
    neural_cfg = {
        "precursor_sanity_filter": {
            "judge_llm": {"base_url": "http://cfg:9/v1", "model": "cfg-model"}
        }
    }
    judge = _build_sanity_judge_client(neural_cfg, main)
    assert judge.base_url == "http://env-host:1234/v1"
    assert judge.model == "env-model"
    assert judge.reasoning_effort == "minimal"
    # Cap still enforced regardless of endpoint source.
    assert judge.max_retries <= 3
