"""Tests for the chat-completions reasoning-truncation budget-growth fix.

A reasoning model (Qwen3.x via vllm) emits a long ``reasoning_content``
block before any user-visible ``content``/``tool_calls``. When ``max_tokens``
is too small for a turn the stream is cut with ``finish_reason=="length"``
while still inside the reasoning block: ``content`` empty, no ``tool_calls``,
``reasoning_content`` non-empty. The old client flagged that "empty" and
retried the *identical* request up to ``max_retries`` times, re-truncating
each time (~50% of JMC wall-clock).

The fix (core/llm_client.py):
  - ``_empty_is_reasoning_truncation`` distinguishes this from a genuine
    transient empty.
  - ``_grow_budget_for_truncation`` computes a larger output budget bounded
    by real context headroom, or None when growth is impossible.
  - ``_post_chat_completions`` grows the budget and retries *immediately*
    (no backoff) on truncation, but keeps the backoff path for true empties.
"""
import types

import pytest

from core import llm_client as mod
from core.llm_client import (
    LLMClient,
    _empty_is_reasoning_truncation,
    _is_empty_completion,
)


def _completion(content=None, tool_calls=None, reasoning=None,
                finish_reason="stop", prompt_tokens=0):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if reasoning:
        msg["reasoning_content"] = reasoning
    return {
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": prompt_tokens},
    }


# ── _empty_is_reasoning_truncation ────────────────────────────────────


def test_detects_reasoning_truncation():
    data = _completion(content=None, reasoning="thinking...",
                       finish_reason="length", prompt_tokens=12000)
    is_trunc, pt = _empty_is_reasoning_truncation(data)
    assert is_trunc is True
    assert pt == 12000


def test_not_truncation_when_finish_reason_stop():
    # Genuine transient empty: no content, but finish_reason != length.
    data = _completion(content=None, reasoning="x", finish_reason="stop",
                       prompt_tokens=5000)
    is_trunc, _ = _empty_is_reasoning_truncation(data)
    assert is_trunc is False


def test_not_truncation_when_no_reasoning():
    # length-cut but no reasoning block → not the Qwen signature.
    data = _completion(content=None, reasoning=None, finish_reason="length",
                       prompt_tokens=5000)
    is_trunc, _ = _empty_is_reasoning_truncation(data)
    assert is_trunc is False


def test_not_truncation_when_content_present():
    data = _completion(content="answer", reasoning="x", finish_reason="length",
                       prompt_tokens=5000)
    is_trunc, _ = _empty_is_reasoning_truncation(data)
    assert is_trunc is False


def test_truncation_completion_is_also_empty():
    # The truncation case must still register as "empty" to the old check,
    # otherwise it would be returned to the agent as a usable (blank) turn.
    data = _completion(content=None, reasoning="x", finish_reason="length",
                       prompt_tokens=5000)
    assert _is_empty_completion(data) is True


# ── _grow_budget_for_truncation ───────────────────────────────────────


def _client(context_window=65536, max_tokens=8192):
    c = LLMClient.__new__(LLMClient)  # bypass __init__/network
    c.context_window = context_window
    c.max_tokens = max_tokens
    return c


def test_grow_geometric_step():
    c = _client()
    # current 8192, plenty of headroom → doubles to 16384.
    assert c._grow_budget_for_truncation({"max_tokens": 8192}, 12000) == 16384
    assert c._grow_budget_for_truncation({"max_tokens": 16384}, 12000) == 32768


def test_grow_capped_at_headroom():
    c = _client(context_window=65536)
    # prompt 50k → headroom = 65536-50000-2048 = 13488; doubling 8192→16384
    # would exceed it, so cap at headroom.
    assert c._grow_budget_for_truncation({"max_tokens": 8192}, 50000) == 13488


def test_grow_none_when_no_headroom():
    c = _client(context_window=65536)
    # prompt already near the window → no room beyond current budget.
    assert c._grow_budget_for_truncation({"max_tokens": 8192}, 60000) is None


def test_grow_none_without_context_window():
    # gpt-5.5 / Claude: context_window unset → never grows (returns None),
    # so those paths keep their original empty-choices handling.
    c = _client(context_window=None)
    assert c._grow_budget_for_truncation({"max_tokens": 8192}, 12000) is None


def test_grow_none_without_prompt_tokens():
    c = _client()
    # usage carried no prompt_tokens → can't bound growth safely.
    assert c._grow_budget_for_truncation({"max_tokens": 8192}, 0) is None


# ── _post_chat_completions loop behavior ──────────────────────────────


class _FakeResp:
    def raise_for_status(self):
        pass


def _run_loop(client, responses, monkeypatch):
    """Drive _post_chat_completions with a scripted list of stream results.

    `responses` is a list of completion dicts returned in order by the
    stubbed stream reader. Records the payload max_tokens seen on each POST
    and whether time.sleep was called.
    """
    seen_budgets = []
    sleeps = []

    def fake_post(url, json=None, **kw):
        seen_budgets.append(json["max_tokens"])
        return _FakeResp()

    # Bare-constructed client (no __init__): supply the attrs the real method
    # touches. The network session is a stub whose .post records the budget.
    client.session = types.SimpleNamespace(post=fake_post)
    client.base_url = "http://localhost:8000/v1"
    client.model = "qwen3.6-35b-a3b"

    it = iter(responses)
    monkeypatch.setattr(mod, "read_chat_completions_stream",
                        lambda resp, model="": next(it))
    monkeypatch.setattr(client, "_reset_session", lambda: None)
    monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))

    from services.llm_models import ChatRequest
    req = ChatRequest(model="qwen3.6-35b-a3b",
                      messages=[{"role": "user", "content": "hi"}],
                      max_tokens=8192)
    data = client._post_chat_completions(req, req.messages)
    return data, seen_budgets, sleeps


def test_loop_grows_budget_without_sleep_on_truncation(monkeypatch):
    c = _client()
    c.max_retries = 10
    c.retry_delay = 2.0
    c.connect_timeout = 1.0
    c.read_timeout = 1.0
    # First two responses truncate in reasoning; third succeeds.
    responses = [
        _completion(content=None, reasoning="r", finish_reason="length",
                    prompt_tokens=12000),
        _completion(content=None, reasoning="r", finish_reason="length",
                    prompt_tokens=12000),
        _completion(content="done", finish_reason="stop", prompt_tokens=12000),
    ]
    data, budgets, sleeps = _run_loop(c, responses, monkeypatch)
    assert data["choices"][0]["message"]["content"] == "done"
    # Budget grew 8192 -> 16384 -> 32768 across the three POSTs.
    assert budgets == [8192, 16384, 32768]
    # No backoff sleep on a deterministic budget-growth retry.
    assert sleeps == []


def test_loop_keeps_backoff_for_genuine_empty(monkeypatch):
    c = _client()
    c.max_retries = 10
    c.retry_delay = 2.0
    c.connect_timeout = 1.0
    c.read_timeout = 1.0
    # Genuine transient empties (no reasoning, finish_reason=stop): the old
    # backoff path must still run, and the budget must NOT grow.
    responses = [
        _completion(content=None, finish_reason="stop", prompt_tokens=12000),
        _completion(content="ok", finish_reason="stop", prompt_tokens=12000),
    ]
    data, budgets, sleeps = _run_loop(c, responses, monkeypatch)
    assert data["choices"][0]["message"]["content"] == "ok"
    assert budgets == [8192, 8192]   # unchanged
    assert sleeps and sleeps[0] > 0  # backoff happened
