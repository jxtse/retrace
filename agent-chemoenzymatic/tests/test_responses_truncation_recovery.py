"""Regression tests for the Responses (gpt-5.5) truncated-stream recovery.

Context (reproduced 2026-05-30): high reasoning_effort makes the model idle
the SSE connection for >157s between frames; an intermediate proxy on the
4141 reverse tunnel has a ~180s read timeout that severs the stream, leaving
`status=in_progress` with empty output. The old client treated this as a
generic empty completion and applied a 120s proxy-outage backoff — so a
single complex turn could burn 10×120s twice (primary + inline retry) and
stall a whole hard_mol case for hours (idx 9: 314 retries / 5.5h).

The fix splits truncation (status != completed) from a genuine empty answer:
truncation de-escalates reasoning_effort (shorter think → smaller idle gap),
reconnects fast (<=5s, not 120s), shrinks history once effort floors, and
latches a persistent per-client effort ceiling so the session pays the
truncation tax at most once.
"""
import requests
from unittest.mock import MagicMock, patch

import core.llm_client as L


def _seq(states):
    out = []
    for s in states:
        if s == "ok":
            out.append({
                "id": "resp_ok",
                "status": "completed",
                "output": [{"type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": "ANS"}]}],
                "usage": {},
            })
        else:
            out.append({"status": s, "output": [], "usage": {}})
    return out


def _run(client, seq):
    calls = {"n": 0, "efforts": []}

    def fake_post(self, url, json=None, timeout=None, stream=None):
        calls["n"] += 1
        calls["efforts"].append((json.get("reasoning") or {}).get("effort"))
        m = MagicMock()
        m.raise_for_status = lambda: None
        return m

    def fake_read(resp):
        return seq[min(calls["n"] - 1, len(seq) - 1)]

    slept = []
    with patch.object(requests.Session, "post", new=fake_post), \
         patch.object(L, "read_responses_stream", side_effect=fake_read), \
         patch.object(L.time, "sleep", side_effect=lambda s: slept.append(s)):
        req = L._build_chat_request(
            model="gpt-5.5", messages=[{"role": "user", "content": "plan"}],
            tools=None, temperature=0.2, max_tokens=300000, reasoning_effort="high")
        out = client._post_responses_with_retry(
            req, [{"role": "user", "content": "plan"}], use_state=False)
    return calls, slept, out


# ── helper-level ──────────────────────────────────────────────────────────

def test_effort_ladder():
    assert L._next_lower_effort("xhigh") == "high"
    assert L._next_lower_effort("high") == "medium"
    assert L._next_lower_effort("medium") == "low"
    assert L._next_lower_effort("low") is None
    assert L._next_lower_effort("none") is None  # unknown tier → already-minimal


def test_truncation_detection():
    empty = {"choices": [{"message": {}}]}
    full = {"choices": [{"message": {"content": "hi"}}]}
    assert L._is_truncated_stream({"status": "in_progress"}, empty) is True
    assert L._is_truncated_stream({"status": "incomplete"}, empty) is True
    # genuine empty answer (model finished, said nothing) is NOT truncation
    assert L._is_truncated_stream({"status": "completed"}, empty) is False
    # has content → not empty, not truncation
    assert L._is_truncated_stream({"status": "in_progress"}, full) is False
    # unknown status → don't misclassify as truncation
    assert L._is_truncated_stream({}, empty) is False


def test_with_responses_effort_is_pure():
    p = {"reasoning": {"effort": "high", "summary": "auto"}, "model": "gpt-5.5"}
    q = L.LLMClient._with_responses_effort(p, "medium")
    assert q["reasoning"]["effort"] == "medium"
    assert q["stream"] is True
    assert p["reasoning"]["effort"] == "high"  # original untouched


# ── loop-level ──────────────────────────────────────────────────────────

def _client():
    return L.LLMClient(base_url="http://x/v1", model="gpt-5.5",
                       reasoning_effort="high", max_retries=10,
                       retry_delay=0.01, max_tokens=300000)


def test_truncation_deescalates_and_fast_reconnects():
    c = _client()
    calls, slept, out = _run(c, _seq(["in_progress", "in_progress", "ok"]))
    assert out["choices"][0]["message"]["content"] == "ANS"
    # high → medium → low, succeeding on the third attempt
    assert calls["efforts"] == ["high", "medium", "low"]
    # truncation must NOT use the 120s proxy-outage backoff
    assert all(s <= 5.0 for s in slept), slept


def test_effort_ceiling_latches_across_turns():
    c = _client()
    calls1, _, _ = _run(c, _seq(["in_progress", "ok"]))
    assert calls1["efforts"] == ["high", "medium"]
    assert c._responses_effort_ceiling == "medium"
    # second turn starts at the latched ceiling, not back at high
    calls2, _, _ = _run(c, _seq(["ok"]))
    assert calls2["efforts"] == ["medium"]


def test_genuine_empty_uses_backoff_not_deescalation():
    """status=completed + empty output is a real empty answer; keep the
    cache-bust + exponential-backoff path, do not de-escalate effort."""
    c = _client()
    # completed-but-empty twice, then a real answer
    calls, slept, out = _run(c, _seq(["completed", "completed", "ok"]))
    assert out["choices"][0]["message"]["content"] == "ANS"
    # effort stays high (no de-escalation on genuine empties)
    assert calls["efforts"] == ["high", "high", "high"]
    # genuine-empty path uses exponential backoff (retry_delay * 2**attempt),
    # NOT the truncation fast-reconnect (retry_delay * truncated_count). With
    # retry_delay=0.01 both stay small, so assert on the growth shape: the
    # backoff path doubles (0.01, 0.02), and crucially effort never lowered
    # and the ceiling was never latched.
    assert slept == [0.01, 0.02], slept
    assert c._responses_effort_ceiling is None


# ── native-state correctness after shrink (P2#1 regression) ───────────────

def _run_stateful(client, seq, messages, *, effort="low"):
    """Drive _post_responses_with_retry with use_state=True and capture the
    `store` value of every request, so we can assert which response (if any)
    became the native-state anchor."""
    calls = {"n": 0, "stores": []}

    def fake_post(self, url, json=None, timeout=None, stream=None):
        calls["n"] += 1
        calls["stores"].append(json.get("store"))
        m = MagicMock()
        m.raise_for_status = lambda: None
        return m

    def fake_read(resp):
        return seq[min(calls["n"] - 1, len(seq) - 1)]

    with patch.object(requests.Session, "post", new=fake_post), \
         patch.object(L, "read_responses_stream", side_effect=fake_read), \
         patch.object(L.time, "sleep", side_effect=lambda s: None):
        req = L._build_chat_request(
            model="gpt-5.5", messages=messages,
            tools=None, temperature=0.2, max_tokens=300000,
            reasoning_effort=effort)
        out = client._post_responses_with_retry(req, messages, use_state=True)
    return calls, out


def _long_messages(n_filler=10):
    """A transcript long enough (>8 msgs) that _shrink_history drops a real
    middle section rather than returning it unchanged."""
    msgs = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "the target molecule"},
    ]
    for i in range(n_filler):
        msgs.append({"role": "assistant", "content": f"step {i}"})
        msgs.append({"role": "user", "content": f"continue {i}"})
    return msgs


def test_shrink_recovery_does_not_anchor_native_state():
    """P2#1: when effort is already floored, a truncated stream triggers a
    history shrink — sent with store=None, so the server never persisted the
    response NOR saw the full transcript. The client must NOT capture that
    response as the previous_response_id anchor; doing so would make the next
    delta silently omit the dropped middle history."""
    c = _client()
    msgs = _long_messages()
    # effort='low' => first truncation goes straight to the shrink branch
    # (no lower tier to step down to). Then it succeeds.
    calls, out = _run_stateful(
        c, _seq(["in_progress", "ok"]), msgs, effort="low")
    assert out["choices"][0]["message"]["content"] == "ANS"
    # request 1: initial attempt, store=True (native-state candidate).
    # request 2: post-shrink replay, store=None (stateless).
    assert calls["stores"][0] is True
    assert calls["stores"][1] is None
    # The successful response came from the store=None shrink replay, so NO
    # native-state anchor may be set.
    assert c._responses_previous_response_id is None
    assert c._responses_request_fingerprint is None


def test_clean_state_capture_still_anchors_on_store_true():
    """Control: a first-try success (store=True) DOES anchor native state —
    the P2#1 guard must not suppress the normal happy path."""
    c = _client()
    msgs = _long_messages()
    calls, out = _run_stateful(c, _seq(["ok"]), msgs, effort="low")
    assert out["choices"][0]["message"]["content"] == "ANS"
    assert calls["stores"][0] is True
    assert c._responses_previous_response_id is not None

