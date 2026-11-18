"""vLLM rejects requests whose ``prompt + max_tokens`` exceeds the model's
context window with HTTP 400 *before* generating anything:

    This model's maximum context length is 65536 tokens. However, you
    requested 32768 output tokens and your prompt contains at least 32769
    input tokens, for a total of at least 65537 tokens. ...
    (parameter=input_tokens, value=32769)

wirev6 (2026-09-16) died on exactly this after 21 successful calls: the
prompt grew past 32.8k while ``max_tokens`` stayed at the configured 32768.
A 400 is not retryable as-is, but this one is deterministic and fixable by
shrinking the output budget to the real headroom. The client must do that
instead of failing the whole case.
"""
import types

import pytest
import requests

from core import llm_client as mod
from core.llm_client import LLMClient, _parse_context_overflow


VLLM_MSG = (
    "This model's maximum context length is 65536 tokens. However, you "
    "requested 32768 output tokens and your prompt contains at least 32769 "
    "input tokens, for a total of at least 65537 tokens. Please reduce the "
    "length of the input prompt or the number of requested output tokens. "
    "(parameter=input_tokens, value=32769)"
)


def test_parse_context_overflow_extracts_window_and_prompt():
    body = {"error": {"message": VLLM_MSG, "type": "BadRequestError", "code": 400}}
    parsed = _parse_context_overflow(body)
    assert parsed == (65536, 32769, 32768)


def test_parse_context_overflow_ignores_other_400s():
    assert _parse_context_overflow({"error": {"message": "tools not supported"}}) is None
    assert _parse_context_overflow({"detail": "bad"}) is None
    assert _parse_context_overflow("not json") is None


def _client(max_tokens=32768, context_window=65536):
    c = LLMClient.__new__(LLMClient)
    c.max_tokens = max_tokens
    c.context_window = context_window
    return c


def test_shrink_budget_for_overflow_fits_headroom():
    c = _client()
    payload = {"max_tokens": 32768}
    new = c._shrink_budget_for_overflow(payload, window=65536, prompt_tokens=32769)
    # window - prompt - margin(256) = 32511
    assert new == 65536 - 32769 - 256
    assert new < 32768


def test_shrink_budget_returns_none_when_no_useful_room():
    c = _client()
    # prompt nearly fills the window: fewer than min_output tokens would remain
    assert c._shrink_budget_for_overflow({"max_tokens": 32768}, window=65536,
                                         prompt_tokens=65536 - 1000) is None


def _http_400(message):
    import json as _json
    resp = requests.Response()
    resp.status_code = 400
    resp._content = _json.dumps(
        {"error": {"message": message, "type": "BadRequestError", "code": 400}}).encode()
    return requests.HTTPError("400 Client Error", response=resp)


def test_post_chat_completions_shrinks_and_retries_on_context_overflow(monkeypatch):
    """First POST → vLLM 400 overflow; second POST (smaller max_tokens) → OK."""
    c = _client()
    c.base_url = "http://x/v1"
    c.model = "m"
    c.max_retries = 3
    c.retry_delay = 0.0
    c.connect_timeout = 1
    c.read_timeout = 1
    c.reasoning_effort = "low"
    c.temperature = 0.7
    c.responses_state = "manual"
    seen = []

    class FakeResp:
        def __init__(self, status, body=None, err=None):
            self.status_code = status
            self._err = err
            self._body = body

        def raise_for_status(self):
            if self._err:
                raise self._err

    class FakeTokenizeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"count": 33381}   # real prompt is LONGER than the 400's lower bound

    class FakeSession:
        def post(self, url, json=None, timeout=None, stream=None):
            if url.endswith("/tokenize"):
                assert json["messages"] and json.get("add_generation_prompt") is True
                return FakeTokenizeResp()
            seen.append(dict(json))
            if len(seen) == 1:
                return FakeResp(400, err=_http_400(VLLM_MSG))
            return FakeResp(200)

    c.session = FakeSession()
    monkeypatch.setattr(mod, "read_chat_completions_stream",
                        lambda resp, model=None: {"choices": [{"index": 0, "message": {
                            "role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 32769}})
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(c, "_post_chat_with_recovery",
                        lambda *a, **k: pytest.fail("should not reach recovery"))

    msgs = [{"role": "user", "content": "hi"}]
    req = mod._build_chat_request(model="m", messages=msgs, tools=None,
                                  temperature=0.7, max_tokens=32768,
                                  reasoning_effort="low")
    out = c._post_chat_completions(req, msgs)
    assert out["choices"][0]["message"]["content"] == "ok"
    assert len(seen) == 2
    assert seen[0]["max_tokens"] == 32768
    # shrink uses the EXACT tokenize count (33381), not the 400's "at least 32769"
    assert seen[1]["max_tokens"] == 65536 - 33381 - 256


def test_shrink_uses_conservative_margin_when_tokenize_unavailable(monkeypatch):
    c = _client()
    c.base_url = "http://x/v1"; c.model = "m"; c.max_retries = 3; c.retry_delay = 0.0
    c.connect_timeout = 1; c.read_timeout = 1; c.reasoning_effort = "low"; c.temperature = 0.7
    c.responses_state = "manual"
    seen = []

    class FakeResp:
        def __init__(self, err=None): self._err = err; self.status_code = 400 if err else 200
        def raise_for_status(self):
            if self._err: raise self._err

    class FakeSession:
        def post(self, url, json=None, timeout=None, stream=None):
            if url.endswith("/tokenize"):
                raise requests.ConnectionError("no tokenize endpoint")
            seen.append(dict(json))
            return FakeResp(_http_400(VLLM_MSG)) if len(seen) == 1 else FakeResp()

    c.session = FakeSession()
    monkeypatch.setattr(mod, "read_chat_completions_stream",
                        lambda resp, model=None: {"choices": [{"index": 0, "message": {
                            "role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 33000}})
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    msgs = [{"role": "user", "content": "hi"}]
    req = mod._build_chat_request(model="m", messages=msgs, tools=None, temperature=0.7,
                                  max_tokens=32768, reasoning_effort="low")
    c._post_chat_completions(req, msgs)
    assert seen[1]["max_tokens"] == 65536 - 32769 - 2048
