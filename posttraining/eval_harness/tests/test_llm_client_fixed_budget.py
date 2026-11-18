"""Fixed-output-budget protocol (keep max_tokens constant at 32k across attempts).

Under ``fixed_output_budget=True`` the chat-completions client must never
change ``max_tokens`` across attempts: a reasoning-truncation
(finish_reason=length, empty content) is retried with the IDENTICAL payload,
every attempt is wire-logged, and after ``max_retries`` the call fails loudly
instead of rewriting history (cache-bust / shrink) or growing the budget.
"""
import types

import pytest

from core import llm_client as mod
from core.llm_client import LLMClient
from tests.test_llm_client_reasoning_truncation import _completion, _FakeResp


def _client(fixed, max_retries=10):
    c = LLMClient.__new__(LLMClient)
    c.context_window = 131072
    c.max_tokens = 32768
    c.max_retries = max_retries
    c.retry_delay = 2.0
    c.connect_timeout = 1.0
    c.read_timeout = 1.0
    c.fixed_output_budget = fixed
    c.wire_log_path = None
    return c


def _run(client, responses, monkeypatch):
    budgets, sleeps = [], []

    def fake_post(url, json=None, **kw):
        budgets.append(json["max_tokens"])
        return _FakeResp()

    client.session = types.SimpleNamespace(post=fake_post)
    client.base_url = "http://localhost:8000/v1"
    client.model = "qwen3.8-27b-student"
    it = iter(responses)
    monkeypatch.setattr(mod, "read_chat_completions_stream", lambda resp, model="": next(it))
    monkeypatch.setattr(client, "_reset_session", lambda: None)
    monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))

    def no_recovery(*a, **k):
        raise AssertionError("history-rewriting recovery must not run under a fixed budget")
    monkeypatch.setattr(client, "_post_chat_with_recovery", no_recovery)
    from services.llm_models import ChatRequest
    req = ChatRequest(model=client.model, messages=[{"role": "user", "content": "hi"}], max_tokens=32768)
    return client._post_chat_completions(req, req.messages), budgets, sleeps


def test_fixed_budget_retries_identical_payload(monkeypatch):
    c = _client(fixed=True)
    responses = [
        _completion(content=None, reasoning="r", finish_reason="length", prompt_tokens=18185),
        _completion(content="done", finish_reason="stop", prompt_tokens=18185),
    ]
    data, budgets, sleeps = _run(c, responses, monkeypatch)
    assert data["choices"][0]["message"]["content"] == "done"
    assert budgets == [32768, 32768]
    assert sleeps == []


def test_fixed_budget_exhaustion_fails_loudly(monkeypatch):
    c = _client(fixed=True, max_retries=2)
    responses = [
        _completion(content=None, reasoning="r", finish_reason="length", prompt_tokens=18185),
        _completion(content=None, reasoning="r", finish_reason="length", prompt_tokens=18185),
    ]
    with pytest.raises(RuntimeError, match="fixed output budget"):
        _run(c, responses, monkeypatch)


def test_growth_still_default_when_not_fixed(monkeypatch):
    c = _client(fixed=False)
    responses = [
        _completion(content=None, reasoning="r", finish_reason="length", prompt_tokens=18185),
        _completion(content="done", finish_reason="stop", prompt_tokens=18185),
    ]
    _, budgets, _ = _run(c, responses, monkeypatch)
    assert budgets == [32768, 65536]


def test_config_projects_fixed_budget_to_client(tmp_path):
    import yaml
    from core.config import load_config
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"llm": {"model": "qwen3.8-27b-student", "max_tokens": 32768,
                                            "fixed_output_budget": True, "context_window": 131072}}))
    config = load_config(str(path), thinking="fast")
    assert config["llm_fixed_output_budget"] is True
    # default remains off so eval/other profiles keep the growth behaviour
    default = load_config(None, thinking="fast")
    assert default["llm_fixed_output_budget"] is False
    client = LLMClient(base_url="http://x/v1", model="m", max_tokens=32768,
                       fallback_enabled=False, fixed_output_budget=True)
    assert client.fixed_output_budget is True
    assert LLMClient(base_url="http://x/v1", model="m", fallback_enabled=False).fixed_output_budget is False


def test_factory_forwards_fixed_budget(monkeypatch, tmp_path):
    import core.agent_factory as factory
    from tests.test_agent_e2e_loop import _make_config
    seen = {}

    class _Stub:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.model = kwargs.get("model")
            self.max_tokens = kwargs.get("max_tokens")
            self.temperature = 0.0
    monkeypatch.setattr(factory, "make_llm_client", lambda **kw: _Stub(**kw))
    config = _make_config(str(tmp_path))
    config["llm"]["fixed_output_budget"] = True
    config["llm_fixed_output_budget"] = True
    try:
        factory.build_agent(config)
    except Exception:
        pass  # heavy tool wiring may fail offline; the client kwargs were captured first
    assert seen.get("fixed_output_budget") is True
