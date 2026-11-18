from __future__ import annotations

import pytest

from core.llm_client import LLMClient


def test_anthropic_fallback_disabled_does_not_try_openrouter(monkeypatch):
    client = LLMClient(
        base_url="http://localhost:3000/v1",
        model="claude-opus-4-8",
        fallback_enabled=False,
    )

    def fail_main(*args, **kwargs):
        raise RuntimeError("main path failed")

    def fail_if_called(*args, **kwargs):  # pragma: no cover - assertion guard
        raise AssertionError("fallback should not be called")

    monkeypatch.setattr(client, "_post_anthropic_with_retry", fail_main)
    monkeypatch.setattr(client, "_try_anthropic_fallback", fail_if_called)

    with pytest.raises(RuntimeError, match="main path failed"):
        client.chat([{"role": "user", "content": "hello"}])


def test_anthropic_fallback_enabled_still_uses_openrouter(monkeypatch):
    client = LLMClient(
        base_url="http://localhost:3000/v1",
        model="claude-opus-4-8",
        fallback_enabled=True,
    )

    def fail_main(*args, **kwargs):
        raise RuntimeError("main path failed")

    def fallback(*args, **kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(client, "_post_anthropic_with_retry", fail_main)
    monkeypatch.setattr(client, "_try_anthropic_fallback", fallback)

    out = client.chat([{"role": "user", "content": "hello"}])

    assert out["choices"][0]["message"]["content"] == "ok"
