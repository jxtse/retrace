from __future__ import annotations

import json

import requests

from core.llm_client import LLMClient
from services.anthropic_stream import read_chat_completions_stream


class _FakeStreamResponse:
    def __init__(self, chunks: list[dict], status_code: int = 200, text: str = ""):
        self.chunks = chunks
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def iter_lines(self, decode_unicode=True):
        if self.status_code >= 400:
            return
        for chunk in self.chunks:
            yield "data: " + json.dumps(chunk)
            yield ""
        yield "data: [DONE]"


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.headers = {}

    def post(self, url, json=None, timeout=None, stream=False):
        self.requests.append({
            "url": url,
            "json": json,
            "timeout": timeout,
            "stream": stream,
        })
        if not self.responses:
            raise AssertionError("no fake response queued")
        return self.responses.pop(0)

    def close(self):
        pass


def _substrate_chunks() -> list[dict]:
    return [
        {
            "id": "chatcmpl_1",
            "model": "substrate-test",
            "choices": [{
                "delta": {
                    "reasoning_content": "checking ",
                    "content": "hel",
                }
            }],
        },
        {
            "id": "chatcmpl_1",
            "model": "substrate-test",
            "choices": [{
                "delta": {
                    "reasoning": "tools",
                    "content": "lo",
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": "{\"query\":\"asp",
                        },
                    }],
                }
            }],
        },
        {
            "id": "chatcmpl_1",
            "model": "substrate-test",
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": "irin\"}"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        },
        {
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "completion_tokens_details": {"reasoning_tokens": 3},
                "total_tokens": 18,
            },
            "choices": [],
        },
    ]


def test_read_chat_completions_stream_accumulates_substrate_deltas():
    raw = read_chat_completions_stream(
        _FakeStreamResponse(_substrate_chunks()),
        model="fallback-model",
    )

    msg = raw["choices"][0]["message"]
    assert raw["id"] == "chatcmpl_1"
    assert raw["model"] == "substrate-test"
    assert msg["content"] == "hello"
    assert msg["reasoning_content"] == "checking tools"
    assert msg["tool_calls"] == [{
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": "{\"query\":\"aspirin\"}"},
    }]
    assert raw["choices"][0]["finish_reason"] == "tool_calls"
    assert raw["usage"]["prompt_tokens"] == 11
    assert raw["usage"]["completion_tokens"] == 7
    assert raw["usage"]["reasoning_tokens"] == 3


def test_chat_completions_route_defaults_to_stream_with_tools_and_usage():
    session = _FakeSession([_FakeStreamResponse(_substrate_chunks())])
    client = LLMClient(
        base_url="http://proxy/v1",
        model="substrate-test",
        max_tokens=128,
        max_retries=1,
    )
    client.session = session

    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a molecule",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    }
    raw = client.chat(
        [{"role": "user", "content": "lookup aspirin"}],
        tools=[tool],
    )

    req = session.requests[0]
    assert req["stream"] is True
    assert req["json"]["stream"] is True
    assert req["json"]["stream_options"] == {"include_usage": True}
    assert req["json"]["tools"] == [tool]
    assert req["json"]["tool_choice"] == "auto"
    assert raw["choices"][0]["message"]["tool_calls"][0]["id"] == "call_1"
