from __future__ import annotations

import json

import requests

from core.llm_client import LLMClient


class _FakeResponse:
    def __init__(self, response: dict | None = None, status_code: int = 200, text: str = ""):
        self.response = response or {}
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def iter_lines(self, decode_unicode=True):
        if self.status_code >= 400:
            return
        yield "event: response.completed"
        yield "data: " + json.dumps({"response": self.response})
        yield ""


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


def _client_with_session(session: _FakeSession) -> LLMClient:
    client = LLMClient(
        base_url="http://proxy/v1",
        model="gpt-5.5",
        max_tokens=64000,
        reasoning_effort="high",
    )
    client.session = session
    client._reset_session = lambda: None
    return client


def _tool():
    return [{
        "type": "function",
        "function": {
            "name": "suggest_templates",
            "description": "Suggest",
            "parameters": {
                "type": "object",
                "properties": {"product_smiles": {"type": "string"}},
            },
        },
    }]


def _function_response(resp_id: str, call_id: str = "call_1") -> dict:
    return {
        "id": resp_id,
        "model": "gpt-5.5",
        "output": [{
            "type": "function_call",
            "call_id": call_id,
            "name": "suggest_templates",
            "arguments": '{"product_smiles":"CCO"}',
        }],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }


def _message_response(resp_id: str, text: str = "done") -> dict:
    return {
        "id": resp_id,
        "model": "gpt-5.5",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }],
        "usage": {"input_tokens": 3, "output_tokens": 4},
    }


def test_responses_state_sends_delta_after_first_response():
    session = _FakeSession([
        _FakeResponse(_function_response("resp_1")),
        _FakeResponse(_message_response("resp_2")),
    ])
    client = _client_with_session(session)

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "target"},
    ]
    first = client.chat(messages, _tool())
    assistant = first["choices"][0]["message"]

    messages2 = messages + [
        assistant,
        {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'},
    ]
    client.chat(messages2, _tool())

    first_payload = session.requests[0]["json"]
    second_payload = session.requests[1]["json"]
    assert first_payload["store"] is True
    assert "previous_response_id" not in first_payload
    assert second_payload["previous_response_id"] == "resp_1"
    assert second_payload["store"] is True
    assert second_payload["instructions"] == ""
    assert [item["type"] for item in second_payload["input"]] == [
        "function_call_output"
    ]
    assert second_payload["input"][0]["call_id"] == "call_1"


def test_responses_state_resets_to_full_replay_on_prefix_mismatch():
    session = _FakeSession([
        _FakeResponse(_function_response("resp_1")),
        _FakeResponse(_message_response("resp_2")),
    ])
    client = _client_with_session(session)

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "target"},
    ]
    first = client.chat(messages, _tool())
    assistant = first["choices"][0]["message"]
    compacted = [
        {"role": "system", "content": "rewritten system prompt"},
        {"role": "user", "content": "target"},
        assistant,
        {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'},
    ]

    client.chat(compacted, _tool())

    second_payload = session.requests[1]["json"]
    assert "previous_response_id" not in second_payload
    assert second_payload["instructions"] == "rewritten system prompt"
    assert [item["type"] for item in second_payload["input"]] == [
        "message",
        "function_call",
        "function_call_output",
    ]


def test_previous_response_failure_retries_full_replay_and_disables_state():
    session = _FakeSession([
        _FakeResponse(_function_response("resp_1")),
        _FakeResponse(
            status_code=404,
            text='{"error":"previous_response_id resp_1 not found"}',
        ),
        _FakeResponse(_message_response("resp_2")),
    ])
    client = _client_with_session(session)

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "target"},
    ]
    first = client.chat(messages, _tool())
    assistant = first["choices"][0]["message"]
    messages2 = messages + [
        assistant,
        {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'},
    ]

    client.chat(messages2, _tool())

    state_payload = session.requests[1]["json"]
    retry_payload = session.requests[2]["json"]
    assert state_payload["previous_response_id"] == "resp_1"
    assert "previous_response_id" not in retry_payload
    assert "store" not in retry_payload
    assert retry_payload["instructions"] == "system prompt"
    assert client._responses_native_state_disabled is True


def test_store_unsupported_retries_without_store_and_disables_state():
    session = _FakeSession([
        _FakeResponse(
            status_code=400,
            text='{"error":{"message":"store is not supported","code":"unsupported_value","param":"store"}}',
        ),
        _FakeResponse(_function_response("resp_1")),
    ])
    client = _client_with_session(session)

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "target"},
    ]

    client.chat(messages, _tool())

    first_payload = session.requests[0]["json"]
    retry_payload = session.requests[1]["json"]
    assert first_payload["store"] is True
    assert "store" not in retry_payload
    assert "previous_response_id" not in retry_payload
    assert client._responses_store_unsupported is True
    assert client._responses_native_state_disabled is True


def test_chat_with_override_is_stateless_by_default():
    session = _FakeSession([
        _FakeResponse(_function_response("resp_1")),
        _FakeResponse(_message_response("resp_2")),
    ])
    client = _client_with_session(session)

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "target"},
    ]
    first = client.chat(messages, _tool())
    assistant = first["choices"][0]["message"]
    messages2 = messages + [
        assistant,
        {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'},
    ]

    client.chat_with_override(messages2, _tool(), max_tokens=1000)

    second_payload = session.requests[1]["json"]
    assert "previous_response_id" not in second_payload
    assert "store" not in second_payload
    assert second_payload["instructions"] == "system prompt"
    assert second_payload["max_output_tokens"] == 1000


def test_manual_responses_state_never_uses_previous_response_id():
    session = _FakeSession([
        _FakeResponse(_function_response("resp_1")),
        _FakeResponse(_message_response("resp_2")),
    ])
    client = _client_with_session(session)
    client.responses_state = "manual"

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "target"},
    ]
    first = client.chat(messages, _tool())
    assistant = first["choices"][0]["message"]
    messages2 = messages + [
        assistant,
        {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'},
    ]

    client.chat(messages2, _tool())

    assert "previous_response_id" not in session.requests[1]["json"]
    assert "store" not in session.requests[1]["json"]
