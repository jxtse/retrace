"""Regression: read_responses_stream must accept the substrate (:3000) SSE
shape, which puts the response snapshot at the TOP LEVEL of each event's data
(no canonical OpenAI ``data["response"]`` wrapper) and emits the full output
inline on ``response.completed`` with ``response.output_text.done`` (no
``.delta`` events).
"""

from __future__ import annotations

from services.anthropic_stream import read_responses_stream


class _FakeSSEResponse:
    """Minimal stand-in for requests.Response exposing iter_lines()."""

    def __init__(self, lines):
        self._lines = lines
        self.encoding = None

    def iter_lines(self, decode_unicode=True):  # noqa: ARG002
        yield from self._lines


# Substrate-shaped stream: top-level snapshot, output_text.done, completed inline.
_SUBSTRATE_LINES = [
    "event: response.created",
    'data: {"id": "resp_1", "object": "response", "model": "dev-gpt-55-reasoning", '
    '"status": "in_progress", "output": [{"type": "message", "id": "m1", '
    '"role": "assistant", "content": [{"type": "output_text", "text": ""}]}]}',
    "",
    "event: response.output_text.done",
    'data: {"type": "response.output_text.done", "text": "pong"}',
    "",
    "event: response.completed",
    'data: {"id": "resp_1", "object": "response", "model": "dev-gpt-55-reasoning", '
    '"status": "completed", "output": [{"type": "message", "id": "m1", '
    '"role": "assistant", "content": [{"type": "output_text", "text": "pong"}]}], '
    '"usage": {"input_tokens": 5, "output_tokens": 2}}',
    "",
    "data: [DONE]",
]

# Canonical OpenAI-shaped stream: snapshot wrapped under data["response"].
_CANONICAL_LINES = [
    "event: response.completed",
    'data: {"type": "response.completed", "response": {"id": "resp_2", '
    '"object": "response", "model": "gpt-5.5", "status": "completed", '
    '"output": [{"type": "message", "id": "m2", "role": "assistant", '
    '"content": [{"type": "output_text", "text": "hi"}]}], "usage": {}}}',
    "",
    "data: [DONE]",
]


def test_substrate_unwrapped_shape_yields_text():
    out = read_responses_stream(_FakeSSEResponse(_SUBSTRATE_LINES))
    assert out["status"] == "completed"
    assert out["id"] == "resp_1"
    text = out["output"][0]["content"][0]["text"]
    assert text == "pong"


def test_canonical_wrapped_shape_still_works():
    out = read_responses_stream(_FakeSSEResponse(_CANONICAL_LINES))
    assert out["status"] == "completed"
    assert out["id"] == "resp_2"
    assert out["output"][0]["content"][0]["text"] == "hi"
