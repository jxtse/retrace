from __future__ import annotations

import json

from pydantic import TypeAdapter

from services.config_models import LLMConfig
from services.llm_models import (
    ChatRequest,
    ResponsesResponse,
    ToolCall,
    ToolCallFunction,
    ToolDef,
    ToolDefFunction,
    ToolMessage,
    UserMessage,
    AssistantMessage,
    from_responses,
    to_anthropic,
    to_responses,
)


def test_to_responses_projects_tools_reasoning_store_and_sanitizes_schema():
    req = ChatRequest(
        model="gpt-5.5",
        reasoning_effort="minimal",
        max_tokens=64000,
        messages=[
            UserMessage(content="target"),
            AssistantMessage(
                content="checking",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=ToolCallFunction(
                            name="suggest_templates",
                            arguments=json.dumps({"smiles": "CCO"}),
                        ),
                    )
                ],
            ),
            ToolMessage(tool_call_id="call_1", content='{"ok": true}'),
        ],
        tools=[
            ToolDef(
                function=ToolDefFunction(
                    name="suggest_templates",
                    description="Suggest disconnections",
                    parameters={
                        "type": "object",
                        "properties": {"smiles": {"type": "string"}},
                        "oneOf": [{"required": ["smiles"]}],
                        "not": {"required": ["bad"]},
                    },
                )
            )
        ],
    )

    payload = to_responses(
        req,
        previous_response_id="resp_prev",
        store=True,
    ).model_dump(exclude_none=True)

    assert payload["model"] == "gpt-5.5"
    assert payload["reasoning"] == {"effort": "minimal", "summary": "auto"}
    assert payload["max_output_tokens"] == 64000
    assert payload["store"] is True
    assert payload["previous_response_id"] == "resp_prev"
    assert [item["type"] for item in payload["input"]] == [
        "message",
        "message",
        "function_call",
        "function_call_output",
    ]
    params = payload["tools"][0]["parameters"]
    assert params == {"type": "object", "properties": {"smiles": {"type": "string"}}}


def test_to_anthropic_projects_reasoning_effort():
    req = ChatRequest(
        model="claude-opus-4-8",
        reasoning_effort="xhigh",
        max_tokens=32000,
        messages=[UserMessage(content="review route")],
    )

    payload = to_anthropic(req).model_dump(exclude_none=True)

    assert payload["model"] == "claude-opus-4-8"
    assert payload["reasoning_effort"] == "xhigh"


def test_to_responses_clamps_max_output_tokens_to_substrate_cap():
    # The agent's config-level max_tokens (300000) doubles as the responses
    # output budget. Substrate's gpt-5.5 path 400s ("max_tokens is too large:
    # 300000 ... supports at most 128000") and re-wraps it as a 502, which is
    # the empty-iteration-1 stall we hit on the standard preset. to_responses
    # must clamp to 128000 (the :4141 proxy tolerated 300000; substrate does not).
    req = ChatRequest(
        model="gpt-5.5",
        reasoning_effort="high",
        max_tokens=300000,
        messages=[UserMessage(content="target")],
    )
    payload = to_responses(req).model_dump(exclude_none=True)
    assert payload["max_output_tokens"] == 128000

    # A budget already under the cap passes through unchanged.
    req_small = ChatRequest(
        model="gpt-5.5",
        reasoning_effort="high",
        max_tokens=64000,
        messages=[UserMessage(content="target")],
    )
    assert to_responses(req_small).model_dump()["max_output_tokens"] == 64000


def test_to_responses_mirrors_tool_result_into_content_for_substrate():
    # Substrate's /v1/responses shim re-projects each input item into a chat
    # message sourcing the text from `content`. The tool RESULT must land in
    # `content`, or substrate drops it and the model loops re-calling the tool
    # (root cause of the retro-190 substrate gpt-5.5 perception loop, 2026-06-13).
    req = ChatRequest(
        model="gpt-5.5",
        reasoning_effort="medium",
        max_tokens=64000,
        messages=[
            UserMessage(content="groups?"),
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        function=ToolCallFunction(
                            name="analyze_molecule",
                            arguments='{"smiles": "CCO"}',
                        ),
                    )
                ],
            ),
            ToolMessage(
                tool_call_id="c1",
                content='{"functional_groups": [{"name": "hydroxyl"}]}',
            ),
        ],
    )
    payload = to_responses(req).model_dump(exclude_none=True)
    by_type = {it["type"]: it for it in payload["input"]}
    # The tool output is mirrored verbatim into content (NOT empty).
    assert by_type["function_call_output"]["content"] == (
        '{"functional_groups": [{"name": "hydroxyl"}]}'
    )
    assert by_type["function_call_output"]["content"], "content must be non-empty"
    # The function_call carries a readable echo, never an empty string.
    assert by_type["function_call"]["content"] == 'analyze_molecule({"smiles": "CCO"})'


def test_minimal_reasoning_effort_is_accepted_by_config_schema():
    cfg = TypeAdapter(LLMConfig).validate_python({
        "model": "gpt-5.5",
        "reasoning_effort": "minimal",
        "responses_state": "manual",
    })

    assert cfg.reasoning_effort == "minimal"
    assert cfg.responses_state == "manual"


def test_from_responses_restores_text_function_call_and_reasoning_tokens():
    raw = {
        "id": "resp_1",
        "model": "gpt-5.5",
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Think done."}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "save_best_route",
                "arguments": '{"label":"r1"}',
            },
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 7},
        },
    }

    out = from_responses(
        ResponsesResponse.model_validate(raw),
        model="gpt-5.5",
    ).model_dump(exclude_none=True)

    choice = out["choices"][0]
    assert out["id"] == "resp_1"
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "Think done."
    assert choice["message"]["tool_calls"][0]["id"] == "call_1"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "save_best_route"
    assert out["usage"]["reasoning_tokens"] == 7
