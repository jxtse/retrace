"""Pydantic v2 models for the three LLM wire protocols + the unified
internal shape the rest of the agent consumes.

Three wire protocols, one internal shape:
  /v1/chat/completions  ↔  ChatCompletionsPayload / ChatCompletion
  /v1/messages          ↔  AnthropicPayload      / AnthropicResponse
  /v1/responses         ↔  ResponsesPayload      / ResponsesResponse

`to_<protocol>(req)` and `from_<protocol>(resp)` are pure functions
that translate between an internal `ChatRequest` / `ChatCompletion`
and the wire-shape pydantic models. The wire models use
`extra="allow"` (proxy adds fields we don't model) but our internal
models use `extra="forbid"` so malformed LLM payloads surface immediately.
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ─── Internal (route-agnostic) shapes ───────────────────────────────


class ToolCallFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    # JSON-encoded arguments string. Empty → "{}" per Responses contract.
    arguments: str = "{}"


class ToolCall(BaseModel):
    """Mirrors OpenAI tool_call shape (the lowest-common-denominator
    inside the agent)."""

    model_config = ConfigDict(extra="forbid")
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class SystemMessage(BaseModel):
    # `extra="ignore"` (the pydantic default): the agent carries
    # internal metadata on message dicts (`name`, `tool_name`,
    # `elapsed_ms`, `_provenance`, …) that the wire protocols don't
    # need. Silently drop them rather than fail validation — the
    # message model isn't the right typo-detection boundary; the
    # wire-shape models and `ChatRequest`/`ToolCall`/`ToolDef` are.
    model_config = ConfigDict(extra="ignore")
    role: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["user"] = "user"
    # The agent uses plain strings on the user channel; preserve that.
    content: str


class AssistantMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # Model reasoning / thinking trace. Captured for transcript logging only;
    # callers must NOT echo this back into the multi-turn message stream
    # (the Responses / Anthropic APIs manage reasoning continuity natively,
    # and re-sending summaries is both wasteful and API-invalid).
    reasoning_content: str | None = None


class ToolMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str


ChatMessage = Annotated[
    Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage],
    Field(discriminator="role"),
]


class ToolDefFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolDef(BaseModel):
    """Wrapped OpenAI tool definition — matches what
    `tools/base.py:to_openai_tool()` emits."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["function"] = "function"
    function: ToolDefFunction


class ChatRequest(BaseModel):
    """Internal request shape used by `LLMClient.chat()`. Translators
    project this into protocol-specific payloads."""

    model_config = ConfigDict(extra="forbid")
    model: str
    messages: list[ChatMessage]
    tools: list[ToolDef] = Field(default_factory=list)
    temperature: float = 0.2
    max_tokens: int = 4096
    # Honoured only on the Responses route; ignored elsewhere.
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "medium"


class Usage(BaseModel):
    # The 4141 proxy and OpenRouter both add fields we don't model
    # (`prompt_tokens_details.cached_tokens`, …); allow extras to keep
    # the schema future-proof.
    model_config = ConfigDict(extra="allow")
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Populated only for Responses.
    reasoning_tokens: int = 0


class Choice(BaseModel):
    model_config = ConfigDict(extra="allow")
    index: int = 0
    message: AssistantMessage
    finish_reason: str = "stop"


class ChatCompletion(BaseModel):
    """Internal response shape. `model_dump()` of this is what
    LLMClient hands back to core.agent / verifier (dict shape unchanged
    from today)."""

    # Allow extras for proxy-added fields (object, created, system_fingerprint, …).
    model_config = ConfigDict(extra="allow")
    id: str = ""
    object: Literal["chat.completion"] = "chat.completion"
    model: str = ""
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


# ─── Wire shapes — Anthropic /v1/messages ───────────────────────────


class AnthropicTextBlock(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["text"]
    text: str


class AnthropicToolUseBlock(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class AnthropicToolResultBlock(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[dict] = ""


AnthropicContentBlock = Annotated[
    Union[AnthropicTextBlock, AnthropicToolUseBlock, AnthropicToolResultBlock],
    Field(discriminator="type"),
]


class AnthropicTool(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class AnthropicMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["user", "assistant"]
    # The wire shape mixes plain strings (initial user nudges) and
    # content-block lists (tool use / result turns).
    content: str | list[AnthropicContentBlock]


class AnthropicPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    system: str | None = None
    messages: list[AnthropicMessage]
    tools: list[AnthropicTool] | None = None
    tool_choice: dict[str, Any] | None = None
    max_tokens: int = 4096
    temperature: float = 0.2
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "medium"


class AnthropicUsage(BaseModel):
    model_config = ConfigDict(extra="allow")
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = ""
    model: str = ""
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicContentBlock] = Field(default_factory=list)
    stop_reason: str | None = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


# ─── Wire shapes — OpenAI /v1/chat/completions ──────────────────────


class ChatCompletionsPayload(BaseModel):
    """Already OpenAI-shaped — messages + tools pass through unchanged."""

    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    temperature: float = 0.2
    max_tokens: int = 4096


# ─── Wire shapes — OpenAI /v1/responses (gpt-5.5 family) ────────────


class ResponsesReasoning(BaseModel):
    model_config = ConfigDict(extra="forbid")
    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"]
    # Request a reasoning summary so it can be captured for transcript
    # logging. Without this the Responses API returns reasoning *token
    # counts* but no summary text, so the output has no summary to extract.
    # Serialised only when set (exclude_none at dump time).
    summary: Optional[str] = None


class ResponsesInputMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["message"] = "message"
    role: Literal["user", "assistant"]
    content: str


# Substrate's (:3000) `/v1/responses` shim internally re-projects each input
# item into a chat `message`, sourcing that message's text from the item's
# `content` field. The OpenAI-canonical responses format omits `content` on
# function_call / function_call_output items (the payload lives in `arguments` /
# `output`), so two things break on substrate:
#   1. A missing `content` 400s ("Invalid value for 'content': expected a
#      string, got null", param messages.[N].content). The :4141 proxy tolerated
#      the omission.
#   2. WORSE — an *empty* `content` makes substrate project the tool-result
#      message as empty text, silently DROPPING the real `output`. The model
#      then never sees any tool result, re-calls the same tool every turn, and
#      the agent loops forever (analyze_molecule ×400, never saves a route).
#      Verified 2026-06-13: with content="" the model re-calls the tool; with
#      content=<the real output> it answers from the result.
# Fix: carry `content` and MIRROR the real payload into it — `output` for a
# function_call_output, a readable "name(arguments)" summary for a function_call.
# A model_validator backfills it so call sites don't change. Spec-correct readers
# ignore the extra field; :4141 tolerated everything, so this is safe on every
# channel.
class ResponsesFunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["function_call"] = "function_call"
    call_id: str
    name: str
    arguments: str  # JSON string, "{}" if empty
    content: str = ""  # substrate-shim text; backfilled below

    @model_validator(mode="after")
    def _mirror_content(self) -> "ResponsesFunctionCall":
        if not self.content:
            # A readable echo of the call so substrate's projected message is
            # non-empty and self-describing (the canonical signal is still in
            # `name`/`arguments`).
            self.content = f"{self.name}({self.arguments})"
        return self


class ResponsesFunctionCallOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str
    content: str = ""  # substrate-shim text; backfilled below

    @model_validator(mode="after")
    def _mirror_content(self) -> "ResponsesFunctionCallOutput":
        if not self.content:
            # The tool result MUST land in `content` or substrate drops it and
            # the agent loops re-calling the tool (see class note).
            self.content = self.output
        return self


ResponsesInputItem = Union[
    ResponsesInputMessage,
    ResponsesFunctionCall,
    ResponsesFunctionCallOutput,
]


class ResponsesTool(BaseModel):
    """Flat (unwrapped) tool definition the Responses API expects."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["function"] = "function"
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ResponsesPayload(BaseModel):
    # Wire payload — allow extras so future proxy additions don't trip
    # us at serialisation time.
    model_config = ConfigDict(extra="allow")
    model: str
    instructions: str = ""
    input: list[ResponsesInputItem]
    tools: list[ResponsesTool] | None = None
    max_output_tokens: int = 4096
    reasoning: ResponsesReasoning
    store: bool | None = None
    previous_response_id: str | None = None
    # No `temperature` — Responses rejects it on gpt-5.5.


class ResponsesOutputText(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["output_text"]
    text: str = ""


class ResponsesOutputMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["message"]
    role: Literal["assistant"] = "assistant"
    content: list[ResponsesOutputText] = Field(default_factory=list)


class ResponsesOutputFunctionCall(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["function_call"]
    call_id: str
    name: str
    arguments: str = "{}"
    status: str | None = None


class ResponsesOutputReasoning(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["reasoning"]
    id: str = ""
    summary: list[Any] = Field(default_factory=list)


ResponsesOutputBlock = Annotated[
    Union[
        ResponsesOutputMessage,
        ResponsesOutputFunctionCall,
        ResponsesOutputReasoning,
    ],
    Field(discriminator="type"),
]


class ResponsesUsageDetails(BaseModel):
    model_config = ConfigDict(extra="allow")
    reasoning_tokens: int = 0


class ResponsesUsage(BaseModel):
    model_config = ConfigDict(extra="allow")
    input_tokens: int = 0
    output_tokens: int = 0
    output_tokens_details: ResponsesUsageDetails = Field(
        default_factory=ResponsesUsageDetails
    )


class ResponsesResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = ""
    model: str = ""
    output: list[ResponsesOutputBlock] = Field(default_factory=list)
    usage: ResponsesUsage = Field(default_factory=ResponsesUsage)


# ─── Helpers ────────────────────────────────────────────────────────


def _parse_tool_arguments(raw: str | dict | None) -> dict:
    """Mirror the leniency of the previous helper:
    accept dicts, strings, and empty values; never crash on bad JSON."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
        except Exception:
            return {"_raw": raw}
        return obj if isinstance(obj, dict) else {"_value": obj}
    return {"_value": raw}


def _new_tool_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:12]}"


# ─── Anthropic translators ──────────────────────────────────────────


def to_anthropic(req: ChatRequest) -> AnthropicPayload:
    """Project the internal request into an Anthropic Messages payload.

    Matches the behaviour of llm_client._openai_messages_to_anthropic +
    _openai_tools_to_anthropic exactly:

      • system messages concatenate (\\n\\n) into the top-level `system`
      • adjacent same-role content blocks merge into a single message
      • assistant `tool_calls` emit as `tool_use` blocks (after any text)
      • tool messages emit as user `tool_result` blocks
      • the first message must be `user`; otherwise prepend a placeholder
    """

    system_chunks: list[str] = []
    msgs: list[AnthropicMessage] = []

    def _push(role: Literal["user", "assistant"], block: AnthropicContentBlock | str) -> None:
        if msgs and msgs[-1].role == role:
            cur = msgs[-1]
            content = cur.content
            if isinstance(content, list):
                if isinstance(block, str):
                    content.append(AnthropicTextBlock(type="text", text=block))
                else:
                    content.append(block)
                return
            blocks: list[AnthropicContentBlock] = []
            if isinstance(content, str) and content:
                blocks.append(AnthropicTextBlock(type="text", text=content))
            if isinstance(block, str):
                blocks.append(AnthropicTextBlock(type="text", text=block))
            else:
                blocks.append(block)
            cur.content = blocks
            return
        if isinstance(block, str):
            msgs.append(AnthropicMessage(role=role, content=block))
        else:
            msgs.append(AnthropicMessage(role=role, content=[block]))

    for m in req.messages:
        if isinstance(m, SystemMessage):
            if m.content:
                system_chunks.append(m.content)
            continue
        if isinstance(m, UserMessage):
            if isinstance(m.content, str):
                _push("user", m.content)
            continue
        if isinstance(m, AssistantMessage):
            if m.content:
                _push("assistant", AnthropicTextBlock(type="text", text=m.content))
            for tc in m.tool_calls:
                input_obj = _parse_tool_arguments(tc.function.arguments)
                _push(
                    "assistant",
                    AnthropicToolUseBlock(
                        type="tool_use",
                        id=tc.id or _new_tool_id(),
                        name=tc.function.name,
                        input=input_obj,
                    ),
                )
            continue
        if isinstance(m, ToolMessage):
            content = m.content
            if not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False, default=str)
                except Exception:
                    content = str(content)
            _push(
                "user",
                AnthropicToolResultBlock(
                    type="tool_result",
                    tool_use_id=m.tool_call_id or "",
                    content=content,
                ),
            )
            continue

    # Anthropic requires the first message to be `user`.
    if msgs and msgs[0].role != "user":
        msgs.insert(0, AnthropicMessage(role="user", content="(continue)"))

    anth_tools: list[AnthropicTool] | None = None
    if req.tools:
        anth_tools = [
            AnthropicTool(
                name=t.function.name,
                description=t.function.description or "",
                input_schema=_sanitize_tool_parameters(
                    t.function.parameters
                    or {"type": "object", "properties": {}}
                ),
            )
            for t in req.tools
        ]

    payload = AnthropicPayload(
        model=req.model,
        system="\n\n".join(system_chunks) if system_chunks else None,
        messages=msgs,
        tools=anth_tools,
        tool_choice={"type": "auto"} if anth_tools else None,
        max_tokens=_clamp_anthropic_max_tokens(req.model, req.max_tokens),
        temperature=req.temperature,
        reasoning_effort=req.reasoning_effort,
    )
    return payload


# Substrate enforces the real per-model output-token ceiling and 400s above it.
# Both wire paths hit the SAME 128000 cap:
#   • Anthropic: "max_tokens: 300000 > 128000, which is the maximum allowed
#     number of output tokens for claude-opus-4-8"
#   • Responses (gpt-5.5): "max_tokens is too large: 300000. This model supports
#     at most 128000 completion tokens" (substrate re-wraps the upstream 400 as
#     a 502 Bad Gateway, which is what the agent sees).
# The :4141 proxy silently accepted our config-level 300000, so the agent's
# large `llm.max_tokens` (it doubles as the output budget) must be clamped to
# the model cap here rather than forcing a config edit per channel.
_SUBSTRATE_MAX_OUTPUT_TOKENS = 128000
# Back-compat alias (name predates the responses path needing the same clamp).
_ANTHROPIC_MAX_OUTPUT_TOKENS = _SUBSTRATE_MAX_OUTPUT_TOKENS


def _clamp_output_tokens(requested: int) -> int:
    """Clamp an output-token budget to the substrate per-model ceiling.

    Shared by the Anthropic (`to_anthropic`) and Responses (`to_responses`)
    paths — both upstreams reject the agent's config-level 300000.
    """
    cap = _SUBSTRATE_MAX_OUTPUT_TOKENS
    try:
        req = int(requested)
    except (TypeError, ValueError):
        return cap
    return min(req, cap) if req > 0 else cap


def _clamp_anthropic_max_tokens(model: str, requested: int) -> int:
    return _clamp_output_tokens(requested)


def from_anthropic(resp: AnthropicResponse, model: str) -> ChatCompletion:
    """Translate an Anthropic Messages response to the internal shape.
    Mirrors llm_client._from_anthropic_response."""

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for b in resp.content:
        if isinstance(b, AnthropicTextBlock):
            if b.text:
                text_parts.append(b.text)
        elif isinstance(b, AnthropicToolUseBlock):
            tool_calls.append(
                ToolCall(
                    id=b.id or _new_tool_id(),
                    type="function",
                    function=ToolCallFunction(
                        name=b.name or "",
                        arguments=json.dumps(
                            b.input or {}, ensure_ascii=False, default=str
                        ),
                    ),
                )
            )
        # tool_result blocks should never appear on the assistant side; skip.

    msg = AssistantMessage(
        role="assistant",
        content="\n".join(text_parts) if text_parts else None,
        tool_calls=tool_calls,
    )

    finish_map = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }
    finish_reason = finish_map.get(resp.stop_reason or "", resp.stop_reason or "stop")

    return ChatCompletion(
        id=resp.id or "",
        object="chat.completion",
        model=resp.model or model,
        choices=[Choice(index=0, message=msg, finish_reason=finish_reason)],
        usage=Usage(
            prompt_tokens=resp.usage.input_tokens or 0,
            completion_tokens=resp.usage.output_tokens or 0,
            total_tokens=(resp.usage.input_tokens or 0)
            + (resp.usage.output_tokens or 0),
        ),
    )


# ─── Chat Completions translator ────────────────────────────────────


def to_chat_completions(req: ChatRequest) -> ChatCompletionsPayload:
    """Pass-through projection — agent already emits OpenAI shape.

    We dump each typed message into a plain dict, dropping the empty
    `tool_calls` list when the assistant didn't call any tools so that
    the wire payload looks the same as the hand-rolled one this client
    used to produce.
    """
    messages: list[dict[str, Any]] = []
    for m in req.messages:
        d = m.model_dump(exclude_none=True)
        if isinstance(m, AssistantMessage) and not m.tool_calls:
            d.pop("tool_calls", None)
        messages.append(d)

    tools: list[dict[str, Any]] | None = None
    if req.tools:
        tools = [t.model_dump(exclude_none=True) for t in req.tools]

    return ChatCompletionsPayload(
        model=req.model,
        messages=messages,
        tools=tools,
        tool_choice="auto" if tools else None,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )


# ─── Responses translators ──────────────────────────────────────────


def _sanitize_tool_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Strip JSON-Schema constructs the upstream APIs reject at the top
    level of a tool's parameters block.

    Both the Responses API and the substrate (:3000) Anthropic Messages
    endpoint reject a top-level combinator with HTTP 400::

        # Responses
        Invalid schema for function '<name>': schema must have type
        'object' and not have 'oneOf'/'anyOf'/'allOf'/'enum'/'not'
        at the top level.
        # substrate /v1/messages
        tools.N.function.parameters: input_schema does not support
        oneOf, allOf, or anyOf at the top level

    (The older :4141 proxy's Messages path happened to tolerate a
    top-level `anyOf`; substrate does not, so the Anthropic path now
    sanitizes too — not just Responses.)

    For tools that still express mutual-exclusion intent through a top-level
    combinator, the per-property descriptions remain visible to the model.
    Schema-level enforcement was a defense-in-depth nicety on
    `/v1/chat/completions`; the other two endpoints can't enforce it, so we
    drop it rather than crash. Nested schemas (inside `properties.*`) are not
    touched — only the top-level keys."""
    if not isinstance(params, dict):
        return {"type": "object", "properties": {}}
    out = {k: v for k, v in params.items()
           if k not in ("oneOf", "anyOf", "allOf", "enum", "not")}
    out.setdefault("type", "object")
    out.setdefault("properties", {})
    return out


# Back-compat alias: the responses path and its tests reference the old name.
_sanitize_responses_parameters = _sanitize_tool_parameters


def to_responses(
    req: ChatRequest,
    *,
    previous_response_id: str | None = None,
    store: bool | None = None,
) -> ResponsesPayload:
    """Project the internal request into a `/v1/responses` payload.

    Layout:
      • system messages concatenate (\\n\\n) into the top-level
        `instructions` field — there is no system-role item in `input`.
      • each user / assistant text turn becomes a `message` item.
      • assistant `tool_calls` emit as separate `function_call` items
        *after* the assistant message (Responses keeps them as siblings).
      • tool messages emit as `function_call_output` items.
      • tools are flat (`{type:function, name, description, parameters}`).
      • empty arguments become "{}" (the API rejects "").
    """

    instructions_chunks: list[str] = []
    items: list[ResponsesInputItem] = []

    for m in req.messages:
        if isinstance(m, SystemMessage):
            if m.content:
                instructions_chunks.append(m.content)
            continue
        if isinstance(m, UserMessage):
            if m.content:
                items.append(ResponsesInputMessage(role="user", content=m.content))
            continue
        if isinstance(m, AssistantMessage):
            if m.content:
                items.append(
                    ResponsesInputMessage(role="assistant", content=m.content)
                )
            for tc in m.tool_calls:
                args = tc.function.arguments
                if not isinstance(args, str) or not args.strip():
                    args = "{}"
                items.append(
                    ResponsesFunctionCall(
                        call_id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )
            continue
        if isinstance(m, ToolMessage):
            items.append(
                ResponsesFunctionCallOutput(
                    call_id=m.tool_call_id,
                    output=m.content if isinstance(m.content, str) else str(m.content),
                )
            )
            continue

    tools: list[ResponsesTool] | None = None
    if req.tools:
        tools = [
            ResponsesTool(
                name=t.function.name,
                description=t.function.description or "",
                parameters=_sanitize_responses_parameters(
                    t.function.parameters
                    or {"type": "object", "properties": {}}
                ),
            )
            for t in req.tools
        ]

    return ResponsesPayload(
        model=req.model,
        instructions="\n\n".join(instructions_chunks) if instructions_chunks else "",
        input=items,
        tools=tools,
        # Substrate's gpt-5.5 responses path 400s (re-wrapped as a 502) on the
        # agent's config-level 300000; clamp to the 128000 model cap. The :4141
        # proxy tolerated 300000, so this only bites on the substrate channel.
        max_output_tokens=_clamp_output_tokens(req.max_tokens),
        reasoning=ResponsesReasoning(effort=req.reasoning_effort, summary="auto"),
        store=store,
        previous_response_id=previous_response_id,
    )


def from_responses(resp: ResponsesResponse, model: str) -> ChatCompletion:
    """Translate a `/v1/responses` payload into the internal shape.

    Walk `output[]`:
      • `reasoning` blocks: skip (we don't echo them in multi-turn).
      • `message` blocks: collect `content[*].text` (output_text).
      • `function_call` blocks: translate into `ToolCall`.
    """
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    reasoning_parts: list[str] = []

    for block in resp.output:
        if isinstance(block, ResponsesOutputReasoning):
            # Capture reasoning summary text for transcript logging. The
            # summary is a list of items, typically {type: "summary_text",
            # text: ...}; be tolerant of plain strings / other shapes.
            for item in block.summary or []:
                if isinstance(item, str):
                    reasoning_parts.append(item)
                elif isinstance(item, dict):
                    t = item.get("text") or item.get("summary") or ""
                    if t:
                        reasoning_parts.append(str(t))
                else:
                    t = getattr(item, "text", None)
                    if t:
                        reasoning_parts.append(str(t))
            continue
        if isinstance(block, ResponsesOutputMessage):
            for c in block.content:
                if isinstance(c, ResponsesOutputText) and c.text:
                    text_parts.append(c.text)
            continue
        if isinstance(block, ResponsesOutputFunctionCall):
            args = block.arguments or "{}"
            if not isinstance(args, str) or not args.strip():
                args = "{}"
            tool_calls.append(
                ToolCall(
                    id=block.call_id or _new_tool_id(),
                    type="function",
                    function=ToolCallFunction(
                        name=block.name or "",
                        arguments=args,
                    ),
                )
            )
            continue

    msg = AssistantMessage(
        role="assistant",
        content="\n".join(text_parts) if text_parts else None,
        tool_calls=tool_calls,
        reasoning_content="\n".join(reasoning_parts) if reasoning_parts else None,
    )
    finish_reason = "tool_calls" if tool_calls else "stop"

    reasoning_tokens = (
        resp.usage.output_tokens_details.reasoning_tokens
        if resp.usage and resp.usage.output_tokens_details
        else 0
    )
    return ChatCompletion(
        id=resp.id or "",
        object="chat.completion",
        model=resp.model or model,
        choices=[Choice(index=0, message=msg, finish_reason=finish_reason)],
        usage=Usage(
            prompt_tokens=resp.usage.input_tokens or 0,
            completion_tokens=resp.usage.output_tokens or 0,
            total_tokens=(resp.usage.input_tokens or 0)
            + (resp.usage.output_tokens or 0),
            reasoning_tokens=reasoning_tokens or 0,
        ),
    )


__all__ = [
    # Internal shapes
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "ChatMessage",
    "ToolCall",
    "ToolCallFunction",
    "ToolDef",
    "ToolDefFunction",
    "ChatRequest",
    "ChatCompletion",
    "Choice",
    "Usage",
    # Anthropic wire
    "AnthropicTextBlock",
    "AnthropicToolUseBlock",
    "AnthropicToolResultBlock",
    "AnthropicContentBlock",
    "AnthropicTool",
    "AnthropicMessage",
    "AnthropicPayload",
    "AnthropicUsage",
    "AnthropicResponse",
    # Chat Completions wire
    "ChatCompletionsPayload",
    # Responses wire
    "ResponsesReasoning",
    "ResponsesInputMessage",
    "ResponsesFunctionCall",
    "ResponsesFunctionCallOutput",
    "ResponsesInputItem",
    "ResponsesTool",
    "ResponsesPayload",
    "ResponsesOutputText",
    "ResponsesOutputMessage",
    "ResponsesOutputFunctionCall",
    "ResponsesOutputReasoning",
    "ResponsesOutputBlock",
    "ResponsesUsage",
    "ResponsesUsageDetails",
    "ResponsesResponse",
    # Translators
    "to_anthropic",
    "from_anthropic",
    "to_chat_completions",
    "to_responses",
    "from_responses",
]
