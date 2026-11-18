"""
SSE stream readers for Anthropic /v1/messages, OpenAI/Substrate
/v1/chat/completions, and OpenAI-compat /v1/responses (gpt-5.5).

Why streaming: long reasoning outputs (high effort, big toolchains) hold
the connection idle past intermediate proxy read timeouts and get killed
mid-flight. Streaming keeps bytes flowing so proxies don't drop the
socket, and lets us rebuild the same final response shape that the
non-streaming path produced.

The functions here take an already-issued `requests.Response` (with
`stream=True`) and return a dict in the same shape the corresponding
non-streaming endpoint would have returned — caller-visible behavior is
unchanged.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


def _iter_sse_events(resp: requests.Response):
    """Yield (event_type, data_dict) tuples from a SSE response.

    Tolerates the two ways Anthropic / OpenAI emit events:
      • explicit `event: <name>` line followed by `data: {...}`
      • single `data: {...}` line whose JSON has a `type` field.
    """
    # Force UTF-8 for line decoding. SSE bodies are UTF-8, but when the
    # response omits an explicit charset, requests falls back to ISO-8859-1
    # for `iter_lines(decode_unicode=True)`, which mangles every multibyte
    # character (e.g. a U+2019 apostrophe in reasoning text becomes "â\x80\x99").
    # Setting encoding makes requests use an incremental UTF-8 decoder that
    # also handles multibyte sequences split across chunk boundaries.
    resp.encoding = "utf-8"
    cur_event: str | None = None
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        if raw == "":
            cur_event = None
            continue
        if raw.startswith(":"):
            continue
        if raw.startswith("event:"):
            cur_event = raw[6:].strip() or None
            continue
        if not raw.startswith("data:"):
            continue
        data_str = raw[5:].lstrip()
        if data_str == "[DONE]":
            return
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            logger.debug("skipping non-JSON SSE data: %s", data_str[:120])
            continue
        ev = cur_event or (data.get("type") if isinstance(data, dict) else None)
        yield ev, data


def read_anthropic_stream(resp: requests.Response) -> dict:
    """Consume an Anthropic /v1/messages SSE stream and return a dict
    shaped like the non-streaming response body (i.e. what
    `AnthropicResponse.model_validate` expects).
    """
    msg_id = ""
    model = ""
    role = "assistant"
    stop_reason: str | None = None
    usage = {"input_tokens": 0, "output_tokens": 0}

    # index → block dict
    blocks: dict[int, dict[str, Any]] = {}
    # index → accumulating partial JSON for tool_use input
    tool_json_buf: dict[int, str] = {}

    for ev, data in _iter_sse_events(resp):
        if ev == "message_start":
            msg = data.get("message", {})
            msg_id = msg.get("id", msg_id)
            model = msg.get("model", model)
            role = msg.get("role", role)
            u = msg.get("usage") or {}
            usage["input_tokens"] = int(u.get("input_tokens", usage["input_tokens"]) or 0)
            usage["output_tokens"] = int(u.get("output_tokens", usage["output_tokens"]) or 0)
        elif ev == "content_block_start":
            idx = int(data.get("index", 0))
            block = dict(data.get("content_block") or {})
            btype = block.get("type")
            if btype == "text":
                block.setdefault("text", "")
            elif btype == "tool_use":
                block.setdefault("input", {})
                tool_json_buf[idx] = ""
            blocks[idx] = block
        elif ev == "content_block_delta":
            idx = int(data.get("index", 0))
            delta = data.get("delta") or {}
            dtype = delta.get("type")
            block = blocks.setdefault(idx, {"type": "text", "text": ""})
            if dtype == "text_delta":
                block["text"] = (block.get("text") or "") + (delta.get("text") or "")
            elif dtype == "input_json_delta":
                tool_json_buf[idx] = tool_json_buf.get(idx, "") + (delta.get("partial_json") or "")
            elif dtype == "thinking_delta":
                # extended-thinking blocks: accumulate but the agent loop
                # ignores them (we don't surface thinking content blocks).
                block.setdefault("thinking", "")
                block["thinking"] = block["thinking"] + (delta.get("thinking") or "")
        elif ev == "content_block_stop":
            idx = int(data.get("index", 0))
            block = blocks.get(idx)
            if block and block.get("type") == "tool_use":
                buf = tool_json_buf.get(idx, "")
                if buf:
                    try:
                        block["input"] = json.loads(buf)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            "tool_use partial_json failed to parse (idx=%d, len=%d): %s",
                            idx, len(buf), e,
                        )
                        # keep raw fragment so caller sees something
                        block["input"] = {"_raw_partial_json": buf}
        elif ev == "message_delta":
            delta = data.get("delta") or {}
            if "stop_reason" in delta and delta["stop_reason"] is not None:
                stop_reason = delta["stop_reason"]
            u = data.get("usage") or {}
            if "output_tokens" in u and u["output_tokens"] is not None:
                usage["output_tokens"] = int(u["output_tokens"])
            if "input_tokens" in u and u["input_tokens"] is not None:
                usage["input_tokens"] = int(u["input_tokens"])
        elif ev == "message_stop":
            pass
        elif ev == "error":
            err = data.get("error") or data
            raise requests.HTTPError(
                f"anthropic stream error: {err}", response=resp,
            )
        elif ev == "ping":
            pass
        # Unknown events are ignored — Anthropic adds new ones over time.

    # Finalize tool_use blocks whose `content_block_stop` never arrived
    # (connection cut mid-stream with no `error` event). Without this, the
    # accumulated `tool_json_buf` is discarded and `input` stays the empty
    # `{}` default — the caller would then execute a tool with empty/wrong
    # args silently (a result with tool_calls is not "empty", so the client
    # would not retry). Parse any leftover buffer; if a tool_use block has
    # neither a stop nor any buffered args AND the stream was truncated
    # (no stop_reason), raise so the client's retry path kicks in.
    truncated = stop_reason is None
    for idx, block in blocks.items():
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        if block.get("input"):
            continue
        buf = tool_json_buf.get(idx, "")
        if buf:
            try:
                block["input"] = json.loads(buf)
            except json.JSONDecodeError:
                if truncated:
                    raise requests.HTTPError(
                        "anthropic stream truncated mid tool_use "
                        f"(idx={idx}, unparseable partial args, len={len(buf)})",
                        response=resp,
                    )
                block["input"] = {"_raw_partial_json": buf}
        elif truncated:
            raise requests.HTTPError(
                f"anthropic stream truncated before tool_use args (idx={idx})",
                response=resp,
            )

    ordered_content = [blocks[k] for k in sorted(blocks.keys())]
    return {
        "id": msg_id,
        "type": "message",
        "role": role,
        "model": model,
        "content": ordered_content,
        "stop_reason": stop_reason,
        "usage": usage,
    }


def read_chat_completions_stream(resp: requests.Response, *, model: str = "") -> dict:
    """Consume an OpenAI/Substrate chat-completions SSE stream and return
    a normal `/v1/chat/completions`-shaped dict.

    Substrate-style streams can emit long reasoning/content/tool-call
    deltas over many chunks. Buffer them here so callers keep the same
    synchronous `LLMClient.chat()` contract while the HTTP request stays
    streaming on the wire.
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
    }
    response_id = ""
    response_model = model
    finish_reason: str | None = None

    for ev, data in _iter_sse_events(resp):
        if ev == "error":
            err = data.get("error") or data
            raise requests.HTTPError(
                f"chat-completions stream error: {err}", response=resp,
            )

        response_id = data.get("id", response_id) or response_id
        response_model = data.get("model", response_model) or response_model

        u = data.get("usage")
        if isinstance(u, dict):
            prompt_tokens = int(
                u.get("prompt_tokens", u.get("input_tokens", 0)) or 0
            )
            completion_tokens = int(
                u.get("completion_tokens", u.get("output_tokens", 0)) or 0
            )
            details = (
                u.get("completion_tokens_details")
                or u.get("output_tokens_details")
                or {}
            )
            reasoning_tokens = (
                int(details.get("reasoning_tokens", 0) or 0)
                if isinstance(details, dict)
                else 0
            )
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": int(
                    u.get("total_tokens", prompt_tokens + completion_tokens) or 0
                ),
                "reasoning_tokens": reasoning_tokens,
            }

        choices = data.get("choices") or []
        if not choices:
            continue

        choice = choices[0] or {}
        if choice.get("finish_reason") is not None:
            finish_reason = choice.get("finish_reason")
        delta = choice.get("delta") or {}

        content = delta.get("content")
        if content:
            content_parts.append(str(content))

        reasoning = (
            delta.get("reasoning_content")
            or delta.get("reasoning")
            or delta.get("cot_summary")
            or ""
        )
        if reasoning:
            reasoning_parts.append(str(reasoning))

        for tc in delta.get("tool_calls") or []:
            idx = int(tc.get("index", 0) or 0)
            entry = tool_calls.setdefault(
                idx,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if tc.get("id"):
                entry["id"] = tc["id"]
            if tc.get("type"):
                entry["type"] = tc["type"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                entry["function"]["name"] = fn["name"]
            if fn.get("arguments") is not None:
                entry["function"]["arguments"] += str(fn["arguments"])

    ordered_tool_calls = [
        tool_calls[idx] for idx in sorted(tool_calls.keys())
    ]
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts) if content_parts else None,
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if ordered_tool_calls:
        message["tool_calls"] = ordered_tool_calls
        finish_reason = finish_reason or "tool_calls"

    return {
        "id": response_id,
        "object": "chat.completion",
        "model": response_model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason or "stop",
        }],
        "usage": usage,
    }


def read_responses_stream(resp: requests.Response) -> dict:
    """Consume an OpenAI Responses (gpt-5.5) SSE stream and return a dict
    shaped like the non-streaming response body — i.e. what
    `ResponsesResponse.model_validate` expects.

    Schema reference:
      response.created            — full snapshot, may be partial
      response.output_item.added  — new output item (message / tool call)
      response.output_text.delta  — text delta inside a message item
      response.function_call_arguments.delta — args delta for tool call
      response.completed          — terminal snapshot, has full output
    """
    final_snapshot: dict | None = None
    # Track items by index for incremental rebuild fallback
    items_by_index: dict[int, dict[str, Any]] = {}
    text_buf: dict[tuple[int, int], str] = {}    # (item_idx, content_idx) → text
    args_buf: dict[int, str] = {}                # item_idx → arguments text
    response_id = ""
    model = ""
    usage: dict[str, Any] = {}
    status: str | None = None
    # Preserve upstream diagnostics so an empty output isn't a silent symptom.
    # Responses API sets `incomplete_details.reason` to one of
    # max_output_tokens / content_filter / run_failed / cancelled when
    # status="incomplete"; preserving it lets the caller log why an empty
    # response was empty rather than blindly retrying.
    incomplete_details: dict[str, Any] | None = None

    for ev, data in _iter_sse_events(resp):
        if ev in ("response.created", "response.in_progress", "response.completed"):
            # Canonical OpenAI shape wraps the snapshot under data["response"].
            # The substrate (:3000) API instead puts the response fields at the
            # top level of the event data (no "response" wrapper) and ships the
            # full output inline on response.completed. Accept both: prefer the
            # wrapper, else treat data itself as the snapshot when it looks like
            # one (object=="response" or carries an output list).
            r = data.get("response") or {}
            if not r and (data.get("object") == "response" or "output" in data):
                r = {k: v for k, v in data.items() if k != "type"}
            if r:
                response_id = r.get("id", response_id)
                model = r.get("model", model)
                if r.get("usage"):
                    usage = r["usage"]
                if r.get("status"):
                    status = r["status"]
                if r.get("incomplete_details"):
                    incomplete_details = r["incomplete_details"]
                if ev == "response.completed":
                    final_snapshot = r
        elif ev == "response.output_item.added":
            idx = int(data.get("output_index", len(items_by_index)))
            item = dict(data.get("item") or {})
            if item.get("type") == "message":
                item.setdefault("content", [])
            elif item.get("type") in ("function_call", "tool_call"):
                item.setdefault("arguments", "")
                args_buf[idx] = item.get("arguments") or ""
            items_by_index[idx] = item
        elif ev == "response.output_text.delta":
            idx = int(data.get("output_index", 0))
            cidx = int(data.get("content_index", 0))
            key = (idx, cidx)
            text_buf[key] = text_buf.get(key, "") + (data.get("delta") or "")
        elif ev == "response.function_call_arguments.delta":
            idx = int(data.get("output_index", 0))
            args_buf[idx] = args_buf.get(idx, "") + (data.get("delta") or "")
        elif ev == "response.output_item.done":
            idx = int(data.get("output_index", 0))
            item = data.get("item") or {}
            if item:
                items_by_index[idx] = dict(item)
        elif ev == "error" or ev == "response.failed":
            err = data.get("error") or data
            raise requests.HTTPError(
                f"responses stream error: {err}", response=resp,
            )
        # ignore deltas we don't need (refusal, audio, reasoning summaries…)

    if final_snapshot is not None:
        # final snapshot already includes status/incomplete_details if upstream
        # populated them; nothing to splice.
        return final_snapshot

    # Rebuild from incremental events when no final snapshot arrived.
    output: list[dict] = []
    for idx in sorted(items_by_index.keys()):
        item = dict(items_by_index[idx])
        if item.get("type") == "message":
            content = list(item.get("content") or [])
            # Inject accumulated text deltas under the right content index.
            keys = sorted([k for k in text_buf if k[0] == idx], key=lambda k: k[1])
            for (_, cidx) in keys:
                while len(content) <= cidx:
                    content.append({"type": "output_text", "text": ""})
                blk = dict(content[cidx])
                if blk.get("type") in (None, "output_text"):
                    blk["type"] = "output_text"
                    # The output_item.done item is authoritative for any text
                    # it already carries. Only inject the buffered deltas when
                    # the done block is empty (or absent) — otherwise a done
                    # item that survived while response.completed was dropped
                    # would have its text duplicated by re-appending the deltas.
                    existing = blk.get("text") or ""
                    if existing:
                        blk["text"] = existing
                    else:
                        blk["text"] = text_buf[(idx, cidx)]
                content[cidx] = blk
            item["content"] = content
        elif item.get("type") in ("function_call", "tool_call"):
            if idx in args_buf:
                item["arguments"] = args_buf[idx]
        output.append(item)

    rebuilt: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "model": model,
        "status": status or "completed",
        "output": output,
        "usage": usage,
    }
    if incomplete_details is not None:
        rebuilt["incomplete_details"] = incomplete_details
    return rebuilt
