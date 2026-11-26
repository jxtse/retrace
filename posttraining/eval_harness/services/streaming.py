"""
Streaming LLM client for the retrosynthesis agent.
Supports Server-Sent Events (SSE) streaming from OpenAI-compatible APIs.
"""

import json
import logging
from typing import Generator

import requests

from services.llm_http import LLMHTTPSession

logger = logging.getLogger(__name__)


def _auth_headers(base_url: str) -> dict:
    from core.llm_client import primary_api_key

    key = primary_api_key(base_url)
    return {"Authorization": f"Bearer {key}"} if key else {}


def stream_chat_completion(
    base_url: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> Generator[dict, None, None]:
    """
    Stream chat completions via SSE. Yields delta events.
    Each yielded dict has: {"type": "content"|"tool_call"|"done", ...}
    """
    # Guard against routes this streamer cannot handle. The Responses API
    # (gpt-5.5) has a different SSE event schema; the main agent loop is
    # non-streaming so no caller in this repo hits it today, but raise
    # explicitly rather than silently posting an unsupported payload.
    from core.llm_client import _is_responses_model  # local import: avoid cycle
    if _is_responses_model(model):
        raise NotImplementedError(
            "Streaming is not implemented for the Responses API "
            "(gpt-5.5/gpt-5.6 family). The agent's main loop uses non-streaming "
            "chat(); ad-hoc streaming callers must use a different model."
        )

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    url = f"{base_url.rstrip('/')}/chat/completions"

    try:
        with LLMHTTPSession() as session:
            resp = session.post(url, json=payload, headers=_auth_headers(base_url),
                                stream=True, timeout=300)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Stream request failed: %s", e)
        yield {"type": "error", "error": str(e)}
        return

    content_buffer = ""
    tool_calls_buffer = {}  # id -> {function: {name, arguments}}
    finish_reason = None

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        for choice in data.get("choices", []):
            delta = choice.get("delta", {})
            fr = choice.get("finish_reason")
            if fr:
                finish_reason = fr

            # Content delta
            if "content" in delta and delta["content"]:
                content_buffer += delta["content"]
                yield {"type": "content_delta", "content": delta["content"]}

            # Tool call deltas
            if "tool_calls" in delta:
                for tc_delta in delta["tool_calls"]:
                    idx = tc_delta.get("index", 0)
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {
                            "id": tc_delta.get("id", ""),
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    tc = tool_calls_buffer[idx]
                    if tc_delta.get("id"):
                        tc["id"] = tc_delta["id"]
                    fn = tc_delta.get("function", {})
                    if fn.get("name"):
                        tc["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        tc["function"]["arguments"] += fn["arguments"]

    # Emit final results
    if content_buffer:
        yield {"type": "content", "content": content_buffer}
    if tool_calls_buffer:
        yield {"type": "tool_calls", "tool_calls": list(tool_calls_buffer.values())}
    yield {"type": "done", "finish_reason": finish_reason}
