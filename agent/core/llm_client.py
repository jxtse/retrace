"""
OpenAI-compatible LLM client for the retrosynthesis agent.

External interface is OpenAI-shaped (`messages` with role/content +
`tools` with {type:function, function:{...}} + responses with
`choices[].message.{content, tool_calls}`) regardless of the upstream
endpoint. This keeps `core.agent` and `tools/verifier/llm_judge.py`
single-shape; the wire protocol is chosen here.

Wire-protocol routing rule (4141 proxy reality):
  - Anthropic Messages (`/v1/messages`) for Claude models, since the
    4141 proxy's `/v1/chat/completions` path 400s on ANY `tools`
    payload for claude-opus/sonnet.
  - Responses (`/v1/responses`) for the gpt-5.5 family — Chat
    Completions returns `unsupported_api_for_model`.
  - OpenAI Chat Completions (`/v1/chat/completions`) for everything else.

The translation layer lives in `services/llm_models.py` (pydantic v2 +
pure translator functions). This module is only the HTTP glue + retry
policy.
"""

import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any

import requests

from pydantic import TypeAdapter

from services.llm_models import (
    AnthropicResponse,
    ChatMessage,
    ChatRequest,
    ResponsesResponse,
    ToolDef,
    from_anthropic,
    from_responses,
    to_anthropic,
    to_chat_completions,
    to_responses,
)

_CHAT_MESSAGE_ADAPTER = TypeAdapter(ChatMessage)
from services.streaming import stream_chat_completion
from services.anthropic_stream import (
    read_anthropic_stream,
    read_chat_completions_stream,
    read_responses_stream,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

ANTHROPIC_VERSION = "2023-06-01"


def _is_anthropic_model(model: str) -> bool:
    """True if the model should be routed through `/v1/messages`."""
    return bool(model) and model.lower().startswith("claude")


def _anthropic_implied_reasoning_tier(model: str) -> str:
    """Extract a legacy reasoning tier encoded in a Claude model id suffix.

    Current substrate deployments expect the base Claude model id plus an
    explicit `reasoning_effort` parameter. This helper only detects older
    `-high` / `-xhigh` model ids so the client can warn about the legacy form.
    """
    if not model:
        return "medium"
    lower = model.lower()
    if lower.endswith("-xhigh"):
        return "xhigh"
    if lower.endswith("-high"):
        return "high"
    return "medium"


def _is_responses_model(model: str) -> bool:
    """True if the model should be routed through `/v1/responses`.

    Currently only the gpt-5.5 family — the 4141 proxy returns
    `unsupported_api_for_model` on chat completions for these models.
    """
    if not model:
        return False
    m = model.lower()
    return m.startswith("gpt-5.5") or m.startswith("gpt-5-5")


# Exported settings take precedence over local dotenv entries.
def _load_dotenv_once() -> None:
    if os.environ.get("_AGENT_DOTENV_LOADED"):
        return
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, ".env"), ".env"):
        if not os.path.isfile(cand):
            continue
        try:
            with open(cand, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    os.environ.setdefault(k, v)
        except OSError as exc:
            logger.debug(".env load skipped (%s): %s", cand, exc)
    os.environ["_AGENT_DOTENV_LOADED"] = "1"


def primary_api_key(base_url: str) -> str | None:
    from urllib.parse import urlsplit

    _load_dotenv_once()
    origin = urlsplit(base_url)
    key = os.getenv("LLM_API_KEY")
    if origin.scheme == "https" and origin.netloc == "openrouter.ai":
        key = key or os.getenv("OPENROUTER_API_KEY")
    if origin.scheme == "https" and origin.netloc == "api.openai.com":
        key = key or os.getenv("OPENAI_API_KEY")
    return key


def _apply_auth(session: "requests.Session", base_url: str) -> None:
    key = primary_api_key(base_url)
    session.headers.setdefault("anthropic-version", ANTHROPIC_VERSION)
    if not key:
        return
    session.headers["Authorization"] = f"Bearer {key}"
    session.headers["x-api-key"] = key
    session.headers.setdefault("HTTP-Referer", "https://github.com/local/retrosynthesis-agent")
    session.headers.setdefault("X-Title", "retrosynthesis-agent")


DEFAULT_BASE_URL = None
DEFAULT_MODEL = "claude-opus-4-8"


# ---------------------------------------------------------------------------
# Pydantic boundary helpers
# ---------------------------------------------------------------------------


def _build_chat_request(
    *,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
) -> ChatRequest:
    """Coerce the legacy dict-shaped inputs into a typed `ChatRequest`.

    This is the single boundary where dict ➜ pydantic happens. Pydantic's
    discriminated-union `ChatMessage` enforces the four valid roles
    (system/user/assistant/tool); any malformed payload from core.agent
    raises here with a useful error rather than several layers deeper.
    """
    typed_msgs: list[ChatMessage] = []
    for m in messages:
        # Tolerate the same shape variations the old client did:
        # • assistant with `tool_calls` but no `content` → drop `content: None`
        # • tool messages where `content` is non-string → stringify
        m = dict(m)
        role = m.get("role")
        if role == "tool" and not isinstance(m.get("content"), str):
            try:
                m["content"] = json.dumps(m.get("content"), ensure_ascii=False, default=str)
            except Exception:
                m["content"] = str(m.get("content"))
        if role == "assistant" and m.get("tool_calls") is None:
            m.pop("tool_calls", None)
        typed_msgs.append(_CHAT_MESSAGE_ADAPTER.validate_python(m))

    typed_tools: list[ToolDef] = []
    for t in tools or []:
        # Accept either OpenAI-wrapped or already-Anthropic shape.
        if isinstance(t, dict) and "function" in t:
            typed_tools.append(ToolDef.model_validate(t))
        elif isinstance(t, dict) and "name" in t and "input_schema" in t:
            typed_tools.append(
                ToolDef.model_validate({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", "") or "",
                        "parameters": t.get("input_schema") or {},
                    },
                })
            )

    return ChatRequest(
        model=model,
        messages=typed_msgs,
        tools=typed_tools,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
    )


def _is_empty_completion(data: dict) -> bool:
    """Same emptiness semantics the old client used on Chat Completions /
    Anthropic responses (after translation to the internal dict shape)."""
    choices = data.get("choices") or []
    if not choices:
        return True
    for ch in choices:
        m = ch.get("message", {}) or {}
        if m.get("content") or m.get("tool_calls"):
            return False
    return True


def _empty_is_reasoning_truncation(data: dict) -> tuple[bool, int]:
    """Detect a reasoning model that ran out of *output* budget mid-thought.

    A reasoning model (e.g. Qwen3.x) emits a long `reasoning_content` block
    before any user-visible `content` / `tool_calls`. When `max_tokens` is
    too small for a given turn, the stream is cut with
    ``finish_reason=="length"`` while still inside the reasoning block:
    `content` is empty, `tool_calls` absent, but `reasoning_content` is
    non-empty. `_is_empty_completion` then flags it "empty" and the caller
    retries the *identical* request — which, being only weakly stochastic
    at low temperature, truncates again. Empirically (Qwen3.6 JMC run) this
    burned ~50% of wall-clock: 67 full 8192-token generations per case that
    never produced a usable token.

    Returns ``(is_reasoning_truncation, prompt_tokens)``. The prompt-token
    count (0 if the stream carried no usage) lets the caller size the next
    attempt's budget within real context headroom. The genuine transient
    empty (no reasoning, finish_reason != length) returns ``(False, _)`` so
    its existing backoff-retry path is preserved untouched.
    """
    choices = data.get("choices") or []
    if not choices:
        return False, 0
    ch = choices[0] or {}
    if ch.get("finish_reason") != "length":
        return False, 0
    m = ch.get("message", {}) or {}
    if m.get("content") or m.get("tool_calls"):
        return False, 0
    if not (m.get("reasoning_content") or m.get("reasoning")):
        return False, 0
    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    return True, prompt_tokens



# Responses (gpt-5.5) reasoning-effort tiers, strongest → weakest. Used to
# de-escalate effort when a stream is truncated mid-reasoning: a lower tier
# spends less wall-clock thinking, so the per-frame idle gap stays under the
# intermediate proxy's ~180s read timeout that was severing high-effort
# streams (reproduced 2026-05-30: high maxgap 157s → cut; medium 73-79s;
# low ≤14s always completes).
_EFFORT_LADDER = ["xhigh", "high", "medium", "low"]


def _next_lower_effort(effort: str) -> str | None:
    """Return the next weaker reasoning effort, or None if already lowest."""
    try:
        i = _EFFORT_LADDER.index(effort)
    except ValueError:
        # Unknown tier (none/minimal) — treat as already-minimal.
        return None
    return _EFFORT_LADDER[i + 1] if i + 1 < len(_EFFORT_LADDER) else None


def _responses_status(raw: dict) -> str | None:
    """Extract the Responses lifecycle status from a raw stream-rebuild dict.

    `read_responses_stream` preserves `status` (and `incomplete_details`) on
    the dict it returns. status ∈ {completed, in_progress, incomplete, ...};
    anything other than `completed` on an *empty* output means the stream was
    severed mid-flight (idle-timeout truncation) rather than the model
    genuinely answering with nothing.
    """
    if not isinstance(raw, dict):
        return None
    st = raw.get("status")
    return str(st) if st else None


def _is_truncated_stream(raw: dict, normalized: dict) -> bool:
    """True when an empty completion is the result of a severed stream
    (status != completed) rather than a real empty answer. These warrant a
    fast reconnect + effort de-escalation, NOT a 120s proxy-outage backoff.
    """
    if not _is_empty_completion(normalized):
        return False
    st = _responses_status(raw)
    return st is not None and st != "completed"


class LLMClient:
    """OpenAI-compatible chat completion client with tool-calling support."""

    def __init__(
        self,
        base_url: str | None = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_retries: int = 6,   # 2026-06-11: 10→6 + 120s→30s cap below. The
                                 # 10/120s policy (built to span full proxy
                                 # outages) turned every intermittent 4141
                                 # gateway 502 into ~10min of backoff sleep —
                                 # 2265 such 502s in one retro-190 run, all
                                 # header-time gateway rejects (not 429 rate-
                                 # limit, not stream truncation, so streaming
                                 # can't prevent them). Fail-fast: 6 attempts,
                                 # ~90s worst-case backoff, so a flaky call
                                 # gives up and the case moves on instead of
                                 # sleeping minutes. A real multi-min outage
                                 # window still burns the case wallclock either
                                 # way, so spanning it bought little.
        retry_delay: float = 2.0,
        connect_timeout: float = 10.0,
        read_timeout: float = 300.0,
        fallback_enabled: bool = False,
        fallback_base_url: str | None = "https://openrouter.ai/api/v1",
        fallback_model: str | None = "anthropic/claude-opus-4.7",
        reasoning_effort: str = "medium",
        responses_state: str = "auto",
        context_window: int | None = None,
    ):
        if not base_url:
            raise ValueError("llm.base_url is required")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.reasoning_effort = reasoning_effort
        # Total model context window (input+output share it), used by the
        # chat-completions empty-recovery to grow the output budget within
        # real headroom when a reasoning model truncates mid-thought
        # (finish_reason=length). None disables that path (gpt-5.5 / Claude
        # don't exhibit it and route through other methods anyway).
        self.context_window = int(context_window) if context_window else None
        # Last assistant reasoning/thinking trace, set by chat_with_tools for
        # transcript logging. Not part of any return value or message stream.
        self.last_reasoning: str | None = None
        self.responses_state = responses_state if responses_state in ("auto", "manual") else "auto"
        self.session = requests.Session()
        _apply_auth(self.session, self.base_url)
        # OpenRouter fallback session, lazily constructed.
        # Activated only when the main Anthropic path raises after both
        # retry rounds (see chat / chat_with_override). Anthropic-only.
        self.fallback_enabled = fallback_enabled
        self.fallback_base_url = (fallback_base_url or "").rstrip("/") or None
        self.fallback_model = fallback_model
        self._fallback_session: "requests.Session | None" = None
        self._fallback_unavailable_logged = False
        # GPT-5.5 /v1/responses native state. This is per-client, which
        # lines up with the agent's per-session LLMClient lifecycle.
        self._responses_previous_response_id: str | None = None
        self._responses_request_message_count: int = 0
        self._responses_request_fingerprint: str | None = None
        self._responses_assistant_snapshot: dict[str, Any] | None = None
        self._responses_native_state_disabled = False
        self._responses_store_unsupported = False
        # Persistent reasoning-effort ceiling, lowered the first time a stream
        # is truncated mid-reasoning (the ~180s idle-timeout cut). Without
        # this, every complex turn would pay the truncation tax anew — burn
        # one ~180s cut, de-escalate, succeed, then reset to high next turn
        # and cut again. Latching the lowered ceiling means a session pays
        # the tax at most once. None = no cap (use the request's effort).
        self._responses_effort_ceiling: str | None = None
        # Current substrate Claude routing expects the base model id and an
        # explicit reasoning_effort parameter. Keep a soft warning for older
        # suffix-style ids because they may not be authorized by the gateway.
        if _is_anthropic_model(self.model):
            implied = _anthropic_implied_reasoning_tier(self.model)
            if implied != "medium":
                logger.warning(
                    "Anthropic model id %r uses legacy reasoning suffix %r; "
                    "prefer the base model id plus reasoning_effort=%r.",
                    self.model,
                    implied,
                    self.reasoning_effort,
                )

    # ---- fallback session --------------------------------------------------

    def _get_fallback_session(self) -> "requests.Session | None":
        """Return the lazily-constructed OpenRouter fallback session, or
        None if the fallback is disabled / unconfigured.

        We don't reuse `_apply_auth` because that helper prefers
        `LLM_API_KEY` (the 4141 proxy key). The OpenRouter endpoint needs
        `OPENROUTER_API_KEY` specifically; otherwise it would send a
        proxy key to OpenRouter and 401.
        """
        if not self.fallback_enabled or not self.fallback_base_url:
            return None
        if self._fallback_session is not None:
            return self._fallback_session
        _load_dotenv_once()
        from urllib.parse import urlsplit
        origin = urlsplit(self.fallback_base_url)
        if origin.scheme == "https" and origin.netloc == "openrouter.ai":
            key = os.getenv("OPENROUTER_API_KEY")
        else:
            key = os.getenv("LLM_FALLBACK_API_KEY")
        if not key:
            if not self._fallback_unavailable_logged:
                logger.info(
                    "Anthropic fallback disabled: no credential for the configured provider"
                )
                self._fallback_unavailable_logged = True
            return None
        sess = requests.Session()
        sess.headers["Authorization"] = f"Bearer {key}"
        sess.headers.setdefault("anthropic-version", ANTHROPIC_VERSION)
        sess.headers.setdefault("HTTP-Referer", "https://github.com/local/retrosynthesis-agent")
        sess.headers.setdefault("X-Title", "retrosynthesis-agent")
        self._fallback_session = sess
        return sess

    # ---- strict-validator sanitizers (OpenRouter / Bedrock) ----------------

    def _sanitize_anthropic_messages_for_strict_validator(
        self,
        anth_msgs: list[dict],
    ) -> tuple[list[dict], int, int]:
        """Repair orphaned tool_use / tool_result blocks before posting to a
        validator (OpenRouter) that's stricter than the 4141 proxy.

        Strategy: walk through and (1) drop tool_use blocks whose IDs never
        get a matching tool_result later in the conversation, (2) drop
        tool_result blocks whose tool_use_id has no matching tool_use
        earlier.
        """
        result_ids: set[str] = set()
        for m in anth_msgs:
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "tool_result":
                    tid = blk.get("tool_use_id")
                    if tid:
                        result_ids.add(tid)
        cleaned: list[dict] = []
        seen_use_ids: set[str] = set()
        dropped_uses = 0
        dropped_results = 0
        for m in anth_msgs:
            content = m.get("content")
            if not isinstance(content, list):
                cleaned.append(m)
                continue
            new_blocks: list = []
            for blk in content:
                if not isinstance(blk, dict):
                    new_blocks.append(blk)
                    continue
                btype = blk.get("type")
                if btype == "tool_use":
                    bid = blk.get("id")
                    if bid and bid in result_ids:
                        seen_use_ids.add(bid)
                        new_blocks.append(blk)
                    else:
                        dropped_uses += 1
                elif btype == "tool_result":
                    tid = blk.get("tool_use_id")
                    if tid and tid in seen_use_ids:
                        new_blocks.append(blk)
                    else:
                        dropped_results += 1
                else:
                    new_blocks.append(blk)
            if new_blocks:
                new_msg = dict(m)
                new_msg["content"] = new_blocks
                cleaned.append(new_msg)
            elif m.get("role") == "assistant":
                cleaned.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": "(continuing)"}],
                })
        if cleaned and cleaned[0].get("role") != "user":
            cleaned.insert(0, {"role": "user", "content": "(continue)"})
        if cleaned and cleaned[-1].get("role") == "assistant":
            cleaned.append({
                "role": "user",
                "content": "Continue from this state.",
            })
        return cleaned, dropped_uses, dropped_results

    def _sanitize_anthropic_tools_for_strict_validator(
        self,
        anth_tools: list[dict] | None,
    ) -> tuple[list[dict] | None, int]:
        """Strip top-level JSON Schema combinators (oneOf/allOf/anyOf) from
        each tool's `input_schema` before posting to Bedrock via OpenRouter.
        """
        if not anth_tools:
            return anth_tools, 0
        BAD = ("oneOf", "allOf", "anyOf")
        cleaned: list[dict] = []
        dropped = 0
        for t in anth_tools:
            if not isinstance(t, dict):
                cleaned.append(t)
                continue
            schema = t.get("input_schema")
            if not isinstance(schema, dict):
                cleaned.append(t)
                continue
            top_combinators = [k for k in BAD if k in schema]
            if not top_combinators:
                cleaned.append(t)
                continue
            new_schema = dict(schema)
            for k in top_combinators:
                new_schema.pop(k, None)
                dropped += 1
            new_t = dict(t)
            new_t["input_schema"] = new_schema
            cleaned.append(new_t)
        return cleaned, dropped

    def _try_anthropic_fallback(
        self,
        req: ChatRequest,
    ) -> dict:
        """Single-shot rescue call to OpenRouter's Anthropic Messages
        endpoint after the main 4141 path has exhausted its retries.
        """
        sess = self._get_fallback_session()
        if sess is None:
            raise RuntimeError("anthropic fallback unavailable (missing provider credential)")
        payload = to_anthropic(req).model_dump(exclude_none=True)
        anth_msgs = payload.get("messages") or []
        cleaned, dropped_uses, dropped_results = (
            self._sanitize_anthropic_messages_for_strict_validator(anth_msgs)
        )
        if dropped_uses or dropped_results:
            logger.info(
                "OpenRouter fallback sanitizer dropped %d orphan tool_use(s) and "
                "%d orphan tool_result(s) (msgs %d -> %d)",
                dropped_uses, dropped_results, len(anth_msgs), len(cleaned),
            )
            payload["messages"] = cleaned
        anth_tools = payload.get("tools")
        cleaned_tools, dropped_combinators = (
            self._sanitize_anthropic_tools_for_strict_validator(anth_tools)
        )
        if dropped_combinators:
            logger.info(
                "OpenRouter fallback tool sanitizer dropped %d top-level "
                "combinator(s) from tool schema(s)",
                dropped_combinators,
            )
            payload["tools"] = cleaned_tools
        payload["model"] = self.fallback_model
        payload["stream"] = True
        url = f"{self.fallback_base_url}/messages"
        logger.info(
            "falling back to OpenRouter (model=%s) for failed Anthropic call",
            self.fallback_model,
        )
        resp = sess.post(
            url, json=payload,
            timeout=(self.connect_timeout, self.read_timeout),
            stream=True,
        )
        if resp.status_code >= 400:
            body_preview = ""
            try:
                body_preview = resp.text[:600]
            except Exception:
                pass
            logger.warning(
                "OpenRouter fallback HTTP %s body=%s",
                resp.status_code, body_preview,
            )
        resp.raise_for_status()
        raw = read_anthropic_stream(resp)
        normalized = from_anthropic(
            AnthropicResponse.model_validate(raw),
            model=self.fallback_model or self.model,
        ).model_dump(exclude_none=True)
        if _is_empty_completion(normalized):
            raise RuntimeError(
                f"empty choices from OpenRouter fallback: {str(raw)[:200]}"
            )
        logger.info("OpenRouter fallback succeeded (model=%s)", self.fallback_model)
        return normalized

    # ---- session + cache-bust + history-shrink -----------------------------

    def _reset_session(self) -> None:
        """Drop the keep-alive pool — needed after RemoteDisconnected /
        ConnectionError because the cached socket is dead and reuse will
        immediately fail again."""
        try:
            self.session.close()
        except Exception as exc:
            logger.debug("session.close() failed: %s", exc, exc_info=True)
        self.session = requests.Session()
        _apply_auth(self.session, self.base_url)

    def _is_empty_choices_error(self, err: BaseException | None) -> bool:
        return isinstance(err, RuntimeError) and "empty choices" in str(err)

    def _bust_cache(self, messages: list[dict], attempt: int) -> list[dict]:
        """Prepend an invisible marker to the SYSTEM message to break upstream
        prompt-prefix caching.

        Called only on retry, so happy-path cache hits are preserved.
        """
        marker = f"​<retry-cache-bust {attempt} {int(time.time())}​>"
        out = list(messages)
        for i, m in enumerate(out):
            if m.get("role") != "system":
                continue
            new_m = dict(m)
            content = new_m.get("content")
            if isinstance(content, str):
                new_m["content"] = marker + "\n" + content
            elif isinstance(content, list):
                new_m["content"] = [{"type": "text", "text": marker}] + list(content)
            else:
                new_m["content"] = marker
            out[i] = new_m
            return out
        out.insert(0, {"role": "system", "content": marker})
        return out

    @staticmethod
    def _drop_orphan_tool_pairs_openai(messages: list[dict]) -> list[dict]:
        """Remove orphaned tool_call / tool_result pairs from OpenAI-shaped
        messages so a truncated history doesn't 400 the API.

        Drops (1) `tool` messages whose `tool_call_id` has no preceding
        assistant `tool_calls` entry in this list, and (2) `tool_calls`
        entries (or the whole assistant message) whose ids are never
        answered by a following `tool` message. Also drops malformed
        `tool_calls` whose `function.arguments` are not valid JSON; vLLM's
        chat renderer parses historical arguments with `json.loads`, so
        replaying a model-emitted malformed call turns the next request into
        an unrecoverable HTTP 400.
        """
        # Normalize typed (pydantic) message objects to plain dicts first.
        # `ChatRequest.messages` elements may be Model instances (e.g. the
        # reasoning-truncation loop passes `req.messages` straight in), and
        # this filter uses dict access throughout. (Codex review 2026-07-18.)
        normalized: list[dict] = []
        for m in messages:
            if isinstance(m, dict):
                normalized.append(m)
            elif hasattr(m, "model_dump"):
                normalized.append(m.model_dump(exclude_none=True))
            else:
                normalized.append(dict(m))
        messages = normalized

        def _tool_call_valid(tc: Any) -> bool:
            if not isinstance(tc, dict):
                return False
            fn = tc.get("function")
            if not isinstance(fn, dict):
                return False
            args = fn.get("arguments")
            if not isinstance(args, str):
                return False
            try:
                json.loads(args)
            except Exception:
                return False
            return True

        # First pass: collect tool_call ids emitted by assistants and ids
        # answered by tool messages.
        answered: set[str] = set()
        for m in messages:
            if m.get("role") == "tool":
                tcid = m.get("tool_call_id")
                if tcid:
                    answered.add(tcid)
        cleaned: list[dict] = []
        emitted: set[str] = set()
        for m in messages:
            role = m.get("role")
            if role == "assistant" and m.get("tool_calls"):
                kept_calls = []
                for tc in m["tool_calls"]:
                    tcid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    if tcid and tcid in answered and _tool_call_valid(tc):
                        kept_calls.append(tc)
                        emitted.add(tcid)
                if kept_calls:
                    nm = dict(m)
                    nm["tool_calls"] = kept_calls
                    cleaned.append(nm)
                elif m.get("content"):
                    # Keep the text, drop the now-empty tool_calls field.
                    nm = {k: v for k, v in m.items() if k != "tool_calls"}
                    cleaned.append(nm)
                # else: assistant had only orphaned tool_calls — drop entirely.
            elif role == "tool":
                tcid = m.get("tool_call_id")
                if tcid and tcid in emitted:
                    cleaned.append(m)
                # else: orphaned tool result — drop.
            else:
                cleaned.append(m)
        return cleaned

    def _shrink_history(self, messages: list[dict]) -> list[dict]:
        """Aggressive last-resort: keep system + first user (target) + last 6
        messages. Used only if cache-busting alone didn't recover."""
        if len(messages) <= 8:
            return self._drop_orphan_tool_pairs_openai(list(messages))
        kept: list[dict] = []
        sys_msg = next((m for m in messages if m.get("role") == "system"), None)
        first_user = next((m for m in messages if m.get("role") == "user"), None)
        if sys_msg:
            kept.append(sys_msg)
        if first_user and first_user is not sys_msg:
            kept.append(first_user)
        kept.append({"role": "user", "content": (
            "[CONTEXT SHRUNK after repeated empty-choices upstream errors. "
            "Earlier exploration is dropped. Continue from current state.]"
        )})
        tail = messages[-6:]
        while tail and tail[0].get("role") == "tool":
            tail = tail[1:]
        kept.extend(tail)
        # The first-user message may itself be an assistant-with-tool_calls
        # whose results landed in the dropped middle, and the tail may carry
        # tool results whose calls were dropped. Strip any such orphans so the
        # shrunk payload validates on BOTH the main and fallback paths.
        return self._drop_orphan_tool_pairs_openai(kept)

    # ---- /v1/chat/completions path -----------------------------------------

    def _grow_budget_for_truncation(
        self, payload: dict, prompt_tokens: int
    ) -> int | None:
        """Compute a larger output budget for a reasoning-truncation retry.

        Returns a new `max_tokens` strictly greater than the current one,
        bounded by real context headroom
        (``context_window - prompt_tokens - margin``), or ``None`` when
        growth is impossible/pointless — in which case the caller falls
        back to the normal empty-choices backoff path. ``None`` is returned
        if: the client has no known `context_window` (e.g. gpt-5.5 — its
        truncation is handled on the Responses path, not here), usage gave
        no `prompt_tokens` to bound against, or the prompt already fills the
        window so no extra output room exists (history compaction, not a
        bigger budget, is the right move — handled by the recovery path).
        """
        if not self.context_window or prompt_tokens <= 0:
            return None
        current = int(payload.get("max_tokens") or self.max_tokens)
        margin = 2048  # keep input+output strictly under the window
        headroom = self.context_window - prompt_tokens - margin
        if headroom <= current:
            return None
        # Escalate geometrically (at least +4096), capped at headroom, so a
        # case needing the full window converges in a few deterministic
        # steps (e.g. 8192->16384->32768->headroom) rather than coin-flipping
        # the same budget up to `max_retries` times.
        grown = min(headroom, max(current * 2, current + 4096))
        return grown if grown > current else None

    def _post_chat_completions(self, req: ChatRequest, messages: list[dict]) -> dict:
        """POST a `/v1/chat/completions` payload with retry + empty-choices
        recovery. Returns the (already OpenAI-shaped) dict verbatim — no
        translation needed."""
        url = f"{self.base_url}/chat/completions"
        messages = self._drop_orphan_tool_pairs_openai(list(messages))
        req = _build_chat_request(
            model=req.model,
            messages=messages,
            tools=[t.model_dump(exclude_none=True) for t in req.tools] or None,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            reasoning_effort=req.reasoning_effort,
        )
        payload = to_chat_completions(req).model_dump(exclude_none=True)
        # Stream by default: long substrate/chat-completions outputs can
        # sit idle long enough for the backend/proxy to kill buffered calls.
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

        last_error = None
        # Track an escalating output budget across attempts: a reasoning
        # model that truncates mid-thought (finish_reason=length, empty
        # content, non-empty reasoning) needs MORE output budget, not an
        # identical retry. We grow `payload["max_tokens"]` toward real
        # context headroom instead of backing off and re-truncating.
        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(
                    url, json=payload,
                    timeout=(self.connect_timeout, self.read_timeout),
                    stream=True,
                )
                resp.raise_for_status()
                data = read_chat_completions_stream(resp, model=self.model)
                if _is_empty_completion(data):
                    is_trunc, prompt_tokens = _empty_is_reasoning_truncation(data)
                    grown = self._grow_budget_for_truncation(
                        payload, prompt_tokens) if is_trunc else None
                    if grown is not None:
                        # Deterministic fix: the model exhausted its output
                        # budget inside the reasoning block. Re-request with a
                        # larger budget and NO backoff sleep — this is not a
                        # flaky-proxy empty, so waiting buys nothing.
                        logger.warning(
                            "chat empty=reasoning-truncation (attempt %d/%d): "
                            "prompt=%d, max_tokens %d->%d; retrying immediately",
                            attempt + 1, self.max_retries, prompt_tokens,
                            payload["max_tokens"], grown)
                        payload["max_tokens"] = grown
                        self._reset_session()
                        continue
                    last_error = RuntimeError(
                        f"empty choices in response: {str(data)[:200]}")
                    wait = min(30.0, self.retry_delay * (2 ** attempt))
                    logger.warning(
                        "chat empty choices (attempt %d/%d). Retrying in %.1fs",
                        attempt + 1, self.max_retries, wait)
                    self._reset_session()
                    time.sleep(wait)
                    continue
                return data
            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
                self._reset_session()
                wait = min(30.0, self.retry_delay * (2 ** attempt))
                logger.warning("Request failed (attempt %d/%d): %s. Retrying in %.1fs",
                              attempt + 1, self.max_retries, e, wait)
                time.sleep(wait)
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status is None or status in (408, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529):
                    if status is not None and status >= 500:
                        self._reset_session()
                    last_error = e
                    wait = min(30.0, self.retry_delay * (2 ** attempt))
                    logger.warning("HTTP %s (attempt %d/%d): %s. Retrying in %.1fs",
                                  status, attempt + 1, self.max_retries, e, wait)
                    time.sleep(wait)
                else:
                    raise
            except requests.RequestException as e:
                last_error = e
                self._reset_session()
                wait = min(30.0, self.retry_delay * (2 ** attempt))
                logger.warning("Request error (attempt %d/%d): %s. Retrying in %.1fs",
                              attempt + 1, self.max_retries, e, wait)
                time.sleep(wait)
        return self._post_chat_with_recovery(url, payload, messages, req, last_error)

    def _post_chat_with_recovery(
        self,
        url: str,
        payload: dict,
        messages: list[dict],
        req: ChatRequest,
        last_error: BaseException | None,
    ) -> dict:
        """Last-ditch recovery for upstream-cached empty-choices responses.
        Two extra attempts: cache-bust marker, then compact history."""
        if not self._is_empty_choices_error(last_error):
            raise last_error  # type: ignore
        for attempt, transform in enumerate((
            lambda m: self._bust_cache(m, attempt=1),
            lambda m: self._bust_cache(self._shrink_history(m), attempt=2),
        ), start=1):
            try:
                new_msgs = transform(messages)
                # Re-translate with the new messages.
                new_req = _build_chat_request(
                    model=req.model,
                    messages=new_msgs,
                    tools=[t.model_dump(exclude_none=True) for t in req.tools] or None,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    reasoning_effort=req.reasoning_effort,
                )
                new_payload = to_chat_completions(new_req).model_dump(exclude_none=True)
                new_payload["stream"] = True
                new_payload["stream_options"] = {"include_usage": True}
                logger.warning(
                    "LLM empty-choices recovery attempt %d (msgs %d -> %d)",
                    attempt, len(messages), len(new_msgs),
                )
                self._reset_session()
                resp = self.session.post(
                    url, json=new_payload,
                    timeout=(self.connect_timeout, self.read_timeout),
                    stream=True,
                )
                resp.raise_for_status()
                data = read_chat_completions_stream(resp, model=self.model)
                if not _is_empty_completion(data):
                    logger.info("LLM empty-choices recovery attempt %d succeeded", attempt)
                    return data
                last_error = RuntimeError(
                    f"empty choices after recovery attempt {attempt}: {str(data)[:200]}")
            except Exception as e:
                last_error = e
                logger.warning("LLM empty-choices recovery attempt %d raised: %s", attempt, e)
        raise last_error  # type: ignore

    # ---- /v1/messages (Anthropic) path -------------------------------------

    def _post_anthropic_with_retry(self, req: ChatRequest, messages: list[dict]) -> dict:
        """POST an Anthropic Messages payload with retry / empty-choices
        recovery / 502 cache-bust escalation."""
        url = f"{self.base_url}/messages"
        # Drop orphaned tool_calls/tool_result pairs BEFORE translation. The
        # happy path translates verbatim and HTTP 400 is not retriable here,
        # so an orphan produced upstream (e.g. by history compaction) would
        # otherwise crash the run with an unrecoverable 400. This guard makes
        # the main path as robust as the shrink / fallback paths.
        messages = self._drop_orphan_tool_pairs_openai(list(messages))
        req = _build_chat_request(
            model=req.model,
            messages=messages,
            tools=[t.model_dump(exclude_none=True) for t in req.tools] or None,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            reasoning_effort=req.reasoning_effort,
        )
        payload = to_anthropic(req).model_dump(exclude_none=True)
        # Stream by default: long reasoning outputs hold the connection
        # idle past intermediate-proxy read timeouts when buffered.
        payload["stream"] = True

        last_error: BaseException | None = None
        empty_count = 0
        cur_messages = messages
        cur_payload = payload
        cur_req = req
        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(
                    url, json=cur_payload,
                    timeout=(self.connect_timeout, self.read_timeout),
                    stream=True,
                )
                resp.raise_for_status()
                raw = read_anthropic_stream(resp)
                normalized = from_anthropic(
                    AnthropicResponse.model_validate(raw), model=self.model
                ).model_dump(exclude_none=True)
                # Surface extended-thinking text for transcript logging.
                # AnthropicResponse drops `thinking` blocks (only text/tool_use
                # are modelled), so pull them straight off the raw stream dict
                # and attach to the normalized assistant message. Logging only
                # — never re-sent in the message stream.
                try:
                    think_parts = [
                        b.get("thinking", "")
                        for b in (raw.get("content") or [])
                        if isinstance(b, dict) and b.get("type") == "thinking"
                    ]
                    think_text = "".join(p for p in think_parts if p)
                    if think_text and normalized.get("choices"):
                        normalized["choices"][0].setdefault("message", {})
                        normalized["choices"][0]["message"]["reasoning_content"] = think_text
                except Exception:
                    pass
                if _is_empty_completion(normalized):
                    empty_count += 1
                    last_error = RuntimeError(
                        f"empty choices in anthropic response: {str(raw)[:200]}")
                    wait = min(30.0, self.retry_delay * (2 ** attempt))
                    logger.warning(
                        "anthropic chat empty content (attempt %d/%d, empty_count=%d). Retrying in %.1fs",
                        attempt + 1, self.max_retries, empty_count, wait)
                    try:
                        if empty_count >= 2:
                            new_msgs = self._bust_cache(
                                self._shrink_history(cur_messages),
                                attempt=empty_count,
                            )
                        else:
                            new_msgs = self._bust_cache(cur_messages, attempt=empty_count)
                        cur_req = _build_chat_request(
                            model=req.model,
                            messages=new_msgs,
                            tools=[t.model_dump(exclude_none=True) for t in req.tools] or None,
                            temperature=req.temperature,
                            max_tokens=req.max_tokens,
                            reasoning_effort=req.reasoning_effort,
                        )
                        cur_payload = to_anthropic(cur_req).model_dump(exclude_none=True)
                        cur_messages = new_msgs
                        logger.info(
                            "anthropic empty-choices recovery: cache-bust attempt %d (msgs %d -> %d)",
                            empty_count, len(messages), len(new_msgs),
                        )
                    except Exception as transform_err:
                        logger.warning(
                            "anthropic cache-bust transform failed: %s — retrying identical payload",
                            transform_err,
                        )
                    self._reset_session()
                    time.sleep(wait)
                    continue
                if empty_count > 0:
                    logger.info(
                        "anthropic empty-choices recovery succeeded after %d empty attempt(s)",
                        empty_count,
                    )
                return normalized
            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
                self._reset_session()
                wait = min(30.0, self.retry_delay * (2 ** attempt))
                logger.warning(
                    "anthropic request failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1, self.max_retries, e, wait)
                time.sleep(wait)
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status is not None and 400 <= status < 500:
                    body = ""
                    try:
                        body = e.response.text[:600] if e.response is not None else ""
                    except Exception:
                        pass
                    logger.error(
                        "anthropic HTTP %s body=%s payload=%s",
                        status, body, json.dumps(payload, default=str)[:4000])
                elif status is not None and status >= 500 and attempt == 0:
                    # 5xx bodies are otherwise never surfaced; log once (first
                    # attempt only) with payload size so a persistent 500 can be
                    # diagnosed instead of silently retried into exhaustion.
                    body = ""
                    try:
                        body = e.response.text[:800] if e.response is not None else ""
                    except Exception:
                        pass
                    logger.error(
                        "anthropic HTTP %s body=%s payload_bytes=%d n_msgs=%d",
                        status, body,
                        len(json.dumps(cur_payload, default=str)),
                        len(cur_payload.get("messages", [])))
                if status is None or status in (408, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529):
                    if status is not None and status >= 500:
                        self._reset_session()
                    last_error = e
                    wait = min(30.0, self.retry_delay * (2 ** attempt))
                    logger.warning(
                        "anthropic HTTP %s (attempt %d/%d). Retrying in %.1fs",
                        status, attempt + 1, self.max_retries, wait)
                    is_likely_ghost_502 = (status == 502)
                    if is_likely_ghost_502:
                        try:
                            empty_count += 1
                            if empty_count >= 2:
                                new_msgs = self._bust_cache(
                                    self._shrink_history(cur_messages),
                                    attempt=empty_count,
                                )
                            else:
                                new_msgs = self._bust_cache(
                                    cur_messages, attempt=empty_count,
                                )
                            cur_req = _build_chat_request(
                                model=req.model,
                                messages=new_msgs,
                                tools=[t.model_dump(exclude_none=True) for t in req.tools] or None,
                                temperature=req.temperature,
                                max_tokens=req.max_tokens,
                                reasoning_effort=req.reasoning_effort,
                            )
                            cur_payload = to_anthropic(cur_req).model_dump(exclude_none=True)
                            cur_messages = new_msgs
                            logger.info(
                                "anthropic 502 recovery: cache-bust attempt %d (msgs %d -> %d)",
                                empty_count, len(messages), len(new_msgs),
                            )
                        except Exception as transform_err:
                            logger.warning(
                                "anthropic 502 cache-bust transform failed: %s — retrying identical payload",
                                transform_err,
                            )
                    time.sleep(wait)
                else:
                    raise
            except requests.RequestException as e:
                last_error = e
                self._reset_session()
                wait = min(30.0, self.retry_delay * (2 ** attempt))
                logger.warning(
                    "anthropic request error (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1, self.max_retries, e, wait)
                time.sleep(wait)
        if last_error is None:
            raise RuntimeError("anthropic chat exhausted retries with no error captured")
        raise last_error

    # ---- /v1/responses (gpt-5.5) path --------------------------------------

    def _canonical_fingerprint(self, obj: Any) -> str:
        data = json.dumps(
            obj,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def _normalise_assistant_snapshot(self, msg: dict | None) -> dict[str, Any] | None:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            return None
        snap: dict[str, Any] = {"role": "assistant"}
        content = msg.get("content")
        if content:
            snap["content"] = content
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            snap["tool_calls"] = tool_calls
        return snap if len(snap) > 1 else None

    def _clear_responses_state(self) -> None:
        self._responses_previous_response_id = None
        self._responses_request_message_count = 0
        self._responses_request_fingerprint = None
        self._responses_assistant_snapshot = None

    def _responses_state_ready(self) -> bool:
        return (
            self.responses_state == "auto"
            and not self._responses_native_state_disabled
            and bool(self._responses_previous_response_id)
            and bool(self._responses_request_fingerprint)
            and self._responses_assistant_snapshot is not None
            and self._responses_request_message_count > 0
        )

    def _responses_delta_messages(self, messages: list[dict]) -> list[dict] | None:
        """Return only the new user/tool messages after the last Responses
        output, or None when the local history no longer matches the state.

        The agent stores a full chat-style transcript locally. With
        previous_response_id, the next Responses request should not replay
        the previous assistant function_call; it only needs the new
        function_call_output / user messages appended after it.
        """
        if not self._responses_state_ready():
            return None
        n = self._responses_request_message_count
        if len(messages) <= n:
            return None
        if self._canonical_fingerprint(messages[:n]) != self._responses_request_fingerprint:
            return None
        assistant = self._normalise_assistant_snapshot(messages[n])
        if (
            assistant is None
            or self._canonical_fingerprint(assistant)
            != self._canonical_fingerprint(self._responses_assistant_snapshot)
        ):
            return None
        delta = list(messages[n + 1:])
        return delta or None

    def _responses_payload_for_messages(
        self,
        req: ChatRequest,
        messages: list[dict],
        *,
        use_state: bool,
    ) -> tuple[dict, bool]:
        """Build either a full replay payload or a previous_response_id
        delta payload. The bool says whether native state was used.
        """
        if use_state and self._responses_state_ready():
            delta = self._responses_delta_messages(messages)
            if delta:
                delta_req = _build_chat_request(
                    model=req.model,
                    messages=delta,
                    tools=[t.model_dump(exclude_none=True) for t in req.tools] or None,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    reasoning_effort=req.reasoning_effort,
                )
                payload = to_responses(
                    delta_req,
                    previous_response_id=self._responses_previous_response_id,
                    store=True,
                ).model_dump(exclude_none=True)
                return payload, True
            # Local history changed shape (usually compaction); the next
            # successful full replay becomes the new native-state anchor.
            self._clear_responses_state()
        store = True if (use_state and not self._responses_store_unsupported) else None
        payload = to_responses(req, store=store).model_dump(exclude_none=True)
        return payload, False

    def _capture_responses_state(self, messages: list[dict], normalized: dict) -> None:
        if self.responses_state != "auto" or self._responses_native_state_disabled:
            return
        resp_id = normalized.get("id") or ""
        choices = normalized.get("choices") or []
        if not resp_id or not choices:
            self._clear_responses_state()
            return
        msg = (choices[0] or {}).get("message") or {}
        snapshot = self._normalise_assistant_snapshot(msg)
        if snapshot is None:
            self._clear_responses_state()
            return
        self._responses_previous_response_id = str(resp_id)
        self._responses_request_message_count = len(messages)
        self._responses_request_fingerprint = self._canonical_fingerprint(messages)
        self._responses_assistant_snapshot = snapshot

    def _is_previous_response_error(self, status: int | None, body: str) -> bool:
        if status not in (400, 404, 409):
            return False
        text = (body or "").lower()
        return (
            "previous_response_id" in text
            or "previous response" in text
            or "not found" in text
            or "expired" in text
            or "state" in text
        )

    def _is_store_unsupported_error(self, status: int | None, body: str) -> bool:
        if status != 400:
            return False
        text = (body or "").lower()
        return "store" in text and (
            "not supported" in text
            or "unsupported" in text
            or "unsupported_value" in text
        )

    def _bust_cache_responses(self, payload: dict, attempt: int) -> dict:
        """Cache-bust for Responses: there is no system message, so we
        prepend an invisible marker to `instructions` instead.
        """
        marker = f"​<retry-cache-bust {attempt} {int(time.time())}​>"
        out = dict(payload)
        out["instructions"] = marker + "\n" + (out.get("instructions") or "")
        return out

    @staticmethod
    def _with_responses_effort(payload: dict, effort: str) -> dict:
        """Return a copy of a Responses payload with reasoning effort changed.

        Used to de-escalate effort after a truncated stream so the next
        attempt thinks for less wall-clock and keeps the SSE idle gap under
        the intermediate proxy's read timeout.
        """
        out = dict(payload)
        reasoning = dict(out.get("reasoning") or {})
        reasoning["effort"] = effort
        out["reasoning"] = reasoning
        out["stream"] = True
        return out

    def _post_responses_with_retry(
        self,
        req: ChatRequest,
        messages: list[dict],
        *,
        use_state: bool,
    ) -> dict:
        """POST a `/v1/responses` payload with retry + cache-bust escalation.

        Mirrors `_post_anthropic_with_retry` but:
          • the cache-bust nonce goes into `instructions`, not a system msg
          • no history-shrink (gpt-5.5 has ample context; not a failure mode)
          • no OpenRouter fallback (Anthropic-only by user decision)
        """
        url = f"{self.base_url}/responses"
        full_payload = to_responses(req, store=True).model_dump(exclude_none=True)
        stateless_full_payload = to_responses(req, store=None).model_dump(exclude_none=True)
        payload, used_native_state = self._responses_payload_for_messages(
            req, messages, use_state=use_state,
        )
        # Stream gpt-5.5: high reasoning_effort responses can idle the
        # connection long enough for upstream proxies to drop it.
        payload["stream"] = True
        full_payload["stream"] = True
        stateless_full_payload["stream"] = True

        last_error: BaseException | None = None
        empty_count = 0
        truncated_count = 0
        history_shrunk = False
        # Start from the persistent ceiling if a prior turn already de-escalated
        # due to truncation; otherwise the request's own effort. If the ceiling
        # is stronger-or-equal to the request, the request wins (don't raise).
        cur_effort = req.reasoning_effort
        if self._responses_effort_ceiling is not None:
            ceil = self._responses_effort_ceiling
            try:
                if _EFFORT_LADDER.index(ceil) > _EFFORT_LADDER.index(cur_effort):
                    cur_effort = ceil
            except ValueError:
                pass
        # If the ceiling lowered our effort below what the payload was built
        # with, rebuild the payload to match before the first attempt.
        if cur_effort != req.reasoning_effort:
            payload = self._with_responses_effort(payload, cur_effort)
        cur_payload = payload
        cur_used_native_state = used_native_state
        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(
                    url, json=cur_payload,
                    timeout=(self.connect_timeout, self.read_timeout),
                    stream=True,
                )
                resp.raise_for_status()
                raw = read_responses_stream(resp)
                normalized = from_responses(
                    ResponsesResponse.model_validate(raw), model=self.model
                ).model_dump(exclude_none=True)
                if _is_empty_completion(normalized):
                    empty_count += 1
                    diag = ""
                    raw_status = _responses_status(raw)
                    if isinstance(raw, dict):
                        if raw_status:
                            diag += f" status={raw_status}"
                        inc = raw.get("incomplete_details")
                        if inc:
                            diag += f" incomplete_details={inc}"
                        usg = raw.get("usage") or {}
                        if isinstance(usg, dict):
                            ot = usg.get("output_tokens_details") or {}
                            rt_n = ot.get("reasoning_tokens") if isinstance(ot, dict) else None
                            if rt_n is not None:
                                diag += f" reasoning_tokens={rt_n}"
                            ot_n = usg.get("output_tokens")
                            if ot_n is not None:
                                diag += f" output_tokens={ot_n}"
                    last_error = RuntimeError(
                        f"empty choices in responses response:{diag} raw_head={str(raw)[:160]}")

                    if _is_truncated_stream(raw, normalized):
                        # Stream was SEVERED mid-reasoning (status != completed),
                        # not a genuine empty answer. Root cause (reproduced
                        # 2026-05-30): high reasoning_effort idles the SSE long
                        # enough that an intermediate proxy's ~180s read timeout
                        # cuts it. A 120s proxy-outage backoff is exactly wrong
                        # here — the proxy is fine, the model just thought too
                        # long between frames. De-escalate the reasoning effort
                        # (shorter think → smaller idle gap), and once effort
                        # bottoms out, shrink the input so there's less to reason
                        # over. Reconnect fast (no long backoff).
                        truncated_count += 1
                        lowered = _next_lower_effort(cur_effort)
                        action = ""
                        if lowered is not None:
                            cur_effort = lowered
                            # Latch the ceiling so later turns in this session
                            # start here instead of re-paying the truncation tax.
                            self._responses_effort_ceiling = lowered
                            cur_payload = self._with_responses_effort(cur_payload, cur_effort)
                            # Native-state delta payloads can't safely change
                            # effort mid-thread; fall back to a full stateless
                            # replay carrying the lowered effort.
                            if cur_used_native_state:
                                self._clear_responses_state()
                                cur_payload = self._with_responses_effort(
                                    dict(stateless_full_payload), cur_effort)
                                cur_used_native_state = False
                            action = f"lowering reasoning_effort→{cur_effort}"
                        elif not history_shrunk:
                            # Effort already at floor — shrink the conversation
                            # so the model has less to chew on before answering.
                            shrunk = self._shrink_history(messages)
                            shrink_req = _build_chat_request(
                                model=req.model, messages=shrunk,
                                tools=[t.model_dump(exclude_none=True) for t in req.tools] or None,
                                temperature=req.temperature, max_tokens=req.max_tokens,
                                reasoning_effort=cur_effort,
                            )
                            cur_payload = to_responses(
                                shrink_req, store=None).model_dump(exclude_none=True)
                            cur_payload["stream"] = True
                            self._clear_responses_state()
                            cur_used_native_state = False
                            history_shrunk = True
                            action = f"shrinking history ({len(messages)}→{len(shrunk)} msgs)"
                        else:
                            action = "effort+history floored; reconnecting"
                        wait = min(5.0, self.retry_delay * truncated_count)
                        logger.warning(
                            "responses TRUNCATED stream (attempt %d/%d, truncated=%d).%s "
                            "%s. Fast-reconnect in %.1fs",
                            attempt + 1, self.max_retries, truncated_count, diag,
                            action, wait)
                        self._reset_session()
                        time.sleep(wait)
                        continue

                    # Genuine empty (status=completed but no content/tools) or
                    # status unknown — the original cache-bust + backoff path.
                    wait = min(30.0, self.retry_delay * (2 ** attempt))
                    logger.warning(
                        "responses empty output (attempt %d/%d, empty_count=%d).%s Retrying in %.1fs",
                        attempt + 1, self.max_retries, empty_count, diag, wait)
                    try:
                        cur_payload = self._bust_cache_responses(cur_payload, attempt=empty_count)
                    except Exception as transform_err:
                        logger.warning(
                            "responses cache-bust transform failed: %s — retrying identical payload",
                            transform_err,
                        )
                    self._reset_session()
                    time.sleep(wait)
                    continue
                if empty_count > 0:
                    logger.info(
                        "responses empty-output recovery succeeded after %d empty attempt(s)",
                        empty_count,
                    )
                # Only re-anchor native Responses state when THIS successful
                # request was actually persisted server-side (store=True). The
                # truncation-recovery branches fall back to a stateless replay
                # (effort de-escalation) or shrink the history — both send
                # store=None, so the server neither persisted the response nor
                # saw the full transcript. Capturing previous_response_id from
                # such a response would make the NEXT delta silently omit the
                # dropped middle history (the client would believe the prior
                # response saw everything). In every store=True path — native
                # delta or full replay — the server's cumulative view equals
                # `messages`, so capturing against the full transcript is right.
                if use_state and cur_payload.get("store") is True:
                    self._capture_responses_state(messages, normalized)
                return normalized
            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
                self._reset_session()
                wait = min(30.0, self.retry_delay * (2 ** attempt))
                logger.warning(
                    "responses request failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1, self.max_retries, e, wait)
                time.sleep(wait)
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                body = ""
                if status is not None and 400 <= status < 500:
                    try:
                        body = e.response.text[:600] if e.response is not None else ""
                    except Exception:
                        pass
                    logger.error(
                        "responses HTTP %s body=%s payload=%s",
                        status, body, json.dumps(cur_payload, default=str)[:400])
                if self._is_store_unsupported_error(status, body):
                    logger.warning(
                        "Responses store parameter is unsupported by this proxy; "
                        "disabling native Responses state and retrying without store",
                    )
                    self._responses_store_unsupported = True
                    self._responses_native_state_disabled = True
                    self._clear_responses_state()
                    cur_payload = dict(stateless_full_payload)
                    cur_used_native_state = False
                    empty_count = 0
                    self._reset_session()
                    continue
                if cur_used_native_state and self._is_previous_response_error(status, body):
                    logger.warning(
                        "Responses previous_response_id failed (HTTP %s); "
                        "disabling native state for this LLMClient and retrying full replay",
                        status,
                    )
                    self._responses_native_state_disabled = True
                    self._clear_responses_state()
                    cur_payload = dict(stateless_full_payload)
                    cur_used_native_state = False
                    empty_count = 0
                    self._reset_session()
                    continue
                if status is None or status in (408, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529):
                    if status is not None and status >= 500:
                        self._reset_session()
                    last_error = e
                    wait = min(30.0, self.retry_delay * (2 ** attempt))
                    logger.warning(
                        "responses HTTP %s (attempt %d/%d). Retrying in %.1fs",
                        status, attempt + 1, self.max_retries, wait)
                    if status == 502:
                        try:
                            empty_count += 1
                            cur_payload = self._bust_cache_responses(cur_payload, attempt=empty_count)
                            logger.info(
                                "responses 502 recovery: cache-bust attempt %d",
                                empty_count,
                            )
                        except Exception as transform_err:
                            logger.warning(
                                "responses 502 cache-bust transform failed: %s — retrying identical payload",
                                transform_err,
                            )
                    time.sleep(wait)
                else:
                    raise
            except requests.RequestException as e:
                last_error = e
                self._reset_session()
                wait = min(30.0, self.retry_delay * (2 ** attempt))
                logger.warning(
                    "responses request error (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1, self.max_retries, e, wait)
                time.sleep(wait)
        if last_error is None:
            raise RuntimeError("responses exhausted retries with no error captured")
        raise last_error

    # ---- Public chat API ---------------------------------------------------

    def _chat_dispatch(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        temperature: float,
        max_tokens: int,
        reasoning_effort: str,
        use_responses_state: bool,
    ) -> dict:
        """Shared dispatcher used by both `chat` and `chat_with_override`."""
        req = _build_chat_request(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        if _is_anthropic_model(self.model):
            logger.debug(
                "POST /messages model=%s msgs=%d (anthropic route)",
                self.model, len(messages),
            )
            try:
                return self._post_anthropic_with_retry(req, messages)
            except Exception as main_exc:
                if not self.fallback_enabled:
                    raise main_exc
                try:
                    return self._try_anthropic_fallback(req)
                except Exception as fb_exc:
                    logger.warning(
                        "OpenRouter fallback also failed: %s — re-raising main-path error",
                        fb_exc,
                    )
                    raise main_exc
        if _is_responses_model(self.model):
            logger.debug(
                "POST /responses model=%s msgs=%d effort=%s",
                self.model, len(messages), reasoning_effort,
            )
            return self._post_responses_with_retry(
                req,
                messages,
                use_state=use_responses_state and self.responses_state == "auto",
            )
        logger.debug(
            "POST /chat/completions model=%s msgs=%d",
            self.model, len(messages),
        )
        return self._post_chat_completions(req, messages)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Send a chat completion request. Returns the full response dict.

        Routes to `/v1/messages` for Claude models, `/v1/responses` for
        the gpt-5.5 family, and `/v1/chat/completions` otherwise. Output
        shape is always OpenAI chat-completion regardless of route.
        """
        return self._chat_dispatch(
            messages, tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            use_responses_state=True,
        )

    def chat_with_override(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        use_responses_state: bool = False,
    ) -> dict:
        """Thread-safe variant of `chat` with per-call overrides.

        The default is deliberately stateless: side LLM calls used for
        compaction or judging should not inherit or mutate the main
        GPT-5.5 Responses conversation.
        """
        return self._chat_dispatch(
            messages, tools,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            reasoning_effort=(
                reasoning_effort if reasoning_effort is not None else self.reasoning_effort
            ),
            use_responses_state=use_responses_state,
        )

    def chat_stream(self, messages: list[dict], tools: list[dict] | None = None):
        """Stream chat completions. Yields delta events."""
        return stream_chat_completion(
            base_url=self.base_url,
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def chat_with_tools(self, messages: list[dict], tools: list[dict] | None = None):
        """Higher-level helper: returns (content, tool_calls, finish_reason).

        Any model reasoning / thinking trace is stashed on
        ``self.last_reasoning`` (str | None) for transcript logging — it is
        deliberately NOT part of the return tuple so callers can't
        accidentally splice it back into the multi-turn message stream.
        """
        resp = self.chat(messages, tools)
        choices = resp.get("choices", [])
        content = None
        tool_calls = None
        finish_reason = ""
        reasoning = None
        for choice in choices:
            msg = choice.get("message", {})
            if msg.get("content"):
                content = msg["content"]
            if msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]
            if msg.get("reasoning_content"):
                reasoning = msg["reasoning_content"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
        self.last_reasoning = reasoning
        return content, tool_calls, finish_reason
