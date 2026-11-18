"""
Substrate LLM client — wraps the local "substrate" API on port 3000.

The substrate API (replacing the copilot-api reverse-tunnel at :4141) is a
drop-in OpenAI/Anthropic-compatible endpoint exposing the SAME three wire
protocols the base :class:`~core.llm_client.LLMClient` already routes to:

  - ``/v1/messages``          (Anthropic-native, Claude models — tool_use OK)
  - ``/v1/responses``         (gpt-5.5 family)
  - ``/v1/chat/completions``  (everything else)

So substrate is NOT a protocol rewrite. The only two things that differ from
the :4141 proxy and need handling here:

  1. **Model-id naming.** Substrate authorizes the *hyphenated* Claude id
     ``claude-opus-4-8`` and returns HTTP 403 ("User unauthorized for model
     'dev-anthropic-claude-opus-4.8'") on the *dotted* ``claude-opus-4.8``
     that the rest of the codebase / run scripts use. We normalise the dotted
     Claude id to the hyphenated wire form here. ``gpt-5.5`` stays dotted (it
     works as-is), and the routing predicates in ``core.llm_client``
     (``startswith("claude")`` / ``startswith("gpt-5.5")``) fire on both forms,
     so this only rewrites the *wire* model id, never the routing decision.

  2. **Base URL** is ``http://localhost:3000/v1``.

Everything else — streaming, truncation recovery, retry policy, auth headers —
is inherited unchanged from :class:`LLMClient`.
"""

from __future__ import annotations

import logging
import re

from core.llm_client import LLMClient

logger = logging.getLogger(__name__)

SUBSTRATE_BASE_URL = "http://localhost:4141/v1"

# Ports that host the substrate API (Anthropic-native + unwrapped-responses SSE
# shims). Historically :3000; the substrate moved to :4141 (2026-06-22) when the
# :3000 listener was retired. Both are recognised so make_llm_client() routes
# either to SubstrateLLMClient (which applies the model-id normalisation +
# 128k max_tokens clamp + unwrapped-responses handling).
_SUBSTRATE_PORTS = (":4141", ":3000")

# Matches a Claude model id with a dotted major.minor version, capturing the
# name prefix, the two version numbers, and any trailing suffix (e.g. -high).
#   claude-opus-4.8        -> ("claude-opus-", "4", "8", "")
#   claude-opus-4.8-xhigh  -> ("claude-opus-", "4", "8", "-xhigh")
#   claude-sonnet-4.6      -> ("claude-sonnet-", "4", "6", "")
_DOTTED_CLAUDE_RE = re.compile(r"^(claude-.*?)(\d+)\.(\d+)(.*)$", re.IGNORECASE)


def _substrate_model_id(model: str) -> str:
    """Normalise a Claude model id to the hyphenated form substrate authorizes.

    Only ``claude-*`` ids with a dotted major.minor version are rewritten
    (``claude-opus-4.8`` -> ``claude-opus-4-8``). ``gpt-5.5``, already-
    hyphenated Claude ids, and any non-Claude id pass through untouched.
    """
    if not model:
        return model
    m = _DOTTED_CLAUDE_RE.match(model)
    if not m:
        return model
    prefix, major, minor, suffix = m.groups()
    return f"{prefix}{major}-{minor}{suffix}"


class SubstrateLLMClient(LLMClient):
    """:class:`LLMClient` pinned to the substrate API on port 3000.

    Defaults ``base_url`` to :data:`SUBSTRATE_BASE_URL` and normalises the
    Claude model id to the hyphenated wire form before the base constructor
    runs (so the reasoning-tier warning and all wire methods see the final id).
    """

    def __init__(self, *, base_url: str | None = None, model: str, **kwargs):
        resolved_base = base_url or SUBSTRATE_BASE_URL
        wire_model = _substrate_model_id(model)
        if wire_model != model:
            logger.info(
                "substrate: normalised model id %r -> %r for the port-3000 API",
                model,
                wire_model,
            )
        super().__init__(base_url=resolved_base, model=wire_model, **kwargs)


def make_llm_client(*, base_url: str | None, model: str, **kwargs) -> LLMClient:
    """Construct the right LLM client for the configured channel.

    Returns a :class:`SubstrateLLMClient` when ``base_url`` points at the
    port-3000 substrate API, otherwise a plain :class:`LLMClient`. Centralising
    the choice here keeps every construction site (main agent client, sanity
    judge, feasibility judge) on the same channel without each duplicating the
    branch.
    """
    if base_url and any(p in base_url for p in _SUBSTRATE_PORTS):
        return SubstrateLLMClient(base_url=base_url, model=model, **kwargs)
    # Avoid passing base_url=None into LLMClient (it would .rstrip() a None);
    # let the base default apply when no base_url was resolved.
    if base_url is None:
        return LLMClient(model=model, **kwargs)
    return LLMClient(base_url=base_url, model=model, **kwargs)
