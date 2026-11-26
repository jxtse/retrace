"""Normalize Claude model IDs for explicitly configured Substrate endpoints."""

from __future__ import annotations

import logging
import re

from core.llm_client import LLMClient

logger = logging.getLogger(__name__)

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

    Uses the supplied endpoint and normalises the
    Claude model id to the hyphenated wire form before the base constructor
    runs (so the reasoning-tier warning and all wire methods see the final id).
    """

    def __init__(self, *, base_url: str | None = None, model: str, **kwargs):
        wire_model = _substrate_model_id(model)
        if wire_model != model:
            logger.info(
                "substrate: normalised model id %r -> %r for the port-3000 API",
                model,
                wire_model,
            )
        super().__init__(base_url=base_url, model=wire_model, **kwargs)


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
    return LLMClient(base_url=base_url, model=model, **kwargs)
