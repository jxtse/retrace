"""
Deprecated: context compaction shim.

The implementation moved to `services.autocompact`. This module is kept as
a thin re-export so that existing callers (e.g. `core/repl.py`) continue to
work without code changes. New code should import from `services.autocompact`
directly.
"""

from __future__ import annotations

import warnings
from typing import Any

from services.autocompact import (
    CompactionConfig,
    estimate_tokens,  # exact char/4 formula — bit-identical with legacy version
    maybe_compact,
)

__all__ = ["estimate_tokens", "compact_messages", "CompactionConfig"]

# Same char/token heuristic as the legacy module, exported for callers that
# imported it by name.
CHARS_PER_TOKEN = 4


def compact_messages(
    messages: list[dict],
    max_tokens: int = 100_000,
    keep_recent: int = 10,
    llm_client: Any = None,
) -> list[dict]:
    """
    Backward-compatible wrapper around `services.autocompact.maybe_compact`.

    Legacy signature: (messages, max_tokens, keep_recent, llm_client).
    Internally translated to a CompactionConfig with:
      - max_tokens               = max_tokens
      - keep_recent_tail         = keep_recent
      - auto_compact_threshold   = 1.0 (legacy compacted unconditionally once over budget)
      - micro_compact_threshold  = 1.0 (legacy had no micro stage)
    """
    warnings.warn(
        "services.compaction.compact_messages is deprecated; "
        "use services.autocompact.maybe_compact + CompactionConfig instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    cfg = CompactionConfig(
        max_tokens=max_tokens,
        keep_recent_tail=keep_recent,
        auto_compact_threshold=1.0,
        micro_compact_threshold=1.0,
    )
    return maybe_compact(messages, cfg=cfg, llm_client=llm_client)
