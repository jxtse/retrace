"""
Enhanced context compaction for long retrosynthesis conversations.

Inspired by Claude Code's dual-layer approach:
  - microCompact: cheap, deterministic cleanup of superseded tool-results
    (idempotent read-only tools whose old snapshots can be safely dropped).
  - autoCompact: LLM-driven summarization of the older half of the transcript
    once a token-budget threshold is crossed.

This module is designed to be drop-in alongside services/compaction.py.
Callers typically invoke `maybe_compact(messages, llm_client=...)` once per
agent iteration, just before sending the next LLM request.

Token accounting uses the same chars-per-token heuristic as compaction.py
for portability (no tokenizer dependency).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

# Same heuristic as services/compaction.py so the two modules agree.
CHARS_PER_TOKEN = 4

# ---------------------------------------------------------------------------
# Tool-result policy
# ---------------------------------------------------------------------------
# Tools whose output is idempotent / cheap to recompute / irrelevant once
# superseded. microCompact will replace their OLD tool_result messages with a
# short placeholder, preserving only the most-recent N occurrences per tool.
#
# Rules-of-thumb for inclusion:
#   - Deterministic on its input (same args -> same result)
#   - Safe to re-run if the LLM really needs the data again
#   - The result does not itself contain the final route / scoring verdict
DROPPABLE_TOOLS: set[str] = {
    "validate_smiles",
    "analyze_molecule",
    "compare_molecules",
    "identify_bonds",
    "retro_decompose",
    "apply_template",
    "check_buyability",
    # internal ML tools: their results get summarized into skill outputs
    "neural_retro",
    "_classify_reaction",
    "_recommend_enzyme",
    # precedent recall: route exemplars are inspirational at decision
    # time; once the LLM has acted on them they can be dropped under
    # pressure (full payload survives on disk via result_store).
    "recall_similar_routes",
}

# Tools whose results are load-bearing for the final answer — never drop.
PROTECTED_TOOLS: set[str] = {
    "score_route",
    "todo_write",             # plan state
    "todo_read",
}

# Per-tool: how many most-recent occurrences to keep intact.
KEEP_RECENT_PER_TOOL: dict[str, int] = {
    # molecules get re-validated a lot; keep more
    "validate_smiles": 4,
    "check_buyability": 4,
    # heavy outputs — keep recent picks intact so the LLM can cross-reference
    # multiple disconnection rounds; raised from 1→3 (neural_retro) /
    # 1→3 (recall_similar_routes) on 2026-05-18 — at compaction.max_tokens
    # 180k there's room, and dropping the second-most-recent neural_retro
    # was forcing redundant re-queries on branchy targets.
    "neural_retro": 3,
    "retro_decompose": 2,
    "analyze_molecule": 3,
    "recall_similar_routes": 3,
}
DEFAULT_KEEP_RECENT = 2


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------

def estimate_tokens(messages: Iterable[dict]) -> int:
    """Rough token estimate (chars/4). Matches services.compaction."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                total_chars += len(str(part))
        if "tool_calls" in msg and msg["tool_calls"]:
            try:
                total_chars += len(json.dumps(msg["tool_calls"]))
            except (TypeError, ValueError):
                total_chars += sum(len(str(tc)) for tc in msg["tool_calls"])
    return total_chars // CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CompactionConfig:
    """Controls when and how compaction triggers."""
    # Maximum tokens the context should occupy.
    max_tokens: int = 120_000
    # Fraction of max_tokens at which autoCompact runs (Claude-Code style 0.9).
    auto_compact_threshold: float = 0.85
    # Fraction at which microCompact runs (cheap, run earlier and more often).
    micro_compact_threshold: float = 0.60
    # How many tail messages to protect from any compaction.
    keep_recent_tail: int = 12
    # When autoCompact runs, how many messages at the head to keep verbatim
    # (in addition to the system message which is always kept).
    keep_head_messages: int = 2
    # Max length of each summarized block returned by the LLM.
    summary_max_tokens: int = 1500
    # If True, fall back to heuristic summarization when no llm_client is
    # provided. If False, skip the summary step entirely and only microCompact.
    allow_heuristic_summary: bool = True
    # Optional custom token estimator; if None, uses estimate_tokens().
    # Hook for future tiktoken/anthropic accurate counting.
    token_estimator: Callable[[list[dict]], int] | None = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _count_tokens(messages: list[dict], cfg: CompactionConfig) -> int:
    if cfg.token_estimator is not None:
        try:
            return int(cfg.token_estimator(messages))
        except Exception as e:
            logger.warning("custom token_estimator raised %s; falling back to heuristic", e)
    return estimate_tokens(messages)


def maybe_compact(
    messages: list[dict],
    cfg: CompactionConfig | None = None,
    llm_client: Any = None,
    trace: Callable[..., None] | None = None,
) -> list[dict]:
    """
    Run the appropriate compaction strategy based on current token usage.
    Returns a (possibly new) messages list. Safe to call every iteration.

    Order of escalation:
      1) tokens < micro_threshold  -> no-op
      2) micro_threshold <= tokens < auto_threshold -> microCompact
      3) tokens >= auto_threshold  -> microCompact then autoCompact
    """
    cfg = cfg or CompactionConfig()
    trace = trace or (lambda *a, **kw: None)

    est = _count_tokens(messages, cfg)
    micro_limit = int(cfg.max_tokens * cfg.micro_compact_threshold)
    auto_limit = int(cfg.max_tokens * cfg.auto_compact_threshold)

    if est < micro_limit:
        return messages

    original_count = len(messages)
    result = messages

    # Stage 1: microCompact — always try first, it's cheap.
    result = micro_compact(result, cfg=cfg)
    after_micro = _count_tokens(result, cfg)
    logger.info(
        "microCompact: %d msgs (~%d tok) -> %d msgs (~%d tok)",
        original_count, est, len(result), after_micro,
    )
    trace("micro_compact", before_tokens=est, after_tokens=after_micro,
          before_msgs=original_count, after_msgs=len(result))

    # Stage 2: autoCompact — only if still over the big threshold.
    if after_micro >= auto_limit:
        result = auto_compact(result, cfg=cfg, llm_client=llm_client, trace=trace)
        after_auto = _count_tokens(result, cfg)
        logger.info(
            "autoCompact: %d msgs (~%d tok) -> %d msgs (~%d tok)",
            len(messages), after_micro, len(result), after_auto,
        )
        trace("auto_compact", before_tokens=after_micro, after_tokens=after_auto,
              before_msgs=len(messages), after_msgs=len(result))

    return result


# ---------------------------------------------------------------------------
# microCompact: drop superseded tool results
# ---------------------------------------------------------------------------

_PLACEHOLDER = "[Old tool result cleared by microCompact — re-run tool if needed.]"


def _extract_tool_name_from_call(tool_calls: list) -> str | None:
    """Given an assistant message's tool_calls list, return the first tool name."""
    if not tool_calls:
        return None
    tc = tool_calls[0]
    # OpenAI dict-shape
    if isinstance(tc, dict):
        fn = tc.get("function") or tc
        name = fn.get("name") if isinstance(fn, dict) else None
        return name
    # SDK object-shape
    fn = getattr(tc, "function", None)
    return getattr(fn, "name", None) if fn is not None else getattr(tc, "name", None)


def _build_tool_call_index(messages: list[dict]) -> dict[str, str]:
    """Map tool_call_id -> tool_name by scanning assistant messages."""
    index: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tcs = msg.get("tool_calls") or []
        for tc in tcs:
            if isinstance(tc, dict):
                tc_id = tc.get("id")
                fn = tc.get("function") or {}
                name = fn.get("name") if isinstance(fn, dict) else None
            else:
                tc_id = getattr(tc, "id", None)
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) if fn is not None else None
            if tc_id and name:
                index[tc_id] = name
    return index


def micro_compact(messages: list[dict], cfg: CompactionConfig | None = None) -> list[dict]:
    """
    Replace content of old `tool` role messages for droppable tools with a
    placeholder, keeping the most recent N occurrences per tool intact.

    We never drop:
      - tool results inside the protected tail (last `keep_recent_tail` msgs)
      - tool results for PROTECTED_TOOLS
      - tool results that already look like errors (leaving them helps the LLM
        learn from failures)
    """
    cfg = cfg or CompactionConfig()
    if len(messages) <= cfg.keep_recent_tail + 1:
        return messages

    id2name = _build_tool_call_index(messages)
    tail_start = len(messages) - cfg.keep_recent_tail

    # Walk in REVERSE and count occurrences per tool; drop once we exceed
    # the per-tool keep budget AND we're before the protected tail.
    kept_count: dict[str, int] = {}
    new_messages = [None] * len(messages)  # type: ignore[assignment]

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "tool":
            new_messages[i] = msg
            continue

        tcid = msg.get("tool_call_id", "")
        # Prefetch results (synthetic call_prefetch_* IDs from
        # agent._prefetch_disconnections) are session-start anchors and must
        # never be dropped — the LLM rewrites no equivalent and the prefetch
        # is the only place suggest_templates/neural_retro/recall_similar_routes
        # fire on the target.
        if tcid.startswith("call_prefetch_"):
            new_messages[i] = msg
            continue

        tool_name = id2name.get(tcid, "")
        keep_budget = KEEP_RECENT_PER_TOOL.get(tool_name, DEFAULT_KEEP_RECENT)

        protected_by_tail = i >= tail_start
        protected_by_policy = (
            tool_name in PROTECTED_TOOLS or tool_name not in DROPPABLE_TOOLS
        )
        content = msg.get("content", "")
        protected_by_error = (
            isinstance(content, str) and '"error"' in content[:200]
        )

        already_kept = kept_count.get(tool_name, 0)
        should_keep = (
            protected_by_tail
            or protected_by_policy
            or protected_by_error
            or already_kept < keep_budget
        )

        if should_keep:
            kept_count[tool_name] = already_kept + 1
            new_messages[i] = msg
        else:
            new_messages[i] = {
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id", ""),
                "content": json.dumps({
                    "compacted": True,
                    "original_tool": tool_name,
                    "note": _PLACEHOLDER,
                }),
            }

    return [m for m in new_messages if m is not None]


# ---------------------------------------------------------------------------
# autoCompact: LLM-driven summarization of the older half
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM = """\
You are a context compactor for a chemoenzymatic retrosynthesis agent.
You will receive a slice of an in-progress conversation. Write a concise,
information-dense summary that preserves EVERYTHING the agent will need to
continue planning:

Required sections (omit a section only if it has no content):
  ## Target & constraints
  ## Confirmed reaction steps (product >> reactants, with reaction type / EC)
  ## Verified buyable leaf molecules
  ## Verified NOT-buyable molecules (still need decomposition)
  ## Candidate disconnections considered (and why rejected, if any)
  ## Pending TODOs / leaves to explore
  ## Scoring feedback received so far (if any)
  ## Open questions / uncertainties

Rules:
  - Preserve exact SMILES strings verbatim — do not paraphrase them.
  - Preserve EC numbers verbatim.
  - Preserve numeric scores verbatim.
  - Do NOT add information not present in the transcript.
  - Be terse; drop filler prose, keep facts.
"""


def _collect_tool_call_ids(messages: list[dict]) -> list[set[str]]:
    """
    For each message index, return the set of tool_call_ids the message
    EMITS (assistant) or REFERENCES (tool result). Non-tool messages get
    an empty set. Used to compute safe split boundaries.
    """
    per_msg: list[set[str]] = []
    for msg in messages:
        ids: set[str] = set()
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id:
                    ids.add(tc_id)
        elif role == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id:
                ids.add(tc_id)
        per_msg.append(ids)
    return per_msg


def _find_safe_split(
    messages: list[dict],
    head_end: int,
    tail_start: int,
) -> tuple[int, int]:
    """
    Adjust (head_end, tail_start) so that no assistant→tool pair crosses
    the middle/tail OR head/middle boundary.

    - tail_start is moved back (made smaller) so an assistant whose tool
      results sit in tail is pulled into tail with them.
    - head_end is moved back (made smaller) so an assistant in the kept head
      whose tool results sit in the middle (which gets replaced by the
      synthetic summary) is pushed into the middle with them — otherwise its
      `tool_use` block would survive with no matching `tool_result`, which
      the Anthropic and OpenAI APIs reject.

    Returns possibly-narrowed boundaries; if no safe split exists, returns
    the inputs unchanged.
    """
    n = len(messages)
    if head_end >= tail_start or tail_start > n:
        return head_end, tail_start

    ids_per = _collect_tool_call_ids(messages)

    # Both invariants must hold simultaneously:
    #   (A) no assistant in [head_end, tail_start) references a tail id
    #       (else its tool_result in tail orphans the tool_use);
    #   (B) no assistant in [0, head_end) references a middle id
    #       (else summarizing the middle orphans the kept-head tool_use).
    # Enforcing B can pull a head assistant INTO the middle; if that
    # assistant also referenced tail ids, A is now violated for it. So the
    # two closures must be interleaved to a fixpoint rather than run once
    # each — otherwise a head assistant whose tool_calls span middle AND
    # tail gets summarized while its tail tool_result survives orphaned.
    #
    # "Give up" must return a NON-splittable boundary (head_end >= tail_start)
    # so the caller's `if head_end >= tail_start: return messages` guard skips
    # compaction entirely. Returning the original (still-unsafe) boundary would
    # let an orphan through — the exact 400 this function exists to prevent.
    no_split = (tail_start, tail_start)
    for _ in range(2 * n + 1):  # bounded — converges; +1 guards tiny n
        # (A) middle/tail closure.
        tail_ids: set[str] = set()
        for i in range(tail_start, n):
            tail_ids |= ids_per[i]
        moved_a = False
        for i in range(head_end, tail_start):
            if messages[i].get("role") == "assistant" and ids_per[i] & tail_ids:
                if i <= head_end:
                    return no_split  # cannot pull tail back without splitting
                tail_start = i
                moved_a = True
                break

        # (B) head/middle closure. Recompute middle ids since head_end may
        # have shrunk on a prior pass.
        middle_ids: set[str] = set()
        for i in range(head_end, tail_start):
            middle_ids |= ids_per[i]
        moved_b = False
        for i in range(head_end - 1, -1, -1):
            if messages[i].get("role") == "assistant" and ids_per[i] & middle_ids:
                head_end = i
                moved_b = True
                break
        if moved_b and head_end >= tail_start:
            return no_split

        if not (moved_a or moved_b):
            break

    if head_end >= tail_start:
        return no_split
    return head_end, tail_start


def auto_compact(
    messages: list[dict],
    cfg: CompactionConfig | None = None,
    llm_client: Any = None,
    trace: Callable[..., None] | None = None,
) -> list[dict]:
    """
    Summarize the middle section of the transcript. Structure:

        [system]                              (kept verbatim)
        [first keep_head_messages user/assistant]  (kept verbatim)
        [synthetic user msg: '## Compacted summary ...']
        [last keep_recent_tail messages]      (kept verbatim)

    If `llm_client` is provided, uses it to produce the summary. Otherwise
    falls back to a heuristic extractor (or skips, per config).
    """
    cfg = cfg or CompactionConfig()
    trace = trace or (lambda *a, **kw: None)

    if len(messages) <= cfg.keep_recent_tail + cfg.keep_head_messages + 2:
        return messages

    # Identify system message (if any)
    system_msg = None
    head_start = 0
    if messages and messages[0].get("role") == "system":
        system_msg = messages[0]
        head_start = 1

    head_end = head_start + cfg.keep_head_messages
    tail_start = len(messages) - cfg.keep_recent_tail

    if head_end >= tail_start:
        # Nothing meaningful in the middle.
        return messages

    # Walk tail boundary back past any `tool` result at its head (simple case).
    while tail_start < len(messages) and messages[tail_start].get("role") == "tool":
        tail_start -= 1
        if tail_start <= head_end:
            return messages  # bail out — cannot cleanly split

    # Tool-call-id closure: ensure no assistant's tool_calls cross into tail.
    head_end, tail_start = _find_safe_split(messages, head_end, tail_start)
    if head_end >= tail_start:
        return messages

    head = messages[head_start:head_end]
    middle = messages[head_end:tail_start]
    tail = messages[tail_start:]

    summary_text = _summarize_middle(middle, cfg=cfg, llm_client=llm_client)

    compacted: list[dict] = []
    if system_msg:
        compacted.append(system_msg)
    compacted.extend(head)
    compacted.append({
        "role": "user",
        "content": (
            f"## Compacted summary of earlier conversation ({len(middle)} messages)\n\n"
            f"{summary_text}\n\n"
            "---\n"
            "End of summary. Continue planning from here using the summary above "
            "and the most recent exchanges below."
        ),
    })
    compacted.extend(tail)
    return compacted


def _summarize_middle(
    middle: list[dict],
    cfg: CompactionConfig,
    llm_client: Any,
) -> str:
    """Produce a summary for the middle slice of messages."""
    if llm_client is not None:
        try:
            return _llm_summary(middle, cfg, llm_client)
        except Exception as e:
            logger.warning("LLM summarization failed (%s); falling back to heuristic.", e)

    if cfg.allow_heuristic_summary:
        return _heuristic_summary(middle)

    return f"({len(middle)} older messages omitted to fit context budget.)"


def _llm_summary(middle: list[dict], cfg: CompactionConfig, llm_client: Any) -> str:
    """Ask the LLM to summarize `middle` into a structured block."""
    # Flatten messages into a plain transcript to feed the summarizer.
    lines: list[str] = []
    for msg in middle:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(p) for p in content)
        content = (content or "").strip()

        if role == "assistant":
            tcs = msg.get("tool_calls") or []
            if tcs:
                for tc in tcs:
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        name = fn.get("name", "?") if isinstance(fn, dict) else "?"
                        args = fn.get("arguments", "") if isinstance(fn, dict) else ""
                    else:
                        fn = getattr(tc, "function", None)
                        name = getattr(fn, "name", "?") if fn else "?"
                        args = getattr(fn, "arguments", "") if fn else ""
                    if isinstance(args, str) and len(args) > 400:
                        args = args[:400] + "..."
                    lines.append(f"[assistant -> tool] {name}({args})")
            if content:
                lines.append(f"[assistant] {content[:2000]}")
        elif role == "tool":
            snippet = content if len(content) <= 800 else content[:800] + "..."
            lines.append(f"[tool_result] {snippet}")
        elif role == "user":
            lines.append(f"[user] {content[:1200]}")
        elif role == "system":
            # Skip — the outer context has its own system message.
            continue

    transcript = "\n".join(lines)

    summary_messages = [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {"role": "user", "content": (
            "Summarize this in-progress retrosynthesis transcript:\n\n"
            "```\n"
            f"{transcript}\n"
            "```"
        )},
    ]

    # Prefer chat_with_override (thread-safe, no mutation of client state).
    # Fall back to chat() for clients that don't implement it, temporarily
    # only when not under concurrent use.
    if hasattr(llm_client, "chat_with_override"):
        resp = llm_client.chat_with_override(
            summary_messages,
            tools=None,
            max_tokens=cfg.summary_max_tokens,
        )
    else:
        resp = llm_client.chat(summary_messages, tools=None)

    choices = resp.get("choices", [])
    for choice in choices:
        msg = choice.get("message", {})
        content = msg.get("content")
        if content:
            return content.strip()
    return _heuristic_summary(middle)


def _heuristic_summary(middle: list[dict]) -> str:
    """Fallback: extract tool-call traces and confirmed facts without an LLM."""
    tools_called: list[str] = []
    molecules_buyable: list[str] = []
    molecules_not_buyable: list[str] = []
    route_lines: list[str] = []
    scoring_notes: list[str] = []

    for msg in middle:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    name = fn.get("name", "?") if isinstance(fn, dict) else "?"
                    args = fn.get("arguments", "") if isinstance(fn, dict) else ""
                    tools_called.append(f"{name}({str(args)[:80]})")
            if isinstance(content, str):
                for line in content.splitlines():
                    ls = line.strip()
                    if ls.lower().startswith("step ") and ">>" in ls:
                        route_lines.append(ls[:160])

        elif role == "tool" and isinstance(content, str):
            # Try to parse as JSON and extract key fields
            try:
                data = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(data, dict):
                smi = data.get("smiles") or data.get("input_smiles")
                buyable = data.get("buyable")
                if smi and buyable is True:
                    molecules_buyable.append(smi[:80])
                elif smi and buyable is False:
                    molecules_not_buyable.append(smi[:80])
                if "total_score" in data:
                    scoring_notes.append(
                        f"total_score={data.get('total_score')} "
                        f"fail={data.get('fail_reason', '')[:80]}"
                    )

    parts: list[str] = []
    if route_lines:
        parts.append("## Route steps proposed\n" + "\n".join(f"- {l}" for l in route_lines[:20]))
    if molecules_buyable:
        parts.append("## Confirmed buyable\n" + "\n".join(f"- {m}" for m in dict.fromkeys(molecules_buyable)))
    if molecules_not_buyable:
        parts.append("## Confirmed NOT buyable\n" + "\n".join(f"- {m}" for m in dict.fromkeys(molecules_not_buyable)))
    if tools_called:
        parts.append(f"## Tools invoked ({len(tools_called)})\n" + "; ".join(tools_called[:30]))
    if scoring_notes:
        parts.append("## Scoring feedback\n" + "\n".join(f"- {s}" for s in scoring_notes))
    if not parts:
        parts.append(f"({len(middle)} older messages of general planning discussion.)")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Milestone-triggered compaction (called by the agent loop after save_best_route)
# ---------------------------------------------------------------------------
def compact_to_checkpoint(
    messages: list[dict],
    checkpoint_after_index: int,
    cfg: CompactionConfig | None = None,
) -> tuple[list[dict], int]:
    """Replace `tool` message contents older than `checkpoint_after_index`
    with the standard placeholder. Preserves the assistant `tool_calls`
    pairing (cache-key safe) — the call stays, only the result content
    is demoted.

    Returns ``(new_messages, n_demoted)``.
    """
    cfg = cfg or CompactionConfig()
    if checkpoint_after_index <= 0 or checkpoint_after_index >= len(messages):
        return messages, 0
    new = list(messages)
    n = 0
    for i in range(checkpoint_after_index):
        msg = new[i]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content == _PLACEHOLDER:
            continue
        # leave error tool results alone — useful diagnostic
        if isinstance(content, str) and (
            content.startswith('{"error"') or '"error":' in content[:80]
        ):
            continue
        new[i] = {**msg, "content": _PLACEHOLDER}
        n += 1
    return new, n
