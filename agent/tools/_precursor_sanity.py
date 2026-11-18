"""LLM-judge precursor sanity filter for `neural_retro`.

Single-step models (rxngraphormer, ChemBART) sometimes emit precursor
sets where the only reactant is a tiny leaving group (methylamine,
ethane, water, CO2, ...) that cannot supply the product's skeleton.
This module asks the main LLM to flag such candidates and drops them
before the agent ever sees them.

Design notes:
- **Fail-open.** A judge error must never reduce the number of
  candidates the agent has to work with — the filter is an optimisation,
  not a correctness gate. Any exception or unparseable response keeps
  every candidate.
- **Cache by content.** Key on the canonical target + sorted canonical
  reactants set so the same call is not paid twice.
- **Single batched LLM call.** Up to `batch_size` candidates per call;
  the agent never invokes this directly — `neural_retro.execute` does.
"""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any

from rdkit import Chem

logger = logging.getLogger(__name__)


_SYSTEM = (
    "You are a careful organic-chemistry reviewer screening proposed "
    "retrosynthesis precursor sets emitted by ML models. You will be "
    "shown a target product SMILES and a numbered list of candidate "
    "precursor sets. For EACH candidate, decide if it is chemically "
    "plausible as a one-step retrosynthesis disconnection of the target."
)

_INSTRUCTIONS = (
    "Mark a candidate IMPLAUSIBLE if AND ONLY IF any of these hold:\n"
    "  (a) The candidate is a single very small molecule (≤ 3 heavy atoms) "
    "that CANNOT account for the product's skeleton (e.g. methylamine, "
    "methane, ethane, methanol, CO2, water, NH3 alone).\n"
    "  (b) Total reactant heavy-atom count is far smaller than the product's "
    "(< 50%) AND no condensation/leaving-group rationale is obvious.\n"
    "  (c) The reactants contain a chemically nonsensical SMILES.\n"
    "Otherwise mark PLAUSIBLE — err on the side of keeping candidates. "
    "Open-ring / close-ring / multi-step-looking disconnections are PLAUSIBLE.\n\n"
    "Return ONLY a JSON array (no prose) of objects "
    '{"idx": <int>, "plausible": <bool>, "reason": <short string>}. '
    "Cover every candidate exactly once."
)


def _canon(smiles: str) -> str:
    try:
        m = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(m) if m is not None else smiles
    except Exception:
        return smiles


def _cache_key(target: str, candidates: list[dict]) -> str:
    target_c = _canon(target)
    cand_canon = sorted(
        ".".join(sorted(_canon(p) for p in c.get("reactants", "").split(".")))
        for c in candidates
    )
    h = hashlib.sha256()
    h.update(target_c.encode())
    h.update(b"|")
    h.update("\n".join(cand_canon).encode())
    return h.hexdigest()


def _extract_text(resp: Any) -> str:
    """Pull the assistant message text out of whatever `chat()` returns.

    `LLMClient.chat()` returns the FULL OpenAI-shaped completion dict
    (``{"choices": [{"message": {"content": ...}}]}``), not a bare
    string — feeding that dict straight to `_parse_judge` made the judge
    silently fail-open for every model. Accept a dict (extract
    ``choices[0].message.content``) or a plain string. Content may itself
    be a list of typed parts (Anthropic-style); join their text.
    """
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        try:
            content = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    t = part.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            return "".join(parts)
    return ""


def _parse_judge(text: str, n: int) -> list[dict] | None:
    """Best-effort JSON extraction. Returns None on any structural problem."""
    try:
        # Try a strict parse first; otherwise look for the first '['..']'.
        try:
            data = json.loads(text)
        except Exception:
            i = text.find("[")
            j = text.rfind("]")
            if i == -1 or j == -1 or j <= i:
                return None
            data = json.loads(text[i : j + 1])
        if not isinstance(data, list) or len(data) != n:
            return None
        for entry in data:
            if not isinstance(entry, dict):
                return None
            if "idx" not in entry or "plausible" not in entry:
                return None
        return data
    except Exception:
        return None


def filter_precursors(
    target_smiles: str,
    candidates: list[dict],
    llm_client: Any,
    *,
    batch_size: int = 20,
    cache: dict | None = None,
) -> tuple[list[dict], list[dict], list[str]]:
    """Filter `candidates` via the LLM judge.

    Returns (kept, dropped, reasons_for_dropped). On any failure path
    returns (candidates, [], []) so the caller never loses candidates.
    """
    if not candidates:
        return [], [], []

    if cache is not None:
        key = _cache_key(target_smiles, candidates)
        cached = cache.get(key)
        if cached is not None:
            # Return a deep copy: callers (ensemble clustering) mutate the
            # returned dicts in place (adding `bucket` / `cluster_size`), so
            # handing out the cached objects directly would pollute the cache
            # and leak stale clustering internals into later cache hits.
            kept, dropped, reasons = cached
            return deepcopy(kept), deepcopy(dropped), list(reasons)

    # Split into batches to keep prompts bounded. Batches are independent
    # (each is a self-contained LLM judge call), so they run concurrently —
    # this is the dominant cost in neural_retro's ensemble path (N batches
    # × ~15s serial collapses to ~15s wall-clock).
    batches = [
        candidates[start : start + batch_size]
        for start in range(0, len(candidates), batch_size)
    ]

    def _judge_batch(batch: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
        """Judge one batch. Returns (kept, dropped, reasons). Fail-open:
        on any LLM/parse failure the whole batch is kept."""
        kept: list[dict] = []
        dropped: list[dict] = []
        reasons: list[str] = []
        listing = "\n".join(
            f"  {i}. {c.get('reactants', '')}"
            for i, c in enumerate(batch)
        )
        user = (
            f"Target product SMILES: {target_smiles}\n\n"
            f"Candidate precursor sets ({len(batch)}):\n{listing}\n\n"
            f"{_INSTRUCTIONS}"
        )
        try:
            resp = llm_client.chat(
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user},
                ]
            )
        except Exception as e:
            logger.warning(
                "precursor_sanity: LLM call failed (%s); fail-open keeping batch", e
            )
            return list(batch), [], []

        text = _extract_text(resp)
        decisions = _parse_judge(text, len(batch))
        if decisions is None:
            logger.warning(
                "precursor_sanity: unparseable judge output; fail-open keeping batch"
            )
            return list(batch), [], []

        for entry in decisions:
            i = entry["idx"]
            if not isinstance(i, int) or i < 0 or i >= len(batch):
                # Malformed entry: keep the candidate it would have addressed.
                continue
            cand = batch[i]
            if entry.get("plausible", True):
                kept.append(cand)
            else:
                dropped.append(cand)
                reasons.append(str(entry.get("reason", "implausible")))

        # Any candidate not addressed by a decision is kept (defensive).
        addressed = {entry["idx"] for entry in decisions if isinstance(entry.get("idx"), int)}
        for i, cand in enumerate(batch):
            if i not in addressed and cand not in kept and cand not in dropped:
                kept.append(cand)
        return kept, dropped, reasons

    kept_total: list[dict] = []
    dropped_total: list[dict] = []
    reasons_total: list[str] = []

    if len(batches) <= 1:
        # No concurrency win for a single batch — call inline.
        batch_results = [_judge_batch(b) for b in batches]
    else:
        # Bound workers so a flood of batches can't open unbounded LLM
        # connections; 8 is plenty for the candidate counts we see.
        max_workers = min(8, len(batches))
        batch_results: list[tuple[list[dict], list[dict], list[str]] | None] = (
            [None] * len(batches)
        )
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_judge_batch, b): i for i, b in enumerate(batches)}
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    batch_results[idx] = fut.result()
                except Exception as e:
                    # Defensive: _judge_batch already fails open, but if the
                    # future itself raised, keep that batch verbatim.
                    logger.warning(
                        "precursor_sanity: batch %d future failed (%s); "
                        "fail-open keeping batch", idx, e
                    )
                    batch_results[idx] = (list(batches[idx]), [], [])

    # Concatenate in original batch order so `dropped` and `reasons` stay
    # aligned and results are deterministic regardless of completion order.
    for res in batch_results:
        if res is None:
            continue
        kept, dropped, reasons = res
        kept_total.extend(kept)
        dropped_total.extend(dropped)
        reasons_total.extend(reasons)

    result = (kept_total, dropped_total, reasons_total)
    if cache is not None:
        # Store deep copies so the miss-path caller's downstream mutations
        # (clustering adds keys to these dicts) don't corrupt the cached entry.
        cache[key] = (deepcopy(kept_total), deepcopy(dropped_total), list(reasons_total))
    return result
