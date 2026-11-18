"""
Scratchpad + backtrack tools — long-horizon memory for the retrosynthesis
agent.

Retrosynthesis is inherently a search: the LLM often tries a disconnection,
scores it, finds it mediocre, then needs to explore a different strategy.
Without explicit memory, the old conversation turns (failed tool outputs,
dead-end decompositions) sit in context and burn tokens for no benefit.

This module adds three capabilities:

1. **Scratchpad** — ``save_best_route`` / ``list_best_routes`` /
   ``get_best_route``: the agent can persist candidate routes with their
   score and notes. Top-K (by ``total_score``) are kept; older/worse
   entries are evicted.

2. **Lessons-learned** — part of the store: ``backtrack_and_summarize``
   appends a short terse note (``failed_disconnection``, ``why``,
   ``lowest_sub_score``) to a running list.

3. **Context compaction hook** — ``backtrack_and_summarize`` returns a
   compact JSON snapshot (target, current todos, top-N routes, lessons).
   The agent loop in ``core.agent`` uses a marker in the tool result to
   trigger an aggressive ``maybe_compact`` that rewrites the conversation
   to keep **only**: system prompt + target + this snapshot. Everything
   else (raw tool call/tool result turns for dead branches) is dropped.

Design inspired by Claude Code's own patterns:
  - Task store (TaskCreate/TaskUpdate/TaskList) for the CRUD shape
  - TodoWriteTool for the single-in-progress-item discipline
  - services/compact for the recover-best-N-recent approach
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tools.base import Tool
from tools._route_fp import canonicalize_smiles as _canonicalize_smiles
from tools._route_fp import route_fingerprint as _route_fingerprint

logger = logging.getLogger(__name__)


# Marker string embedded in the BacktrackAndSummarize tool result; the
# agent loop watches for this to trigger the forced compaction.
BACKTRACK_MARKER = "__BACKTRACK_COMPACT__"


def _validate_steps_smiles(steps: List[str]) -> Optional[dict]:
    """Reject routes whose steps aren't real ``product>>reactants`` SMILES.

    The LLM occasionally writes free-text descriptors like
    ``"target>>chloro-pteridinone.NCC(F)(F)F (reaction: SNAr ...)"`` —
    the trailing reactant is a valid SMILES, but the LHS and one
    reactant are placeholder words. Forward T5 round-trip then fails on
    every step and feasibility falls back to the LLM judge cap (0.85),
    so the route earns a high score without real verification.

    Returns ``None`` if every step parses; otherwise an error dict
    describing the first bad step. Strips the trailing
    ``" (reaction: …)"`` annotation the same way ``_route_fingerprint``
    does. Fail-open if RDKit is unavailable.
    """
    try:
        from rdkit import Chem
    except Exception:
        return None
    for idx, step in enumerate(steps or []):
        if not isinstance(step, str) or ">>" not in step:
            return {
                "status": "invalid_step_rejected",
                "bad_step_index": idx,
                "bad_step": str(step)[:200],
                "reason": "step missing '>>' separator; use 'product_smiles>>reactant1.reactant2 (reaction: name)'",
            }
        prod, _, react = step.partition(">>")
        react = react.split(" (", 1)[0]
        prod = prod.strip()
        if not prod or Chem.MolFromSmiles(prod) is None:
            return {
                "status": "invalid_step_rejected",
                "bad_step_index": idx,
                "bad_step": step[:200],
                "bad_token": prod[:80],
                "reason": "product is not parseable SMILES (placeholder text like 'target' / 'intermediate' is not allowed — use canonical SMILES)",
            }
        for r in react.split("."):
            r = r.strip()
            if not r:
                continue
            if Chem.MolFromSmiles(r) is None:
                return {
                    "status": "invalid_step_rejected",
                    "bad_step_index": idx,
                    "bad_step": step[:200],
                    "bad_token": r[:80],
                    "reason": "reactant is not parseable SMILES (use canonical SMILES, not common names like 'chloro-pteridinone')",
                }
    return None


def derive_leaf_smiles(steps: List[str]) -> Optional[List[str]]:
    """Derive a route's true leaves structurally from its steps.

    A leaf is a reactant SMILES that is never produced as the product of any
    step — i.e. a raw starting material the route actually bottoms out on.
    Steps are ``product>>reactant1.reactant2 (reaction: name)`` strings (the
    trailing ``" (reaction: …)"`` annotation is stripped, same as
    ``_validate_steps_smiles`` / ``_route_fingerprint``).

    This is the structural ground truth the buyability gate should check,
    independent of the LLM-supplied ``starting_materials`` field — a route can
    declare a couple of trivially-buyable blocks while its steps decompose to
    entirely different, non-buyable precursors.

    Returns a sorted, de-duplicated list of canonical leaf SMILES, or ``None``
    if RDKit is unavailable or any step is unparseable (fail-soft → caller
    falls back to the declared field rather than crashing).
    """
    try:
        from rdkit import Chem
    except Exception:
        return None

    def _canon(smi: str) -> Optional[str]:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m) if m is not None else None

    products: set[str] = set()
    reactants: set[str] = set()
    for step in steps or []:
        if not isinstance(step, str) or ">>" not in step:
            return None
        prod, _, react = step.partition(">>")
        react = react.split(" (", 1)[0]
        cp = _canon(prod.strip())
        if cp is None:
            return None
        products.add(cp)
        for r in react.split("."):
            r = r.strip()
            if not r:
                continue
            cr = _canon(r)
            if cr is None:
                return None
            reactants.add(cr)
    # Leaves = reactants never produced by any step.
    return sorted(reactants - products)


def _canon_smiles(smi: str) -> Optional[str]:
    """Canonicalize one SMILES, or None if RDKit missing / unparseable."""
    try:
        from rdkit import Chem
    except Exception:
        return None
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m) if m is not None else None


def check_route_grounding(
    target_smiles: str,
    step_products: List[str],
    step_reactants: List[List[str]],
) -> dict:
    """Structural "is this route actually a tree rooted at the target?" check.

    Pure topology, NO chemistry judgement — it never asks whether a reaction is
    feasible, only whether the route's reaction graph is connected and points at
    the requested target. Two conditions:

      1. root == target: at least one step's product canonically equals the
         target molecule (the route actually synthesises what was asked).
      2. no dangling intermediates: every step's product is either the target
         OR consumed as a reactant in some other step. A product that nothing
         downstream uses is a disconnected fragment — the "tree" is really two
         unrelated pieces.

    This is necessary-but-insufficient: it catches the broken-tree and
    wrong-final-product classes, NOT the written-target-string fake (product
    string == target but the declared reactants don't actually give it) — that
    needs per-step forward re-derivation, a separate layer.

    Returns a JSON-serialisable dict::

        {"ok": bool, "checked": bool, "root_matches_target": bool,
         "n_dangling": int, "dangling": [smi, ...], "reason": str}

    ``checked`` is False (and ``ok`` True, fail-open) when RDKit is unavailable
    or any SMILES is unparseable — grounding cannot be asserted, so we do not
    block on it; the per-step SMILES validator already rejects unparseable
    routes upstream.
    """
    out = {
        "ok": True, "checked": False, "root_matches_target": False,
        "n_dangling": 0, "dangling": [], "reason": "",
    }
    ct = _canon_smiles(target_smiles)
    if ct is None:
        out["reason"] = "target unparseable or RDKit unavailable; grounding not checked"
        return out

    products: List[str] = []
    for p in step_products:
        cp = _canon_smiles(p)
        if cp is None:
            out["reason"] = "a step product is unparseable; grounding not checked"
            return out
        products.append(cp)

    reactant_set: set[str] = set()
    for rs in step_reactants:
        for r in (rs or []):
            cr = _canon_smiles(r)
            if cr is None:
                out["reason"] = "a step reactant is unparseable; grounding not checked"
                return out
            reactant_set.add(cr)

    out["checked"] = True
    product_set = set(products)
    out["root_matches_target"] = ct in product_set
    # A product is dangling if it is neither the target nor used downstream.
    dangling = sorted(p for p in product_set if p != ct and p not in reactant_set)
    out["n_dangling"] = len(dangling)
    out["dangling"] = dangling

    problems = []
    if not out["root_matches_target"]:
        problems.append("no step produces the target molecule (root != target)")
    if dangling:
        problems.append(
            f"{len(dangling)} dangling intermediate(s) produced but never consumed"
        )
    out["ok"] = not problems
    out["reason"] = "; ".join(problems) if problems else "route graph grounded at target"
    return out


def check_route_grounding_from_steps(target_smiles: str, steps: List[str]) -> dict:
    """``check_route_grounding`` adapter for ``product>>reactants`` step strings.

    Parses the same way as ``derive_leaf_smiles`` / ``_validate_steps_smiles``.
    Returns ``checked=False`` (fail-open) if any step is malformed.
    """
    products: List[str] = []
    reactants: List[List[str]] = []
    for step in steps or []:
        if not isinstance(step, str) or ">>" not in step:
            return {"ok": True, "checked": False, "root_matches_target": False,
                    "n_dangling": 0, "dangling": [],
                    "reason": "a step is not 'product>>reactants'; grounding not checked"}
        prod, _, react = step.partition(">>")
        react = react.split(" (", 1)[0]
        products.append(prod.strip())
        reactants.append([r.strip() for r in react.split(".") if r.strip()])
    return check_route_grounding(target_smiles, products, reactants)


@dataclass
class RouteEntry:
    label: str
    total_score: float
    steps: List[str] = field(default_factory=list)
    sub_scores: Dict[str, float] = field(default_factory=dict)
    starting_materials: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: float = field(default_factory=time.time)
    # Provenance: which retrieval / generation path produced this route.
    # Populated by the agent loop based on the tool_calls that preceded
    # the save_best_route. One of: "unknown" (default / legacy),
    # "rag_anchored", "template_suggested", "neural_consensus",
    # "zero_shot". Tool schema does NOT expose this — internal only.
    provenance: str = "unknown"
    # When provenance == "rag_anchored", list of precedent route_ids
    # that were in context.
    rag_source_ids: Optional[List[str]] = None
    # Exact per-step template provenance from score_route, aligned with
    # ``steps``. None preserves compatibility with legacy saved entries that
    # predate this field; individual unmatched steps are represented by null.
    step_provenance: Optional[List[Optional[Dict[str, Any]]]] = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "total_score": round(self.total_score, 4),
            "steps": self.steps,
            "sub_scores": self.sub_scores,
            "starting_materials": self.starting_materials,
            "notes": self.notes,
            "created_at": self.created_at,
            "provenance": self.provenance,
            "rag_source_ids": self.rag_source_ids,
            "step_provenance": self.step_provenance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RouteEntry":
        """Rebuild from a ``to_dict()`` payload, ignoring unknown keys.

        ``to_dict`` keys map 1:1 to the constructor, so this round-trips
        cleanly; extra keys (e.g. ``num_steps`` from a ``summary()`` blob)
        are dropped rather than raising. Used to rehydrate the scratchpad
        on a continuation run.
        """
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def summary(self) -> dict:
        return {
            "label": self.label,
            "total_score": round(self.total_score, 4),
            "num_steps": len(self.steps),
            "num_starting_materials": len(self.starting_materials),
            "sub_scores": self.sub_scores,
            "notes_excerpt": (self.notes[:160] + "…") if len(self.notes) > 160 else self.notes,
            "provenance": self.provenance,
        }


@dataclass
class Lesson:
    reason: str
    failed_disconnection: str = ""
    lowest_sub_score: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "reason": self.reason,
            "failed_disconnection": self.failed_disconnection,
            "lowest_sub_score": self.lowest_sub_score,
            "created_at": self.created_at,
        }


class ScratchpadStore:
    """Per-run store for candidate routes + lessons-learned."""

    # Capacity history: 5 (original) → 10 (post forced-refinement sweep) →
    # 25 (current). With max_iterations=150 and ensemble neural_retro
    # spawning 12+ branches per target, runs were routinely hitting the
    # 10-slot ceiling and silently evicting mid-tier candidates the LLM
    # later wanted to revisit. 25 covers the worst hard-target observed
    # (≈18 distinct disconnections explored before convergence) with
    # negligible memory overhead — a RouteEntry is a few hundred bytes.
    DEFAULT_MAX_ROUTES = 25

    def __init__(self, max_routes: int = DEFAULT_MAX_ROUTES):
        self._lock = threading.Lock()
        self._routes: Dict[str, RouteEntry] = {}
        self._lessons: List[Lesson] = []
        # Structural fingerprint -> label, so SaveBestRouteTool can
        # short-circuit duplicate inserts (same canonical SMILES,
        # order-insensitive at both step and reactant level).
        self._fingerprints: Dict[str, str] = {}
        self.max_routes = max(1, int(max_routes))

    # Back-compat alias for callers/tests that still read MAX_ROUTES.
    @property
    def MAX_ROUTES(self) -> int:
        return self.max_routes

    # ---- routes ----
    def save(self, entry: RouteEntry) -> tuple[RouteEntry, str]:
        """Insert entry. Returns (entry, status).

        status is one of:
        - ``"inserted"``:  entry was added (possibly evicting a worse one).
        - ``"updated"``:   same fingerprint AND same label → existing entry's
                           ``total_score`` / ``sub_scores`` / ``notes`` /
                           ``starting_materials`` were overwritten in place.
                           Supports the legitimate "save draft → rescore →
                           re-save" workflow without re-opening the
                           same-route-under-two-labels gaming hole.
        - ``"duplicate"``: a structurally identical route was already
                           stored under a DIFFERENT label; returned
                           ``entry`` is the existing one (NOT the input).
        - ``"rejected"``:  at capacity and entry score is no better than
                           the worst kept route.
        """
        fp = _route_fingerprint(entry.steps)
        with self._lock:
            existing_label = self._fingerprints.get(fp)
            if existing_label and existing_label in self._routes:
                if existing_label == entry.label:
                    # Same route, same label → in-place rescore. Preserve
                    # the original ``created_at`` so ordering stays stable;
                    # the fingerprint→label mapping is already correct.
                    existing = self._routes[existing_label]
                    existing.total_score = entry.total_score
                    existing.sub_scores = entry.sub_scores
                    existing.notes = entry.notes
                    existing.starting_materials = entry.starting_materials
                    # Only overwrite provenance when the caller supplied a
                    # real attribution — never demote a known provenance
                    # back to "unknown" on a rescore.
                    if entry.provenance and entry.provenance != "unknown":
                        existing.provenance = entry.provenance
                        existing.rag_source_ids = entry.rag_source_ids
                    if entry.step_provenance is not None:
                        existing.step_provenance = entry.step_provenance
                    return existing, "updated"
                existing = self._routes[existing_label]
                if existing.provenance == "deterministic_closure":
                    # The stored entry was auto-captured from a score_route
                    # payload (opaque label, generic notes). A later explicit
                    # model-authored save of the SAME chemistry carries the
                    # meaningful label / notes / starting_materials /
                    # provenance and must be allowed to take over the entry
                    # rather than bounce off as "duplicate" (Codex review
                    # 2026-07-18). Re-key the entry to the explicit label.
                    del self._routes[existing_label]
                    existing.label = entry.label
                    existing.total_score = entry.total_score
                    existing.sub_scores = entry.sub_scores
                    existing.notes = entry.notes
                    existing.starting_materials = entry.starting_materials
                    if entry.provenance and entry.provenance != "unknown":
                        existing.provenance = entry.provenance
                        existing.rag_source_ids = entry.rag_source_ids
                    if entry.step_provenance is not None:
                        existing.step_provenance = entry.step_provenance
                    self._routes[entry.label] = existing
                    self._fingerprints[fp] = entry.label
                    return existing, "updated"
                return existing, "duplicate"

            # If this label already holds a *different* route, the old
            # fingerprint(s) pointing at it must be dropped now —
            # otherwise a later save of that old chemistry under a fresh
            # label would be misclassified as ``duplicate_of=<this_label>``
            # (the label still exists, but its chemistry has moved on).
            # We sweep all fingerprint→label entries for this label
            # rather than re-deriving the old fingerprint, in case the
            # in-place ``"updated"`` branch above ever leaves more than
            # one fingerprint pointing at the same label (defence in
            # depth — it shouldn't, but the cost is O(N) per overwrite).
            if entry.label in self._routes:
                self._fingerprints = {
                    fp_: lbl for fp_, lbl in self._fingerprints.items()
                    if lbl != entry.label
                }

            if (
                entry.label not in self._routes
                and len(self._routes) >= self.MAX_ROUTES
            ):
                worst_score = min(r.total_score for r in self._routes.values())
                if entry.total_score <= worst_score:
                    return entry, "rejected"

            self._routes[entry.label] = entry
            self._fingerprints[fp] = entry.label

            if len(self._routes) > self.MAX_ROUTES:
                sorted_items = sorted(
                    self._routes.items(),
                    key=lambda kv: kv[1].total_score,
                    reverse=True,
                )
                kept = dict(sorted_items[: self.MAX_ROUTES])
                kept_labels = set(kept)
                self._fingerprints = {
                    fp_: lbl for fp_, lbl in self._fingerprints.items()
                    if lbl in kept_labels
                }
                self._routes = kept
            return entry, "inserted"

    def list(self) -> List[RouteEntry]:
        with self._lock:
            return sorted(self._routes.values(), key=lambda r: r.total_score, reverse=True)

    def all(self) -> List[RouteEntry]:
        """Return every entry currently in the store (unordered copy)."""
        with self._lock:
            return list(self._routes.values())

    def get(self, label: str) -> Optional[RouteEntry]:
        with self._lock:
            return self._routes.get(label)

    def best(self) -> Optional[RouteEntry]:
        routes = self.list()
        return routes[0] if routes else None

    # ---- lessons ----
    def add_lesson(self, lesson: Lesson) -> None:
        with self._lock:
            self._lessons.append(lesson)

    def lessons(self) -> List[Lesson]:
        with self._lock:
            return list(self._lessons)

    def clear(self) -> None:
        with self._lock:
            self._routes.clear()
            self._lessons.clear()
            self._fingerprints.clear()

    def restore(self, entries: List[dict]) -> int:
        """Replace the route set with rehydrated ``to_dict()`` payloads.

        Clears the store, then re-``save()``s each entry so the structural
        ``_fingerprints`` index rebuilds automatically (dedup keeps working
        after a continuation). Persist/restore from ``to_dict()`` — NOT
        ``summary()``, which drops ``steps`` and would make every entry
        fingerprint identically. Returns the count actually stored.
        """
        self.clear()
        restored = 0
        for e in entries or []:
            if not isinstance(e, dict):
                continue
            try:
                entry = RouteEntry.from_dict(e)
            except Exception:
                continue
            _, status = self.save(entry)
            if status in ("inserted", "updated"):
                restored += 1
        return restored

    def snapshot(self, keep_best_n: int = 5) -> dict:
        """Return a compact dict suitable for feeding back to the LLM."""
        routes = self.list()[:keep_best_n]
        return {
            "best_routes": [r.to_dict() for r in routes],
            "lessons_learned": [l.to_dict() for l in self.lessons()],
        }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class SaveBestRouteTool(Tool):
    name = "save_best_route"
    description = (
        "Save a candidate retrosynthesis route to the scratchpad with its "
        "`total_score`. Call AFTER every `score_route` so the agent can "
        "compare attempts and return the best one even if later attempts "
        "are worse. The scratchpad keeps the top-N (default 25); duplicates "
        "(canonical-SMILES, order-insensitive) return `status='duplicate'` "
        "instead of inserting again. Pass `score_route_result_id` from the "
        "matching score_route call so the saved score can be cross-checked "
        "against the persisted payload (mismatches >1e-3 are rejected; "
        "MISSING result_id clamps the score to 0.0 — the entry is archived "
        "but cannot win rank-1). Routes with zero steps are rejected outright "
        "(if the target is buyable, that is a buyability finding, not a "
        "synthesis route). Cost: <50ms, free. Pairs with `score_route` "
        "(always score first, save second)."
    )
    tier = 1
    parameters = {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": "Short human-readable label, e.g. 'suzuki-2-step' or 'enzymatic-v1'.",
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": (
                    "Route steps as strings. Use `product>>reactants (reaction: name)` "
                    "format. MUST contain at least one step — a 'route' with zero steps "
                    "is not a synthesis. If the target is itself directly buyable, that "
                    "is a buyability finding, not a route, and must not be saved."
                ),
            },
            "total_score": {
                "type": "number",
                "default": 0.0,
                "description": (
                    "Aggregate route score (0-1) returned by score_route. "
                    "Optional: defaults to sub_scores.total_score if present, "
                    "else 0.0. Pass the exact number from the most recent "
                    "score_route call to enable correct ranking."
                ),
            },
            "sub_scores": {
                "type": "object",
                "description": "Optional: {binding_score, molecule_score, route_validity, sm_availability, step_penalty, convergence, balance, sa_score, total_score, ...}. Free-form numeric map.",
                "additionalProperties": {"type": "number"},
            },
            "starting_materials": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Buyable leaf SMILES.",
            },
            "notes": {"type": "string", "description": "Why this route, what was tried, gaps, etc."},
            "score_route_result_id": {
                "type": "string",
                "description": (
                    "result_id from the score_route call that produced "
                    "`total_score`. When provided, the persisted payload is "
                    "re-read and the LLM-supplied `total_score` is rejected "
                    "if it differs from the persisted value by more than "
                    "1e-3. Strongly recommended — without it the score is "
                    "trusted on the LLM's word and the rank-1 route can be "
                    "gamed. Get the id from any recent score_route stub's "
                    "`_result_id` field."
                ),
            },
        },
        "required": ["label", "steps"],
    }

    def __init__(self, store: ScratchpadStore, result_store: Any | None = None):
        self._store = store
        # Optional ResultStore handle — when wired, save_best_route can
        # cross-check the LLM's total_score against the persisted
        # score_route payload. Leaving it None preserves the legacy
        # always-trust behaviour for callers that haven't migrated.
        self._result_store = result_store
        # Current run's target SMILES, set per-run by the agent via
        # set_target(). When present, save_best_route runs a structural
        # route-grounding gate (root==target + no dangling intermediates) and
        # refuses to save a route whose graph isn't actually a tree rooted at
        # the target. None → gate dormant (legacy callers that never set it).
        self._target: Optional[str] = None

    def set_target(self, target_smiles: Optional[str]) -> None:
        """Set the target for the current run's grounding gate."""
        self._target = target_smiles or None

    # Tolerance for the cross-check. score_route rounds to 4 decimals so
    # 1e-3 lets the LLM round-trip the printed value without false rejects.
    _SCORE_TOLERANCE = 1e-3

    def _verify_score_against_result_id(
        self,
        result_id: str,
        claimed_score: float,
        steps_list: Optional[List[str]] = None,
    ) -> Optional[dict]:
        """Cross-check the LLM-supplied total_score against the persisted
        score_route payload AND confirm the cited score_route call actually
        scored the route now being saved.

        Two independent checks:
          1. ``total_score`` matches the persisted value (within
             ``_SCORE_TOLERANCE``).
          2. The persisted ``route_fp`` (a structural fingerprint stamped
             by score_route) matches the fingerprint of ``steps_list``. This
             closes the gaming gap where an agent scores route A (high
             score), captures its ``_result_id``, then saves a *different*
             route B citing A's id + A's score — the value check alone would
             pass. When the payload predates the fingerprint stamp, fall
             back to comparing ``n_steps`` against ``len(steps_list)`` as a
             cheap mismatch guard.

        Returns None if verifiable and consistent. Returns a
        JSON-serialisable error dict on any failure (missing/wrong result,
        score mismatch, or route mismatch).
        """
        if self._result_store is None:
            # No store wired → cannot verify; fall through to trust.
            return None
        try:
            blob = self._result_store.load(result_id)
        except Exception as e:
            return {"error": f"score_route_result_id load failed: {e}"}
        if not isinstance(blob, dict):
            return {"error": "score_route_result_id load returned non-dict"}
        if "error" in blob and "tool" not in blob:
            return {"error": f"score_route_result_id not found: {blob.get('error')}"}
        tool_name = blob.get("tool")
        if tool_name != "score_route":
            return {
                "error": (
                    f"score_route_result_id points at a {tool_name!r} result, "
                    "not a score_route call"
                ),
            }
        payload = blob.get("content")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = None
        if not isinstance(payload, dict):
            return {"error": "score_route payload is not a dict"}
        persisted = payload.get("total_score")
        if persisted is None:
            return {"error": "score_route payload has no total_score field"}
        try:
            persisted_f = float(persisted)
        except (TypeError, ValueError):
            return {"error": f"persisted total_score is not numeric: {persisted!r}"}
        if abs(persisted_f - float(claimed_score)) > self._SCORE_TOLERANCE:
            return {
                "error": (
                    f"total_score mismatch: claimed {claimed_score!r} but "
                    f"score_route_result_id={result_id!r} persisted "
                    f"{round(persisted_f, 4)}. Resave with the persisted value."
                ),
                "persisted_total_score": round(persisted_f, 4),
                "claimed_total_score": float(claimed_score),
            }

        # Route-binding check: prove the cited score_route scored *this*
        # route, not a different one whose id+score were reused.
        if steps_list is not None:
            persisted_fp = payload.get("route_fp")
            if persisted_fp:
                saved_fp = _route_fingerprint(list(steps_list))
                if saved_fp != persisted_fp:
                    return {
                        "error": (
                            f"route mismatch: score_route_result_id={result_id!r} "
                            f"scored a different route than the one being saved "
                            f"(fingerprint {persisted_fp} != {saved_fp}). Score "
                            f"THIS route with score_route, then resave citing "
                            f"that call's _result_id."
                        ),
                        "persisted_route_fp": persisted_fp,
                        "saved_route_fp": saved_fp,
                    }
            else:
                # Pre-fingerprint payload — fall back to the cheap n_steps
                # guard so the most obvious cross-route reuse is still caught.
                persisted_n = payload.get("n_steps")
                if isinstance(persisted_n, int) and persisted_n != len(steps_list):
                    return {
                        "error": (
                            f"route step-count mismatch: "
                            f"score_route_result_id={result_id!r} scored "
                            f"{persisted_n} steps but the route being saved has "
                            f"{len(steps_list)}. Score THIS route, then resave "
                            f"citing that call's _result_id."
                        ),
                        "persisted_n_steps": persisted_n,
                        "saved_n_steps": len(steps_list),
                    }
        return None

    def execute(
        self,
        label: str,
        steps: List[str],
        total_score: Optional[float] = None,
        sub_scores: Optional[dict] = None,
        starting_materials: Optional[List[str]] = None,
        notes: str = "",
        score_route_result_id: Optional[str] = None,
        step_provenance: Optional[List[Optional[Dict[str, Any]]]] = None,
        **kwargs,
    ) -> str:
        # Internal-only provenance kwargs supplied by the agent loop —
        # NOT in the JSON schema, so the LLM cannot set them directly.
        provenance = kwargs.pop("_provenance", None) or "unknown"
        rag_source_ids = kwargs.pop("_rag_source_ids", None)

        if isinstance(steps, str):
            try:
                steps = json.loads(steps)
            except Exception:
                steps = [steps]
        steps_list = list(steps or [])
        if len(steps_list) == 0:
            return json.dumps({
                "status": "empty_route_rejected",
                "error": (
                    "Refusing to save a route with zero steps. If the target is "
                    "directly buyable, that is a buyability finding (call "
                    "check_buyability) — it is not a synthesis route. Synthesis "
                    "routes must have ≥ 1 step. Do not re-save with steps=[]."
                ),
            })

        # Permit total_score to be omitted: fall back to sub_scores.total_score,
        # then 0.0. This stops the LLM from crashing on a missing-arg TypeError
        # mid-loop when it forgets the field — the entry still saves and the
        # next score_route round can update it.
        if total_score is None and isinstance(sub_scores, dict):
            total_score = sub_scores.get("total_score")
        if total_score is None:
            total_score = 0.0
        try:
            score = float(total_score)
        except (TypeError, ValueError):
            return json.dumps({"error": "total_score must be a number"})

        # Cross-check against persisted score_route payload. Without a
        # result_id we cannot trust the LLM's number — it has been observed
        # writing total_score=1.0 directly (e.g. a 0-step "commercial-target-
        # leaf" save after score_route refused the empty payload). Policy:
        # require a result_id whenever a result_store is wired; if missing,
        # clamp the score to 0.0 and surface a warning so the entry can still
        # be archived (label/notes carry useful info) but cannot win rank-1.
        score_verification_warning: Optional[str] = None
        if score_route_result_id:
            err = self._verify_score_against_result_id(
                str(score_route_result_id).strip(), score, steps_list
            )
            if err is not None:
                err.setdefault("status", "score_mismatch_rejected")
                err.setdefault(
                    "hint",
                    (
                        "Re-read the score_route stub's `total_score` field, "
                        "or load_result(result_id=...) for the full payload, "
                        "then resave with the exact persisted number."
                    ),
                )
                return json.dumps(err)
        elif self._result_store is not None:
            score_verification_warning = (
                f"no score_route_result_id supplied — total_score was clamped "
                f"from {round(score, 4)} to 0.0 because it could not be cross-"
                f"checked against a persisted score_route payload. Call "
                f"score_route first, then resave with the stub's `_result_id` "
                f"in `score_route_result_id` to register the real score."
            )
            score = 0.0

        # Reject steps that aren't real product>>reactants SMILES.
        # Without this guard the LLM can save placeholder-text routes
        # like "target>>chloro-pteridinone.NCC(F)(F)F" — T5 round-trip
        # then fails on every step and feasibility silently falls back
        # to the LLM judge cap (0.85), so the route earns a high score
        # without ever being verified.
        validation_err = _validate_steps_smiles(steps_list)
        if validation_err is not None:
            validation_err["hint"] = (
                "Rewrite every step as `canonical_product_SMILES>>"
                "canonical_reactant1_SMILES.canonical_reactant2_SMILES "
                "(reaction: name)`. No placeholder words like 'target', "
                "'intermediate', or chemical names."
            )
            return json.dumps(validation_err)

        # Structural route-grounding gate. Pure topology (no chemistry): the
        # route's reaction graph must be a tree rooted at the run's target —
        # some step must produce the target, and no step may produce a molecule
        # nothing downstream consumes. This is the one deterministic check that
        # catches broken-tree / wrong-final-product fakes; without it such a
        # route earns a high feasibility score (every isolated step may be
        # plausible) yet doesn't actually synthesise the requested molecule.
        # Dormant when no target was set (legacy callers); fail-open when the
        # graph can't be canonicalised (the SMILES validator above already
        # rejected unparseable steps, so this only no-ops on RDKit-missing).
        if self._target:
            grounding = check_route_grounding_from_steps(self._target, steps_list)
            if grounding.get("checked") and not grounding.get("ok"):
                return json.dumps({
                    "status": "ungrounded_route_rejected",
                    "error": (
                        "Route is not a connected tree rooted at the target: "
                        + grounding.get("reason", "")
                    ),
                    "root_matches_target": grounding.get("root_matches_target"),
                    "n_dangling": grounding.get("n_dangling"),
                    "dangling_intermediates": grounding.get("dangling", [])[:8],
                    "hint": (
                        "Ensure exactly one step's PRODUCT is the canonical "
                        "target SMILES, and every intermediate product is "
                        "consumed as a reactant in a later step. Do not save a "
                        "route that synthesises a different molecule or whose "
                        "steps don't connect."
                    ),
                })

        entry = RouteEntry(
            label=str(label).strip() or f"route-{int(time.time())}",
            total_score=score,
            steps=steps_list,
            sub_scores=dict(sub_scores or {}),
            starting_materials=list(starting_materials or []),
            notes=str(notes or ""),
            provenance=str(provenance),
            rag_source_ids=list(rag_source_ids) if rag_source_ids else None,
            step_provenance=(
                list(step_provenance) if step_provenance is not None else None
            ),
        )
        # Backend dedup happens inside store.save() under its lock — the
        # tristate status distinguishes "duplicate" (structurally identical
        # route already on the scratchpad) from "rejected" (capacity-evicted
        # because the new entry didn't beat the current worst). Doing the
        # fingerprint check pre-save outside the lock would race with
        # concurrent calls.
        saved_entry, status = self._store.save(entry)
        routes = self._store.list()
        if status == "duplicate":
            return json.dumps({
                "status": "duplicate",
                "duplicate_of": saved_entry.label,
                "message": (
                    "Route is structurally identical to an earlier saved "
                    "route. Try a fundamentally different first-step "
                    "disconnection (e.g. switch chemistry class, or pick "
                    "a different bond from identify_bonds)."
                ),
            })
        if status == "updated":
            payload = {
                "ok": True,
                "status": "updated",
                "saved": saved_entry.summary(),
                "message": (
                    f"Re-scored existing route in place; total_score now "
                    f"{round(saved_entry.total_score, 4)}."
                ),
                "best_so_far": self._store.best().summary() if self._store.best() else None,
                "total_saved": len(routes),
            }
            if score_verification_warning:
                payload["warning"] = score_verification_warning
            return json.dumps(payload)
        if status == "rejected":
            payload = {
                "ok": False,
                "evicted": True,
                "reason": (
                    f"Scratchpad already holds {self._store.MAX_ROUTES} routes scoring higher; "
                    "this entry was not saved. Try a strategy that beats the current best."
                ),
                "rejected": saved_entry.summary(),
                "best_so_far": self._store.best().summary() if self._store.best() else None,
                "total_saved": len(routes),
            }
            if score_verification_warning:
                payload["warning"] = score_verification_warning
            return json.dumps(payload)
        payload = {
            "ok": True,
            "saved": saved_entry.summary(),
            "best_so_far": self._store.best().summary() if self._store.best() else None,
            "total_saved": len(routes),
        }
        if score_verification_warning:
            payload["warning"] = score_verification_warning
        return json.dumps(payload)


class ListBestRoutesTool(Tool):
    name = "list_best_routes"
    description = (
        "List candidate routes currently saved in the scratchpad, ordered "
        "by `total_score` descending (label, score, sub-scores, step count). "
        "Call before emitting `COMPLETE ROUTE FOUND` to confirm you're "
        "returning the best-scoring attempt, and during long searches to "
        "remind yourself what's already in the pool. Cost: <50ms, free. "
        "Pairs with `get_best_route` (list to find the label, get for the "
        "full payload)."
    )
    tier = 1
    parameters = {"type": "object", "properties": {}}

    def __init__(self, store: ScratchpadStore):
        self._store = store

    def execute(self, **kwargs) -> str:
        routes = self._store.list()
        return json.dumps({
            "count": len(routes),
            "routes": [r.summary() for r in routes],
            "lessons_learned": [l.to_dict() for l in self._store.lessons()],
        })


class GetBestRouteTool(Tool):
    name = "get_best_route"
    description = (
        "Return the full saved route payload (all steps, starting materials, "
        "notes) for a given label, or — with no label — the highest-scoring "
        "saved route. Call when you want to inspect or quote the exact step "
        "list of a saved attempt (e.g. before declaring `COMPLETE ROUTE "
        "FOUND` so you can transcribe it). Cost: <50ms, free. Pairs with "
        "`list_best_routes` (list first to pick the label)."
    )
    tier = 1
    parameters = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "description": "Omit to get the best route."},
        },
    }

    def __init__(self, store: ScratchpadStore):
        self._store = store

    def execute(self, label: Optional[str] = None, **kwargs) -> str:
        entry = self._store.get(label) if label else self._store.best()
        if entry is None:
            # Return a structured null instead of an error envelope so the LLM
            # doesn't infer the tool itself failed.
            return json.dumps({
                "ok": True,
                "route": None,
                "note": (
                    f"No route saved{' under label ' + label if label else ''}. "
                    "Save a route with `save_best_route` first."
                ),
            })
        return json.dumps({"ok": True, "route": entry.to_dict()})


class BacktrackAndSummarizeTool(Tool):
    name = "backtrack_and_summarize"
    description = (
        "Record a lesson-learned for a strategy that isn't working, then "
        "trigger context compaction to drop the failed branch's tool turns. "
        "Call when (a) you've exhausted a strategy and want to try a "
        "fundamentally different disconnection, or (b) the conversation is "
        "getting long and you want a clean slate without losing the best "
        "saved route. After this call, your context contains: system prompt, "
        "target, current todos, top saved routes, lessons. Cost: <100ms, "
        "free. Pairs with `save_best_route` (always save before backtracking "
        "so the route survives compaction)."
    )
    tier = 1
    parameters = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why the current strategy isn't working (one sentence).",
            },
            "failed_disconnection": {
                "type": "string",
                "description": "Short name of the disconnection that failed, e.g. 'direct-Suzuki-at-C3'.",
            },
            "lowest_sub_score": {
                "type": "string",
                "description": "Which sub-score tanked, e.g. 'binding_score=0.31'.",
            },
            "keep_best_n": {
                "type": "integer",
                "default": 5,
                "description": "How many top routes to keep in the compacted snapshot.",
            },
        },
        "required": ["reason"],
    }

    def __init__(self, store: ScratchpadStore):
        self._store = store

    def execute(
        self,
        reason: str,
        failed_disconnection: str = "",
        lowest_sub_score: str = "",
        keep_best_n: int = 5,
        **kwargs,
    ) -> str:
        lesson = Lesson(
            reason=str(reason or "").strip(),
            failed_disconnection=str(failed_disconnection or "").strip(),
            lowest_sub_score=str(lowest_sub_score or "").strip(),
        )
        self._store.add_lesson(lesson)

        snapshot = self._store.snapshot(keep_best_n=keep_best_n)
        payload = {
            "ok": True,
            "lesson_saved": lesson.to_dict(),
            "snapshot": snapshot,
            # Agent loop watches for this marker to force compaction.
            "_marker": BACKTRACK_MARKER,
            "keep_best_n": keep_best_n,
        }
        return json.dumps(payload)


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_scratchpad_tools(
    registry,
    max_routes: Optional[int] = None,
    result_store: Any | None = None,
) -> ScratchpadStore:
    """Register the scratchpad tool family on ``registry``.

    ``result_store`` (optional) is forwarded to ``SaveBestRouteTool`` so it
    can validate the LLM-supplied ``total_score`` against the persisted
    ``score_route`` payload pointed at by ``score_route_result_id``. Without
    it the verification is dormant — the LLM can still claim arbitrary
    scores. Wire it through from ``build_agent``.
    """
    store = (
        ScratchpadStore(max_routes=max_routes)
        if max_routes is not None
        else ScratchpadStore()
    )
    registry.register(SaveBestRouteTool(store, result_store=result_store))
    registry.register(ListBestRoutesTool(store))
    registry.register(GetBestRouteTool(store))
    registry.register(BacktrackAndSummarizeTool(store))
    from tools.builtins.rank_routes import RankRoutesTool
    registry.register(RankRoutesTool(scratchpad_store=store))
    return store
