"""
explore_branches — cross-source consensus ranker for candidate disconnections.

The agent supplies K candidate first-step disconnections (each a
reactant SMILES set). The tool reverse-looks-up the agent's prior
`neural_retro`, `suggest_templates`, and explicit `enzymatic_retro`
results in the ResultStore to attach a `support` record per candidate,
then ranks them with a
verifier-aligned score:

    score = 0.70 * feasibility + 0.15 * cost + 0.15 * greenness

    feasibility = 0.60 * support_signal           # cross-source agreement
                + 0.30 * carbon_conservation      # no fabricated skeletal carbon
                + 0.10 * stereo_ok

    cost        = 0.60 * leaf_buyability
                + 0.40 * sa_decrease

    greenness   = atom_economy

These are the same three pillars (and the same default 0.70/0.15/0.15
weighting) used by score_route, so candidates that win the pre-filter
are statistically the same that would win an expensive full-route
verifier pass.

Iteration budget: this tool does NOT count against
agent.iteration_billable_tools — only suggest_templates and
neural_retro do. Use it freely as a parallel beam pre-filter
before committing to a disconnection.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from rdkit import Chem

from tools.base import Tool
from services.concurrent_tools import run_parallel_branches

# Verifier sub-modules we reuse so pre-filter and verifier rank by the
# same axes. See spec §2.3.
from tools.verifier.carbon_skeleton import carbon_delta
from tools.verifier.stereo_check import warning_for_step
from tools.verifier.scscore import sa_decrease_score
from tools.verifier.score_route import atom_economy

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────


def _canon(smi: str) -> Optional[str]:
    """Canonical SMILES; None on parse failure."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def _canon_set(smiles_list: List[str]) -> Optional[frozenset[str]]:
    """Frozen set of canonical SMILES; None if any SMILES fails to parse."""
    canon: List[str] = []
    for s in smiles_list:
        c = _canon(s)
        if c is None:
            return None
        canon.append(c)
    return frozenset(canon)


def _normalize_branch(raw: Any, idx: int) -> Optional[dict]:
    """Accept dict-shaped or raw-list branches. Returns the dict shape
    {name, reactants, notes} or None on failure."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return None
    if isinstance(raw, list):
        return {"name": f"branch-{idx}", "reactants": list(raw), "notes": ""}
    if isinstance(raw, dict):
        reactants = raw.get("reactants") or raw.get("precursors") or []
        if isinstance(reactants, str):
            reactants = [p for p in reactants.split(".") if p]
        if not reactants:
            return None
        return {
            "name": str(raw.get("name") or raw.get("label") or f"branch-{idx}"),
            "reactants": list(reactants),
            "notes": str(raw.get("notes") or raw.get("reaction") or ""),
        }
    return None


# ── reverse-lookup: gather support evidence from prior tool calls ─────


def _collect_prior_predictions(
    result_store: Any,
    product_canon: str,
) -> List[Dict[str, Any]]:
    """Return the union of (source, rank, reactants_canon_set) records
    from every prior neural_retro / suggest_templates / enzymatic_retro
    call whose product canonicalises to `product_canon`.

    Each record:
        {"source": "neural_retro:graph_retrosyn", "rank": 2,
         "reactants": frozenset({"CCO", "CC=O"})}

    The (engine|template-source) → "source" mapping is what gives us
    cross-source counting in support_signal — multiple ranks from the
    SAME source key collapse to one match (see _support_signal).
    """
    if result_store is None:
        return []
    out: List[Dict[str, Any]] = []
    for tool_name in ("neural_retro", "suggest_templates", "enzymatic_retro"):
        try:
            iterator = result_store.iter_by_tool(tool_name)
        except Exception as e:
            logger.debug("iter_by_tool(%s) failed: %s", tool_name, e)
            continue
        for _, payload in iterator:
            if not isinstance(payload, dict):
                continue
            # Product key differs per tool: neural_retro uses "target",
            # suggest_templates uses "product".
            persisted_product = payload.get("target") or payload.get("product")
            if not persisted_product:
                continue
            persisted_canon = _canon(persisted_product)
            if persisted_canon != product_canon:
                continue
            if tool_name == "neural_retro":
                out.extend(_extract_neural(payload))
            else:
                out.extend(_extract_templates(payload, source_prefix=tool_name))
    return out


def _extract_neural(payload: dict) -> List[Dict[str, Any]]:
    """neural_retro predictions → support records.

    Ensemble payloads put the engine name in each prediction's `support`
    list. Single-engine calls put it at the envelope level.
    """
    envelope_engine = payload.get("engine") or "unknown"
    records: List[Dict[str, Any]] = []
    for i, p in enumerate(payload.get("predictions") or []):
        if not isinstance(p, dict):
            continue
        # Reactants may be a list or a "A.B.C" string.
        rxs = p.get("reactants")
        if isinstance(rxs, str):
            parts = [x for x in rxs.split(".") if x]
        elif isinstance(rxs, list):
            parts = list(rxs)
        else:
            continue
        cset = _canon_set(parts)
        if cset is None:
            continue
        sources = p.get("support") or [envelope_engine]
        for src in sources:
            records.append({
                "source": f"neural_retro:{src}",
                "rank": i + 1,
                "reactants": cset,
            })
    return records


def _extract_templates(
    payload: dict,
    *,
    source_prefix: str = "suggest_templates",
) -> List[Dict[str, Any]]:
    """Template-shaped suggestions → support records."""
    records: List[Dict[str, Any]] = []
    for i, s in enumerate(payload.get("suggestions") or []):
        if not isinstance(s, dict):
            continue
        rxs = s.get("reactants_smiles") or s.get("reactants")
        if isinstance(rxs, str):
            parts = [x for x in rxs.split(".") if x]
        elif isinstance(rxs, list):
            parts = list(rxs)
        else:
            continue
        cset = _canon_set(parts)
        if cset is None:
            continue
        kind = s.get("template_kind") or s.get("template_source") or "any"
        records.append({
            "source": f"{source_prefix}:{kind}",
            "rank": i + 1,
            "reactants": cset,
        })
    return records


def _classify_match(branch_set: frozenset[str], pred_set: frozenset[str]) -> str:
    """Strict / partial / none."""
    if branch_set == pred_set:
        return "strict"
    if branch_set < pred_set or pred_set < branch_set:
        return "partial"
    return "none"


def _support_for_branch(
    branch_set: frozenset[str],
    prior: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Best (highest-priority) match per source.

    Returns a typed list:
        [{"source": ..., "match": "strict"|"partial", "rank": int}, ...]

    Sort key: strict > partial; lower rank > higher rank.
    """
    rank_by_source: Dict[str, Dict[str, Any]] = {}
    for rec in prior:
        match = _classify_match(branch_set, rec["reactants"])
        if match == "none":
            continue
        candidate = {"source": rec["source"], "match": match, "rank": rec["rank"]}
        existing = rank_by_source.get(rec["source"])
        if existing is None or _better_match(candidate, existing):
            rank_by_source[rec["source"]] = candidate
    return list(rank_by_source.values())


def _better_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """a beats b if it's strict-vs-partial, or same match-type with lower rank."""
    if a["match"] != b["match"]:
        return a["match"] == "strict"
    return a["rank"] < b["rank"]


def _support_signal(support: List[Dict[str, Any]]) -> float:
    """0..1 — see spec §2.4.1."""
    n_strict = sum(1 for s in support if s["match"] == "strict")
    n_partial = sum(1 for s in support if s["match"] == "partial")
    return min(1.0, 0.40 * n_strict + 0.20 * n_partial)


# ── tool ──────────────────────────────────────────────────────────────


class ExploreBranchesTool(Tool):
    """Rank K candidate first-step disconnections by verifier-aligned pillars."""

    name = "explore_branches"
    description = (
        "Score K alternative first-step disconnections of the same product "
        "in a single turn, ranked by the SAME three pillars score_route "
        "uses (feasibility 0.70 / cost 0.15 / greenness 0.15). Cross-source "
        "agreement with prior neural_retro / suggest_templates / explicit "
        "enzymatic_retro calls is the "
        "cheap proxy for reaction_rater feasibility — no extra model calls, no "
        "iteration budget consumed. Use this AFTER you have run "
        "suggest_templates and neural_retro at least once on the same product; "
        "explicit enzymatic_retro results for that product also contribute "
        "source/rank support."
    )
    tier = 2
    parameters = {
        "type": "object",
        "properties": {
            "product_smiles": {
                "type": "string",
                "description": "Product SMILES being disconnected.",
            },
            "branches": {
                "type": "array",
                "description": (
                    "Candidate disconnections. Each item: "
                    "{label: str, reactants: [SMILES, ...], notes: str}. "
                    "Minimum 1, max 12."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "reactants": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "notes": {"type": "string"},
                    },
                    "required": ["reactants"],
                },
            },
            "top_k": {
                "type": "integer",
                "default": 3,
                "description": "How many top-ranked branches to return.",
            },
            "max_workers": {
                "type": "integer",
                "default": 8,
                "description": "Parallelism cap.",
            },
        },
        "required": ["product_smiles", "branches"],
    }

    def __init__(
        self,
        buyability_chain: Any,
        result_store: Any | None = None,
    ):
        self._buyability = buyability_chain
        self._result_store = result_store

    # ── per-branch scoring ────────────────────────────────────────────

    def _score_branch(
        self,
        branch: dict,
        product_canon: str,
        prior: List[Dict[str, Any]],
    ) -> dict:
        reactants_raw = branch.get("reactants") or []
        canon: List[str] = []
        for r in reactants_raw:
            c = _canon(r)
            if c is None:
                return {
                    "name": branch.get("name"),
                    "error": f"Invalid reactant SMILES: {r}",
                    "score": 0.0,
                }
            canon.append(c)
        if not canon:
            return {"name": branch.get("name"), "error": "No reactants",
                    "score": 0.0}

        branch_set = frozenset(canon)

        # ── feasibility proxy ────────────────────────────────────────
        if self._result_store is not None:
            support = _support_for_branch(branch_set, prior)
            sup_signal = _support_signal(support)
            sup_note = None
        else:
            support = []
            sup_signal = 0.5
            sup_note = "no result_store — neutral support_signal"

        # Carbon-skeleton sanity: a POSITIVE carbon_delta means the product has
        # more skeletal carbons than every precursor combined (carbon fabricated
        # from nowhere) — a broken disconnection. A negative delta (a fragment
        # leaves) is benign. This is NOT mass balance; only fabricated carbon is
        # penalised.
        carbon_score = 0.0 if carbon_delta(canon, product_canon) > 0 else 1.0
        stereo_warn = warning_for_step(canon, product_canon)
        stereo_score = 1.0 if stereo_warn is None else 0.5
        feasibility = (
            0.60 * sup_signal
            + 0.30 * carbon_score
            + 0.10 * stereo_score
        )

        # ── cost proxy ──────────────────────────────────────────────
        n_buy = 0
        for c in canon:
            try:
                hit = self._buyability.lookup(c)
            except Exception as e:
                logger.debug("buyability.lookup failed for %s: %s", c, e)
                hit = {"source": "none"}
            if hit.get("source") and hit["source"] != "none":
                n_buy += 1
        leaf_buy = n_buy / len(canon)
        try:
            sa_dec = sa_decrease_score(product_canon, canon)
        except (ValueError, RuntimeError) as e:
            logger.warning(
                "explore_branches: sa_decrease_score unavailable for %s: %s "
                "(falling back to neutral 0.5)",
                product_canon, e,
            )
            sa_dec = 0.5
        cost = 0.60 * leaf_buy + 0.40 * sa_dec

        # ── greenness proxy ─────────────────────────────────────────
        ae = atom_economy(canon, product_canon)

        score = max(0.0, min(1.0,
                             0.70 * feasibility
                             + 0.15 * cost
                             + 0.15 * ae))

        out: Dict[str, Any] = {
            "name": branch.get("name"),
            "notes": branch.get("notes", ""),
            "reactants": canon,
            "num_reactants": len(canon),
            "score": round(score, 4),
            "feasibility": round(feasibility, 4),
            "cost": round(cost, 4),
            "greenness": round(ae, 4),
            "sub_scores": {
                "support_signal": round(sup_signal, 4),
                "carbon_conservation": carbon_score,
                "stereo_ok": stereo_score,
                "leaf_buyability": round(leaf_buy, 4),
                "sa_decrease": round(sa_dec, 4),
                "atom_economy": round(ae, 4),
            },
            "support": support,
        }
        if sup_note:
            out["note"] = sup_note
        return out

    # ── execute ───────────────────────────────────────────────────────

    def execute(
        self,
        product_smiles: str,
        branches: list[Any],
        top_k: int = 3,
        max_workers: int = 8,
        **kwargs,
    ) -> str:
        if isinstance(branches, str):
            try:
                branches = json.loads(branches)
            except Exception:
                return json.dumps(
                    {"error": "branches must be a list or JSON array"}
                )
        if not isinstance(branches, list) or len(branches) < 1:
            return json.dumps(
                {"error": "branches must contain at least one candidate"}
            )
        if len(branches) > 12:
            branches = branches[:12]

        product_canon = _canon(product_smiles)
        if product_canon is None:
            return json.dumps(
                {"error": f"Invalid product SMILES: {product_smiles}"}
            )

        normalized: List[dict] = []
        for i, b in enumerate(branches):
            nb = _normalize_branch(b, i)
            if nb is not None:
                normalized.append(nb)
        if not normalized:
            return json.dumps({"error": "No valid branches after normalization"})

        # Pull prior tool results ONCE per execute (not per-branch).
        prior = _collect_prior_predictions(self._result_store, product_canon)

        results = run_parallel_branches(
            branches=normalized,
            scorer=lambda br: self._score_branch(br, product_canon, prior),
            max_workers=max_workers,
        )

        ranked = sorted(results,
                        key=lambda r: r.get("score", 0.0),
                        reverse=True)
        for r in ranked:
            r.pop("branch", None)
            r.pop("_index", None)
        top = ranked[: max(1, int(top_k))]

        return json.dumps({
            "product": product_canon,
            "num_branches_evaluated": len(results),
            "top_k": len(top),
            "ranked": top,
            "all_scores": [
                {"name": r.get("name"), "score": r.get("score", 0.0)}
                for r in ranked
            ],
        })
