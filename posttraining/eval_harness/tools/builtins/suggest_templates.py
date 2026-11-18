"""
suggest_templates — match a product SMILES against the curated template
library (USPTO + enzymatic DBs) and return the top-K most plausible
disconnections with their reactant sets.

This tool replaces the "LLM must invent SMARTS" workflow with a proper
lookup. It uses a cheap RDKit substructure pre-filter to avoid running
every one of the ~300k parsed templates against every product.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem

from tools.base import Tool
from tools.builtins.template_library import (
    classify_source,
    load_all_templates,
)

logger = logging.getLogger(__name__)


# Cache of compiled product-side SMARTS patterns, keyed by template index.
# Populated lazily as we walk the library.
_PATTERN_CACHE: Dict[int, Optional[Chem.Mol]] = {}


def clear_pattern_cache() -> None:
    """Drop all compiled SMARTS patterns. Must be called whenever the
    template library is reloaded/reordered (see template_library.reset_cache
    / set_dataset_dir) — the cache is keyed by integer template index, so a
    stale entry would silently match the SMARTS of whatever template now
    occupies that index."""
    _PATTERN_CACHE.clear()


def _strip_outer_parens(s: str) -> str:
    """Strip balanced outer parentheses from a SMARTS fragment.

    USPTO (aizynthfinder) and RetroRules templates wrap their product side
    in outer parens, e.g. ``([#7;a:4]:[c:3]...)``. RDKit's
    ``MolFromSmarts`` rejects that form. Multi-component product SMARTS
    such as ``(A).(B)`` are left alone — stripping would break them.
    """
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return s
    # Bail if parens aren't a single balanced outer pair.
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i != len(s) - 1:
                return s  # closes early → multiple top-level groups
    return s[1:-1].strip() if depth == 0 else s


def _compile_product_pattern(smarts: str) -> Optional[Chem.Mol]:
    """Compile the PRODUCT side of a retro template SMARTS into an RDKit
    pattern molecule, for use with ``HasSubstructMatch``."""
    try:
        lhs = smarts.split(">>", 1)[0]
        lhs = lhs.split(">")[0]
        lhs_clean = re.sub(r":\d+", "", lhs)
        lhs_clean = _strip_outer_parens(lhs_clean)
        return Chem.MolFromSmarts(lhs_clean)
    except Exception:
        return None


def _get_or_compile(idx: int, raw_templates: List[str]) -> Optional[Chem.Mol]:
    patt = _PATTERN_CACHE.get(idx, "__MISSING__")
    if patt != "__MISSING__":
        return patt  # type: ignore
    compiled = _compile_product_pattern(raw_templates[idx])
    _PATTERN_CACHE[idx] = compiled
    return compiled


def _canon(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    return Chem.MolToSmiles(m)


def _mw_balance(product: Chem.Mol, reactants_smi: str) -> float:
    """0..1 score, 1 = perfect MW balance across reactants."""
    from rdkit.Chem import Descriptors
    try:
        prod_mw = Descriptors.MolWt(product)
        parts = [p for p in reactants_smi.split(".") if p]
        if not parts:
            return 0.0
        mws = []
        for p in parts:
            m = Chem.MolFromSmiles(p)
            if m is None:
                return 0.0
            mws.append(Descriptors.MolWt(m))
        total = sum(mws)
        if total <= 0 or prod_mw <= 0:
            return 0.0
        # Penalize if the reactant set is way bigger/smaller than the product.
        ratio = min(total, prod_mw) / max(total, prod_mw)
        # Reward balanced split across reactants (closer to even is better).
        if len(mws) > 1:
            avg = total / len(mws)
            evenness = 1.0 - (max(mws) - min(mws)) / (total + 1e-9)
            evenness = max(0.0, min(1.0, evenness))
        else:
            evenness = 0.5
        return 0.7 * ratio + 0.3 * evenness
    except Exception:
        return 0.0


class SuggestTemplatesTool(Tool):
    name = "suggest_templates"
    description = (
        "Match the product against the curated retrosynthesis template "
        "library (USPTO organic + RetroBioCat / BRENDA / RetroRules / BKMS "
        "enzymatic) and return the top-K ranked disconnections, each with "
        "reactants, source database, and organic/enzymatic kind. Switch to "
        "`kind='enzymatic'` when the target is a chiral β-hydroxy / aldol / "
        "sugar / terpene / peptide scaffold where biocatalysis is the right "
        "play. Cost: ~0.5-2s/call, **counts toward the iteration budget**. "
        "Pairs with `neural_retro` (run both per non-leaf — templates know "
        "precedent, neural interpolates beyond it)."
    )
    tier = 2
    # Default cap on enzymatic suggestions in the final top-K when
    # kind="any". The library is dominated by enzymatic sources (BRENDA
    # ~280k templates), so without a cap the prefetch (kind unset)
    # routinely fills the top-K with enzymatic candidates even when the
    # chemistry should be organic. Caller can override via the
    # ``enzymatic_max_in_any`` argument; explicit ``kind="enzymatic"``
    # bypasses the cap.
    DEFAULT_ENZYMATIC_MAX_IN_ANY = 3

    def __init__(self, enzymatic_max_in_any: int = DEFAULT_ENZYMATIC_MAX_IN_ANY):
        self._enzymatic_max_in_any = int(enzymatic_max_in_any)

    parameters = {
        "type": "object",
        "properties": {
            "product_smiles": {
                "type": "string",
                "description": "Product SMILES to decompose.",
            },
            "kind": {
                "type": "string",
                "enum": ["any", "organic", "enzymatic"],
                "default": "any",
                "description": "Restrict to organic or enzymatic templates.",
            },
            "top_k": {
                "type": "integer",
                "default": 15,
                "description": "Number of ranked suggestions to return.",
            },
            "max_templates_to_try": {
                "type": "integer",
                "default": 4000,
                "description": (
                    "Cap on templates actually executed after pre-filter "
                    "(protects latency on large libraries)."
                ),
            },
            "enzymatic_max_in_any": {
                "type": "integer",
                "default": DEFAULT_ENZYMATIC_MAX_IN_ANY,
                "description": (
                    "When kind='any', cap the number of enzymatic suggestions "
                    "in the final top-K. The rest are organic. Negative or "
                    "null disables the cap. Ignored when kind='enzymatic' or "
                    "'organic'."
                ),
            },
        },
        "required": ["product_smiles"],
    }

    def execute(
        self,
        product_smiles: str,
        kind: str = "any",
        top_k: int = 15,
        max_templates_to_try: int = 4000,
        enzymatic_max_in_any: Optional[int] = None,
        **kwargs,
    ) -> str:
        if enzymatic_max_in_any is None:
            enzymatic_max_in_any = self._enzymatic_max_in_any
        mol = Chem.MolFromSmiles(product_smiles)
        if mol is None:
            return json.dumps({"error": f"Invalid SMILES: {product_smiles}"})

        try:
            raw_templates, sources = load_all_templates()
        except Exception as e:
            return json.dumps({
                "error": f"Template library not loaded: {e}",
                "hint": "Set templates.dataset_dir via config override or RETRO_TEMPLATE_DATASET_DIR.",
            })

        if not raw_templates:
            return json.dumps({
                "error": "No templates available.",
                "hint": (
                    "Template dataset not found. See "
                    "tools/builtins/template_library.py for search paths."
                ),
            })

        # Pre-filter: iterate the (kind-filtered) slice of the library and
        # compile product-side patterns lazily. Templates whose product
        # pattern doesn't substructure-match the target are dropped before
        # we pay for rdchiralReaction construction.
        #
        # Budgeting: when the enzymatic cap is active in kind="any" mode,
        # a single ``n_matched >= max_templates_to_try`` cutoff would let
        # an enzymatic-heavy library prefix saturate the budget before any
        # organic template is scanned — and the downstream per-kind
        # rdchiral pool can't backfill what the prefilter never collected.
        # Use per-kind budgets in that case so organic is guaranteed a
        # share of the prefilter budget.
        prefilter_capped = (
            kind == "any"
            and enzymatic_max_in_any is not None
            and enzymatic_max_in_any >= 0
        )
        if prefilter_capped:
            # Reserve enough organic slots to backfill the cap, plus a
            # comfortable oversampling factor so ranking has room. The
            # enzymatic share is whatever's left of the total budget.
            org_prefilter_budget = max(
                max_templates_to_try // 2,
                (top_k - enzymatic_max_in_any) * 6,
                top_k * 3,
            )
            enz_prefilter_budget = max(
                max_templates_to_try - org_prefilter_budget,
                enzymatic_max_in_any * 6,
            )
        else:
            org_prefilter_budget = max_templates_to_try
            enz_prefilter_budget = max_templates_to_try

        n_scanned = 0
        n_matched = 0
        n_enz_matched = 0
        n_org_matched = 0
        candidates: List[Tuple[int, str, str]] = []  # (idx, kind, smarts)
        for i, smarts in enumerate(raw_templates):
            src_kind = classify_source(sources[i])
            if kind != "any" and src_kind != kind:
                continue
            n_scanned += 1
            patt = _get_or_compile(i, raw_templates)
            if patt is None:
                continue
            try:
                if not mol.HasSubstructMatch(patt):
                    continue
            except Exception:
                continue
            # Per-kind budget enforcement: skip (don't break) once a
            # pool is full so the other pool can keep collecting.
            is_enz = src_kind == "enzymatic"
            if is_enz and n_enz_matched >= enz_prefilter_budget:
                # Both pools full → done.
                if n_org_matched >= org_prefilter_budget:
                    break
                continue
            if (not is_enz) and n_org_matched >= org_prefilter_budget:
                if n_enz_matched >= enz_prefilter_budget:
                    break
                continue
            n_matched += 1
            if is_enz:
                n_enz_matched += 1
            else:
                n_org_matched += 1
            candidates.append((i, src_kind, smarts))
            # Total-budget early exit (covers kind != "any" and the
            # cap-disabled case via the per-kind budgets above).
            if not prefilter_capped and n_matched >= max_templates_to_try:
                break

        logger.info(
            "suggest_templates: scanned=%d matched=%d (of %d total, kind=%s)",
            n_scanned, n_matched, len(raw_templates), kind,
        )

        # Run rdchiral on surviving candidates — parsing one SMARTS at a
        # time to avoid the 100k-upfront-parse that kills cold-start.
        try:
            from rdchiral.main import rdchiralRun
            from rdchiral.initialization import rdchiralReaction, rdchiralReactants
        except ImportError:
            return json.dumps({
                "error": "rdchiral not installed — cannot apply templates.",
            })

        rdc_product = rdchiralReactants(product_smiles)
        seen_sets: set = set()
        suggestions: List[dict] = []
        # The cap is applied at RANK time, not scan time. Scan-time
        # truncation drops enzymatic candidates in library scan order,
        # which discards the better-ranked enzymatic that happen to
        # appear later in the library. Instead we collect uncapped
        # (with separate per-kind budgets so a dense enzymatic prefix
        # doesn't starve organic backfill), sort by ``rank_score``,
        # then apply the cap by walking the sorted suggestions.
        enzymatic_capped = (
            kind == "any"
            and enzymatic_max_in_any is not None
            and enzymatic_max_in_any >= 0
        )
        # Collect at least 6× as many candidates per pool as we'll
        # surface so ranking has room to pick the top items. When the
        # cap is active, the enzymatic pool needs ≥ enzymatic_max_in_any
        # candidates AT MINIMUM but we still oversample to give ranking
        # something to work with.
        enz_pool_budget = max(top_k * 6, (enzymatic_max_in_any or 0) * 6)
        org_pool_budget = top_k * 6
        n_enz_collected = 0
        n_org_collected = 0
        # Count templates skipped due to bad SMARTS / rdchiral failures so a
        # systematically-broken library slice is visible instead of silently
        # shrinking the candidate pool (one summary line, not per-template).
        n_smarts_compile_fail = 0
        n_rdchiral_run_fail = 0

        for idx, src_kind, smarts in candidates:
            try:
                rxn = rdchiralReaction(smarts)
            except Exception:
                n_smarts_compile_fail += 1
                continue
            try:
                outcomes = rdchiralRun(rxn, rdc_product)
            except Exception:
                n_rdchiral_run_fail += 1
                continue
            for outcome in outcomes:
                try:
                    canonical_parts = sorted({_canon(p) for p in outcome.split(".") if p})
                except Exception:
                    continue
                if not canonical_parts:
                    continue
                reactants_key = ".".join(canonical_parts)
                if reactants_key in seen_sets:
                    continue

                is_enz = src_kind == "enzymatic"
                # Per-pool budget: stop adding to a pool that's full so
                # we don't waste time scanning past it. We DON'T cap by
                # ``enzymatic_max_in_any`` here — that's a ranking
                # concern, applied below.
                if is_enz and n_enz_collected >= enz_pool_budget:
                    continue
                if (not is_enz) and n_org_collected >= org_pool_budget:
                    continue

                seen_sets.add(reactants_key)
                balance = _mw_balance(mol, reactants_key)
                # Scoring:
                #  * enzymatic boost when the user asked for enzymatic
                #  * prefer 2-reactant disconnections
                #  * weight by MW balance
                n_r = len(canonical_parts)
                n_pref_bonus = 0.25 if n_r == 2 else (0.1 if n_r == 3 else 0.0)
                kind_bonus = 0.15 if (kind == "enzymatic" and src_kind == "enzymatic") else 0.0
                score = 0.6 * balance + n_pref_bonus + kind_bonus
                suggestions.append({
                    "template_source": sources[idx],
                    "template_kind": src_kind,
                    "reactants": canonical_parts,
                    "reactants_smiles": reactants_key,
                    "num_reactants": n_r,
                    "rank_score": round(score, 4),
                    "mw_balance": round(balance, 4),
                })
                if is_enz:
                    n_enz_collected += 1
                else:
                    n_org_collected += 1
            # Outer break: stop scanning once both pools are saturated.
            # When the cap is inactive we only need ``top_k * 10`` total
            # to match the historical budget.
            if enzymatic_capped:
                if (
                    n_enz_collected >= enz_pool_budget
                    and n_org_collected >= org_pool_budget
                ):
                    break
            else:
                if len(suggestions) >= top_k * 10:
                    break

        if n_smarts_compile_fail or n_rdchiral_run_fail:
            logger.warning(
                "suggest_templates: skipped %d templates (%d SMARTS compile "
                "failures, %d rdchiral run failures) of %d candidates scanned",
                n_smarts_compile_fail + n_rdchiral_run_fail,
                n_smarts_compile_fail, n_rdchiral_run_fail, len(candidates),
            )

        suggestions.sort(key=lambda s: s["rank_score"], reverse=True)

        # Cap the enzymatic count during top_k selection: walk the
        # ranked list and take suggestions in score order, dropping
        # enzymatic candidates once we already have
        # ``enzymatic_max_in_any`` of them. This keeps the highest-
        # ranked enzymatic regardless of where they appeared in the
        # library scan.
        if enzymatic_capped:
            top: List[dict] = []
            enz_in_top = 0
            for s in suggestions:
                if s["template_kind"] == "enzymatic":
                    if enz_in_top >= enzymatic_max_in_any:
                        continue
                    enz_in_top += 1
                top.append(s)
                if len(top) >= top_k:
                    break
        else:
            top = suggestions[: top_k]

        enzymatic_in_top = sum(
            1 for s in top if s.get("template_kind") == "enzymatic"
        )

        return json.dumps({
            "product": _canon(product_smiles),
            "library_size": len(raw_templates),
            "candidates_after_prefilter": len(candidates),
            "num_suggestions": len(top),
            "enzymatic_in_top_k": enzymatic_in_top,
            "suggestions": top,
        })
