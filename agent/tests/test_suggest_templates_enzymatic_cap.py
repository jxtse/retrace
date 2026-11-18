"""Enzymatic cap regression for ``suggest_templates``.

Bug: the cap was applied AFTER candidate collection stopped at
``top_k * 10``. If the first 10×top_k matches happened to all be
enzymatic, the cap dropped them to ≤ ``enzymatic_max_in_any`` with no
second-pass scan to backfill organic candidates — kind="any" returned
far fewer than ``top_k`` items, contradicting the parameter docstring.

Fix: fold the cap into the in-loop collection so we keep scanning the
(organic) library after the cap is hit and reach the requested ``top_k``.

Reproduction strategy: build a synthetic library where the enzymatic
prefix yields enough distinct rdchiral outcomes (after dedup) to trip
the buggy outer break ``len(suggestions) >= top_k * 10`` BEFORE any
organic template is processed.
"""

import json
from unittest.mock import patch


def _generate_leaving_groups(n):
    """Return n leaving-group SMILES designed so each rdchiral run on
    toluene produces a unique canonical reactant set after dedup.

    Strategy: vary only the chain length on a few halogen anchors
    (I, Br, F). Each yields a distinct ``CH3-LG`` SMILES on toluene
    because the chain length is preserved through canonicalization.
    """
    out: list[str] = []
    bases = ["I", "Br", "F"]
    # Three chain-0 forms first so the first three are short.
    out.extend(bases)
    # Then increasing chain lengths on iodide (most distinct), padded
    # with bromide / fluoride if more candidates are needed.
    for k in range(1, n):
        out.append(f"I{'C' * k}")
        if len(out) >= n:
            break
    if len(out) < n:
        for k in range(1, n):
            out.append(f"Br{'C' * k}")
            if len(out) >= n:
                break
    return out[:n]


def _build_synthetic_library(n_enz, n_org):
    """Return (templates, sources) — n_enz enzymatic followed by n_org
    organic, all matching toluene's methyl-arene bond and producing
    distinct reactants."""
    leaving_groups = _generate_leaving_groups(n_enz + n_org)
    templates = [
        f"[c:1][CH3:2]>>[c:1][H].[CH3:2]{lg}"
        for lg in leaving_groups
    ]
    sources = (
        [f"BRENDA:enz_{i}" for i in range(n_enz)]
        + [f"USPTO:org_{i}" for i in range(n_org)]
    )
    return templates, sources


def test_enzymatic_cap_does_not_starve_top_k():
    """kind='any' with a dense enzymatic prefix MUST still return up to
    top_k items by backfilling organic candidates after the cap fires.

    Bug condition: the buggy outer loop breaks at
    ``len(suggestions) >= top_k * 10``. Picking ``top_k=2`` so the
    threshold is 20, then queueing 30 distinct enzymatic outcomes
    BEFORE any organic template, deterministically trips the outer
    break before organic templates are scanned. Under the buggy
    post-hoc cap, the result is then capped to ≤ enzymatic_max_in_any
    with no organic backfill available — the final list is shorter
    than top_k.
    """
    from tools.builtins.suggest_templates import SuggestTemplatesTool
    import tools.builtins.suggest_templates as st

    raw_templates, sources = _build_synthetic_library(n_enz=30, n_org=10)
    tool = SuggestTemplatesTool(enzymatic_max_in_any=1)

    with patch(
        "tools.builtins.suggest_templates.load_all_templates",
        return_value=(raw_templates, sources),
    ):
        # Reset module-level pattern cache: cache is keyed by template
        # index, which now points at toy SMARTS instead of whatever
        # the live library happened to set in a previous test run.
        st._PATTERN_CACHE.clear()
        out = json.loads(tool.execute(
            product_smiles="Cc1ccccc1",
            kind="any",
            top_k=2,
            max_templates_to_try=200,
        ))

    assert "error" not in out, out
    sugg = out["suggestions"]
    assert len(sugg) == 2, (
        f"cap starved top_k: got {len(sugg)}/2 with enzymatic_max_in_any=1 "
        f"(expected the in-loop cap to keep scanning organic templates "
        f"so we backfill to top_k)"
    )
    n_enz = sum(1 for s in sugg if s["template_kind"] == "enzymatic")
    assert n_enz <= 1, f"cap not respected: {n_enz} enzymatic in top-2"
    n_org = sum(1 for s in sugg if s["template_kind"] == "organic")
    assert n_org >= 1, (
        f"organic backfill failed: {n_org} organic suggestions (expected ≥ 1) — "
        "the in-loop cap should keep scanning past the enzymatic prefix"
    )


def test_enzymatic_cap_disabled_when_negative():
    """A negative cap disables the cap entirely — kind='any' returns
    whatever the ranker picks, regardless of how many are enzymatic."""
    from tools.builtins.suggest_templates import SuggestTemplatesTool
    import tools.builtins.suggest_templates as st

    raw_templates, sources = _build_synthetic_library(n_enz=20, n_org=10)
    tool = SuggestTemplatesTool(enzymatic_max_in_any=-1)

    with patch(
        "tools.builtins.suggest_templates.load_all_templates",
        return_value=(raw_templates, sources),
    ):
        st._PATTERN_CACHE.clear()
        out = json.loads(tool.execute(
            product_smiles="Cc1ccccc1",
            kind="any",
            top_k=10,
            max_templates_to_try=50,
        ))

    assert "error" not in out, out
    sugg = out["suggestions"]
    assert len(sugg) == 10
    n_enz = sum(1 for s in sugg if s["template_kind"] == "enzymatic")
    assert n_enz > 3, (
        f"negative cap should not cap: only {n_enz} enzymatic in top-10"
    )


def test_enzymatic_kind_explicit_bypasses_cap():
    """kind='enzymatic' must bypass the cap and return enzymatic-only."""
    from tools.builtins.suggest_templates import SuggestTemplatesTool
    import tools.builtins.suggest_templates as st

    raw_templates, sources = _build_synthetic_library(n_enz=20, n_org=10)
    tool = SuggestTemplatesTool(enzymatic_max_in_any=3)

    with patch(
        "tools.builtins.suggest_templates.load_all_templates",
        return_value=(raw_templates, sources),
    ):
        st._PATTERN_CACHE.clear()
        out = json.loads(tool.execute(
            product_smiles="Cc1ccccc1",
            kind="enzymatic",
            top_k=10,
            max_templates_to_try=50,
        ))

    assert "error" not in out, out
    sugg = out["suggestions"]
    assert len(sugg) == 10
    assert all(s["template_kind"] == "enzymatic" for s in sugg), (
        "kind='enzymatic' must return enzymatic-only suggestions, "
        "regardless of the cap"
    )


def test_prefilter_does_not_starve_organic_when_cap_active():
    """Prefilter must reserve organic budget when the cap is active.

    Bug: even with the rank-time cap, the prefilter loop still stops at
    a single ``n_matched >= max_templates_to_try`` budget. If the
    library's enzymatic prefix is large enough to fill that budget
    before any organic template is scanned, organic templates never
    enter the ``candidates`` list — and the per-kind rdchiral pool
    can't backfill what the prefilter never collected.

    Setup: ``max_templates_to_try=10`` with 15 enzymatic matchers
    BEFORE 5 organic matchers. The buggy prefilter stops at 10
    enzymatic candidates, sees zero organic, and returns ≤ cap items.

    Fix: when the cap is active, the prefilter must keep scanning past
    a saturated enzymatic budget until the organic pool is also
    sufficient (or the library is exhausted).
    """
    from tools.builtins.suggest_templates import SuggestTemplatesTool
    import tools.builtins.suggest_templates as st

    raw_templates, sources = _build_synthetic_library(n_enz=15, n_org=5)
    tool = SuggestTemplatesTool(enzymatic_max_in_any=2)

    with patch(
        "tools.builtins.suggest_templates.load_all_templates",
        return_value=(raw_templates, sources),
    ):
        st._PATTERN_CACHE.clear()
        out = json.loads(tool.execute(
            product_smiles="Cc1ccccc1",
            kind="any",
            top_k=4,
            # Tight prefilter budget: less than the enzymatic prefix so
            # a naive single-budget prefilter starves organic entirely.
            max_templates_to_try=10,
        ))

    assert "error" not in out, out
    sugg = out["suggestions"]
    n_org = sum(1 for s in sugg if s["template_kind"] == "organic")
    assert n_org >= 1, (
        f"prefilter starved organic: got 0 organic in {len(sugg)} "
        f"suggestions (expected ≥ 1 — the prefilter must keep scanning "
        f"past a saturated enzymatic budget when the cap is active)"
    )
    assert len(sugg) >= 3, (
        f"top_k starved by prefilter cap: got {len(sugg)}/4 suggestions "
        f"(expected ≥ 3 — the prefilter must collect enough organic to "
        f"backfill the cap-induced enzymatic shortfall)"
    )


def test_enzymatic_cap_picks_highest_ranked_enzymatic_not_first_seen():
    """Cap must select the top-ranked enzymatic candidates, NOT the
    ones encountered first in the library.

    Bug: the in-loop cap dropped enzymatic candidates in library scan
    order, so a high-``rank_score`` enzymatic that appeared later in
    the library was discarded while a low-``rank_score`` enzymatic
    that came first survived. This violates the documented semantic
    of "top-K ranked disconnections, at most N enzymatic".

    Setup:
      - 5 enzymatic templates yielding LOW-balance outcomes (long-chain
        leaving groups → poor MW match against toluene's modest size).
      - 5 enzymatic templates yielding HIGH-balance outcomes (short
        leaving groups → near-1:1 MW match).
      - cap = 2.

    The buggy code keeps the first 2 enzymatic (low-rank) and drops
    the high-rank ones; the fix should keep the 2 highest-ranked
    enzymatic candidates regardless of library order.
    """
    from tools.builtins.suggest_templates import SuggestTemplatesTool
    import tools.builtins.suggest_templates as st

    # Long-chain (poor MW balance) enzymatic templates first…
    long_lgs = [f"I{'C' * k}" for k in range(20, 28)]   # 8 templates, ICCCC...
    # …then short-chain (excellent MW balance) enzymatic templates.
    short_lgs = ["I", "Br", "F", "IC", "ICC"]            # 5 templates
    # Plus 5 organic so the top-K can still be filled.
    organic_lgs = [f"Br{'C' * k}" for k in range(1, 6)]  # 5 templates, BrCCC...

    enz_lgs = long_lgs + short_lgs
    leaving_groups = enz_lgs + organic_lgs

    templates = [
        f"[c:1][CH3:2]>>[c:1][H].[CH3:2]{lg}"
        for lg in leaving_groups
    ]
    sources = (
        [f"BRENDA:enz_long_{i}" for i in range(len(long_lgs))]
        + [f"BRENDA:enz_short_{i}" for i in range(len(short_lgs))]
        + [f"USPTO:org_{i}" for i in range(len(organic_lgs))]
    )

    tool = SuggestTemplatesTool(enzymatic_max_in_any=2)

    with patch(
        "tools.builtins.suggest_templates.load_all_templates",
        return_value=(templates, sources),
    ):
        st._PATTERN_CACHE.clear()
        out = json.loads(tool.execute(
            product_smiles="Cc1ccccc1",
            kind="any",
            top_k=4,
            max_templates_to_try=200,
        ))

    assert "error" not in out, out
    sugg = out["suggestions"]
    enz = [s for s in sugg if s["template_kind"] == "enzymatic"]
    assert len(enz) <= 2, f"cap not respected: {len(enz)} enzymatic > 2"

    # The kept enzymatic suggestions must come from the SHORT-leaving-
    # group block (high MW balance / high rank_score), not the LONG
    # block. The bug keeps the first 2 long-chain enzymatic and drops
    # the better-ranked short ones.
    enz_sources = [s["template_source"] for s in enz]
    for src in enz_sources:
        assert src.startswith("BRENDA:enz_short_"), (
            f"cap must pick top-ranked enzymatic, not first-seen — got "
            f"{enz_sources} (expected only enz_short_*)"
        )
