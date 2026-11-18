"""Helpers for the multi-engine `neural_retro` ensemble.

Clustering recipe ported from ``分子前体聚类demo.ipynb``:

  * For each candidate, parse dot-split fragments → per-fragment Morgan-2
    bit vectors (radius=2, nBits=2048, **useChirality=True**).
  * OR-merge the fragment fps into a single combined fingerprint per
    candidate (so "A.B" and "B.A" hash identically).
  * Pairwise Tanimoto distance lower-triangle → ``RDKit Butina.ClusterData``
    at ``distThresh=0.5`` with ``reordering=True``.
  * Take **one representative per cluster** (highest aggregated score,
    ties broken by support count) and globally rerank, capped at
    ``return_topk``. This is the "filter by structural difference to
    reduce LLM context pressure" step requested by the user.

The legacy ``bucket_by_classifier`` / ``pick_per_bucket`` names are kept
as thin shims so the existing call site in ``tools/neural_retro.py``
keeps working without edits.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonicalization + dedup
# ---------------------------------------------------------------------------


def _canon_one(smi: str) -> Optional[str]:
    try:
        from rdkit import Chem
    except ImportError:
        return smi
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def canonicalize_reactants(reactants_smi: str) -> Optional[str]:
    if not reactants_smi:
        return None
    parts = []
    for frag in reactants_smi.split("."):
        frag = frag.strip()
        if not frag:
            continue
        c = _canon_one(frag)
        if c is None:
            return None
        parts.append(c)
    if not parts:
        return None
    parts.sort()
    return ".".join(parts)


def dedup_predictions(
    engine_results: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for engine, preds in engine_results.items():
        for entry in preds or []:
            raw = entry.get("reactants", "") or ""
            key = canonicalize_reactants(raw)
            if key is None:
                logger.warning("Dropping prediction with invalid reactants SMILES: %s", raw)
                continue
            score = float(entry.get("score") or 0.0)
            template = entry.get("template")

            cur = merged.get(key)
            if cur is None:
                merged[key] = {
                    "reactants": key,
                    "score": score,
                    "support": [engine],
                    "templates": [template] if template else [],
                }
            else:
                if score > cur["score"]:
                    cur["score"] = score
                if engine not in cur["support"]:
                    cur["support"].append(engine)
                if template and template not in cur["templates"]:
                    cur["templates"].append(template)
    out = list(merged.values())
    for e in out:
        e["support"].sort()
    return out


# ---------------------------------------------------------------------------
# Clustering — notebook recipe (Morgan2 useChirality=True, OR-merge, Butina)
# ---------------------------------------------------------------------------


def _combined_morgan_fp(reactants_smi: str, radius: int = 2, n_bits: int = 2048):
    """OR-merge per-fragment Morgan FPs into a single combined fingerprint.

    Mirrors ``_combine_morgan_fps(method='or')`` from the notebook.
    Uses ``useChirality=True`` so stereo-distinct precursors don't collide.
    Returns ``None`` if no fragment parses.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.DataStructs.cDataStructs import ExplicitBitVect

    bits = ExplicitBitVect(n_bits)
    any_ok = False
    for frag in reactants_smi.split("."):
        frag = frag.strip()
        if not frag:
            continue
        m = Chem.MolFromSmiles(frag)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(
            m, radius, nBits=n_bits, useChirality=True
        )
        bits |= fp
        any_ok = True
    return bits if any_ok else None


def cluster_candidates(
    candidates: Sequence[Dict[str, Any]],
    distance_threshold: float = 0.65,
    radius: int = 2,
    n_bits: int = 2048,
    return_topk: int = 15,
    k_per_cluster: int = 2,
    return_meta: bool = False,
):
    """Cluster candidates by structural similarity, return up to k_per_cluster
    reps per cluster.

    Recipe: combined Morgan fingerprint per candidate (per-fragment Morgan-2
    OR-merged) → pairwise Tanimoto distances → ``Butina.ClusterData(...,
    isDistData=True, reordering=True)``. From each cluster keep the top
    ``k_per_cluster`` entries ranked by ``(score, len(support))``, tag each
    with a ``bucket`` id (descending cluster size) and ``cluster_size``,
    then globally rerank and truncate to ``return_topk``. Candidates that
    fail to fingerprint are kept as singleton clusters so nothing is
    silently dropped.

    Defaults relative to the original notebook recipe (May 2026 tuning):
      * ``distance_threshold`` 0.5 → 0.65: at 0.5 the OR-merged Morgan FP
        collapses chemically-related candidates that share the leftover
        scaffold into very few clusters (often 2-4), leaving the LLM with
        only 2-3 predictions even when 30-60 unique candidates entered.
      * ``k_per_cluster`` 1 → 2: keep one runner-up per cluster so a
        ring-broken vs ring-kept variant survives even when the FP says
        they're close.
      * ``return_topk`` 12 → 15: small bump to absorb the extra reps.

    When ``return_meta=True``, returns ``(reps, meta)`` where ``meta`` is
    ``{"n_clusters": int, "n_input": int, "n_returned": int}``. This is
    what the result-store stub uses to surface the funnel
    ``n_unique → n_clusters → n_in_stub`` to the model.
    """
    if not candidates:
        if return_meta:
            return [], {"n_clusters": 0, "n_input": 0, "n_returned": 0}
        return []

    try:
        from rdkit import DataStructs
        from rdkit.ML.Cluster import Butina
    except Exception as e:
        logger.warning("RDKit clustering unavailable (%s); returning top-K by score.", e)
        ranked = sorted(
            candidates,
            key=lambda e: (e.get("score", 0.0), len(e.get("support", []))),
            reverse=True,
        )
        for c in ranked:
            c.setdefault("bucket", 0)
        out = ranked[:return_topk]
        if return_meta:
            return out, {"n_clusters": len(candidates), "n_input": len(candidates),
                         "n_returned": len(out)}
        return out

    fps = []
    valid_idx: List[int] = []
    invalid_idx: List[int] = []
    for i, c in enumerate(candidates):
        fp = _combined_morgan_fp(c.get("reactants", ""), radius=radius, n_bits=n_bits)
        if fp is None:
            invalid_idx.append(i)
        else:
            fps.append(fp)
            valid_idx.append(i)

    n = len(fps)
    if n == 0:
        ranked = sorted(
            candidates,
            key=lambda e: (e.get("score", 0.0), len(e.get("support", []))),
            reverse=True,
        )
        for c in ranked:
            c.setdefault("bucket", 0)
        out = ranked[:return_topk]
        if return_meta:
            return out, {"n_clusters": len(candidates), "n_input": len(candidates),
                         "n_returned": len(out)}
        return out

    # Lower-triangle Tanimoto distances
    dists: List[float] = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)

    clusters = Butina.ClusterData(
        dists, n, distance_threshold, isDistData=True, reordering=True,
    )

    # Build [list of original indices] per cluster, ordered by cluster size desc.
    sized = sorted(
        ([valid_idx[m] for m in c] for c in clusters),
        key=len,
        reverse=True,
    )
    # Each invalid candidate is its own singleton cluster (nothing dropped).
    for j in invalid_idx:
        sized.append([j])

    n_clusters = len(sized)
    keep_per = max(1, int(k_per_cluster))

    reps: List[Dict[str, Any]] = []
    for cid, member_idxs in enumerate(sized):
        members = [candidates[i] for i in member_idxs]
        # Top ``keep_per`` within the cluster by (score, support) — was 1.
        members.sort(
            key=lambda e: (float(e.get("score") or 0.0), len(e.get("support", []))),
            reverse=True,
        )
        for r in members[:keep_per]:
            r["bucket"] = cid
            r["cluster_size"] = len(member_idxs)
            reps.append(r)

    reps.sort(
        key=lambda e: (float(e.get("score") or 0.0), len(e.get("support", []))),
        reverse=True,
    )
    out = reps[:return_topk]
    if return_meta:
        return out, {"n_clusters": n_clusters, "n_input": len(candidates),
                     "n_returned": len(out)}
    return out


# ---------------------------------------------------------------------------
# Backward-compat shims for the existing call site in tools/neural_retro.py
# ---------------------------------------------------------------------------


def bucket_by_classifier(
    candidates: Sequence[Dict[str, Any]],
    target_smiles: str,  # unused; kept for signature parity
    n_buckets: int = 10,  # unused; clusters are determined by distance threshold
) -> List[Dict[str, Any]]:
    """Compat wrapper: tag each candidate with a Butina cluster id.

    The legacy two-step ``bucket → pick_per_bucket`` pipeline is replaced
    by a single ``cluster_candidates`` call done lazily inside
    ``pick_per_bucket``. To avoid clustering twice, this wrapper just
    annotates a fingerprint cache on each entry; the real work happens
    in ``pick_per_bucket``.
    """
    for c in candidates:
        c.setdefault("bucket", 0)
    return list(candidates)


def pick_per_bucket(
    candidates: Sequence[Dict[str, Any]],
    k_per_bucket: int = 2,
    return_topk: int = 15,
    distance_threshold: float = 0.65,
    return_meta: bool = False,
):
    """Cluster + select using the Morgan/Butina recipe.

    ``k_per_bucket`` is the number of representatives kept per cluster
    (was hard-coded to 1 in the legacy recipe — now honored). When
    ``return_meta=True`` the call returns ``(reps, meta)`` where
    ``meta`` is ``{"n_clusters": int, "n_input": int, "n_returned": int}``;
    that's what the result-store stub uses to surface the funnel
    ``n_unique → n_clusters → n_in_stub`` to the LLM.
    """
    return cluster_candidates(
        list(candidates),
        distance_threshold=distance_threshold,
        radius=2,
        n_bits=2048,
        return_topk=return_topk,
        k_per_cluster=k_per_bucket,
        return_meta=return_meta,
    )
