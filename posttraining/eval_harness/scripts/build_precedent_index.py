#!/usr/bin/env python
"""Build the precedent route index consumed by services/precedent_db.py.

Inputs:
  - `--source uspto`         download/parse AiZynthFinder USPTO route hdf5
  - `--source retrobiocat`   clone RetroBioCat rule set, assemble pseudo-routes
  - `--source both`          do both, merge into one index

Outputs (under `--out-dir`):
  - routes.jsonl           one JSON route per line (schema = spec §2.1)
  - target_fps.npy         uint8 (N, 256) packed Morgan FPs (r=2, 2048-bit)
  - target_index.json      {canonical_smiles: row_idx}
  - source_masks.npz       {'uspto': bool[N], 'enzyme': bool[N]}

USPTO source
------------
PRIMARY (default): Retro* / LLM-Syn-Planner ``routes_*.pkl`` — a
``list[list[str]]`` where each route is an ordered list of reaction
SMILES ``"product>>reactant1.reactant2"``. 299,202 multi-step routes
extracted from USPTO by Chen et al. (Retro*, ICML 2020), repackaged
and shipped by Wang et al. (LLM-Syn-Planner, ICML 2025) at
https://www.dropbox.com/scl/fi/dmmypid2ooohp3freiox8/dataset.zip
(rlkey ``fmrhvds6fmxck2cp8h94albpc``). Drop ``routes_train.pkl`` (and
optionally ``routes_val.pkl`` / ``routes_test.pkl``) into
``<cache_dir>/`` or set ``USPTO_ROUTES_PKL`` to an absolute path.
We canonicalise each step's product + reactants, drop steps where
RDKit can't parse, pack Morgan FPs over the route's TARGET (= step 0
product), and emit the four output files above.

FALLBACK (legacy): AiZynthFinder ``uspto_*_routes.hdf5`` — pickled
DataFrame whose ``tree`` column holds the nested AiZynthFinder
reaction tree JSON. Kept around so the AiZynthFinder route artefacts
(if you have them) still work. The tree walker / ``_tree_to_route``
flattens those into our schema.

RetroBioCat source
------------------
RetroBioCat ships a curated YAML/JSON rule set
(github.com/william-finnigan/RetroBioCat). Without a multi-step solver
we assemble single-step pseudo-routes from each rule application
matched against the training reactions file. Pragmatic but sufficient
to seed `is_enzymatic=True` exemplars for the agent.

Usage
-----
    # Smoke (1k USPTO routes) — assumes routes_train.pkl is in cache dir
    .venv/bin/python scripts/build_precedent_index.py \\
        --source uspto --limit 1000 \\
        --out-dir /path/to/precedent-db \\
        --cache-dir /path/to/precedent-db/_downloads/extracted/dataset

    # Full (~300k USPTO + RetroBioCat 52 enzyme pathways)
    .venv/bin/python scripts/build_precedent_index.py \\
        --source both \\
        --out-dir /path/to/precedent-db \\
        --cache-dir /path/to/precedent-db/_downloads/extracted/dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Iterator

import numpy as np

# Local import of FP helpers — keep on-disk format in sync with the loader.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _REPO_ROOT)
from services.precedent_db import (  # noqa: E402
    MORGAN_NBITS,
    PACKED_BYTES_PER_FP,
    canonicalize,
    make_morgan_fp,
)

logger = logging.getLogger("build_precedent_index")

# Genheden et al. AiZynthFinder published route artefacts on Figshare
# (DOI 10.6084/m9.figshare.20618655). The Figshare per-file numeric ids
# rotate when authors re-upload, so we don't hard-code them — operators
# point the script at a directory that already contains the hdf5(s)
# Where to look for USPTO route inputs in the cache dir:
#   PRIMARY: Retro* / LLM-Syn-Planner *.pkl  (list[list[str]] of reaction SMILES)
#   FALLBACK: AiZynthFinder *.hdf5            (pickled DataFrame w/ 'tree' col)
# To populate the cache (PRIMARY path):
#   curl -L 'https://www.dropbox.com/scl/fi/dmmypid2ooohp3freiox8/dataset.zip?rlkey=fmrhvds6fmxck2cp8h94albpc&dl=1' -o dataset.zip
#   unzip dataset.zip 'dataset/routes_*.pkl' -d <cache_dir_parent>/
USPTO_PKL_FILENAMES = ("routes_train.pkl", "routes_val.pkl", "routes_test.pkl")
USPTO_HDF5_FILENAMES = ("uspto_train_routes.hdf5",)
RETROBIOCAT_REPO = "https://github.com/willfinnigan/RetroBioCat.git"


# ---------------------------------------------------------------------------
# USPTO loader
# ---------------------------------------------------------------------------

def _locate_uspto_inputs(cache_dir: str) -> tuple[list[str], list[str]]:
    """Return ``(pkl_paths, hdf5_paths)`` operator pre-downloaded under
    ``cache_dir``. Resolution order:

      1) ``$USPTO_ROUTES_PKL`` env var (single .pkl, takes priority).
      2) ``$USPTO_ROUTES_HDF5`` env var (single .hdf5, legacy path).
      3) ``routes_train.pkl`` under ``cache_dir`` (PRIMARY). Other
         ``routes_*.pkl`` splits (val/test/possible_test_hard) are
         IGNORED by default to prevent benchmark leakage — your
         downstream evaluations on USPTO-190 / hard-test would silently
         retrieve their own ground truth from the index. Set
         ``USPTO_INCLUDE_NON_TRAIN_SPLITS=1`` to opt in (e.g. for an
         ablation that explicitly wants the full corpus).
      4) Any ``uspto_*_routes.hdf5`` under ``cache_dir`` (FALLBACK).

    Raises FileNotFoundError if neither is found.
    """
    env_pkl = os.environ.get("USPTO_ROUTES_PKL")
    env_hdf5 = os.environ.get("USPTO_ROUTES_HDF5")
    if env_pkl:
        if not os.path.exists(env_pkl):
            raise FileNotFoundError(f"USPTO_ROUTES_PKL set but not found: {env_pkl}")
        return [env_pkl], []
    if env_hdf5:
        if not os.path.exists(env_hdf5):
            raise FileNotFoundError(f"USPTO_ROUTES_HDF5 set but not found: {env_hdf5}")
        return [], [env_hdf5]
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(
            f"cache_dir does not exist: {cache_dir}. Download "
            f"the LLM-Syn-Planner dataset.zip (see module docstring) "
            f"and drop routes_train.pkl into {cache_dir}."
        )
    include_other_splits = os.environ.get(
        "USPTO_INCLUDE_NON_TRAIN_SPLITS", ""
    ).strip().lower() in ("1", "true", "yes", "on")
    all_pkls = sorted(
        f for f in os.listdir(cache_dir)
        if f.startswith("routes_") and f.endswith(".pkl")
    )
    if include_other_splits:
        kept_pkls = all_pkls
        if all_pkls:
            logger.warning(
                "USPTO_INCLUDE_NON_TRAIN_SPLITS=1 — including %d pkl(s) "
                "incl. val/test/possible_test_hard. Make sure your "
                "evaluation does not query against routes that are in "
                "the index, or you will see inflated solve rates.",
                len(all_pkls),
            )
    else:
        kept_pkls = [f for f in all_pkls if f == "routes_train.pkl"]
        skipped = [f for f in all_pkls if f != "routes_train.pkl"]
        if skipped:
            logger.info(
                "Skipping non-train pkl(s) to avoid eval leakage: %s. "
                "Set USPTO_INCLUDE_NON_TRAIN_SPLITS=1 to opt in.",
                skipped,
            )
    pkls = [os.path.join(cache_dir, f) for f in kept_pkls]
    hdf5s = sorted(
        os.path.join(cache_dir, f)
        for f in os.listdir(cache_dir)
        if f.startswith("uspto") and f.endswith(".hdf5") and "routes" in f
    )
    if not pkls and not hdf5s:
        raise FileNotFoundError(
            f"No routes_train.pkl (preferred) or uspto_*_routes.hdf5 "
            f"(legacy) found in {cache_dir}. See module docstring for "
            f"download URL. If you only have val/test/possible_test_hard "
            f"splits, set USPTO_INCLUDE_NON_TRAIN_SPLITS=1 to use them "
            f"(but understand the leakage risk for your benchmarks)."
        )
    return pkls, hdf5s


def _walk_aizynth_tree(node: dict) -> Iterator[dict]:
    """Depth-first walk over an AiZynthFinder route tree, yielding reaction
    nodes (type='reaction') in execution order (leaves first).

    Schema:
      mol  -> {type='mol', smiles, in_stock, children=[reaction?]}
      rxn  -> {type='reaction', metadata?, children=[mol, mol, ...]}
    """
    if not isinstance(node, dict):
        return
    children = node.get("children") or []
    for c in children:
        yield from _walk_aizynth_tree(c)
    if node.get("type") == "reaction":
        yield node


def _tree_to_route(tree: dict, route_id: str) -> dict | None:
    """Flatten an AiZynthFinder route tree to our schema.

    Returns None when the tree has no reactions or any reaction node
    lacks SMILES for product / reactants.
    """
    if not isinstance(tree, dict) or tree.get("type") != "mol":
        return None
    target = tree.get("smiles")
    if not target:
        return None
    canon_target = canonicalize(target)
    # Build a single rxn -> parent_mol_smiles map by walking the tree
    # once (id(rxn_dict) is stable for the duration of this call). Avoids
    # the O(N²) re-traversal _find_parent_product previously did.
    parent_map: dict[int, str] = {}

    def _index_parents(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "mol":
            for c in node.get("children") or []:
                if isinstance(c, dict) and c.get("type") == "reaction":
                    parent_map[id(c)] = node.get("smiles") or ""
                _index_parents(c)
        else:
            for c in node.get("children") or []:
                _index_parents(c)

    _index_parents(tree)

    steps: list[dict] = []
    for rxn in _walk_aizynth_tree(tree):
        product = parent_map.get(id(rxn))
        if not product:
            continue
        reactants = []
        for child in rxn.get("children") or []:
            if isinstance(child, dict) and child.get("smiles"):
                reactants.append(canonicalize(child["smiles"]))
        if not reactants:
            continue
        meta = rxn.get("metadata") or {}
        steps.append({
            "product": canonicalize(product),
            "reactants": reactants,
            "reaction_smarts": meta.get("smarts") or meta.get("template"),
            "rxn_class": meta.get("classification") or meta.get("rxn_class"),
        })
    if not steps:
        return None
    rxn_classes = sorted({s["rxn_class"] for s in steps if s.get("rxn_class")})
    return {
        "route_id": route_id,
        "source": "uspto",
        "target_smiles": canon_target,
        "n_steps": len(steps),
        "steps": steps,
        "rxn_classes": rxn_classes,
        "is_enzymatic": False,
        "max_step_complexity": None,
    }


def _find_parent_product(node: dict, rxn: dict) -> str | None:
    """DEPRECATED: kept only for the test in test_build_precedent_index.py
    that hand-walks a toy tree. _tree_to_route no longer uses this."""
    if not isinstance(node, dict):
        return None
    if node.get("type") == "mol":
        for c in node.get("children") or []:
            if c is rxn:
                return node.get("smiles")
            found = _find_parent_product(c, rxn)
            if found:
                return found
    else:
        for c in node.get("children") or []:
            found = _find_parent_product(c, rxn)
            if found:
                return found
    return None


def _step_str_to_dict(step_smarts: str) -> dict | None:
    """Parse a single reaction SMILES of the form
    ``"product>>reactant1.reactant2"`` (Retro* / LLM-Syn-Planner format)
    into our step dict. Returns None if either side fails to parse.
    """
    if not isinstance(step_smarts, str) or ">>" not in step_smarts:
        return None
    lhs, rhs = step_smarts.split(">>", 1)
    product_raw = lhs.strip()
    reactants_raw = [r.strip() for r in rhs.split(".") if r.strip()]
    if not product_raw or not reactants_raw:
        return None
    product = canonicalize(product_raw)
    reactants = [canonicalize(r) for r in reactants_raw]
    if not product or not all(reactants):
        return None
    return {
        "product": product,
        "reactants": reactants,
        # The pickled corpus carries raw reaction SMILES only — no
        # template / classification. ``reaction_smarts`` is the
        # round-tripped form; ``rxn_class`` is left None and can be
        # back-filled later via ``idx2template_retro.json`` if desired.
        "reaction_smarts": f"{product}>>{'.'.join(reactants)}",
        "rxn_class": None,
    }


def _route_steps_to_dict(steps: list[str], route_id: str) -> dict | None:
    """Flatten a Retro* / LLM-Syn-Planner route (list of reaction SMILES,
    leaves-first order) into our schema. The TARGET is the product of
    step 0 (the last reaction applied — Retro* stores routes in
    application order, not synthesis order). Returns None if step 0 is
    unparseable.
    """
    if not isinstance(steps, list) or not steps:
        return None
    parsed_steps: list[dict] = []
    for s in steps:
        d = _step_str_to_dict(s)
        if d is None:
            # Tolerate broken individual steps — drop them but keep the
            # rest of the route if anything survives. (In practice <1%
            # of LLM-Syn-Planner routes have any unparseable step.)
            continue
        parsed_steps.append(d)
    if not parsed_steps:
        return None
    target = parsed_steps[0]["product"]
    return {
        "route_id": route_id,
        "source": "uspto",
        "target_smiles": target,
        "n_steps": len(parsed_steps),
        "steps": parsed_steps,
        "rxn_classes": [],
        "is_enzymatic": False,
        "max_step_complexity": None,
    }


def _iter_uspto_routes_from_pkl(path: str, limit: int | None,
                                already_yielded: int) -> Iterator[dict]:
    """Yield routes from a Retro* / LLM-Syn-Planner ``routes_*.pkl``.
    ``already_yielded`` is the running count from prior files in the
    same build, so ``--limit`` still acts as a global cap across files.
    """
    import pickle
    logger.info("Reading USPTO pkl: %s ...", path)
    with open(path, "rb") as f:
        routes = pickle.load(f)
    if not isinstance(routes, list):
        raise RuntimeError(
            f"{path}: expected list of routes, got {type(routes).__name__}"
        )
    yielded = 0
    src_tag = os.path.splitext(os.path.basename(path))[0]  # e.g. routes_train
    for i, raw_steps in enumerate(routes):
        if limit is not None and (already_yielded + yielded) >= limit:
            return
        route_id = f"uspto_{src_tag}_{i:07d}"
        route = _route_steps_to_dict(raw_steps, route_id)
        if route is None:
            continue
        yield route
        yielded += 1


def _iter_uspto_routes_from_hdf5(path: str, limit: int | None,
                                 already_yielded: int) -> Iterator[dict]:
    """Yield routes from an AiZynthFinder ``uspto_*_routes.hdf5`` (legacy
    path; kept for operators who already have those artefacts)."""
    try:
        import pandas as pd  # noqa: F401
    except ImportError as e:
        raise RuntimeError("pandas required: pip install pandas") from e
    import pandas as pd
    logger.info("Reading USPTO hdf5: %s ...", path)
    df = pd.read_hdf(path)
    tree_col = None
    for cand in ("tree", "trees", "route"):
        if cand in df.columns:
            tree_col = cand
            break
    if tree_col is None:
        raise RuntimeError(f"no tree column found in {path}: columns={list(df.columns)}")
    yielded = 0
    for i, raw in enumerate(df[tree_col].tolist()):
        if limit is not None and (already_yielded + yielded) >= limit:
            return
        try:
            tree = raw if isinstance(raw, dict) else json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(tree, list):
            tree = tree[0] if tree else None
        if not tree:
            continue
        route_id = f"uspto_{os.path.basename(path).split('.')[0]}_{i:07d}"
        route = _tree_to_route(tree, route_id)
        if route is None:
            continue
        yield route
        yielded += 1


def iter_uspto_routes(cache_dir: str, limit: int | None) -> Iterator[dict]:
    """Yield route dicts from the cached USPTO inputs (pkl preferred,
    hdf5 fallback). ``--limit`` caps the TOTAL yielded across files.
    """
    pkls, hdf5s = _locate_uspto_inputs(cache_dir)
    yielded = 0
    # PRIMARY: Retro* / LLM-Syn-Planner pkl
    for path in pkls:
        for route in _iter_uspto_routes_from_pkl(path, limit, yielded):
            yield route
            yielded += 1
            if limit is not None and yielded >= limit:
                return
    # FALLBACK: AiZynthFinder hdf5 — only consulted when no pkl present
    if not pkls:
        for path in hdf5s:
            for route in _iter_uspto_routes_from_hdf5(path, limit, yielded):
                yield route
                yielded += 1
                if limit is not None and yielded >= limit:
                    return


# ---------------------------------------------------------------------------
# RetroBioCat loader (single-step pseudo-routes)
# ---------------------------------------------------------------------------

def iter_retrobiocat_routes(cache_dir: str, limit: int | None) -> Iterator[dict]:
    """Yield enzymatic pseudo-routes from the RetroBioCat repo.

    Concretely, we read ``scripts/pathway_testing/test_pathways.xlsx`` —
    the 52-pathway benchmark the RetroBioCat authors hand-curated from
    primary literature (each row: target product + comma-separated
    substrates + comma-separated enzymes + DOI). We collapse each row
    into a single "compressed" route entry:

      - ``target_smiles``  = the curated target product
      - ``steps``          = one synthetic step with all substrates as
                             reactants and the enzyme list as ``rxn_class``
      - ``is_enzymatic``   = True
      - ``n_steps``        = the real enzyme count (so LLM sees how
                             long the analog was, even though we don't
                             reconstruct intermediates)

    This is a compromise: the repo no longer ships YAML reaction
    rules — those live in a Mongo dump — so a per-step pseudo-route
    extraction isn't possible without spinning up the mongo container.
    The compressed form still gives the agent enzyme-friendly anchors,
    which is the point of source="mixed" being able to surface ≥1
    enzyme exemplar for a chemo-leaning query.
    """
    import subprocess

    os.makedirs(cache_dir, exist_ok=True)
    repo_dir = os.path.join(cache_dir, "RetroBioCat")
    if not os.path.isdir(repo_dir):
        logger.info("Cloning RetroBioCat -> %s", repo_dir)
        subprocess.run(
            ["git", "clone", "--depth", "1", RETROBIOCAT_REPO, repo_dir],
            check=True,
        )

    try:
        import pandas as pd  # noqa: F401
    except ImportError as e:
        raise RuntimeError("pandas required: pip install pandas") from e
    import pandas as pd

    xlsx_path = os.path.join(
        repo_dir, "scripts", "pathway_testing", "test_pathways.xlsx"
    )
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(
            f"RetroBioCat pathway xlsx not found at {xlsx_path} — has the "
            f"upstream layout changed? Re-check willfinnigan/RetroBioCat."
        )
    df = pd.read_excel(xlsx_path)
    needed = {"Target Product", "Substrates", "Enzymes"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"xlsx missing columns: {missing}; cols={list(df.columns)}")

    yielded = 0
    for i, row in df.iterrows():
        if limit is not None and yielded >= limit:
            return
        target_raw = str(row["Target Product"] or "").strip()
        subs_raw = str(row["Substrates"] or "").strip()
        enzymes_raw = str(row["Enzymes"] or "").strip()
        if not target_raw or not subs_raw:
            continue
        target = canonicalize(target_raw)
        substrates = [
            canonicalize(s.strip()) for s in subs_raw.split(",") if s.strip()
        ]
        substrates = [s for s in substrates if s]
        if not substrates:
            continue
        enzymes = [e.strip() for e in enzymes_raw.split(",") if e.strip()]
        n_enzyme_steps = len(enzymes) if enzymes else 1
        rxn_class = ",".join(enzymes) if enzymes else "enzymatic"
        doi = str(row.get("DOI") or "").strip()
        route_id = f"retrobiocat_pathway_{int(row.get('Num', i)):04d}"
        yield {
            "route_id": route_id,
            "source": "retrobiocat",
            "target_smiles": target,
            "n_steps": n_enzyme_steps,
            "steps": [{
                "product": target,
                "reactants": substrates,
                "reaction_smarts": None,
                "rxn_class": rxn_class,
            }],
            "rxn_classes": enzymes or ["enzymatic"],
            "is_enzymatic": True,
            "max_step_complexity": None,
            "doi": doi or None,
            "notes": "compressed: only target+substrates+enzyme-list known; intermediates not reconstructed",
        }
        yielded += 1


# ---------------------------------------------------------------------------
# Index writer
# ---------------------------------------------------------------------------

def build_index(
    sources: list[str],
    out_dir: str,
    cache_dir: str,
    limit: int | None,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    routes_path = os.path.join(out_dir, "routes.jsonl")
    fps_path = os.path.join(out_dir, "target_fps.npy")
    index_path = os.path.join(out_dir, "target_index.json")
    masks_path = os.path.join(out_dir, "source_masks.npz")

    seen_route_ids: set[str] = set()
    fp_rows: list[np.ndarray] = []
    uspto_mask: list[bool] = []
    enzyme_mask: list[bool] = []
    canon_index: dict[str, int] = {}
    n_kept = 0
    n_skipped_fp = 0
    n_skipped_dup = 0

    with open(routes_path, "w") as rout:
        for src in sources:
            if src == "uspto":
                stream = iter_uspto_routes(cache_dir, limit)
            elif src == "retrobiocat":
                stream = iter_retrobiocat_routes(cache_dir, limit)
            else:
                raise ValueError(f"unknown source: {src}")
            for route in stream:
                rid = route["route_id"]
                if rid in seen_route_ids:
                    n_skipped_dup += 1
                    continue
                target = route["target_smiles"]
                fp = make_morgan_fp(target)
                if fp is None or fp.shape[0] != PACKED_BYTES_PER_FP:
                    n_skipped_fp += 1
                    continue
                seen_route_ids.add(rid)
                fp_rows.append(fp)
                uspto_mask.append(route["source"] == "uspto")
                enzyme_mask.append(bool(route.get("is_enzymatic", False)))
                # First-write wins for the canonical->row index (later
                # duplicates map to their own ids via route_id but not
                # via target_index lookup).
                canon_index.setdefault(target, n_kept)
                rout.write(json.dumps(route, ensure_ascii=False))
                rout.write("\n")
                n_kept += 1
                if n_kept % 10000 == 0:
                    logger.info("  packed %d routes ...", n_kept)

    if n_kept == 0:
        raise RuntimeError("no routes survived parsing — check inputs")

    fps = np.stack(fp_rows, axis=0).astype(np.uint8)
    assert fps.shape == (n_kept, PACKED_BYTES_PER_FP), fps.shape
    np.save(fps_path, fps)
    with open(index_path, "w") as f:
        json.dump(canon_index, f)
    np.savez(
        masks_path,
        uspto=np.array(uspto_mask, dtype=bool),
        enzyme=np.array(enzyme_mask, dtype=bool),
    )

    stats = {
        "n_kept": n_kept,
        "n_uspto": int(sum(uspto_mask)),
        "n_enzyme": int(sum(enzyme_mask)),
        "n_skipped_fp": n_skipped_fp,
        "n_skipped_dup": n_skipped_dup,
        "fps_bytes": fps.nbytes,
        "routes_bytes": os.path.getsize(routes_path),
        "out_dir": out_dir,
    }
    logger.info("Index build complete: %s", stats)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=["uspto", "retrobiocat", "both"], required=True)
    ap.add_argument("--out-dir", required=True, help="Output directory for the four index files")
    ap.add_argument("--cache-dir", default=None,
                    help="Where to cache raw downloads (default: <out-dir>/_cache)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Smoke flag: stop after N routes per source")
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    cache_dir = args.cache_dir or os.path.join(args.out_dir, "_cache")
    sources = ["uspto", "retrobiocat"] if args.source == "both" else [args.source]
    stats = build_index(
        sources=sources, out_dir=args.out_dir, cache_dir=cache_dir, limit=args.limit,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
