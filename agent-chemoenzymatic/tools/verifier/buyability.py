"""Buyability lookup chain.

Order of resolution for each leaf SMILES:
  1. Enamine Building Blocks Stock pickle (~574k canonical SMILES)
  2. molbloom ZINC20 bloom filter (fast probabilistic, may have false positives)
  3. Multi-vendor sqlite DB (source_mols.db) — building_blocks then metabolites tables
  4. Legacy CSV stocks (vendor/legacy_retro_planner/building_block_dataset/*.csv)

Returns the first hit with a tagged ``source``; if all fail, ``source = "none"``.
"""

from __future__ import annotations

import csv
import logging
import os
import pickle
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)


def _canonicalize(s: str) -> str:
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(s)
        if m is None:
            return s
        return Chem.MolToSmiles(m)
    except Exception:
        return s


# ── Enamine pickle index singleton ───────────────────────────────────────
_enamine_index: Optional[Dict[str, Dict[str, Any]]] = None
_enamine_lock = threading.Lock()


def _load_enamine_index(pkl_path: str) -> Dict[str, Dict[str, Any]]:
    global _enamine_index
    if _enamine_index is not None:
        return _enamine_index
    with _enamine_lock:
        if _enamine_index is not None:
            return _enamine_index
        path = Path(pkl_path)
        if not path.exists():
            logger.warning("Enamine index pickle not found: %s", path)
            _enamine_index = {}
            return _enamine_index
        try:
            with open(path, "rb") as fh:
                _enamine_index = pickle.load(fh)
            logger.info("Loaded Enamine index: %d entries", len(_enamine_index))
        except Exception as exc:
            logger.warning("Enamine index load failed: %s", exc)
            _enamine_index = {}
        return _enamine_index


# ── Buyability sqlite singleton ──────────────────────────────────────────
_buyability_indexed_paths: Set[str] = set()
_buyability_lock = threading.Lock()


def _ensure_buyability_indexes(db_path: str, conn: sqlite3.Connection) -> None:
    if db_path in _buyability_indexed_paths:
        return
    with _buyability_lock:
        if db_path in _buyability_indexed_paths:
            return
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bb_smiles ON building_blocks(smiles)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_met_smiles ON metabolites(smiles)")
            conn.commit()
        except Exception as exc:
            logger.debug("buyability index creation skipped: %s", exc)
        _buyability_indexed_paths.add(db_path)


# ── Legacy CSV stocks singleton ──────────────────────────────────────────
_legacy_set: Optional[Set[str]] = None
_legacy_lock = threading.Lock()


def _load_legacy_csvs(csv_dir: str) -> Set[str]:
    """Aggregate canonical SMILES from CSVs in the legacy building-block directory.

    Only reads files that exist. Each CSV is expected to have a column whose
    header is one of {smiles, canonical_smiles, smi}; if not, the first column
    is used. Empty or missing dir → empty set.
    """
    global _legacy_set
    if _legacy_set is not None:
        return _legacy_set
    with _legacy_lock:
        if _legacy_set is not None:
            return _legacy_set
        out: Set[str] = set()
        d = Path(csv_dir)
        if not d.exists():
            _legacy_set = out
            return _legacy_set
        for csv_path in d.glob("*.csv"):
            try:
                with open(csv_path, newline="", encoding="utf-8", errors="ignore") as fh:
                    reader = csv.reader(fh)
                    header = next(reader, None) or []
                    smi_col = 0
                    for i, h in enumerate(header):
                        if h.strip().lower() in {"smiles", "canonical_smiles", "smi"}:
                            smi_col = i
                            break
                    for row in reader:
                        if smi_col < len(row) and row[smi_col]:
                            cs = _canonicalize(row[smi_col].strip())
                            if cs:
                                out.add(cs)
            except Exception as exc:
                logger.debug("Skipping CSV %s: %s", csv_path, exc)
                continue
        logger.info("Loaded legacy stocks: %d unique canonical SMILES", len(out))
        _legacy_set = out
        return _legacy_set


# ── Public interface ─────────────────────────────────────────────────────


class BuyabilityChain:
    """Stateful chain — caches DB connection and indexes per-process."""

    def __init__(
        self,
        enamine_pkl: Optional[str] = None,
        buyability_db: Optional[str] = None,
        legacy_csv_dir: Optional[str] = None,
        use_molbloom: bool = True,
    ):
        self.enamine_pkl = enamine_pkl
        self.buyability_db = buyability_db
        self.legacy_csv_dir = legacy_csv_dir
        self.use_molbloom = use_molbloom

    def lookup(self, smiles: str) -> Dict[str, Any]:
        """Return ``{smiles, source, catalog_id}`` (canonical input)."""
        canon = _canonicalize(smiles)

        # 1. Enamine pickle
        if self.enamine_pkl:
            idx = _load_enamine_index(self.enamine_pkl)
            entry = idx.get(canon)
            if entry:
                return {
                    "smiles": canon,
                    "source": "enamine",
                    "catalog_id": entry.get("catalog_id", ""),
                }

        # 2. molbloom (ZINC20) — best-effort
        if self.use_molbloom:
            try:
                import molbloom
                if molbloom.buy(canon, catalog="zinc20"):
                    return {"smiles": canon, "source": "molbloom", "catalog_id": None}
            except Exception as exc:
                # Don't fail the lookup, but DO surface that the molbloom path
                # is degraded — otherwise a broken install (version skew,
                # missing catalog) silently turns every molbloom check into a
                # miss and buyability quietly under-reports. Warn once.
                if not getattr(self, "_molbloom_warn_emitted", False):
                    logger.warning(
                        "molbloom buyability check failed (%s: %s); "
                        "disabling molbloom for this session and falling back "
                        "to sqlite/stock lookups.",
                        type(exc).__name__, exc,
                    )
                    self._molbloom_warn_emitted = True
                    self.use_molbloom = False

        # 3. Multi-vendor sqlite DB
        if self.buyability_db and os.path.exists(self.buyability_db):
            try:
                conn = sqlite3.connect(self.buyability_db)
                conn.row_factory = sqlite3.Row
                _ensure_buyability_indexes(self.buyability_db, conn)
                row = conn.execute(
                    "SELECT * FROM building_blocks WHERE smiles = ? LIMIT 1",
                    (canon,),
                ).fetchone()
                if row is not None:
                    cid = None
                    for col in ("mcule_id", "sigma_id", "zinc_id", "molport_id"):
                        try:
                            v = row[col]
                            if v:
                                cid = f"{col}={v}"
                                break
                        except Exception:
                            continue
                    conn.close()
                    return {"smiles": canon, "source": "buyability_db", "catalog_id": cid}
                row = conn.execute(
                    "SELECT * FROM metabolites WHERE smiles = ? LIMIT 1",
                    (canon,),
                ).fetchone()
                conn.close()
                if row is not None:
                    return {"smiles": canon, "source": "metabolite_db", "catalog_id": None}
            except Exception as exc:
                logger.debug("buyability DB lookup failed for %s: %s", canon, exc)

        # 4. Legacy CSV stocks
        if self.legacy_csv_dir:
            stocks = _load_legacy_csvs(self.legacy_csv_dir)
            if canon in stocks:
                return {"smiles": canon, "source": "legacy_csv", "catalog_id": None}

        return {"smiles": canon, "source": "none", "catalog_id": None}

    def evaluate_leaves(self, leaf_smiles: Iterable[str]) -> Dict[str, Any]:
        """Look up every leaf, return aggregate buyability and per-leaf detail."""
        leaves: List[Dict[str, Any]] = []
        n_buy = 0
        for s in leaf_smiles:
            res = self.lookup(s)
            leaves.append(res)
            if res["source"] != "none":
                n_buy += 1
        frac = (n_buy / len(leaves)) if leaves else 0.0
        return {"fraction": frac, "leaves": leaves, "n_total": len(leaves), "n_buyable": n_buy}
