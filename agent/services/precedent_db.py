"""Precedent route database for agent-controlled RAG.

A read-only store of N synthesis routes (target SMILES + packed Morgan
fingerprint + full route JSON), partitioned by source (chemo vs enzyme).
The query API returns Tanimoto-similar exemplars selected for diversity
via MMR, plus an adaptive `confidence` band — see spec §3.

The companion offline builder is ``scripts/build_precedent_index.py``.

Memory footprint for 300k routes is ~75 MB packed FPs + ~120 MB route
JSON; everything is loaded once at agent startup and shared across
queries via this process-local instance.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults (override-able via core.config: precedent_recall.*)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {"high": 0.7, "medium": 0.4, "low": 0.2}
DEFAULT_K = 3
DEFAULT_MIN_SIMILARITY = 0.2
DEFAULT_MMR_LAMBDA = 0.5
DEFAULT_TOP_N_FOR_MMR = 20

# Morgan FP geometry. Must match what the index builder packed.
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
PACKED_BYTES_PER_FP = MORGAN_NBITS // 8  # 256


# ---------------------------------------------------------------------------
# Helpers (also reused by the index builder)
# ---------------------------------------------------------------------------

def make_morgan_fp(smiles: str) -> np.ndarray | None:
    """Return a packed uint8 Morgan FP (256 bytes) for ``smiles``, or None
    if RDKit can't parse it. Matches the format on disk."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as e:  # pragma: no cover - rdkit is a hard dep
        raise RuntimeError("RDKit required for precedent_db") from e

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
    bits = np.zeros(MORGAN_NBITS, dtype=np.uint8)
    # GetOnBits returns an iterator of set bit indices.
    for b in bv.GetOnBits():
        bits[b] = 1
    return np.packbits(bits)


def canonicalize(smiles: str) -> str:
    """Best-effort canonical SMILES; falls back to input on parse failure."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol)
    except Exception:
        return smiles


def is_enzyme_friendly(smiles: str) -> bool:
    """Light SMARTS heuristic: does ``smiles`` carry any functional group
    commonly handled by an enzyme template (amine / OH / acid / amide /
    olefin)? Used by the ``source="mixed"`` mode to force at least one
    enzyme exemplar when relevant. Spec §6.1.
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        patterns = (
            "[NX3H2,NX3H1]",  # primary/secondary amine
            "[OH]",            # alcohol/phenol
            "C(=O)O",          # carboxylic acid
            "C(=O)N",          # amide
            "C=C",             # olefin
        )
        for pat in patterns:
            patt = Chem.MolFromSmarts(pat)
            if patt is not None and mol.HasSubstructMatch(patt):
                return True
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Bulk Tanimoto (vectorised over packed FPs)
# ---------------------------------------------------------------------------

# Pre-computed popcount LUT for uint8 (256 entries). With this we can do
# Tanimoto entirely on the packed byte arrays — no full bit unpack of the
# 300k × 2048 DB. Peak transient drops from ~1.8 GB (unpacked-bit & + |)
# to ~150 MB (two int16 popcount arrays of length N).
_POPCOUNT_LUT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)


def _byte_popcount(arr: np.ndarray) -> np.ndarray:
    """Popcount per element of a uint8 array (any shape). Returns uint8."""
    return _POPCOUNT_LUT[arr]


def bulk_tanimoto(query_fp_packed: bytes | np.ndarray,
                  db_fps_packed: np.ndarray,
                  db_popcounts: np.ndarray | None = None) -> np.ndarray:
    """Vectorised Tanimoto over packed uint8 fingerprints.

    ``query_fp_packed``: either raw bytes or a length-256 uint8 array.
    ``db_fps_packed``: shape (N, 256) uint8.
    ``db_popcounts``: optional pre-computed per-row popcounts (uint16).
        Pass when iterating many queries against the same DB to skip the
        ~5ms popcount on each call. ``PrecedentDB`` caches this.

    Returns: (N,) float32 array of Tanimoto similarities.

    Matches ``DataStructs.TanimotoSimilarity`` to within float-32 ulps
    on 100 random pairs (tested).

    Memory: O(N) transients (a single uint16 array of intersect counts).
    Does NOT unpack the 600MB bit matrix.
    """
    if isinstance(query_fp_packed, (bytes, bytearray)):
        q_packed = np.frombuffer(query_fp_packed, dtype=np.uint8)
    else:
        q_packed = np.asarray(query_fp_packed, dtype=np.uint8)
    if q_packed.shape[-1] != PACKED_BYTES_PER_FP:
        raise ValueError(
            f"query FP packed length {q_packed.shape[-1]} != {PACKED_BYTES_PER_FP}"
        )
    if db_fps_packed.ndim != 2 or db_fps_packed.shape[1] != PACKED_BYTES_PER_FP:
        raise ValueError(
            f"db_fps_packed must be (N, {PACKED_BYTES_PER_FP}), got {db_fps_packed.shape}"
        )

    q_pop = int(_byte_popcount(q_packed).sum())
    if db_popcounts is None:
        db_pop = _byte_popcount(db_fps_packed).sum(axis=1).astype(np.uint16)
    else:
        db_pop = db_popcounts

    # Per-row intersection popcount via the bytewise-AND of packed bytes.
    # The AND temp is (N, 256) uint8 = 256·N bytes; for N=300k that's 76 MB.
    # Then popcount → sum gives a (N,) uint16. Total transient ≈ 76 MB +
    # tiny constants — well below the 600 MB the unpack path used to need.
    inter_bytes = db_fps_packed & q_packed  # broadcasts q over rows
    inter = _byte_popcount(inter_bytes).sum(axis=1).astype(np.float32)
    union = (q_pop + db_pop.astype(np.float32)) - inter
    # Guard against the all-zero pathological case (would only happen for
    # an entry that failed FP construction — but the index builder filters
    # those upstream).
    np.maximum(union, 1.0, out=union)
    return inter / union


def _pairwise_tanimoto(fps_packed: np.ndarray,
                       bits: np.ndarray | None = None) -> np.ndarray:
    """Symmetric Tanimoto matrix over a small set of FPs (used in MMR).

    For small N (top-N for MMR = 20) the legacy unpack path is fine —
    20 × 2048 unpacked bits is 40 KB. ``bits`` may be supplied if the
    caller already unpacked (kept for backward compatibility); ignored
    when None.
    """
    n = fps_packed.shape[0]
    out = np.zeros((n, n), dtype=np.float32)
    if bits is None:
        bits = np.unpackbits(fps_packed, axis=1)
    popcounts = bits.sum(axis=1).astype(np.float32)
    for i in range(n):
        inter = (bits[i] & bits).sum(axis=1).astype(np.float32)
        union = popcounts[i] + popcounts - inter
        union = np.maximum(union, 1.0)
        out[i] = inter / union
    return out


def mmr_select(candidate_idxs: np.ndarray,
               candidate_sims: np.ndarray,
               db_fps_packed: np.ndarray,
               k: int,
               lambda_: float = DEFAULT_MMR_LAMBDA) -> list[int]:
    """Maximum Marginal Relevance over a pre-filtered top-N candidate set.

    Returns up to ``k`` indices (into the original DB array) chosen to
    balance query similarity with intra-result diversity.

    mmr(c) = λ·sim(q,c) - (1-λ)·max_{s∈selected} sim(c,s)
    """
    if len(candidate_idxs) == 0 or k <= 0:
        return []
    if len(candidate_idxs) <= k:
        # Nothing to diversify over.
        return list(candidate_idxs)

    # Compute pairwise similarities among candidates only.
    cand_fps = db_fps_packed[candidate_idxs]
    pair = _pairwise_tanimoto(cand_fps)

    selected: list[int] = []
    remaining = set(range(len(candidate_idxs)))

    # First pick: highest sim to query.
    first = int(np.argmax(candidate_sims))
    selected.append(first)
    remaining.discard(first)

    while len(selected) < k and remaining:
        best_cand = -1
        best_score = -np.inf
        for c in remaining:
            redundancy = max(pair[c, s] for s in selected)
            score = lambda_ * candidate_sims[c] - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best_cand = c
        if best_cand < 0:
            break
        selected.append(best_cand)
        remaining.discard(best_cand)

    return [int(candidate_idxs[i]) for i in selected]


# ---------------------------------------------------------------------------
# PrecedentDB
# ---------------------------------------------------------------------------

@dataclass
class PrecedentDBConfig:
    k: int = DEFAULT_K
    min_similarity: float = DEFAULT_MIN_SIMILARITY
    mmr_lambda: float = DEFAULT_MMR_LAMBDA
    top_n_for_mmr: int = DEFAULT_TOP_N_FOR_MMR
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    enzyme_friendly_force_one: bool = True


class PrecedentDB:
    """In-memory precedent route store.

    Files expected on disk (paths supplied to ``__init__`` are absolute):
      - ``routes.jsonl``   — N rows, one route per line (schema §2.1)
      - ``target_fps.npy`` — uint8 (N, 256) packed Morgan FPs
      - ``source_masks.npz`` — npz containing at least ``uspto``,
                               ``enzyme`` bool[N] arrays

    Optional:
      - ``target_index.json`` — ``{canonical_smiles: row_idx}`` for
                                exact-match lookup. Written by the
                                builder but NOT loaded here yet — kept
                                on disk as a side-car for future dedup
                                / exact-hit fast-path features. Safe to
                                delete if disk pressure matters.

    Caller pattern is ``db.query(smiles, ...)`` returning a dict of the
    same shape the tool emits.
    """

    def __init__(
        self,
        db_root: str,
        routes_file: str = "routes.jsonl",
        fps_file: str = "target_fps.npy",
        source_masks_file: str = "source_masks.npz",
        config: PrecedentDBConfig | None = None,
    ):
        self.db_root = db_root
        self.config = config or PrecedentDBConfig()
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple, dict] = OrderedDict()
        self._cache_max = 100

        routes_path = os.path.join(db_root, routes_file)
        fps_path = os.path.join(db_root, fps_file)
        masks_path = os.path.join(db_root, source_masks_file)

        for p in (routes_path, fps_path, masks_path):
            if not os.path.exists(p):
                raise FileNotFoundError(p)

        # Load FPs (mmap when possible — keeps process RSS low until rows
        # are actually touched).
        try:
            self._fps = np.load(fps_path, mmap_mode="r")
        except ValueError:
            # mmap can fail for compressed or non-contiguous arrays —
            # fall back to in-memory load.
            self._fps = np.load(fps_path)
        if self._fps.dtype != np.uint8:
            raise ValueError(
                f"target_fps.npy must be uint8 packed, got {self._fps.dtype}"
            )
        if self._fps.ndim != 2 or self._fps.shape[1] != PACKED_BYTES_PER_FP:
            raise ValueError(
                f"target_fps.npy shape must be (N, {PACKED_BYTES_PER_FP}), "
                f"got {self._fps.shape}"
            )
        n_rows = self._fps.shape[0]

        # Load routes (kept in RAM as a list — random access by index).
        self._routes: list[dict] = []
        with open(routes_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._routes.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("routes.jsonl: skipped malformed row: %s", e)
        if len(self._routes) != n_rows:
            raise ValueError(
                f"Row count mismatch: routes.jsonl has {len(self._routes)} "
                f"but target_fps.npy has {n_rows}"
            )

        # Source masks. Expected keys: 'uspto' and 'enzyme'. Missing keys
        # produce all-False masks so the source selection still works.
        masks = np.load(masks_path)
        self._mask_uspto = masks["uspto"].astype(bool) if "uspto" in masks else np.zeros(n_rows, dtype=bool)
        self._mask_enzyme = masks["enzyme"].astype(bool) if "enzyme" in masks else np.zeros(n_rows, dtype=bool)
        if self._mask_uspto.shape != (n_rows,) or self._mask_enzyme.shape != (n_rows,):
            raise ValueError("source_masks shapes must equal number of routes")

        # Pre-compute per-row popcount once at load time so subsequent
        # bulk_tanimoto calls skip the ~5ms popcount on every query.
        # For 300k × 256-byte FPs this is a one-shot ~300ms touch that
        # also faults the mmap pages in. The cached array itself is
        # ~600 KB (uint16 per row) so it's cheap to keep resident.
        self._popcounts = _byte_popcount(np.asarray(self._fps)).sum(axis=1).astype(np.uint16)

        logger.info(
            "PrecedentDB loaded: %d routes (%d uspto, %d enzyme) from %s",
            n_rows, int(self._mask_uspto.sum()), int(self._mask_enzyme.sum()),
            db_root,
        )

    # ---- introspection -------------------------------------------------

    @property
    def n_rows(self) -> int:
        return len(self._routes)

    def n_by_source(self) -> dict[str, int]:
        return {
            "uspto": int(self._mask_uspto.sum()),
            "enzyme": int(self._mask_enzyme.sum()),
            "total": self.n_rows,
        }

    # ---- query ---------------------------------------------------------

    def query(
        self,
        smiles: str,
        k: int | None = None,
        source: str = "mixed",
        min_similarity: float | None = None,
    ) -> dict:
        """Return up to ``k`` exemplar routes for ``smiles``.

        Result shape matches spec §4.2.
        """
        k = int(k if k is not None else self.config.k)
        if k < 1:
            k = 1
        if k > 10:
            k = 10  # defensive ceiling; tool also caps at 5
        min_sim = float(
            min_similarity if min_similarity is not None else self.config.min_similarity
        )
        source = (source or "mixed").lower()
        if source not in ("any", "chemo", "enzyme", "mixed"):
            source = "mixed"

        canon = canonicalize(smiles)

        cache_key = (canon, k, source, round(min_sim, 6))
        with self._lock:
            hit = self._cache.get(cache_key)
            if hit is not None:
                self._cache.move_to_end(cache_key)
                # Return a shallow copy so callers can mutate freely.
                return {**hit, "_cache_hit": True}

        result = self._query_uncached(canon, k, source, min_sim)

        with self._lock:
            self._cache[cache_key] = result
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
        return result

    def _query_uncached(
        self, canon_smiles: str, k: int, source: str, min_sim: float
    ) -> dict:
        empty_template = {
            "query_canonical": canon_smiles,
            "confidence": "none",
            "n_examples": 0,
            "max_similarity": 0.0,
            "examples": [],
            "advisory": "no useful precedent above threshold; fall back to zero-shot generation",
        }

        q_fp = make_morgan_fp(canon_smiles)
        if q_fp is None:
            return {
                **empty_template,
                "error": "invalid SMILES",
                "advisory": "could not parse SMILES; no retrieval performed",
            }

        sims = bulk_tanimoto(q_fp, self._fps, db_popcounts=self._popcounts)

        # Build per-source candidate pool indices.
        if source == "chemo":
            allowed = self._mask_uspto
        elif source == "enzyme":
            allowed = self._mask_enzyme
        else:  # 'any' or 'mixed'
            allowed = np.ones(self.n_rows, dtype=bool)

        threshold_mask = sims >= min_sim
        pool_mask = allowed & threshold_mask
        pool_idxs = np.nonzero(pool_mask)[0]

        if pool_idxs.size == 0:
            return empty_template

        # Take top-N for MMR.
        top_n = min(self.config.top_n_for_mmr, pool_idxs.size)
        # Sort pool_idxs by similarity descending; take first top_n.
        order = np.argsort(-sims[pool_idxs])
        top_pool_idxs = pool_idxs[order[:top_n]]
        top_sims = sims[top_pool_idxs]

        max_sim = float(top_sims.max())

        # Effective k by adaptive thresholding (spec §3.3).
        th_high = self.config.thresholds.get("high", 0.7)
        th_med = self.config.thresholds.get("medium", 0.4)
        th_low = self.config.thresholds.get("low", 0.2)
        if max_sim >= th_high:
            confidence = "high"
            eff_k = k
        elif max_sim >= th_med:
            confidence = "medium"
            eff_k = k
        elif max_sim >= th_low:
            confidence = "low"
            eff_k = min(k, 2)
        else:
            return empty_template

        selected_idxs = mmr_select(
            top_pool_idxs, top_sims, self._fps, eff_k, self.config.mmr_lambda
        )

        # Enzyme-mixing for source="mixed": force at least one enzyme
        # exemplar when the query carries an enzyme-friendly group.
        if source == "mixed" and self.config.enzyme_friendly_force_one:
            if is_enzyme_friendly(canon_smiles):
                has_enzyme = any(self._mask_enzyme[i] for i in selected_idxs)
                if not has_enzyme:
                    # Find the highest-sim enzyme candidate in the full
                    # pool (above threshold). If none, leave the result.
                    enzyme_pool = pool_idxs[self._mask_enzyme[pool_idxs]]
                    if enzyme_pool.size > 0:
                        best_enzyme = int(enzyme_pool[np.argmax(sims[enzyme_pool])])
                        # Replace the lowest-sim chemo entry.
                        chemo_in_selected = [
                            (idx, sims[idx]) for idx in selected_idxs
                            if not self._mask_enzyme[idx]
                        ]
                        if chemo_in_selected:
                            chemo_in_selected.sort(key=lambda kv: kv[1])
                            drop_idx = chemo_in_selected[0][0]
                            selected_idxs = [
                                best_enzyme if i == drop_idx else i
                                for i in selected_idxs
                            ]
                        elif len(selected_idxs) < eff_k:
                            selected_idxs.append(best_enzyme)

        # Build example payloads.
        #
        # ``route`` carries the FULL route dict from disk (all step fields
        # incl. reaction_smarts, plus any source-specific extras like
        # ``doi`` and ``notes`` written by the index builder). This is the
        # payload ResultStore persists. The projector in
        # ``services/result_store._proj_recall_similar_routes`` strips it
        # down to a stub for the in-context message — ``load_result(rid)``
        # later fetches the full version from disk. Do NOT pre-filter
        # ``route`` here or you silently truncate the data the LLM thinks
        # ``load_result`` will give back.
        examples = []
        for idx in selected_idxs:
            route = self._routes[idx]
            examples.append({
                "route_id": route.get("route_id"),
                "similarity": float(round(sims[idx], 4)),
                "source_target": route.get("target_smiles"),
                "source_db": route.get("source"),
                "n_steps": route.get("n_steps", len(route.get("steps", []) or [])),
                "is_enzymatic": bool(route.get("is_enzymatic", False)),
                "rxn_classes": route.get("rxn_classes") or [],
                # Pass-through of the full on-disk route (steps with all
                # fields, doi, notes, etc.). Projector strips for stub.
                "route": route,
            })
        # Sort returned examples by similarity descending for readability.
        examples.sort(key=lambda e: -e["similarity"])

        return {
            "query_canonical": canon_smiles,
            "confidence": confidence,
            "n_examples": len(examples),
            "max_similarity": float(round(max_sim, 4)),
            "examples": examples,
            "advisory": "use as strategic anchors, not as ground truth; verify each step",
        }

    # ---- raw access (for tests / future verifier-aligned re-rank) ----

    def get_route(self, idx: int) -> dict:
        return self._routes[idx]

    def iter_routes(self) -> Iterable[dict]:
        return iter(self._routes)


# ---------------------------------------------------------------------------
# Factory (graceful absence)
# ---------------------------------------------------------------------------

def maybe_load(
    config_block: dict | None,
    base_dir: str,
) -> PrecedentDB | None:
    """Build a PrecedentDB from a config block, or None if disabled / missing.

    ``config_block`` is the ``precedent_recall`` sub-dict of the runtime config.
    ``base_dir`` is the agent root for resolving relative ``db_root``.
    """
    if not config_block:
        return None
    if not config_block.get("enabled", False):
        return None
    db_root = config_block.get("db_root")
    if not db_root:
        logger.warning("precedent_recall.enabled but db_root unset; tool disabled")
        return None
    if not os.path.isabs(db_root):
        db_root = os.path.join(base_dir, db_root)
    if not os.path.isdir(db_root):
        logger.warning("precedent_recall: db_root not found, tool disabled: %s", db_root)
        return None
    defaults = config_block.get("defaults", {}) or {}
    thresholds = config_block.get("thresholds", {}) or {}
    multi_src = config_block.get("multi_source", {}) or {}
    cfg = PrecedentDBConfig(
        k=int(defaults.get("k", DEFAULT_K)),
        min_similarity=float(defaults.get("min_similarity", DEFAULT_MIN_SIMILARITY)),
        mmr_lambda=float(defaults.get("mmr_lambda", DEFAULT_MMR_LAMBDA)),
        top_n_for_mmr=int(defaults.get("top_n_for_mmr", DEFAULT_TOP_N_FOR_MMR)),
        thresholds={
            "high": float(thresholds.get("high", DEFAULT_THRESHOLDS["high"])),
            "medium": float(thresholds.get("medium", DEFAULT_THRESHOLDS["medium"])),
            "low": float(thresholds.get("low", DEFAULT_THRESHOLDS["low"])),
        },
        enzyme_friendly_force_one=bool(
            multi_src.get("enzyme_friendly_force_one", True)
        ),
    )
    try:
        return PrecedentDB(
            db_root=db_root,
            routes_file=config_block.get("routes_file", "routes.jsonl"),
            fps_file=config_block.get("fps_file", "target_fps.npy"),
            source_masks_file=config_block.get(
                "source_masks_file", "source_masks.npz"
            ),
            config=cfg,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.warning("precedent_recall: PrecedentDB load failed (%s); tool disabled", e)
        return None
