"""ChemPrice-backed leaf pricing for the cost pillar.

Replaces the 0/1 ``leaf_buyability`` term with a price-aware ``leaf_economics``
score. We only price the **starting materials (leaves)** of a route, never the
intermediates — by construction the only molecules that actually have to be
purchased are the leaves.

Architecture
~~~~~~~~~~~~
- Reuses ``backend.clients.ChemPriceClient`` from the sibling
  ``../chemoenzymatic-retrosynthesis-verifier`` repo so we don't duplicate the
  pandas/HTTP/2 monkey-patches and credential plumbing.
- Loads ``MOLPORT_API_KEY / CHEMSPACE_API_KEY / MCULE_API_KEY`` from that
  repo's ``.env`` via ``python-dotenv``.
- Per-session sqlite cache (``runs/_session_*/price_cache.db``) keyed by
  canonical SMILES with a 30-day TTL — prices change but not every run.
- On timeout / missing creds / ChemPrice exception → fall back to the local
  ``BuyabilityChain`` (Enamine pkl / molbloom / sqlite / legacy CSV) and, if
  that also misses, the RDKit-only PubChem-style heuristic ported from
  ``chemoenzymatic-retrosynthesis-verifier`` (MW + atom + ring count → tier →
  conservative USD/g floor) so the cost pillar always produces a number.

Price → score curve
~~~~~~~~~~~~~~~~~~~
``leaf_score = clip(1.0 - log10(max(price, 0.1)) / 4.0, 0.3, 1.0)``

  $1/g  → 1.00   (commodity)
  $10/g → 0.75
  $100/g → 0.50  (fine chemical)
  $1000/g → 0.25 → clipped to 0.30 (specialty)

Available leaves with no quoted price (vendor returns ``USD/g = NaN``) get the
mid value 0.5 so they still beat unbuyable leaves (0.0).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
import threading
import time
from math import log10
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _canonicalize(s: str) -> str:
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(s)
        if m is None:
            return s
        return Chem.MolToSmiles(m)
    except Exception:
        return s


def _price_to_score(price_per_g: Optional[float]) -> float:
    """Map ``USD/g`` → leaf score in [0.3, 1.0]; ``None`` → 0.5 (no quote)."""
    if price_per_g is None:
        return 0.5
    p = max(float(price_per_g), 0.1)  # avoid log10(0)
    raw = 1.0 - log10(p) / 4.0
    return max(0.3, min(1.0, raw))


# ──────────────────────────────────────────────────────────────────────────
# chemprice vendor concurrency monkey-patch
# ──────────────────────────────────────────────────────────────────────────
#
# Upstream `chemprice.utils.collect_vendors` calls MolPort, ChemSpace and
# MCule strictly serially. Each vendor block makes 1–2 HTTPS round-trips
# (SMILES→ID and ID→Price), so a single batch_search is bounded by the
# *sum* of per-vendor latencies — typically 5–15 s. Inside the agent
# loop, score_route fires repeatedly with different leaf sets, and this
# becomes the dominant wall-clock contributor.
#
# This patch replaces collect_vendors with a thread-pool version that
# runs the three vendors concurrently. Each vendor is independent; we
# only merge the resulting DataFrames at the end. The patch is opt-in
# via env var (default on) and idempotent.

_CHEMPRICE_PATCH_APPLIED = False
_CHEMPRICE_PATCH_LOCK = threading.Lock()


def _patch_chemprice_vendor_concurrency() -> None:
    """Replace ``chemprice.utils.collect_vendors`` with a parallel version.

    Idempotent. Disable with ``CHEMPRICE_VENDOR_CONCURRENCY=0``.
    """
    global _CHEMPRICE_PATCH_APPLIED
    if _CHEMPRICE_PATCH_APPLIED:
        return
    if os.environ.get("CHEMPRICE_VENDOR_CONCURRENCY", "1").lower() in ("0", "false", "no"):
        return
    with _CHEMPRICE_PATCH_LOCK:
        if _CHEMPRICE_PATCH_APPLIED:
            return
        try:
            import pandas as _pd
            import chemprice.utils as _cpu
            from concurrent.futures import ThreadPoolExecutor as _TPE

            def _parallel_collect_vendors(
                instance, smiles_list, progress_output=None,
                ChemSpace=True, Molport=True, MCule=True,
            ):
                t0 = time.perf_counter()

                def _do_molport():
                    if not Molport:
                        return None
                    df_ids = _cpu.molport_get_ids(instance, smiles_list)
                    prices = _cpu.molport_collect_prices(instance, df_ids)
                    return ("Molport", prices)

                def _do_chemspace():
                    if not ChemSpace:
                        return None
                    return ("ChemSpace", _cpu.chemspace_collect_prices(instance, smiles_list))

                def _do_mcule():
                    if not MCule:
                        return None
                    df_ids = _cpu.mcule_get_ids(instance, smiles_list)
                    pkg = _cpu.build_packages(instance, df_ids)
                    prices = _cpu.mcule_collect_prices(instance, pkg)
                    prices = _cpu.add_input_smiles_columns(df_ids, prices)
                    return ("MCule", prices)

                tasks = [_do_molport, _do_chemspace, _do_mcule]
                results: List[Any] = []
                with _TPE(max_workers=len(tasks) or 1) as ex:
                    futs = [ex.submit(t) for t in tasks]
                    for f in futs:
                        try:
                            r = f.result()
                            if r is not None:
                                results.append(r)
                        except Exception as exc:
                            logger.warning("vendor concurrency: one provider failed: %s", exc)

                if not results:
                    return _pd.DataFrame([])

                merged = _cpu.merge_dataframes([row[1] for row in results])
                logger.info(
                    "ChemPrice vendor-parallel: %d providers in %.2fs",
                    len(results), time.perf_counter() - t0,
                )
                return merged

            _cpu.collect_vendors = _parallel_collect_vendors
            _CHEMPRICE_PATCH_APPLIED = True
            logger.info("Patched chemprice.utils.collect_vendors → parallel")
        except Exception as exc:
            logger.warning("Could not patch chemprice for vendor concurrency: %s", exc)


# ──────────────────────────────────────────────────────────────────────────
# PubChem-style heuristic (ported from chemoenzymatic-retrosynthesis-verifier
# zinc_client._heuristic_fallback). Pure RDKit — no network. Maps molecular
# complexity to a price tier and an estimated USD/g floor used to feed
# `_price_to_score`. Tier rules verbatim from the upstream client.
# ──────────────────────────────────────────────────────────────────────────

# Tier → estimated USD/g floor (conservative midpoints of the verifier's
# tier descriptions: commodity <$10/g, fine $10–100/g, specialty >$100/g).
_TIER_PRICE_FLOOR: Dict[str, float] = {
    "commodity": 5.0,
    "fine_chemical": 50.0,
    "specialty": 250.0,
}


def _heuristic_tier(smiles: str) -> Dict[str, Any]:
    """RDKit-only complexity heuristic. Returns dict with tier + availability.

    Maps molecular complexity to a tier and a conservative USD/g floor::

        n_atoms < 10 and n_rings <= 1   → commodity     (~$5/g)
        mw < 300   and n_rings <= 3     → fine_chemical  (~$50/g)
        otherwise                       → specialty      (~$250/g)

    Tier-dependent availability. ``commodity`` and ``fine_chemical`` (small,
    structurally simple molecules) are almost always purchasable, so they are
    marked ``available`` against their price floor. ``specialty`` (MW≥300 or
    >3 rings) is the tier the heuristic genuinely *cannot* judge — it is
    exactly the "is this actually in a catalogue?" case — so it is marked
    ``unavailable`` rather than optimistically passed.

    Rationale for the split: marking ALL tiers available (the prior behaviour)
    inflated the cost/buyability pillar offline — a route built entirely from
    exotic, genuinely-unpurchasable leaves still scored ~0.3-0.5 instead of
    0.0, defeating the "can this be made from stock?" check. Marking specialty
    unavailable restores that signal. The opposite extreme (specialty always
    unavailable AND used as the primary pricer) would penalise legitimately
    purchasable complex building blocks, but this heuristic is only the
    *offline fallback*; the 0/1 BuyabilityChain stock lookup is the primary
    path when any stock source is reachable.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"tier": "unknown", "available": False, "price_per_g": None}
        mw = Descriptors.MolWt(mol)
        n_atoms = mol.GetNumHeavyAtoms()
        n_rings = mol.GetRingInfo().NumRings()
        if n_atoms < 10 and n_rings <= 1:
            tier = "commodity"
        elif mw < 300 and n_rings <= 3:
            tier = "fine_chemical"
        else:
            tier = "specialty"
        return {
            "tier": tier,
            # Only the confidently-simple tiers are assumed purchasable; the
            # specialty tier is uncertain and must not inflate the pillar.
            "available": tier != "specialty",
            "price_per_g": _TIER_PRICE_FLOOR.get(tier),
        }
    except Exception as exc:  # pragma: no cover
        logger.debug("heuristic tier failed for %s: %s", smiles, exc)
        return {"tier": "unknown", "available": False, "price_per_g": None}


# ──────────────────────────────────────────────────────────────────────────
# Sqlite cache
# ──────────────────────────────────────────────────────────────────────────


class _PriceCache:
    """Per-session sqlite price cache. Schema: smiles → JSON price record."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str, ttl_seconds: int = 30 * 86400):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        # sqlite + WAL + NFS + concurrent writers can zero the header page.
        # Move a corrupt file aside so _init re-creates a fresh DB instead of
        # crashing every subsequent open with "file is not a database".
        try:
            if self.path.exists() and self.path.stat().st_size >= 16:
                with open(self.path, "rb") as fh:
                    head = fh.read(16)
                if head[:15] != b"SQLite format 3":
                    bad = self.path.with_suffix(
                        self.path.suffix + f".corrupt-{int(time.time())}"
                    )
                    self.path.rename(bad)
        except Exception:
            pass
        self._init()

    def _connect(self) -> sqlite3.Connection:
        # busy_timeout PRAGMA is required for cross-process concurrency
        # (ProcessPoolExecutor bench workers each open their own connection;
        # the threading.Lock above does NOT serialize across processes).
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @staticmethod
    def _retry(fn, attempts: int = 6, base_delay: float = 0.2):
        last: Exception | None = None
        for i in range(attempts):
            try:
                return fn()
            except sqlite3.OperationalError as e:
                msg = str(e).lower()
                if "locked" not in msg and "busy" not in msg:
                    raise
                last = e
                time.sleep(base_delay * (2 ** i))
        raise last  # type: ignore[misc]

    def _init(self) -> None:
        def _do():
            with self._lock, self._connect() as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS price_cache (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        created REAL NOT NULL DEFAULT (strftime('%s','now'))
                    )"""
                )
        self._retry(_do)

    def _key(self, smiles: str) -> str:
        return f"v{self.SCHEMA_VERSION}|{_canonicalize(smiles)}"

    def get(self, smiles: str) -> Optional[Dict[str, Any]]:
        import json
        key = self._key(smiles)
        def _do():
            with self._lock, self._connect() as conn:
                return conn.execute(
                    "SELECT value, created FROM price_cache WHERE key = ?", (key,)
                ).fetchone()
        row = self._retry(_do)
        if row is None:
            return None
        value_json, created = row
        if time.time() - float(created) > self.ttl:
            return None
        try:
            return json.loads(value_json)
        except Exception:
            return None

    def put(self, smiles: str, value: Dict[str, Any]) -> None:
        import json
        key = self._key(smiles)
        payload = json.dumps(value)
        def _do():
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO price_cache(key,value,created) "
                    "VALUES (?, ?, strftime('%s','now'))",
                    (key, payload),
                )
                conn.commit()
        self._retry(_do)


# ──────────────────────────────────────────────────────────────────────────
# Public LeafPricer
# ──────────────────────────────────────────────────────────────────────────


class LeafPricer:
    """Wraps ChemPrice (primary) + local BuyabilityChain (fallback) + sqlite cache."""

    def __init__(
        self,
        cache_db: str,
        env_file: Optional[str] = None,
        verifier_repo_path: Optional[str] = None,
        timeout_s: float = 30.0,
        cache_ttl_days: int = 30,
        buyability_chain: Any = None,  # BuyabilityChain instance for fallback
    ):
        self.cache = _PriceCache(cache_db, ttl_seconds=int(cache_ttl_days * 86400))
        self.timeout_s = timeout_s
        self.fallback = buyability_chain
        self._client = None
        self._client_init_attempted = False
        self._client_init_error: Optional[str] = None
        self.env_file = env_file
        self.verifier_repo_path = verifier_repo_path

    # ── lazy client init ──────────────────────────────────────────────

    def _get_client(self):
        if self._client_init_attempted:
            return self._client
        self._client_init_attempted = True

        # Load .env so the ChemPriceClient can read API keys from os.environ.
        if self.env_file and Path(self.env_file).exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(self.env_file, override=False)
                logger.info("LeafPricer: loaded credentials from %s", self.env_file)
            except Exception as exc:
                self._client_init_error = f"dotenv load failed: {exc}"
                logger.warning(self._client_init_error)
                return None
        elif self.env_file:
            self._client_init_error = f".env not found at {self.env_file}"
            logger.warning(self._client_init_error)

        # Disable MCule unconditionally — its API key has been throttled with
        # 84000-second back-offs (HTTP 429) that hang score_route for 25+
        # min while the storm walks 8 amount tiers × N leaves before the
        # 30 s per-batch timeout fires. dotenv just re-injected the key
        # above, so we strip it here right before the client reads
        # os.environ. MolPort and ChemSpace remain enabled.
        os.environ.pop("MCULE_API_KEY", None)

        # Additional env-driven vendor toggles. Useful when one vendor
        # is rate-limited or unreachable (e.g. ChemSpace's SSL/h2 ALPN
        # negotiation adds a 30 s read-timeout on every init from
        # some networks). Setting CHEMPRICE_DISABLE_MCULE here is
        # belt-and-braces over the unconditional pop above; the other
        # two are the actual on/off switches.
        if os.environ.get("CHEMPRICE_DISABLE_MCULE", "0").lower() in ("1", "true", "yes"):
            os.environ.pop("MCULE_API_KEY", None)
            logger.info("LeafPricer: MCule disabled via CHEMPRICE_DISABLE_MCULE")
        if os.environ.get("CHEMPRICE_DISABLE_CHEMSPACE", "0").lower() in ("1", "true", "yes"):
            os.environ.pop("CHEMSPACE_API_KEY", None)
            logger.info("LeafPricer: ChemSpace disabled via CHEMPRICE_DISABLE_CHEMSPACE")
        if os.environ.get("CHEMPRICE_DISABLE_MOLPORT", "0").lower() in ("1", "true", "yes"):
            os.environ.pop("MOLPORT_API_KEY", None)
            os.environ.pop("MOLPORT_USERNAME", None)
            os.environ.pop("MOLPORT_PASSWORD", None)
            logger.info("LeafPricer: MolPort disabled via CHEMPRICE_DISABLE_MOLPORT")

        # Make the verifier repo importable so we can reuse its hardened
        # ChemPriceClient (compat patches + credential plumbing already done).
        repo = self.verifier_repo_path
        if repo:
            repo_abs = str(Path(repo).resolve())
            if repo_abs not in sys.path:
                sys.path.insert(0, repo_abs)

        try:
            from backend.clients.chemprice_client import ChemPriceClient  # type: ignore
            self._client = ChemPriceClient()
            status = self._client.status()
            if not status.get("has_any_credentials"):
                self._client_init_error = (
                    f"ChemPriceClient: no credentials configured (status={status})"
                )
                logger.warning(self._client_init_error)
                self._client = None
            else:
                # Speed-up: parallelise the three vendors (MolPort / ChemSpace
                # / MCule) inside chemprice.utils.collect_vendors. Upstream is
                # strictly serial which dominates score_route latency under
                # repeated agent calls.
                _patch_chemprice_vendor_concurrency()
        except Exception as exc:
            self._client_init_error = f"ChemPriceClient init failed: {exc}"
            logger.warning(self._client_init_error)
            self._client = None
        return self._client

    # ── leaf scoring ──────────────────────────────────────────────────

    def evaluate_leaves(self, leaves: Iterable[str]) -> Dict[str, Any]:
        """Score every leaf SMILES; return aggregate + per-leaf details.

        Output schema::

          {
            "score": float in [0,1],          # mean of per-leaf leaf_score
            "leaves": [
              {"smiles": str, "available": bool, "price_per_g": float|None,
               "leaf_score": float, "vendor": str, "source": str}
            ],
            "n_total": int,
            "n_priced": int,                  # leaves with a USD/g quote
            "n_available": int,               # priced + buyable-but-unpriced
            "source_summary": "chemprice" | "fallback" | "mixed"
          }
        """
        leaves = [s for s in leaves if s]
        if not leaves:
            return {"score": 0.0, "leaves": [], "n_total": 0, "n_priced": 0,
                    "n_available": 0, "source_summary": "none"}

        canon_leaves = [_canonicalize(s) for s in leaves]

        # 1. Cache lookup first
        cached: Dict[str, Dict[str, Any]] = {}
        to_query: List[str] = []
        for cs in canon_leaves:
            hit = self.cache.get(cs)
            if hit is not None:
                cached[cs] = hit
            else:
                to_query.append(cs)

        # 2. ChemPrice batch for cache misses
        chemprice_results: Dict[str, Dict[str, Any]] = {}
        used_fallback = False
        if to_query:
            client = self._get_client()
            if client is not None:
                try:
                    rows = self._batch_with_timeout(client, to_query)
                    for row in rows:
                        smi = _canonicalize(row.get("smiles", ""))
                        if not smi:
                            continue
                        chemprice_results[smi] = {
                            "available": bool(row.get("available")),
                            "price_per_g": row.get("price_per_g"),
                            "vendor": row.get("vendor", ""),
                            "source": "chemprice" if row.get("available") else "chemprice_miss",
                        }
                except Exception as exc:
                    logger.warning("ChemPrice batch failed (%s); falling back to BuyabilityChain", exc)
                    used_fallback = True
            else:
                used_fallback = True

        # 3. Fallback for any leaf still missing or where ChemPrice said miss
        for cs in to_query:
            if cs not in chemprice_results:
                chemprice_results[cs] = self._fallback_lookup(cs)
                used_fallback = True
            elif not chemprice_results[cs]["available"] and self.fallback is not None:
                # ChemPrice said no — try local stocks too (Enamine etc.)
                fb = self._fallback_lookup(cs)
                if fb["available"]:
                    chemprice_results[cs] = fb
                    used_fallback = True

        # 4. Persist new lookups
        for cs, rec in chemprice_results.items():
            self.cache.put(cs, rec)

        # 5. Build per-leaf rows + aggregate
        leaf_rows: List[Dict[str, Any]] = []
        total_score = 0.0
        n_priced = 0
        n_available = 0
        for orig, cs in zip(leaves, canon_leaves):
            rec = cached.get(cs) or chemprice_results.get(cs) or {
                "available": False, "price_per_g": None,
                "vendor": "", "source": "none",
            }
            available = bool(rec.get("available"))
            price = rec.get("price_per_g")
            if available:
                n_available += 1
                if price is not None:
                    n_priced += 1
                    leaf_score = _price_to_score(float(price))
                else:
                    leaf_score = _price_to_score(None)  # 0.5
            else:
                leaf_score = 0.0
            total_score += leaf_score
            leaf_rows.append({
                "smiles": cs,
                "available": available,
                "price_per_g": price,
                "leaf_score": round(leaf_score, 4),
                "vendor": rec.get("vendor", ""),
                "source": rec.get("source", "none"),
            })

        agg_score = total_score / len(leaf_rows)
        if used_fallback and chemprice_results:
            src = "mixed"
        elif used_fallback:
            src = "fallback"
        else:
            src = "chemprice"
        return {
            "score": round(agg_score, 4),
            "leaves": leaf_rows,
            "n_total": len(leaf_rows),
            "n_priced": n_priced,
            "n_available": n_available,
            "source_summary": src,
        }

    # ── internals ─────────────────────────────────────────────────────

    def _batch_with_timeout(self, client: Any, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """Run ``client.batch_search`` in a thread with a hard timeout."""
        result_holder: Dict[str, Any] = {}

        def runner():
            try:
                result_holder["rows"] = client.batch_search(smiles_list)
            except Exception as exc:
                result_holder["err"] = exc

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=self.timeout_s)
        if t.is_alive():
            raise TimeoutError(f"ChemPrice batch_search exceeded {self.timeout_s}s")
        if "err" in result_holder:
            raise result_holder["err"]
        return result_holder.get("rows", [])

    def _fallback_lookup(self, smiles: str) -> Dict[str, Any]:
        """Three-tier fallback: local BuyabilityChain → PubChem-style heuristic.

        1. **BuyabilityChain** (Enamine pkl / molbloom / sqlite / legacy CSV) —
           strongest signal because it is a real stock-list hit. Emits 0/1
           via ``price_per_g=None``; ``_price_to_score`` maps that to 0.5.
        2. **PubChem-style heuristic** (RDKit MW + atom count + ring count) —
           kicks in when the stock chains say "no". Maps complexity to a
           tier and a conservative USD/g floor so the score never silently
           drops to 0 for clearly purchasable small molecules just because
           ChemPrice was offline / had no key for that vendor.
        """
        # Step 1: real stock chain
        if self.fallback is not None:
            try:
                res = self.fallback.lookup(smiles)
                if res.get("source") not in (None, "none"):
                    return {
                        "available": True,
                        "price_per_g": None,
                        "vendor": res.get("source", "") or "",
                        "source": f"fallback:{res.get('source', 'none')}",
                    }
            except Exception as exc:
                logger.debug("BuyabilityChain lookup failed for %s: %s", smiles, exc)

        # Step 2: heuristic complexity → tier → estimated price floor
        heur = _heuristic_tier(smiles)
        return {
            "available": bool(heur["available"]),
            "price_per_g": heur["price_per_g"],
            "vendor": "heuristic",
            "source": f"heuristic:{heur['tier']}",
        }
