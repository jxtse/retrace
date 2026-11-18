"""Check if a molecule is buyable (in building block datasets)."""

import hashlib
import json
import logging
import os
import pickle
import sqlite3
import threading

import pandas as pd
from rdkit import Chem
from tools.base import Tool

logger = logging.getLogger(__name__)


_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".cache", "stock_canonical"
)


class _StockIndex:
    """On-disk sqlite stock index — keeps per-worker RAM bounded.

    Replaces an in-memory ``set[str]`` of canonical SMILES (~1.9 GB for
    the 10.3M-row Zinc stock per worker). Lookups are constant-time
    sqlite indexed reads, and the sqlite file is shared across workers
    via the kernel page cache — a 3-worker batch now pays ~one copy of
    the working set instead of three.

    Connection is thread-local because ``services/concurrent_tools``
    runs tools from a ThreadPoolExecutor and sqlite connections are not
    thread-safe.
    """

    def __init__(self, sqlite_path: str, *, count: int):
        self._path = sqlite_path
        self._count = count
        self._tls = threading.local()

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._tls, "c", None)
        if c is None:
            c = sqlite3.connect(
                f"file:{self._path}?mode=ro", uri=True, check_same_thread=False
            )
            c.execute("PRAGMA query_only=ON")
            self._tls.c = c
        return c

    def __contains__(self, key) -> bool:
        if not isinstance(key, str) or not key:
            return False
        return self._conn().execute(
            "SELECT 1 FROM smi WHERE s = ? LIMIT 1", (key,)
        ).fetchone() is not None

    def __len__(self) -> int:
        return self._count


def _build_sqlite_from_iter(sqlite_path: str, mols) -> int:
    """Bulk-insert canonical SMILES into a fresh sqlite at ``sqlite_path``.

    Atomic via tmp + rename. ``WITHOUT ROWID`` halves on-disk size for a
    table whose only column is the primary key.
    """
    tmp = sqlite_path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    con = sqlite3.connect(tmp)
    try:
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        con.execute("CREATE TABLE smi (s TEXT PRIMARY KEY) WITHOUT ROWID")
        con.executemany("INSERT OR IGNORE INTO smi(s) VALUES (?)",
                        ((s,) for s in mols if isinstance(s, str) and s))
        con.commit()
        n = con.execute("SELECT COUNT(*) FROM smi").fetchone()[0]
    finally:
        con.close()
    os.replace(tmp, sqlite_path)
    return int(n)


class CheckBuyabilityTool(Tool):
    name = "check_buyability"
    description = (
        "Check whether one or more molecules (SMILES) are commercially-available "
        "starting materials. Accepts EITHER a single `smiles` string OR a "
        "`smiles_list` array — PREFER the batch form when checking multiple "
        "leaves of a route (one call ~2-50ms regardless of list length; N "
        "single calls cost N round-trips). Returns per-molecule buyability "
        "and the stock(s) it was found in."
    )
    parameters = {
        "type": "object",
        "properties": {
            "smiles": {
                "type": "string",
                "description": "SMILES of a single molecule to check. If passed together with smiles_list, the singleton (or its dot-joined parts) is merged into the list."
            },
            "smiles_list": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Batch form: list of SMILES to check in one call. "
                    "Prefer this when checking ≥ 2 leaves — the call is "
                    "near-instantaneous regardless of list length."
                ),
            },
        },
        # Avoid top-level JSON Schema combinators here. The 4141 OpenAI-
        # compatible backend rejects tool schemas containing anyOf (HTTP 502).
        # Runtime validation in run() still enforces that at least one of
        # `smiles` or `smiles_list` is present, and both remain optional so
        # the API retains its single- and batch-input forms.
    }

    def __init__(self, stock_files: dict[str, str]):
        """
        stock_files: dict of {stock_name: file_path}
        """
        # Stocks are sqlite-backed `_StockIndex` objects — no in-memory
        # set copies. Previous implementation held two giant sets per
        # worker (`_stocks[name]` + `_all_mols`), costing ~4 GB / worker
        # on the 10.3M-row Zinc stock and triggering OOM at --workers 3.
        self._stocks: dict[str, _StockIndex] = {}
        # Per-instance memo of canonical SMILES -> JSON result string. The
        # LLM frequently re-checks isosteric variants of the same scaffold
        # (e.g. probing Cl/Br/F halides on the same aryl core); both will
        # canonicalize to the same SMILES once, and we want the second call
        # to be free.
        self._result_cache: dict[str, str] = {}
        # Soft advice: if the LLM hits this tool with single SMILES repeatedly,
        # nudge it toward batching.
        self._recent_singletons = 0
        self._advice_window = 3   # need 3 singletons before advising
        for name, path in stock_files.items():
            # Isolate per-stock load failures: a single malformed stock file
            # (unsupported format, missing SMILES column, corrupt pickle)
            # must not kill check_buyability — it's mandatory on every leaf.
            # Skip the bad stock and keep the rest.
            try:
                self._stocks[name] = self._load_stock_cached(name, path)
                logger.info("Loaded stock '%s': %d molecules (sqlite index)",
                            name, len(self._stocks[name]))
            except Exception as e:
                logger.error("Failed to load stock '%s' from %s: %s; "
                             "skipping this stock", name, path, e)

    @staticmethod
    def _canon_one(s) -> str | None:
        if not isinstance(s, str):
            return None
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)

    @classmethod
    def _canonicalize_set(cls, raw) -> set:
        out = set()
        for s in raw:
            c = cls._canon_one(s)
            if c:
                out.add(c)
        return out

    @classmethod
    def _load_stock_cached(cls, name: str, path: str) -> "_StockIndex":
        """Canonicalize once, cache to disk as a sqlite index, reuse on later loads.

        Cache key includes the input path, size, and mtime so a stock-file
        replacement triggers a rebuild. Legacy pickle caches under the same
        key are also accepted — they're migrated to sqlite on first read.
        """
        try:
            st = os.stat(path)
            key = hashlib.md5(
                f"{path}|{st.st_size}|{int(st.st_mtime)}".encode()
            ).hexdigest()[:16]
        except OSError:
            key = hashlib.md5(path.encode()).hexdigest()[:16]
        sqlite_path = os.path.join(_CACHE_DIR, f"{name}_{key}.sqlite")
        if os.path.exists(sqlite_path):
            try:
                con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
                try:
                    n = con.execute("SELECT COUNT(*) FROM smi").fetchone()[0]
                finally:
                    con.close()
                logger.info("Loaded canonical stock sqlite: %s (%d mols)",
                            sqlite_path, n)
                return _StockIndex(sqlite_path, count=int(n))
            except Exception as e:
                logger.warning("Stock sqlite load failed (%s); rebuilding", e)
        # Migrate from a legacy pickle cache if present — saves the
        # multi-minute pandas-CSV reload on the first run after upgrading.
        legacy_pkl = os.path.join(_CACHE_DIR, f"{name}_{key}.pkl")
        if os.path.exists(legacy_pkl):
            try:
                with open(legacy_pkl, "rb") as f:
                    mols = pickle.load(f)
                os.makedirs(_CACHE_DIR, exist_ok=True)
                n = _build_sqlite_from_iter(sqlite_path, mols)
                logger.info("Migrated legacy pickle %s → %s (%d mols)",
                            legacy_pkl, sqlite_path, n)
                return _StockIndex(sqlite_path, count=n)
            except Exception as e:
                logger.warning("Legacy pickle migration failed (%s); "
                               "falling through to full rebuild", e)
        mols = cls._load_stock(path)
        try:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            n = _build_sqlite_from_iter(sqlite_path, mols)
            logger.info("Wrote canonical stock sqlite: %s (%d mols)",
                        sqlite_path, n)
        except Exception as e:
            logger.warning("Stock sqlite write failed: %s", e)
            # `mols` may be a generator (canonical-CSV streaming path), so
            # len() would raise TypeError; only sized containers have a
            # meaningful count. Unknown -> -1 rather than crashing the whole
            # tool at construction.
            n = len(mols) if isinstance(mols, (set, list, tuple, dict)) else -1
        return _StockIndex(sqlite_path, count=n)

    @classmethod
    def _load_stock(cls, path: str):
        """Return an iterable of canonical SMILES strings.

        For large already-canonical CSVs (Zinc 10.3M rows) this returns a
        chunked iterator — never materialises the full list in RAM, so the
        sqlite-build path stays bounded. Smaller / non-canonical files
        return a ``set`` after RDKit canonicalisation (preserves the old
        behaviour for the BioNav-style 100k stocks).
        """
        already_canonical = "canonical" in os.path.basename(path).lower()
        if path.endswith(".csv"):
            if already_canonical:
                # Stream the CSV in chunks so a 10M-row file never lives
                # in RAM as a single DataFrame.
                def _iter():
                    for chunk in pd.read_csv(path, chunksize=200_000):
                        col = cls._pick_smiles_column(chunk)
                        if col is None:
                            continue
                        for s in chunk[col].dropna().tolist():
                            if isinstance(s, str) and s:
                                yield s
                return _iter()
            df = pd.read_csv(path)
            preferred = cls._pick_smiles_column(df)
            if preferred is None:
                raise ValueError(f"No string SMILES column found in {path}")
            values = [s for s in df[preferred].dropna().tolist() if isinstance(s, str)]
            return cls._canonicalize_set(values)
        elif path.endswith(".pkl"):
            with open(path, "rb") as f:
                data = pickle.load(f)
            if already_canonical:
                # The "canonical" filename is only a hint. Validate it on a
                # sample before trusting it: if the stored SMILES aren't
                # actually canonical, lookups (which canonicalize the query)
                # would silently miss. Cheap check — sample up to 50 entries.
                as_set = data if isinstance(data, set) else set(data)
                sample = [s for s in list(as_set)[:50] if isinstance(s, str)]
                mismatched = sum(
                    1 for s in sample
                    if (c := cls._canon_one(s)) is not None and c != s
                )
                if sample and mismatched > 0:
                    logger.warning(
                        "Stock %s tagged 'canonical' but %d/%d sampled entries "
                        "are not canonical; canonicalizing the full set",
                        path, mismatched, len(sample),
                    )
                    return cls._canonicalize_set(as_set)
                return as_set
            return cls._canonicalize_set(data)
        else:
            raise ValueError(f"Unsupported stock file format: {path}")

    @staticmethod
    def _pick_smiles_column(df: pd.DataFrame) -> str | None:
        for col_name in ("smiles", "SMILES", "mol", "Smiles"):
            if col_name in df.columns:
                sample = df[col_name].dropna()
                if len(sample) and isinstance(sample.iloc[0], str):
                    return col_name
        for col_name in df.columns:
            sample = df[col_name].dropna()
            if len(sample) and isinstance(sample.iloc[0], str):
                return col_name
        return None

    def _canonicalize(self, smiles: str) -> str | None:
        return self._canon_one(smiles)

    def execute(self, smiles: str | None = None,
                smiles_list: list[str] | None = None, **kwargs) -> str:
        # Accept aliases occasionally emitted by the LLM.
        if smiles_list is None:
            smiles_list = kwargs.get("smiles_batch") or kwargs.get("molecules")
        # LLMs frequently emit smiles="" alongside a populated smiles_list;
        # treat empty / whitespace-only strings as absent rather than rejecting.
        if isinstance(smiles, str) and not smiles.strip():
            smiles = None
        # Batch path: process a list of SMILES in one call.
        if smiles_list is not None:
            # Be lenient when both args are populated: prefer smiles_list,
            # absorbing the singleton if it isn't already in the list. The
            # alternative (hard reject) was costing 100-290 wasted iterations
            # per target in long thorough runs because the LLM kept retrying
            # the same shape.
            if smiles is not None and isinstance(smiles_list, list):
                # Split dot-joined singletons so we don't double-add them.
                singletons = [s for s in str(smiles).split(".") if s]
                existing = {s for s in smiles_list if isinstance(s, str)}
                for s in singletons:
                    if s not in existing:
                        smiles_list.append(s)
                        existing.add(s)
                smiles = None
            if not isinstance(smiles_list, list):
                return json.dumps({"error": "smiles_list must be an array of SMILES strings"})
            results = []
            n_buyable = 0
            for one in smiles_list:
                if not isinstance(one, str):
                    results.append({"input_smiles": one, "error": "not a string"})
                    continue
                try:
                    one_out = json.loads(self._execute_one(one))
                except Exception as e:
                    one_out = {"input_smiles": one, "error": str(e)}
                if one_out.get("buyable"):
                    n_buyable += 1
                results.append(one_out)
            # Batch was used — reset the singleton-advice counter.
            self._recent_singletons = 0
            return json.dumps({
                "n": len(smiles_list),
                "count": len(results),
                "n_buyable": n_buyable,
                "results": results,
            })
        # Single SMILES path
        if smiles is None:
            return json.dumps({
                "error": "Missing argument: provide `smiles` (str) or `smiles_list` (list[str]).",
            })
        return self._execute_one(smiles)

    def _execute_one(self, smiles: str) -> str:
        can = self._canonicalize(smiles)
        if can is None:
            # Try a partial-sanitize parse so we can give the LLM a useful hint
            # instead of just "invalid SMILES" (saves a wasted iteration when
            # the model hand-crafted a SMILES with an open valence).
            hint = ""
            try:
                m_raw = Chem.MolFromSmiles(smiles, sanitize=False)
                if m_raw is not None:
                    try:
                        Chem.SanitizeMol(m_raw)
                    except Chem.rdchem.AtomValenceException as ve:
                        hint = f" (valence error: {ve})"
                    except Chem.rdchem.KekulizeException as ke:
                        hint = f" (kekulization failed: {ke})"
                    except Exception as e:
                        hint = f" (sanitize error: {type(e).__name__}: {e})"
                else:
                    hint = " (parser returned None — check brackets, ring closures, atom symbols)"
            except Exception as e:
                hint = f" ({type(e).__name__}: {e})"
            return json.dumps({
                "error": f"Invalid SMILES: {smiles}{hint}",
                "input_smiles": smiles,
                "hint": "Call validate_smiles first to canonicalize/repair before check_buyability.",
            })

        # Memo on canonical SMILES (cheap dedup across the session).
        cached = self._result_cache.get(can)
        if cached is not None:
            out = cached
        else:
            found_in = [name for name, stock in self._stocks.items() if can in stock]
            buyable = len(found_in) > 0
            out = json.dumps({
                "input_smiles": smiles,
                "canonical_smiles": can,
                "smiles": can,
                "buyable": buyable,
                "found_in_stocks": found_in,
            })
            self._result_cache[can] = out

        # Track singleton calls for the soft batching advice hint. Only
        # successful (non-error) payloads carry the hint — error payloads
        # already direct the model to validate_smiles.
        self._recent_singletons += 1
        if self._recent_singletons >= self._advice_window:
            try:
                payload = json.loads(out)
            except Exception:
                payload = None
            if isinstance(payload, dict) and "error" not in payload:
                payload["_advice"] = (
                    "Multiple sequential single-SMILES check_buyability calls "
                    "detected — prefer the `smiles_list` batch form when "
                    "checking ≥ 2 leaves. (This is a soft hint; tool budget "
                    "is unaffected.)"
                )
                out = json.dumps(payload)
        return out
