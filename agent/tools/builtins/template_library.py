"""
Curated retrosynthesis template library — ports
``chemoenzymatic-retrosynthesis-verifier/backend/tools/template_loader.py``
into the agent so that ``suggest_templates`` can rank real,
database-sourced disconnections instead of requiring the LLM to
hand-author SMARTS.

Sources pulled (all optional — whichever files are present on disk are
loaded, the rest are skipped with a warning):

  - RetroBioCat     : enzymatic YAML templates
  - BRENDA/EnzymeMap: enzymatic plain-text SMARTS
  - USPTO (aizynth) : organic HDF5 templates
  - RetroRules      : curated biochemical JSON templates (MetaNetX + Rhea)
  - BKMS            : biochemical HDF5 templates

Templates are in retro format (``product>>reactants``). The module-level
caches mean parsing happens at most once per agent process.

This module is import-safe even when the dataset directory is missing —
``load_all_templates()`` will simply return an empty list in that case
so the registry doesn't refuse to load the tool.
"""

from __future__ import annotations

import gzip
import hashlib
import json as _json
import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

_ALL_TEMPLATES: Optional[List[str]] = None
_TEMPLATE_SOURCES: Optional[List[str]] = None
_PARSED_REACTIONS: Optional[List[Any]] = None
_DATASET_DIR: Optional[Path] = None

# On-disk cache for the parsed (templates, sources) lists. Building them from
# the 2.4 GB raw dataset takes ~4 s on every process start; the parsed output
# is only ~490k short strings, so we persist it and rebuild only when the
# dataset files change (fingerprint = sorted relpath|size|mtime over the
# dataset dir). Mirrors the sqlite stock cache in check_buyability.
_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".cache",
    "template_library",
)


# ---------------------------------------------------------------------------
# Dataset-dir resolution
# ---------------------------------------------------------------------------

_DEFAULT_CANDIDATES = (
    # 1. explicit env var wins
    os.environ.get("RETRO_TEMPLATE_DATASET_DIR", ""),
    # in-repo copy (if the user ever vendors the templates)
    "dataset",
)


def _clear_pattern_cache() -> None:
    """Clear the compiled-SMARTS cache in suggest_templates, which is keyed
    by integer template index and would otherwise survive a dataset reload
    and match the wrong template. Lazy import avoids a circular dependency
    (suggest_templates imports this module)."""
    try:
        from tools.builtins.suggest_templates import clear_pattern_cache
        clear_pattern_cache()
    except Exception:
        # suggest_templates may not be importable in every context (e.g.
        # template_library used standalone); a missed clear is non-fatal.
        pass


def set_dataset_dir(path: Optional[str]) -> None:
    """Override the dataset dir (invalidates cache)."""
    global _DATASET_DIR, _ALL_TEMPLATES, _TEMPLATE_SOURCES, _PARSED_REACTIONS
    _DATASET_DIR = Path(path).resolve() if path else None
    _ALL_TEMPLATES = None
    _TEMPLATE_SOURCES = None
    _PARSED_REACTIONS = None
    _clear_pattern_cache()


def get_dataset_dir() -> Optional[Path]:
    global _DATASET_DIR
    if _DATASET_DIR is not None:
        return _DATASET_DIR
    for cand in _DEFAULT_CANDIDATES:
        if not cand:
            continue
        p = Path(cand)
        if not p.is_absolute():
            p = (Path(__file__).resolve().parents[2] / cand).resolve()
        if p.exists() and p.is_dir():
            _DATASET_DIR = p
            return _DATASET_DIR
    return None


def _find_template_file(relative_path: Path) -> Optional[Path]:
    root = get_dataset_dir()
    if root is None:
        return None
    cand = root / relative_path
    return cand if cand.exists() else None


# ---------------------------------------------------------------------------
# Per-source loaders
# ---------------------------------------------------------------------------

def _load_retrobiocat(templates: List[str], sources: List[str]) -> None:
    yaml_path = _find_template_file(Path("retrobiocat") / "rxns_yaml.yaml")
    if yaml_path is None:
        logger.debug("RetroBioCat template file not found")
        return
    try:
        import yaml  # PyYAML
    except ImportError:
        logger.warning("PyYAML missing; skipping RetroBioCat templates")
        return

    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    loaded = 0
    for reaction_name, reaction_data in data.items():
        smarts_list = (reaction_data or {}).get("smarts", []) or []
        for smarts in smarts_list:
            if smarts and ">>" in smarts:
                templates.append(smarts)
                sources.append(f"RetroBioCat:{reaction_name}")
                loaded += 1
    logger.info("Loaded %d RetroBioCat templates", loaded)


def _load_brenda(templates: List[str], sources: List[str]) -> None:
    brenda_path = _find_template_file(
        Path("enzymemap") / "brenda" /
        "brenda_corrected_template_default_unique_templates.txt"
    )
    if brenda_path is None:
        logger.debug("BRENDA template file not found")
        return

    loaded = 0
    with brenda_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            t = line.strip()
            if t and ">>" in t:
                templates.append(t)
                sources.append(f"BRENDA:template_{line_number}")
                loaded += 1
    logger.info("Loaded %d BRENDA templates", loaded)


def _load_uspto(templates: List[str], sources: List[str]) -> None:
    uspto_path = _find_template_file(
        Path("aizynthfinder") / "full_uspto_03_05_19_unique_templates.hdf5"
    )
    if uspto_path is None:
        logger.debug("USPTO template file not found")
        return
    try:
        import h5py
    except ImportError:
        logger.warning("h5py missing; skipping USPTO templates")
        return

    loaded = 0
    try:
        with h5py.File(uspto_path, "r") as handle:
            ds = None
            if "templates" in handle:
                ds = handle["templates"]
            else:
                for key in handle.keys():
                    if key.endswith("templates"):
                        ds = handle[key]
                        break
            if ds is not None:
                for index, template in enumerate(ds):
                    if isinstance(template, bytes):
                        template = template.decode("utf-8")
                    t = str(template).strip()
                    if ">>" in t:
                        templates.append(t)
                        sources.append(f"USPTO:{index}")
                        loaded += 1
            elif "table" in handle and "block0_values" in handle["table"]:
                import pickle
                try:
                    from numpy.exceptions import VisibleDeprecationWarning
                except ImportError:
                    from numpy import VisibleDeprecationWarning  # type: ignore
                col_names = [
                    item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    for item in handle["table/block0_items"][:]
                ]
                try:
                    tidx = col_names.index("retro_template")
                except ValueError:
                    logger.warning("USPTO HDF5 missing retro_template column")
                    return
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=VisibleDeprecationWarning)
                    block = pickle.loads(bytes(handle["table/block0_values"][0]))
                for index, row in enumerate(block):
                    t = str(row[tidx]).strip()
                    if ">>" in t:
                        templates.append(t)
                        sources.append(f"USPTO:{index}")
                        loaded += 1
    except Exception as exc:
        logger.warning("Failed to read USPTO HDF5: %s", exc)
        return
    logger.info("Loaded %d USPTO templates", loaded)


def _load_retrorules(templates: List[str], sources: List[str]) -> None:
    files = [
        Path("retrorules") / "metanetx_templates.json.gz",
        Path("retrorules") / "rhea_templates.json.gz",
    ]
    total = 0
    for rel in files:
        gz_path = _find_template_file(rel)
        if gz_path is None:
            continue
        try:
            with gzip.open(gz_path, "rt", encoding="utf-8") as handle:
                entries = _json.load(handle)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", gz_path, exc)
            continue
        loaded = 0
        for entry in entries:
            if not entry.get("VALID", False):
                continue
            smarts = entry.get("TEMPLATE", "")
            if not smarts or ">>" not in smarts:
                continue
            template_id = entry.get("TEMPLATE_ID", f"rr_{loaded}")
            templates.append(smarts)
            sources.append(f"RetroRules:{template_id}")
            loaded += 1
        total += loaded
    if total:
        logger.info("Loaded %d RetroRules templates", total)


def _load_doranet_synthetic(templates: List[str], sources: List[str]) -> None:
    """Load DORAnet retro organic SMARTS operators (Reaction_Smarts_Retro.py)."""
    py_path = _find_template_file(Path("doranet") / "Reaction_Smarts_Retro.py")
    if py_path is None:
        logger.debug("DORAnet retro SMARTS file not found")
        return
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_doranet_retro_smarts", str(py_path)
        )
        if spec is None or spec.loader is None:
            logger.warning("Could not build import spec for %s", py_path)
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[arg-type]
    except Exception as exc:
        logger.warning("Failed to import DORAnet retro SMARTS: %s", exc)
        return

    ops = getattr(mod, "op_retro_smarts", ())
    loaded = 0
    for op in ops:
        smarts = getattr(op, "smarts", None)
        name = getattr(op, "name", None)
        if not smarts or ">>" not in smarts:
            continue
        # DORAnet operators with reaction_type 'Enzymatic' would be tagged
        # separately; the synthetic module only contains 'Catalytic'.
        templates.append(smarts)
        sources.append(f"Doranet:{name or loaded}")
        loaded += 1
    logger.info("Loaded %d DORAnet (organic) templates", loaded)


def _load_doranet_enzymatic(templates: List[str], sources: List[str]) -> None:
    """Load DORAnet enzymatic SMARTS rule sets (JN1224MIN + JN3604IMT TSVs)."""
    import csv
    files = [
        ("JN1224MIN", Path("doranet") / "JN1224MIN_rules.tsv"),
        ("JN3604IMT", Path("doranet") / "JN3604IMT_rules.tsv"),
    ]
    total = 0
    for prefix_id, rel in files:
        path = _find_template_file(rel)
        if path is None:
            continue
        loaded = 0
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    smarts = (row.get("SMARTS") or "").strip()
                    name = (row.get("Name") or "").strip() or f"row_{loaded}"
                    if not smarts or ">>" not in smarts:
                        continue
                    templates.append(smarts)
                    sources.append(f"DoranetEnzyme:{prefix_id}:{name}")
                    loaded += 1
        except Exception as exc:
            logger.warning("Failed to parse DORAnet TSV %s: %s", path, exc)
            continue
        total += loaded
    if total:
        logger.info("Loaded %d DORAnet (enzymatic) templates", total)


def _load_bkms(templates: List[str], sources: List[str]) -> None:
    bkms_path = _find_template_file(Path("bkms") / "bkms_templates_only.hdf")
    if bkms_path is None:
        logger.debug("BKMS template file not found")
        return
    try:
        import h5py
        import pickle
    except ImportError:
        logger.warning("h5py/pickle missing; skipping BKMS templates")
        return
    try:
        with h5py.File(bkms_path, "r") as handle:
            col_names = [
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in handle["df/block2_items"][:]
            ]
            if "reaction_smarts" not in col_names:
                logger.warning("BKMS HDF5 missing reaction_smarts column")
                return
            smarts_idx = col_names.index("reaction_smarts")
            id_idx = col_names.index("_id") if "_id" in col_names else None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                block = pickle.loads(bytes(handle["df/block2_values"][0]))
        loaded = 0
        for row_idx in range(block.shape[0]):
            smarts = str(block[row_idx][smarts_idx]).strip()
            if not smarts or ">>" not in smarts:
                continue
            template_id = (
                str(block[row_idx][id_idx]) if id_idx is not None
                else f"bkms_{loaded}"
            )
            templates.append(smarts)
            sources.append(f"BKMS:{template_id}")
            loaded += 1
        logger.info("Loaded %d BKMS templates", loaded)
    except Exception as exc:
        logger.warning("Failed to read BKMS HDF5: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ENZYME_SOURCE_PREFIXES = (
    "RetroBioCat:", "BRENDA:", "RetroRules:", "BKMS:", "DoranetEnzyme:",
)
ORGANIC_SOURCE_PREFIXES = ("USPTO:", "Doranet:")


def _dataset_fingerprint(dataset_dir: Path) -> str:
    """Stat-only fingerprint of the dataset dir: sorted (relpath|size|mtime_ns)
    over every file. Cheap (no content reads) and changes whenever any source
    file is added, removed, or modified — so the cache rebuilds automatically.

    Uses nanosecond mtime (`st_mtime_ns`), not `int(st_mtime)`: whole-second
    truncation would miss a same-second rewrite that preserves byte size (e.g.
    a templated regeneration or a `cp`/`rsync` that keeps the size), serving a
    stale cache indefinitely.
    """
    parts: List[str] = []
    for root, _dirs, files in os.walk(dataset_dir):
        for f in sorted(files):
            p = os.path.join(root, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            rel = os.path.relpath(p, dataset_dir)
            parts.append(f"{rel}|{st.st_size}|{st.st_mtime_ns}")
    parts.sort()
    h = hashlib.md5("\n".join(parts).encode("utf-8")).hexdigest()[:16]
    return h


def _cache_path(dataset_dir: Path, fingerprint: str) -> str:
    key = hashlib.md5(str(dataset_dir).encode("utf-8")).hexdigest()[:8]
    return os.path.join(_CACHE_DIR, f"templates_{key}_{fingerprint}.json")


def _load_cached_templates(
    dataset_dir: Path, fingerprint: str
) -> Optional[Tuple[List[str], List[str]]]:
    path = _cache_path(dataset_dir, fingerprint)
    if not os.path.exists(path):
        return None
    try:
        # JSON, not pickle: the cache holds only two lists of strings, and
        # `pickle.load` of a file anyone with write access to .cache/ could
        # craft is an arbitrary-code-execution vector (executes on load via
        # __reduce__, before any validation). JSON can only yield plain data.
        with open(path, "r", encoding="utf-8") as fh:
            blob = _json.load(fh)
        templates = blob.get("templates")
        sources = blob.get("sources")
        if (
            isinstance(templates, list)
            and isinstance(sources, list)
            and len(templates) == len(sources)
        ):
            logger.info(
                "Template library: loaded %d templates from disk cache %s",
                len(templates), path,
            )
            return templates, sources
        logger.warning("Template cache %s malformed; rebuilding", path)
    except Exception as exc:
        logger.warning("Template cache load failed (%s); rebuilding", exc)
    return None


def _store_cached_templates(
    dataset_dir: Path, fingerprint: str,
    templates: List[str], sources: List[str],
) -> None:
    path = _cache_path(dataset_dir, fingerprint)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            _json.dump({"templates": templates, "sources": sources}, fh)
        os.replace(tmp, path)  # atomic — concurrent workers can't see a partial file
        logger.info("Template library: wrote disk cache %s (%d templates)",
                    path, len(templates))
    except Exception as exc:
        logger.warning("Template cache write failed: %s", exc)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def load_all_templates() -> Tuple[List[str], List[str]]:
    """Lazy-load templates from every available source.

    Returns ``(templates, sources)`` where ``sources[i]`` tags the origin
    of ``templates[i]``. Empty lists when no dataset is found.

    Backed by an on-disk cache keyed on a stat-fingerprint of the dataset
    dir: the first build parses the raw 2.4 GB dataset (~4 s) and persists
    the ~490k parsed strings; later process starts load the cache in
    milliseconds and rebuild only when a dataset file changes.
    """
    global _ALL_TEMPLATES, _TEMPLATE_SOURCES
    if _ALL_TEMPLATES is not None and _TEMPLATE_SOURCES is not None:
        return _ALL_TEMPLATES, _TEMPLATE_SOURCES

    templates: List[str] = []
    sources: List[str] = []
    dataset_dir = get_dataset_dir()
    if dataset_dir is None:
        logger.warning(
            "No template dataset directory found. Set RETRO_TEMPLATE_DATASET_DIR "
            "or provide templates.dataset_dir via config override."
        )
        _ALL_TEMPLATES, _TEMPLATE_SOURCES = templates, sources
        return templates, sources

    # Try the disk cache first.
    fingerprint = _dataset_fingerprint(dataset_dir)
    cached = _load_cached_templates(dataset_dir, fingerprint)
    if cached is not None:
        _ALL_TEMPLATES, _TEMPLATE_SOURCES = cached
        return _ALL_TEMPLATES, _TEMPLATE_SOURCES

    for loader in (
        _load_retrobiocat,
        _load_brenda,
        _load_uspto,
        _load_retrorules,
        _load_bkms,
        _load_doranet_synthetic,
        _load_doranet_enzymatic,
    ):
        try:
            loader(templates, sources)
        except Exception as exc:
            logger.warning("Template loader %s failed: %s", loader.__name__, exc)

    _ALL_TEMPLATES = templates
    _TEMPLATE_SOURCES = sources
    logger.info(
        "Template library: %d total templates from sources %s",
        len(templates),
        sorted({s.split(":", 1)[0] for s in sources}) or "[]",
    )
    # Persist for next time (skip empty builds — likely a transient miss).
    if templates:
        _store_cached_templates(dataset_dir, fingerprint, templates, sources)
    return _ALL_TEMPLATES, _TEMPLATE_SOURCES


def load_parsed_templates() -> Tuple[List[Any], List[str], List[str]]:
    """Pre-parse every template into ``rdchiralReaction`` objects.

    Entries that fail to parse are stored as ``None``. Returns
    ``(parsed, raw_templates, sources)``.
    """
    global _PARSED_REACTIONS
    templates, sources = load_all_templates()
    if _PARSED_REACTIONS is not None:
        return _PARSED_REACTIONS, templates, sources

    try:
        from rdchiral.initialization import rdchiralReaction
    except ImportError:
        logger.warning("rdchiral not installed; templates will not be applied")
        _PARSED_REACTIONS = [None] * len(templates)
        return _PARSED_REACTIONS, templates, sources

    parsed: List[Any] = []
    n_ok = n_fail = 0
    for smarts in templates:
        try:
            parsed.append(rdchiralReaction(smarts))
            n_ok += 1
        except Exception:
            parsed.append(None)
            n_fail += 1
    _PARSED_REACTIONS = parsed
    logger.info(
        "Pre-parsed %d/%d templates (%d failed)",
        n_ok, len(templates), n_fail,
    )
    return _PARSED_REACTIONS, templates, sources


def classify_source(source_tag: str) -> str:
    """Classify a source tag as 'organic', 'enzymatic', or 'other'."""
    if source_tag.startswith(ENZYME_SOURCE_PREFIXES):
        return "enzymatic"
    if source_tag.startswith(ORGANIC_SOURCE_PREFIXES):
        return "organic"
    return "other"


def reset_cache() -> None:
    global _ALL_TEMPLATES, _TEMPLATE_SOURCES, _PARSED_REACTIONS
    _ALL_TEMPLATES = None
    _TEMPLATE_SOURCES = None
    _PARSED_REACTIONS = None
    _clear_pattern_cache()
