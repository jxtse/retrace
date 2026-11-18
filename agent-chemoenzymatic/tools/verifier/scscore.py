"""Synthetic-complexity scoring (vendored, no path scanning, no silent fallback).

Primary scorer: **Coley SCScore** (Coley et al., J. Chem. Inf. Model. 2018, 58, 252).
  - Vendored at ``tools/verifier/scscore_coley/`` (numpy-only standalone model
    + ``weights.json.gz``, the ``full_reaxys_model_1024bool`` checkpoint).
  - Returns scores in [1, 5]; **5 is hardest**, 1 is easiest.
  - Loaded as a process-wide singleton (the model is ~5 MB on disk).

Fallback: **SAScore** (Ertl & Schuffenhauer 2009), vendored at
``tools/verifier/sa_score_vendor/``. Used only if Coley weights are missing or
fail to load. Returns scores in [1, 10]; rescaled to [1, 5] for consistency
with the cost pillar's decrease formula.

If neither scorer is available, the helpers raise ``RuntimeError`` — the cost
pillar no longer drops the SCScore decrease term silently.

Public API
~~~~~~~~~~
``sc_score(smiles) -> float``        — single SMILES, raises on failure.
``sa_score(smiles) -> Optional[float]`` — kept for backwards compat (returns
None on parse failure; never raises).
``sa_decrease_score(target, leaves) -> float``
    sigmoid((sc_target - sc_leaves_mean) * 2.0) ∈ [0, 1]; positive when target
    is more complex than leaves (a good route should reduce complexity).
"""

from __future__ import annotations

import logging
import math
import os
import threading
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Coley SCScore singleton ───────────────────────────────────────────────
_coley_model = None
_coley_lock = threading.Lock()
_coley_load_attempted = False
_coley_load_error: Optional[str] = None

_VENDOR_ROOT = Path(__file__).resolve().parent
_COLEY_DIR = _VENDOR_ROOT / "scscore_coley"
_COLEY_WEIGHTS = _COLEY_DIR / "weights.json.gz"


def _try_load_coley():
    global _coley_model, _coley_load_attempted, _coley_load_error
    if _coley_load_attempted:
        return _coley_model
    with _coley_lock:
        if _coley_load_attempted:
            return _coley_model
        if not _COLEY_WEIGHTS.exists():
            _coley_load_error = (
                f"Coley SCScore weights missing at {_COLEY_WEIGHTS}. "
                "Download with:\n  curl -L -o "
                f"{_COLEY_WEIGHTS} "
                "https://raw.githubusercontent.com/connorcoley/scscore/master/"
                "models/full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz"
            )
            logger.warning(_coley_load_error)
            _coley_load_attempted = True
            return None
        try:
            # Apply np.bool / six compat patches (in scscore_coley/__init__.py)
            import tools.verifier.scscore_coley  # noqa: F401
            from tools.verifier.scscore_coley.standalone_model_numpy import SCScorer

            m = SCScorer()
            # Suppress upstream's "Restored variables from ..." print.
            import builtins as _b
            _orig_print = _b.print
            _b.print = lambda *a, **kw: None
            try:
                m.restore(str(_COLEY_WEIGHTS))
            finally:
                _b.print = _orig_print
            _coley_model = m
            logger.info("Loaded Coley SCScore from %s", _COLEY_WEIGHTS)
            return _coley_model
        except Exception as exc:
            _coley_load_error = f"Coley SCScore load failed: {exc}"
            logger.warning(_coley_load_error)
            return None
        finally:
            # Set the "attempted" flag only after load completes, so that
            # a concurrent caller waiting on the lock — or a caller racing
            # the no-lock early-return at the top of this function — sees
            # either the loaded model or a populated error, never the
            # in-progress (attempted=True, model=None, error=None) state.
            _coley_load_attempted = True


# ── SAScore (vendored) fallback ───────────────────────────────────────────
_sascorer = None
_sa_lock = threading.Lock()
_sa_load_attempted = False


def _try_import_sascorer():
    global _sascorer, _sa_load_attempted
    if _sa_load_attempted:
        return _sascorer
    with _sa_lock:
        if _sa_load_attempted:
            return _sascorer
        try:
            from tools.verifier.sa_score_vendor import sascorer as _sa
            _sascorer = _sa
            logger.info("Loaded vendored SAScore (fallback)")
        except Exception as exc:
            logger.warning("Vendored SAScore import failed: %s", exc)
        finally:
            # Same ordering as Coley: flag flips only after the import
            # finishes, so concurrent callers never observe the
            # in-progress (attempted=True, sascorer=None) state.
            _sa_load_attempted = True
    return _sascorer


# ── Public API ────────────────────────────────────────────────────────────


def sc_score(smiles: str) -> float:
    """Return synthetic complexity for one SMILES, in [1, 5].

    Tries Coley SCScore first (1024-bit bool Morgan FP + MLP), falls back to
    SAScore rescaled from [1, 10] to [1, 5]. Raises ``RuntimeError`` if both
    are unavailable.
    """
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            raise ValueError(f"unparseable SMILES: {smiles!r}")
    except Exception as exc:
        raise RuntimeError(f"RDKit parse failed for {smiles!r}: {exc}") from exc

    coley = _try_load_coley()
    if coley is not None:
        try:
            _, score = coley.get_score_from_smi(smiles)
            # The vendored numpy model returns a shape-(1,) ndarray on newer
            # numpy versions; normalize it before converting to a Python float.
            if hasattr(score, "reshape"):
                score = score.reshape(-1)[0]
            score_f = float(score)
            if score_f <= 0.0:  # upstream returns 0.0 on FP failure
                raise ValueError("Coley SCScore returned 0 (fingerprint failure)")
            # Coley score is already in [1, 5]
            return max(1.0, min(5.0, score_f))
        except Exception as exc:
            logger.warning("Coley SCScore failed for %s: %s; trying SAScore", smiles, exc)

    sa = _try_import_sascorer()
    if sa is not None:
        try:
            raw = float(sa.calculateScore(m))  # [1, 10]
            # Rescale to [1, 5]
            return 1.0 + (raw - 1.0) * (4.0 / 9.0)
        except Exception as exc:
            raise RuntimeError(
                f"Both Coley SCScore and SAScore failed for {smiles!r}: {exc}"
            ) from exc

    raise RuntimeError(
        "No synthetic-complexity scorer available. Coley load error: "
        f"{_coley_load_error or 'not attempted'}. "
        "Install vendored SAScore at tools/verifier/sa_score_vendor/ or download "
        "Coley weights to tools/verifier/scscore_coley/weights.json.gz."
    )


def sa_score(smiles: str) -> Optional[float]:
    """Backwards-compat shim. Returns the raw SAScore in [1, 10], or None on
    parse failure. Never raises.
    """
    sa = _try_import_sascorer()
    if sa is None:
        return None
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        return float(sa.calculateScore(m))
    except Exception:
        return None


def sa_decrease_score(target_smiles: str, leaf_smiles: List[str]) -> float:
    """sigmoid((sc_target - sc_leaves_mean) * 2.0) in [0, 1].

    The `* 2.0` stretches the response curve to compensate for SCScore's
    narrower [1, 5] range vs SAScore's [1, 10] (without it, even a 1-unit
    complexity drop only moves the score by ~0.12).

    Raises ``RuntimeError`` if the underlying scorer is unavailable for the
    target — this is intentional: the cost pillar should not silently drop
    the SC term anymore.
    """
    sc_t = sc_score(target_smiles)
    leaf_vals: List[float] = []
    for s in leaf_smiles:
        try:
            leaf_vals.append(sc_score(s))
        except Exception as exc:
            logger.debug("sc_score failed for leaf %s: %s", s, exc)
    if not leaf_vals:
        # Edge case: target has no leaves (zero-step "route"). Treat as
        # neutral 0.5 rather than raising — cost pillar still has the other
        # two terms.
        return 0.5
    sc_l = sum(leaf_vals) / len(leaf_vals)
    delta = sc_t - sc_l
    return 1.0 / (1.0 + math.exp(-delta * 2.0))
