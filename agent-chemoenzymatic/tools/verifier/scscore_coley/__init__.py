"""Vendored Coley SCScore (1024-bit bool MorganFP + MLP, numpy-only).

Source: https://github.com/connorcoley/scscore (MIT-licensed; upstream LICENSE
  verified 2026-11-18, Copyright (c) 2017 Connor Coley).
  - standalone_model_numpy.py  (verbatim from upstream)
  - weights.json.gz            (full_reaxys_model_1024bool/model.ckpt-10654)

Weights file is .gitignore'd because of size; redownload with:
  curl -L -o weights.json.gz \\
    https://raw.githubusercontent.com/connorcoley/scscore/master/models/full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz

Public API (in this package's parent ``scscore.py``):
  - ``coley_sc_score(smiles) -> float in [1, 5]``
"""
import os
import sys

# Compatibility shim: standalone_model_numpy.py uses ``np.bool`` (removed in
# numpy 1.20) and ``six.iteritems``. Patch both before importing it.
import numpy as _np
if not hasattr(_np, "bool"):
    _np.bool = bool  # type: ignore[attr-defined]
try:
    import six  # noqa: F401
except ImportError:
    import types as _types
    _six = _types.ModuleType("six")
    _six.iteritems = lambda d: iter(d.items())  # type: ignore[attr-defined]
    sys.modules["six"] = _six
