"""Sync bridge to ASKCOS NeuralNetContextRecommender (Gao 2018).

Loads the optional ``tools/condition_predictor`` package (ASKCOS-derived,
MPL-2.0; not redistributed with this snapshot — see LICENSES.md). This class
hard-fails on missing package/weights by design; the verifier treats that
failure as an absent optional component (degraded greenness) rather than a
reason to refuse to construct ``score_route``.

Caches the model at class level (one process-wide singleton) and predictions
on a per-instance LRU.
"""

from __future__ import annotations

import logging
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ConditionPredictor:
    """Sync wrapper. Use ``predict_top1(rxn_smiles)`` to get one (T, solvent, …)."""

    _model: Any = None
    _model_lock = threading.Lock()

    def __init__(self, condition_model_dir: str):
        self.dir = Path(condition_model_dir).resolve()
        if not self.dir.exists():
            raise FileNotFoundError(
                f"condition_model_dir does not exist: {self.dir}. "
                "Symlink tools/condition_predictor → chemoenzymatic-retrosynthesis-verifier "
                "tools/condition_predictor."
            )
        # Make `condition_predictor` package importable
        parent = str(self.dir.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        self.weights_path = self.dir / "data" / "dict_weights.npy"
        self.info_path = self.dir / "data"
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Condition predictor weights not found at {self.weights_path}. "
                "ASKCOS condition predictor is required by the verifier (greenness "
                "depends on T/solvent prediction); no fallback is provided."
            )
        # Eagerly probe import so failure surfaces at agent build, not on first call.
        try:
            from condition_predictor.condition_model import NeuralNetContextRecommender  # noqa: F401
        except Exception as exc:
            raise ImportError(
                f"Could not import condition_predictor: {exc}. "
                f"Check that {parent} is on sys.path and that the symlink target exists."
            ) from exc

    def _ensure_loaded(self) -> Any:
        if ConditionPredictor._model is not None:
            return ConditionPredictor._model
        with ConditionPredictor._model_lock:
            if ConditionPredictor._model is not None:
                return ConditionPredictor._model
            from condition_predictor.condition_model import NeuralNetContextRecommender
            logger.info("Loading ASKCOS NeuralNetContextRecommender from %s", self.dir)
            m = NeuralNetContextRecommender()
            m.load_nn_model(
                info_path=str(self.info_path),
                weights_path=str(self.weights_path),
            )
            ConditionPredictor._model = m
            logger.info("ASKCOS condition predictor loaded")
            return m

    def predict_top1(self, rxn_smiles: str) -> Dict[str, Any]:
        """Return ``{temperature, solvent, reagent, catalyst, score}`` for top hit."""
        return self._predict_cached(rxn_smiles)

    @lru_cache(maxsize=2048)
    def _predict_cached(self, rxn_smiles: str) -> Dict[str, Any]:
        # Failures used to be swallowed and surfaced as a "default" reaction
        # condition (25 °C, unknown solvent), which silently lifted greenness
        # scores when the model actually crashed. Now we raise so the
        # orchestrator can present a real tool_error.
        m = self._ensure_loaded()
        combos, scores = m.get_n_conditions(rxn_smiles, n=1, return_scores=True)
        if not combos:
            raise RuntimeError(
                f"ASKCOS condition predictor returned no candidates for {rxn_smiles!r}"
            )
        combo = combos[0]
        score = float(scores[0]) if len(scores) else 0.0
        T = combo[0] if len(combo) > 0 else None
        solvent = combo[1] if len(combo) > 1 else ""
        reagent = combo[2] if len(combo) > 2 else ""
        catalyst = combo[3] if len(combo) > 3 else ""
        try:
            T_val = float(T) if T is not None and str(T).lower() != "nan" else 25.0
        except (TypeError, ValueError):
            T_val = 25.0
        return {
            "temperature": T_val,
            "solvent": str(solvent) if solvent and str(solvent).lower() != "nan" else "unknown",
            "reagent": str(reagent) if reagent and str(reagent).lower() != "nan" else "",
            "catalyst": str(catalyst) if catalyst and str(catalyst).lower() != "nan" else "",
            "score": score,
        }


def temperature_score(T_celsius: float) -> float:
    """Score a temperature in [0,1]: best near 25 °C, falls off below 0 and above 150 °C.

    Piecewise linear: 1.0 in [0, 80], 0.66 at 100, 0.33 at 150, 0.0 at >=200.
    Below 0 (cryogenic): drops to 0.66 at -20, 0.33 at -40, 0.0 at <=-78.
    """
    if T_celsius is None:
        return 0.5
    T = float(T_celsius)
    if 0.0 <= T <= 80.0:
        return 1.0
    if 80.0 < T <= 100.0:
        return 1.0 - (T - 80.0) * (1.0 - 0.66) / 20.0
    if 100.0 < T <= 150.0:
        return 0.66 - (T - 100.0) * (0.66 - 0.33) / 50.0
    if 150.0 < T < 200.0:
        return 0.33 - (T - 150.0) * 0.33 / 50.0
    if T >= 200.0:
        return 0.0
    # Cryogenic
    if -20.0 <= T < 0.0:
        return 1.0 - (-T) * (1.0 - 0.66) / 20.0
    if -40.0 <= T < -20.0:
        return 0.66 - (-T - 20.0) * (0.66 - 0.33) / 20.0
    if -78.0 < T < -40.0:
        return 0.33 - (-T - 40.0) * 0.33 / 38.0
    return 0.0
