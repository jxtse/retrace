"""recall_similar_routes — agent-controlled precedent RAG tool.

Given a target / intermediate SMILES, return up to k structurally-similar
known synthesis routes from the precedent database, selected for
diversity via MMR. Spec: docs/superpowers/specs/2026-05-17-precedent-recall-adaptive-rag-design.md

The actual retrieval lives in ``services.precedent_db.PrecedentDB``;
this file is just the LLM-facing surface (schema, validation,
JSON-stringification).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from tools.base import Tool

logger = logging.getLogger(__name__)


class RecallSimilarRoutesTool(Tool):
    name = "recall_similar_routes"
    description = (
        "Retrieve up to k known full synthesis routes whose target molecule "
        "is Tanimoto-similar to `query_smiles`. Results are diversity-"
        "selected via MMR and tagged with a `confidence` band (high / "
        "medium / low / none) based on the top similarity. Use the returned "
        "routes as STRATEGIC ANCHORS — they show what a known good route "
        "for an analog looks like in the format you should be producing. "
        "Do not transcribe reagents verbatim; transfer the disconnection "
        "strategy. Cost: <100ms, free (no GPU, no LLM, not iteration-"
        "billable). Call before expanding non-trivial intermediates "
        "(≥3 rings, ≥2 stereocenters, unfamiliar scaffold)."
    )
    tier = 2
    parameters = {
        "type": "object",
        "properties": {
            "query_smiles": {
                "type": "string",
                "description": (
                    "Target or intermediate SMILES to find precedent for. "
                    "Canonicalised internally before lookup."
                ),
            },
            "k": {
                "type": "integer",
                "default": 3,
                "description": (
                    "Max number of exemplar routes to return. 1-5 valid. "
                    "Default 3. Larger k wastes context for diminishing "
                    "marginal information."
                ),
            },
            "source": {
                "type": "string",
                "enum": ["any", "chemo", "enzyme", "mixed"],
                "default": "mixed",
                "description": (
                    "Source filter. 'any' = top-k by similarity ignoring "
                    "source; 'chemo' = USPTO only; 'enzyme' = RetroBioCat / "
                    "BKMS only; 'mixed' (default) = guarantee at least 1 "
                    "enzyme exemplar when the query carries an enzyme-"
                    "friendly functional group, else fall through to 'any'."
                ),
            },
            "min_similarity": {
                "type": "number",
                "default": 0.2,
                "description": (
                    "Hard floor on Tanimoto similarity. Returns fewer (or "
                    "zero) examples if no candidate clears this. 0.2 is "
                    "the noise floor; bumping to 0.4+ asks for near-analogs."
                ),
            },
        },
        "required": ["query_smiles"],
    }

    def __init__(self, db):  # type: ignore[no-untyped-def]
        """db: a ``services.precedent_db.PrecedentDB`` instance."""
        self._db = db

    def execute(
        self,
        query_smiles: str = "",
        k: int = 3,
        source: str = "mixed",
        min_similarity: float = 0.2,
        **kwargs: Any,
    ) -> str:
        if not query_smiles or not isinstance(query_smiles, str):
            return json.dumps({
                "error": "query_smiles is required and must be a string",
                "examples": [],
            })
        try:
            k_int = int(k)
        except (TypeError, ValueError):
            k_int = 3
        # Cap user-supplied k in line with prompt advisory (1–5).
        if k_int < 1:
            k_int = 1
        if k_int > 5:
            k_int = 5
        try:
            min_sim = float(min_similarity)
        except (TypeError, ValueError):
            min_sim = 0.2

        try:
            result = self._db.query(
                smiles=query_smiles,
                k=k_int,
                source=str(source or "mixed"),
                min_similarity=min_sim,
            )
        except Exception as e:
            logger.exception("recall_similar_routes failed")
            return json.dumps({
                "error": f"query failed: {e}",
                "examples": [],
            })
        return json.dumps(result, ensure_ascii=False, default=str)
