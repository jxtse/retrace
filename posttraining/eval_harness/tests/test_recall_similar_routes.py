"""Tests for tools/builtins/recall_similar_routes.py.

Surface-level tests on the LLM-facing tool: arg validation, invalid
SMILES handling, adaptive confidence bands at each threshold, enzyme-
friendly heuristic recall, and source="mixed" forces ≥1 enzyme
exemplar when query is enzyme-friendly.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("rdkit")

from services.precedent_db import (  # noqa: E402
    PrecedentDB,
    PrecedentDBConfig,
    is_enzyme_friendly,
    make_morgan_fp,
)
from tools.builtins.recall_similar_routes import RecallSimilarRoutesTool  # noqa: E402


# Re-use the synthetic fixture from test_precedent_db.
from tests.test_precedent_db import SAMPLE_SMILES, _write_fixture  # noqa: E402


@pytest.fixture
def db(tmp_path):
    _write_fixture(str(tmp_path), SAMPLE_SMILES)
    return PrecedentDB(db_root=str(tmp_path), config=PrecedentDBConfig(top_n_for_mmr=15))


@pytest.fixture
def tool(db):
    return RecallSimilarRoutesTool(db)


def test_invalid_smiles_returns_error(tool):
    out = json.loads(tool.execute(query_smiles="not a smiles"))
    # invalid SMILES bubbles up as either error or empty result
    assert out.get("n_examples", 0) == 0
    assert out["confidence"] == "none"


def test_empty_smiles_arg(tool):
    out = json.loads(tool.execute(query_smiles=""))
    assert "error" in out


def test_k_is_capped_at_5(tool):
    out = json.loads(tool.execute(query_smiles="c1ccccc1", k=100, min_similarity=0.0))
    assert out["n_examples"] <= 5


def test_k_is_floored_at_1(tool):
    out = json.loads(tool.execute(query_smiles="c1ccccc1", k=0, min_similarity=0.0))
    # k=0 -> floor to 1 -> at most 1 example
    assert out["n_examples"] <= 1


def test_confidence_bands(db):
    tool = RecallSimilarRoutesTool(db)
    # Query a SMILES that's in the fixture -> max_sim should be 1.0 (high).
    out = json.loads(tool.execute(query_smiles=SAMPLE_SMILES[5], k=3))
    assert out["confidence"] == "high"
    assert out["max_similarity"] >= 0.99

    # Query a chemically distant SMILES -> low / none confidence.
    distant = "C1CCC2(CC1)OCCO2"  # ketal — not represented
    out2 = json.loads(tool.execute(query_smiles=distant, k=3, min_similarity=0.0))
    assert out2["confidence"] in ("low", "medium", "none")


def test_min_similarity_filter(tool):
    out = json.loads(tool.execute(query_smiles="c1ccccc1", k=5, min_similarity=0.99))
    # Even with k=5, only molecules with sim>=0.99 should be returned.
    for ex in out["examples"]:
        assert ex["similarity"] >= 0.99


def test_enzyme_friendly_heuristic_recall():
    # Hand-labelled enzyme-friendly SMILES (carry OH / NH2 / acid / amide / olefin):
    friendly = [
        "OCC", "CCN", "NCCC", "OC(=O)C", "CC(=O)NCC",
        "CCO", "Nc1ccccc1", "c1ccc(O)cc1", "C=CC",
        "OCC(N)C(=O)O", "NCCNCC", "OCC(O)CN",
        "NC(=O)C", "NC(=O)c1ccccc1", "OC(=O)c1ccccc1",
        "C=C(C)C", "CCC=C", "CC(O)C", "CC(N)C",
        "NCC(O)CO",
    ]
    # Hand-labelled enzyme-UN-friendly (none of the watched groups):
    unfriendly = [
        "ClC(Cl)Cl", "FCCF", "BrCCBr", "C1CCCCC1",
        "Cc1ccccc1", "c1ccccc1", "C(C)(C)C",
        "ClCCCl", "BrCCl", "Clc1ccccc1",
        "c1ccc2ccccc2c1", "C", "CC", "CCC",
        "CCCC", "CC(C)CC", "C1CC1", "C1CCC1", "C1CCCC1", "C1=CC=CC=C1",
    ]
    hits = sum(1 for s in friendly if is_enzyme_friendly(s))
    misses = sum(1 for s in unfriendly if is_enzyme_friendly(s))
    # ≥90% recall on the friendly bucket.
    assert hits / len(friendly) >= 0.9
    # ≥80% specificity on the unfriendly bucket (loose — heuristic is intentionally permissive).
    assert (len(unfriendly) - misses) / len(unfriendly) >= 0.8


def test_mixed_forces_enzyme_when_friendly(db):
    tool = RecallSimilarRoutesTool(db)
    # Phenol is enzyme-friendly (has OH).
    out = json.loads(tool.execute(query_smiles="c1ccc(O)cc1", k=3,
                                  source="mixed", min_similarity=0.0))
    assert any(ex["is_enzymatic"] for ex in out["examples"]), out
