"""Tests for scripts/build_precedent_index.py.

The full builder hits Figshare / clones RetroBioCat — out of scope for
unit tests. Here we hammer the pieces that *don't* need network: the
AiZynthFinder tree -> route flattener, the FP pack/unpack roundtrip,
and the routes.jsonl + npy + npz writer.
"""

from __future__ import annotations

import importlib
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("rdkit")

# Import the script as a module. The script lives outside the package
# tree but we put the repo root on sys.path in conftest.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
build_precedent_index = importlib.import_module("build_precedent_index")

from services.precedent_db import (  # noqa: E402
    PACKED_BYTES_PER_FP,
    bulk_tanimoto,
    make_morgan_fp,
)


# ---- Synthetic AiZynthFinder route trees -----------------------------------
#
# Mirrors the schema we see in `aizynthfinder` output: mol -> reaction -> mol.

def _toy_tree() -> dict:
    """A 2-step toy: target = ester; one step esterification, one step formylation."""
    return {
        "type": "mol",
        "smiles": "CC(=O)OCC",
        "children": [{
            "type": "reaction",
            "metadata": {"classification": "esterification"},
            "children": [
                {
                    "type": "mol",
                    "smiles": "CC(=O)O",
                    "children": [{
                        "type": "reaction",
                        "metadata": {"classification": "oxidation"},
                        "children": [
                            {"type": "mol", "smiles": "CC=O", "in_stock": True},
                            {"type": "mol", "smiles": "O", "in_stock": True},
                        ],
                    }],
                },
                {"type": "mol", "smiles": "CCO", "in_stock": True},
            ],
        }],
    }


def test_tree_to_route_flattens_correctly():
    route = build_precedent_index._tree_to_route(_toy_tree(), route_id="test_0001")
    assert route is not None
    assert route["target_smiles"] == "CCOC(C)=O" or route["target_smiles"] == "CC(=O)OCC"
    assert route["n_steps"] == 2
    # Each step has product + reactants populated.
    for st in route["steps"]:
        assert st["product"]
        assert st["reactants"]
    classes = set(route["rxn_classes"])
    assert {"esterification", "oxidation"}.issubset(classes)
    assert route["is_enzymatic"] is False


def test_tree_to_route_skips_invalid():
    # Tree with no reactions -> None.
    leaf_only = {"type": "mol", "smiles": "CCO", "children": []}
    assert build_precedent_index._tree_to_route(leaf_only, "x") is None
    # Empty tree -> None.
    assert build_precedent_index._tree_to_route({}, "x") is None
    assert build_precedent_index._tree_to_route(None, "x") is None


def test_fp_pack_unpack_roundtrip():
    smiles_set = ["CCO", "c1ccccc1", "OC(=O)c1ccccc1", "NCCN"]
    fps = [make_morgan_fp(s) for s in smiles_set]
    arr = np.stack(fps, axis=0)
    # Save -> load -> Tanimoto remains stable.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "fps.npy")
        np.save(path, arr)
        loaded = np.load(path)
    assert loaded.shape == (len(smiles_set), PACKED_BYTES_PER_FP)
    assert loaded.dtype == np.uint8
    # Cross-check Tanimoto query against itself = 1.0.
    sims = bulk_tanimoto(loaded[0], loaded)
    assert abs(sims[0] - 1.0) < 1e-5


def _seed_fake_retrobiocat_xlsx(cache_dir):
    """Seed cache_dir with a fake cloned RetroBioCat repo containing
    a minimal test_pathways.xlsx in the expected location."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    xlsx_dir = cache_dir / "RetroBioCat" / "scripts" / "pathway_testing"
    xlsx_dir.mkdir(parents=True)
    df = pd.DataFrame([
        {
            "Num": 1,
            "Target Product": "CC(N)C(=O)O",
            "Substrates": "CC(=O)C(=O)O",
            "Enzymes": "transaminase",
            "DOI": "10.1000/foo",
        },
        {
            "Num": 2,
            "Target Product": "CCC(N)C",
            "Substrates": "CCC(=O)C, NH3",
            "Enzymes": "transaminase, reductase",
            "DOI": "",
        },
    ])
    df.to_excel(xlsx_dir / "test_pathways.xlsx", index=False)


def test_retrobiocat_loader_parses_xlsx(tmp_path, monkeypatch):
    """Skip the git clone by pre-seeding cache_dir with a fake xlsx."""
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    cache_dir = tmp_path / "cache"
    _seed_fake_retrobiocat_xlsx(cache_dir)
    routes = list(build_precedent_index.iter_retrobiocat_routes(str(cache_dir), limit=None))
    assert len(routes) == 2
    for r in routes:
        assert r["source"] == "retrobiocat"
        assert r["is_enzymatic"] is True
    # First row: 1 enzyme -> n_steps == 1; second row: 2 enzymes -> n_steps == 2
    assert routes[0]["n_steps"] == 1
    assert routes[1]["n_steps"] == 2
    assert "transaminase" in routes[0]["rxn_classes"]
    assert routes[0]["doi"] == "10.1000/foo"


def test_build_index_writes_all_four_files(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    _seed_fake_retrobiocat_xlsx(cache_dir)
    stats = build_precedent_index.build_index(
        sources=["retrobiocat"],
        out_dir=str(out_dir),
        cache_dir=str(cache_dir),
        limit=None,
    )
    assert stats["n_kept"] >= 1
    assert os.path.exists(out_dir / "routes.jsonl")
    assert os.path.exists(out_dir / "target_fps.npy")
    assert os.path.exists(out_dir / "target_index.json")
    assert os.path.exists(out_dir / "source_masks.npz")
    fps = np.load(out_dir / "target_fps.npy")
    assert fps.shape == (stats["n_kept"], PACKED_BYTES_PER_FP)
    masks = np.load(out_dir / "source_masks.npz")
    assert masks["enzyme"].sum() == stats["n_kept"]  # all retrobiocat -> enzyme
    assert masks["uspto"].sum() == 0
