"""Tests for services/precedent_db.py.

Validates bulk Tanimoto vs RDKit's DataStructs, MMR diversity, source
mask filtering, and the LRU cache hit path. Uses a 50-route synthetic
fixture written by the test itself — no external dependency.
"""

from __future__ import annotations

import json
import os
import random
import tempfile

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem, DataStructs  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from services.precedent_db import (  # noqa: E402
    MORGAN_NBITS,
    PACKED_BYTES_PER_FP,
    PrecedentDB,
    PrecedentDBConfig,
    bulk_tanimoto,
    make_morgan_fp,
    mmr_select,
)


SAMPLE_SMILES = [
    "CCO", "CCN", "CCC", "CCCC", "CCCCC",
    "c1ccccc1", "c1ccccc1O", "c1ccccc1N", "c1ccc(C)cc1", "Cc1ccc(C)cc1",
    "CC(=O)O", "CC(=O)N", "CC(=O)OC", "CC(=O)OCC", "CC(=O)NCC",
    "CCOC(=O)C", "CCOC(=O)CC", "CCOC(=O)CCC", "CCN(CC)CC", "CCN(C)C",
    "OC(=O)c1ccccc1", "OC(=O)c1ccc(O)cc1", "OC(=O)c1ccccn1", "OC(=O)c1ccncc1",
    "NC(=O)c1ccccc1", "NC(=O)c1ccc(N)cc1", "Nc1ccccc1", "Nc1ccc(Cl)cc1",
    "CC(C)(C)O", "CC(C)(C)N", "CC(C)CO", "CC(C)CN",
    "Cn1cnc2c1nc(N)nc2N",  # adenine-ish
    "Cc1cc(C)nc(N)n1",
    "ClCCl", "ClC(Cl)Cl", "FCCF", "BrCCBr",
    "C1CCCCC1", "C1CCCCC1O", "C1CCCCC1N",
    "C1=CC=CC=C1", "C1=CC2=CC=CC=C2C=C1",  # naphthalene
    "OCC(O)CO", "OCC(N)CO",
    "S(=O)(=O)(O)c1ccccc1",
    "C(=O)(O)c1ccc2ccccc2c1",
    "CCCCCC(=O)O", "CCCCCCO",
]


def _write_fixture(tmpdir: str, smiles: list[str]) -> str:
    routes = []
    fps = []
    uspto_mask = []
    enzyme_mask = []
    for i, s in enumerate(smiles):
        fp = make_morgan_fp(s)
        if fp is None:
            continue
        fps.append(fp)
        # Alternate source attribution.
        is_enzyme = (i % 5 == 0)
        is_uspto = not is_enzyme
        uspto_mask.append(is_uspto)
        enzyme_mask.append(is_enzyme)
        routes.append({
            "route_id": f"fix_{i:04d}",
            "source": "retrobiocat" if is_enzyme else "uspto",
            "target_smiles": s,
            "n_steps": 1,
            "steps": [{"product": s, "reactants": ["CC", "CN"],
                       "rxn_class": "enzyme_x" if is_enzyme else "chemo_y"}],
            "rxn_classes": ["enzyme_x"] if is_enzyme else ["chemo_y"],
            "is_enzymatic": is_enzyme,
        })
    fps_arr = np.stack(fps, axis=0).astype(np.uint8)
    np.save(os.path.join(tmpdir, "target_fps.npy"), fps_arr)
    with open(os.path.join(tmpdir, "routes.jsonl"), "w") as f:
        for r in routes:
            f.write(json.dumps(r) + "\n")
    np.savez(
        os.path.join(tmpdir, "source_masks.npz"),
        uspto=np.array(uspto_mask, dtype=bool),
        enzyme=np.array(enzyme_mask, dtype=bool),
    )
    return tmpdir


@pytest.fixture
def db_dir(tmp_path):
    return _write_fixture(str(tmp_path), SAMPLE_SMILES)


@pytest.fixture
def db(db_dir):
    return PrecedentDB(db_root=db_dir, config=PrecedentDBConfig(top_n_for_mmr=10))


def test_bulk_tanimoto_matches_rdkit():
    rng = random.Random(0xC0FFEE)
    a_smiles = rng.sample(SAMPLE_SMILES, 10)
    b_smiles = rng.sample(SAMPLE_SMILES, 10)
    pairs = list(zip(a_smiles * 10, b_smiles * 10))[:100]
    db_fps = np.stack([make_morgan_fp(b) for b in b_smiles], axis=0)
    for a, b in pairs:
        a_fp_packed = make_morgan_fp(a)
        sims = bulk_tanimoto(a_fp_packed, db_fps)
        # Reference: RDKit ExplicitBitVect
        a_bv = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(a), 2, nBits=MORGAN_NBITS)
        for j, bs in enumerate(b_smiles):
            b_bv = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(bs), 2, nBits=MORGAN_NBITS)
            ref = DataStructs.TanimotoSimilarity(a_bv, b_bv)
            assert abs(sims[j] - ref) < 1e-5, (a, bs, sims[j], ref)


def test_mmr_lowers_intra_set_similarity(db):
    # Query something benzene-ish; expect several aromatic neighbors.
    q = "c1ccc(O)cc1"
    q_fp = make_morgan_fp(q)
    all_sims = bulk_tanimoto(q_fp, db._fps)
    top_k = np.argsort(-all_sims)[:5]
    raw_pair_sims = []
    for i in top_k:
        for j in top_k:
            if i != j:
                raw_pair_sims.append(
                    float(bulk_tanimoto(db._fps[i], db._fps[j:j + 1])[0])
                )
    raw_mean = float(np.mean(raw_pair_sims))

    mmr_idxs = mmr_select(
        candidate_idxs=np.arange(db.n_rows),
        candidate_sims=all_sims,
        db_fps_packed=db._fps,
        k=5,
        lambda_=0.5,
    )
    mmr_pair_sims = []
    for i in mmr_idxs:
        for j in mmr_idxs:
            if i != j:
                mmr_pair_sims.append(
                    float(bulk_tanimoto(db._fps[i], db._fps[j:j + 1])[0])
                )
    mmr_mean = float(np.mean(mmr_pair_sims)) if mmr_pair_sims else 0.0
    assert mmr_mean < raw_mean, f"MMR did not increase diversity: {mmr_mean=} {raw_mean=}"


def test_source_mask_filtering(db):
    res_enzyme = db.query("CCO", k=3, source="enzyme", min_similarity=0.0)
    for ex in res_enzyme["examples"]:
        assert ex["source_db"] in ("retrobiocat", "bkms"), ex
        assert ex["is_enzymatic"] is True

    res_chemo = db.query("CCO", k=3, source="chemo", min_similarity=0.0)
    for ex in res_chemo["examples"]:
        assert ex["source_db"] == "uspto", ex
        assert ex["is_enzymatic"] is False


def test_lru_cache_hit(db):
    a = db.query("CCO", k=2, source="any", min_similarity=0.0)
    assert "_cache_hit" not in a
    b = db.query("CCO", k=2, source="any", min_similarity=0.0)
    assert b.get("_cache_hit") is True


def test_invalid_smiles_returns_empty(db):
    res = db.query("not a smiles!!!", k=3)
    assert res["n_examples"] == 0
    assert res["confidence"] == "none"
    assert "error" in res


def test_packed_fp_geometry():
    fp = make_morgan_fp("c1ccccc1")
    assert fp is not None
    assert fp.dtype == np.uint8
    assert fp.shape == (PACKED_BYTES_PER_FP,)


def test_bulk_tanimoto_accepts_precomputed_popcounts():
    """Pre-computed db_popcounts must produce identical results to the
    auto-computed path (the popcount cache is what PrecedentDB uses for
    every query — if these diverge silently every score is wrong)."""
    smiles = SAMPLE_SMILES[:15]
    db_fps = np.stack([make_morgan_fp(s) for s in smiles], axis=0)
    q_fp = make_morgan_fp("c1ccccc1O")
    sims_auto = bulk_tanimoto(q_fp, db_fps)
    # Same path PrecedentDB takes on every query.
    from services.precedent_db import _byte_popcount
    db_pop = _byte_popcount(db_fps).sum(axis=1).astype(np.uint16)
    sims_cached = bulk_tanimoto(q_fp, db_fps, db_popcounts=db_pop)
    assert np.allclose(sims_auto, sims_cached, atol=1e-6), (sims_auto, sims_cached)


def test_recall_examples_carry_full_route(db):
    """The tool result's per-example ``route`` must be the FULL route
    dict (with all step fields + source-specific extras), not just the
    minimal projector view. The projector in result_store strips for
    the in-context stub; ``load_result(rid)`` must be able to recover
    everything from the persisted full payload.
    """
    res = db.query("c1ccc(O)cc1", k=3, source="any", min_similarity=0.0)
    assert res["examples"], res
    ex = res["examples"][0]
    # Top-level metadata still present
    assert ex["route_id"]
    assert ex["similarity"] > 0
    # FULL route shape: a dict with route_id + steps array, not a list
    # of stripped step dicts.
    full_route = ex["route"]
    assert isinstance(full_route, dict), type(full_route)
    assert full_route.get("route_id") == ex["route_id"]
    assert isinstance(full_route.get("steps"), list)
    # The synthetic fixture's steps carry rxn_class — that field must
    # survive untouched (caught the original bug where steps_min only
    # kept product/reactants/rxn_class but stripped reaction_smarts).
    for st in full_route["steps"]:
        assert "product" in st
        assert "reactants" in st
        # rxn_class is part of the on-disk schema; must round-trip.
        assert "rxn_class" in st


def test_precedent_db_caches_popcounts_at_load(db):
    """PrecedentDB must materialise the popcount cache once at load so
    every query path skips per-call popcount work."""
    assert hasattr(db, "_popcounts")
    assert db._popcounts.shape == (db.n_rows,)
    assert db._popcounts.dtype == np.uint16
    # Sanity: popcount matches what bulk_tanimoto would re-derive.
    from services.precedent_db import _byte_popcount
    expected = _byte_popcount(np.asarray(db._fps)).sum(axis=1).astype(np.uint16)
    assert np.array_equal(db._popcounts, expected)
