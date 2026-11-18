"""Verifies _atom_economy is promoted to a public name and its semantics
are unchanged. The redesigned ExploreBranchesTool imports this name."""

import pytest

from tools.verifier import score_route as sr_mod


def test_atom_economy_is_public():
    assert hasattr(sr_mod, "atom_economy"), (
        "Expected public `atom_economy` in tools.verifier.score_route"
    )
    assert callable(sr_mod.atom_economy)


def test_atom_economy_value_aspirin():
    """Aspirin (180.16 g/mol) from salicylic acid (138.12) + acetic anhydride (102.09).
    Trost atom economy = 180.16 / (138.12 + 102.09) = 0.7501."""
    val = sr_mod.atom_economy(
        ["OC(=O)c1ccccc1O", "CC(=O)OC(C)=O"],
        "CC(=O)Oc1ccccc1C(=O)O",
    )
    assert 0.74 < val < 0.76, f"expected ~0.75, got {val}"


def test_atom_economy_invalid_smiles_returns_zero():
    assert sr_mod.atom_economy(["XYZ"], "CCO") == 0.0
    assert sr_mod.atom_economy(["CCO"], "XYZ") == 0.0
    assert sr_mod.atom_economy([], "CCO") == 0.0
