"""Structural leaf derivation for the buyability gate.

derive_leaf_smiles must return the reactants that are never produced by any
step (the true raw starting materials), independent of the LLM-supplied
starting_materials field. Steps are retrosynthetic: product>>reactants.
"""
import pytest

from tools.scratchpad import derive_leaf_smiles


def test_linear_route_leaves():
    # C bottoms out on A and B; intermediate I is produced then consumed.
    # step1:  P  >> I . A        (I, A are reactants; P is product)
    # step2:  I  >> B            (B is reactant; I is product -> not a leaf)
    steps = ["CCCCO>>CCCC.OCC", "CCCC>>CCCCBr"]
    leaves = derive_leaf_smiles(steps)
    # products: {CCCCO, CCCC}; reactants: {CCCC, OCC, CCCCBr}
    # leaves = reactants - products = {OCC, CCCCBr}
    from rdkit import Chem
    canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
    assert set(leaves) == {canon("OCC"), canon("CCCCBr")}


def test_reactant_also_product_is_not_leaf():
    # An intermediate that appears as a reactant in one step and product in
    # another is NOT a leaf.
    steps = ["CCO>>CC=O", "CC=O>>CC(=O)Cl"]
    leaves = derive_leaf_smiles(steps)
    from rdkit import Chem
    canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
    # products {CCO, CC=O}; reactants {CC=O, CC(=O)Cl}; leaves {CC(=O)Cl}
    assert set(leaves) == {canon("CC(=O)Cl")}


def test_reaction_annotation_stripped():
    steps = ["CCO>>CC=O (reaction: reduction)"]
    leaves = derive_leaf_smiles(steps)
    from rdkit import Chem
    canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
    assert set(leaves) == {canon("CC=O")}


def test_unparseable_step_returns_none():
    # placeholder text -> cannot derive -> None (caller falls back)
    assert derive_leaf_smiles(["target>>intermediate.CCO"]) is None


def test_missing_arrow_returns_none():
    assert derive_leaf_smiles(["this is not a step"]) is None


def test_empty_returns_empty_list():
    assert derive_leaf_smiles([]) == []
