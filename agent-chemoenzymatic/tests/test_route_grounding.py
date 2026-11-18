"""Structural route-grounding check (audit #2, first layer).

Pure topology: the route's reaction graph must be a tree rooted at the target.
NO chemistry judgement — only (1) some step produces the target, and (2) every
intermediate product is consumed downstream. Steps are retrosynthetic:
product>>reactants. Fail-open (checked=False, ok=True) on unparseable input.
"""
from rdkit import Chem

from tools.scratchpad import (
    check_route_grounding,
    check_route_grounding_from_steps,
)

canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))


# ── check_route_grounding (product / reactant lists) ─────────────────────────


def test_grounded_linear_route():
    # target CCCCO; step1 produces it from CCCC.OCC; step2 produces CCCC from
    # CCCCBr. Every intermediate (CCCC) is consumed; root == target.
    g = check_route_grounding(
        "CCCCO",
        step_products=["CCCCO", "CCCC"],
        step_reactants=[["CCCC", "OCC"], ["CCCCBr"]],
    )
    assert g["checked"] is True
    assert g["ok"] is True
    assert g["root_matches_target"] is True
    assert g["n_dangling"] == 0


def test_root_not_target_fails():
    # No step produces the target CCCCO.
    g = check_route_grounding(
        "CCCCO",
        step_products=["CCCC"],
        step_reactants=[["CCCCBr"]],
    )
    assert g["checked"] is True
    assert g["ok"] is False
    assert g["root_matches_target"] is False


def test_dangling_intermediate_fails():
    # step2 produces CCCl, which nothing downstream consumes -> disconnected.
    g = check_route_grounding(
        "CCCCO",
        step_products=["CCCCO", "CCCl"],
        step_reactants=[["CCCC", "OCC"], ["CCBr"]],
    )
    assert g["checked"] is True
    assert g["ok"] is False
    assert g["root_matches_target"] is True
    assert g["n_dangling"] == 1
    assert canon("CCCl") in g["dangling"]


def test_unparseable_target_fails_open():
    g = check_route_grounding(
        "not-a-smiles",
        step_products=["CCO"],
        step_reactants=[["CC=O"]],
    )
    assert g["checked"] is False
    assert g["ok"] is True  # fail-open: do not block


def test_unparseable_product_fails_open():
    g = check_route_grounding(
        "CCO",
        step_products=["@@@bad"],
        step_reactants=[["CC=O"]],
    )
    assert g["checked"] is False
    assert g["ok"] is True


# ── check_route_grounding_from_steps (string steps) ──────────────────────────


def test_from_steps_grounded():
    g = check_route_grounding_from_steps(
        "CCCCO", ["CCCCO>>CCCC.OCC", "CCCC>>CCCCBr"]
    )
    assert g["checked"] is True
    assert g["ok"] is True


def test_from_steps_strips_reaction_annotation():
    g = check_route_grounding_from_steps(
        "CCO", ["CCO>>CC=O (reaction: reduction)"]
    )
    assert g["checked"] is True
    assert g["ok"] is True
    assert g["root_matches_target"] is True


def test_from_steps_malformed_fails_open():
    g = check_route_grounding_from_steps("CCO", ["this is not a step"])
    assert g["checked"] is False
    assert g["ok"] is True
