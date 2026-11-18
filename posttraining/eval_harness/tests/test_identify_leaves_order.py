"""score_route._identify_leaves must be order-independent.

lean2 (2026-09-16): the same 3-step route (identical route fingerprint
ae5b5113...) scored 0.6000 / route_complete=false when written target-first
(retro order) and 0.7455 / route_complete=true after the student reordered the
steps. Root cause: the single-pass leaf scan only excluded products of EARLIER
steps, so intermediates consumed before the step that makes them were counted
as starting materials, failing the buyability gate. Leaves are a property of
the route graph, not of the step listing order.
"""
import itertools

from tools.verifier.score_route import _identify_leaves

T = "CCOC(=O)c1nc(N2CC[C@H](NC(=O)c3nc(C(F)(F)F)c(CC)[nH]3)[C@H](OC)C2)sc1C"
I1 = "CCOC(=O)c1nc(N2CC[C@H](N)[C@H](OC)C2)sc1C"      # intermediate amine
I2 = "CCc1[nH]c(C(=O)O)nc1C(F)(F)F"                    # intermediate acid
A = "CCOC(=O)c1nc(Cl)sc1C"
B = "CO[C@@H]1CNCC[C@@H]1N"
C = "CC=O"
D = "O=C(O)c1nc(C(F)(F)F)c[nH]1"

STEPS = [
    {"product": T, "reactants": [I1, I2]},   # final coupling, listed FIRST (retro order)
    {"product": I1, "reactants": [A, B]},
    {"product": I2, "reactants": [C, D]},
]
TRUE_LEAVES = sorted([A, B, C, D])


def test_retro_order_does_not_count_intermediates_as_leaves():
    leaves = _identify_leaves(STEPS)
    assert sorted(leaves) == TRUE_LEAVES
    assert I1 not in leaves and I2 not in leaves


def test_leaves_are_invariant_to_step_permutation():
    fps = {tuple(sorted(_identify_leaves(list(p)))) for p in itertools.permutations(STEPS)}
    assert fps == {tuple(TRUE_LEAVES)}


def test_forward_order_unchanged():
    fwd = [STEPS[1], STEPS[2], STEPS[0]]
    assert sorted(_identify_leaves(fwd)) == TRUE_LEAVES


def test_canonical_spelling_variants_are_merged():
    # non-canonical spelling of B in one step, canonical in another → one leaf
    steps = [
        {"product": T, "reactants": [I1, I2]},
        {"product": I1, "reactants": [A, "N1CC[C@H](N)[C@H](OC)C1"]},
        {"product": I2, "reactants": [C, D]},
    ]
    assert sorted(_identify_leaves(steps)) == TRUE_LEAVES


def test_first_seen_order_preserved_for_reporting():
    leaves = _identify_leaves([STEPS[1], STEPS[2], STEPS[0]])
    assert leaves == [A, B, C, D]
