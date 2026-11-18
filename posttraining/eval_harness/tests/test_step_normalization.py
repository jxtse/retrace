"""Step-string normalisation contract for the scratchpad / fingerprint path.

Regression for wirev5: score_route accepted structured JSON and stamped
route_fp=ae5b5111…, but every save_best_route was rejected with
"route mismatch" because the student wrote reactants separated by ',' or
' + ', or wrote the step forward (`A + B >> P`). The fingerprint must be a
function of the chemistry, not of the separator spelling.
"""
import hashlib
import json

import pytest

from tools._route_fp import normalize_step, route_fingerprint

P = "CCOC(=O)c1nc(N2CC[C@H](N)[C@H](OC)C2)sc1C"
A = "CCOC(=O)c1nc(Cl)sc1C"
B = "N1CC[C@H](N)[C@H](OC)C1"
B_CANON = "CO[C@@H]1CNCC[C@@H]1N"  # same molecule, different spelling

DOT = f"{P}>>{A}.{B}"


@pytest.mark.parametrize(
    "variant",
    [
        f"{P}>>{A},{B}",
        f"{P}>>{B},{A}",
        f"{P}>>{A}, {B}",
        f"{P}>>{A} + {B}",
        f"{P} >> {A} . {B}",
        f"{P}>>{A}.{B_CANON}",
        f"{P}>>{A},{B_CANON} (reaction: SNAr)",
        f"{A} + {B} >> {P}",            # forward notation
        f"{A}.{B}>>{P} (reaction: SNAr)",
        f"{A},{B}>>{P}",
    ],
)
def test_separator_and_direction_variants_share_fingerprint(variant):
    assert route_fingerprint([variant]) == route_fingerprint([DOT])


def test_normalize_step_returns_retro_dot_form():
    prod, reactants, note = normalize_step(f"{A} + {B} >> {P} (reaction: SNAr)")
    assert prod == P
    assert sorted(reactants) == sorted([A, B])
    assert note == "SNAr"


def test_normalize_step_keeps_single_to_single_direction():
    # 1>>1 is direction-ambiguous; we must not silently flip it.
    prod, reactants, _ = normalize_step("CC(=O)O>>CC(=O)OC")
    assert prod == "CC(=O)O" and reactants == ["CC(=O)OC"]


def test_different_chemistry_still_differs():
    assert route_fingerprint([f"{P}>>{A},{B}"]) != route_fingerprint([f"{P}>>{A}"])


def test_malformed_step_without_arrow_is_skipped_not_crashed():
    fp = route_fingerprint(["no arrow here"])
    assert fp == hashlib.md5(("malformed:" + json.dumps(["no arrow here"], sort_keys=True)).encode()).hexdigest()


def test_structured_json_path_matches_string_path():
    """score_route computes the fp from canonical `product>>r1.r2` strings it
    builds from JSON; a save with commas must hit the same fingerprint."""
    from tools.verifier.score_route import _route_fp_from_steps

    fp_json = _route_fp_from_steps([{"product": P, "reactants": [A, B]}])
    assert route_fingerprint([f"{P}>>{A},{B}"]) == fp_json
    assert route_fingerprint([f"{A} + {B} >> {P}"]) == fp_json
