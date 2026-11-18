"""Soft stereo warning per step.

If the product carries any stereo descriptors (chiral centers or double-bond
geometry) that are NOT present on any reactant, emit a warning string —
without failing the step. The verifier surfaces this in feasibility.per_step
so the LLM can decide whether to add a chirality-controlling reagent.

History (2026-05-18): the previous implementation hashed chiral centers as
``(atom_idx, ChiralTag)`` and intersected reactant ∪ product sets. Atom
indices are per-molecule, so the intersection was *always* empty — every
step that kept stereocenters emitted a false-positive warning. Replaced
with R/S-count signatures (and analogous E/Z counts): warn only when the
product introduces strictly more of a category than the reactant pool
supplies. Trade-off: a reaction that inverts one center while creating a
new same-handed one elsewhere produces equal signatures and won't warn —
much less common than the prior false-positive flood.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def _stereo_signature(smi: str) -> Dict[str, int]:
    """Return ``{"R", "S", "E", "Z"}`` counts for the molecule.

    Keys off CIP descriptors so the same physical center is comparable
    across reactant/product mols (atom indices are not). Unassigned
    centers are NOT counted — they aren't a stereo claim by the author.
    """
    sig = {"R": 0, "S": 0, "E": 0, "Z": 0}
    try:
        from rdkit import Chem
    except Exception:
        return sig
    m = Chem.MolFromSmiles(smi or "")
    if m is None:
        return sig
    try:
        Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    except Exception:
        pass
    try:
        for _, code in Chem.FindMolChiralCenters(
            m, includeUnassigned=False, useLegacyImplementation=False
        ):
            if code in ("R", "S"):
                sig[code] += 1
    except Exception:
        pass
    for b in m.GetBonds():
        st = b.GetStereo()
        if st in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS):
            sig["E"] += 1
        elif st in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS):
            sig["Z"] += 1
    return sig


def warning_for_step(reactants: List[str], product: str) -> Optional[str]:
    """Emit a soft warning string if product introduces stereo not present in reactants.

    Returns ``None`` when the step has no new stereochemistry concerns.
    """
    p_sig = _stereo_signature(product)
    if sum(p_sig.values()) == 0:
        return None
    r_sum = {"R": 0, "S": 0, "E": 0, "Z": 0}
    for r in reactants:
        rs = _stereo_signature(r)
        for k in r_sum:
            r_sum[k] += rs[k]
    new = {k: max(0, p_sig[k] - r_sum[k]) for k in p_sig}
    new_chi = new["R"] + new["S"]
    new_bonds = new["E"] + new["Z"]
    if new_chi <= 0 and new_bonds <= 0:
        return None
    parts = []
    if new_chi > 0:
        parts.append(f"{new_chi} new chiral center(s)")
    if new_bonds > 0:
        parts.append(f"{new_bonds} new E/Z bond(s)")
    return ("Stereochemistry introduced in product without matching descriptor "
            "on reactants: " + ", ".join(parts) +
            ". Verify the reaction provides the correct stereo control "
            "(catalyst, chiral auxiliary, kinetic/thermodynamic resolution).")
