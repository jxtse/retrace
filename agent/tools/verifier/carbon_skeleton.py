"""Carbon-skeleton conservation check for a single reaction step.

This is deliberately NOT atom/mass conservation. Full mass balance was removed
because organic reactions are routinely written unbalanced, so balancing every
element false-rejects every catalytically-mediated disconnection (see
[[atom-conservation-not-a-feasibility-signal]]). What survives is the one
balance signal that *is* diagnostic: the carbon skeleton.

``carbon_delta`` returns the signed delta ``product C − Σ precursor C``:

  - POSITIVE → the product has MORE carbons than every listed precursor
    combined, i.e. a skeletal carbon appeared from nowhere. At an H2O/NH3-only
    annulation (Knorr, Hantzsch, …) a nonzero positive delta means a reagent
    homolog is wrong or a carbon was fabricated. The feasibility pillar docks
    the step for this (see ``score_route._score_one_feasibility_step``).
  - NEGATIVE → precursors carry more carbon than the product, i.e. a fragment
    (leaving group, protecting group, co-product) departs. Benign and
    informational only — never penalised.
  - ZERO → carbon conserved.
"""

from __future__ import annotations

from typing import List


def _carbon_count(smiles_list: List[str]) -> int:
    """Total carbon atoms across a list of SMILES; unparseable entries count 0."""
    try:
        from rdkit import Chem
    except Exception:
        return 0
    total = 0
    for s in smiles_list:
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        for atom in m.GetAtoms():
            if atom.GetSymbol() == "C":
                total += 1
    return total


def carbon_delta(reactants: List[str], product: str) -> int:
    """Signed carbon-skeleton delta: ``product C − Σ precursor C``.

    Positive → carbon fabricated (penalised); negative → fragment leaves
    (benign); zero → conserved.
    """
    return _carbon_count([product]) - _carbon_count(reactants)


def _ring_atom_count(smiles_list: List[str]) -> int:
    """Total ring-member atoms (SSSR, summed over rings) across a SMILES list.

    Counts each ring's size and sums them, so two fused 6-rings count 12. Used
    only to detect that a step FORMS a new ring (product ring-atom total exceeds
    the precursors' combined total). Unparseable entries — and any SMILES that
    makes RDKit raise (this runs in the scoring hot path on arbitrary
    model-emitted strings) — count 0.
    """
    try:
        from rdkit import Chem
    except Exception:
        return 0
    total = 0
    for s in smiles_list:
        if not s:
            continue
        try:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            ri = m.GetRingInfo()
            for ring in ri.AtomRings():
                total += len(ring)
        except Exception:
            continue
    return total


def forms_new_ring(reactants: List[str], product: str) -> bool:
    """True if the product carries more ring-member atoms than all precursors
    combined — i.e. the step forms a new ring.

    Deterministic and informational only. Pairs with ``carbon_delta`` to flag
    the one negative-delta case that IS diagnostic: a *bimolecular*
    ring-forming step that simultaneously LOSES skeletal carbon with no leaving
    group to carry it (a wrong-product cycloaddition — e.g. a Diels–Alder
    C5+C4 drawn as a C7 adduct). A 100%-atom-economical cycloaddition must
    conserve carbon, so ``forms_new_ring and carbon_delta < 0`` on two
    independent precursors is mechanistically suspect. NOT used to alter the
    score; surfaced as a warning for the LLM judge / human audit, mirroring the
    negative-delta "informational only" stance documented above.

    KNOWN BLIND SPOT (do not "fix" by loosening the metric): this is a
    ring-ATOM-total comparison, so a cycloaddition that fuses two ALREADY-CYCLIC
    precursors into one product without raising the combined ring-atom total is
    NOT flagged. The canonical RETRACE §2 exemplar — cyclopentadiene (cyclic)
    + maleic-anhydride (cyclic) drawn as a carbon-losing adduct — slips through
    for exactly this reason. Every structural ring metric tried (ring-atom sum,
    ring-count sum, "new ring beyond most-cyclic precursor", "product is
    cyclic") tops out at ~47–52% precision against the chemist audit, because
    legitimate bimolecular ring closures (lactamisation, heteroannulation)
    share the signature; broadening recall only worsens precision. That ceiling
    is why this stays an unscored warning and a real fix needs mechanism
    awareness (reaction class + leaving-group identity), not more ring counting.
    See RETRACE_ERROR_ANALYSIS §2.
    """
    return _ring_atom_count([product]) > _ring_atom_count(reactants)
