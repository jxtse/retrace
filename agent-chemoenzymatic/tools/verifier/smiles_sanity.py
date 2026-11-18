"""Per-molecule SMILES sanitization gate for a single reaction step.

A neural one-step engine (especially the rxngraphormer transformer) occasionally
emits a SMILES that RDKit cannot sanitize — pentavalent carbon, broken
aromaticity, an unclosed ring. Such a string is not a borderline-feasible
disconnection; it is a *malformed route*. Left unchecked it would be handed to
the reaction_rater (which scores the garbage with some arbitrary confidence) and
to atom_economy / SCScore (which silently treat the unparseable mol as 0),
quietly corrupting the feasibility and cost pillars.

This module is the cheap structural gate that catches that case BEFORE the rater
call. The distinction the caller relies on:

  - unparseable SMILES  → the ROUTE is broken → score the step 0, skip the rater.
  - rater endpoint down  → INFRASTRUCTURE is broken → raise (handled in the
    caller, not here).

so a bad disconnection scores 0 while a dead rater fails loudly — the two must
never be conflated.

``Chem.MolFromSmiles`` runs full sanitization by default (valence, aromaticity,
kekulization), so a non-None return is a sufficient validity signal; we do not
need a separate ``SanitizeMol`` pass.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


def first_invalid_smiles(smiles_list: List[str]) -> Optional[str]:
    """Return the first SMILES RDKit cannot parse/sanitize, or None if all valid.

    An empty/whitespace-only entry counts as invalid (a step with a blank
    reactant or product is malformed). If RDKit itself is unavailable the gate
    fails OPEN (returns None) — we never want a missing optional dependency to
    zero out every step.
    """
    try:
        from rdkit import Chem
    except Exception:
        return None
    for s in smiles_list:
        if not s or not s.strip():
            return s if s is not None else ""
        if Chem.MolFromSmiles(s) is None:
            return s
    return None


def check_step_smiles(reactants: List[str], product: str) -> Tuple[bool, Optional[str]]:
    """Validate every SMILES in one step.

    Returns ``(ok, bad_smiles)``: ``ok`` is True when every reactant and the
    product parse; otherwise ``ok`` is False and ``bad_smiles`` is the first
    offending string (for the warning message).
    """
    bad = first_invalid_smiles([*reactants, product])
    if bad is None:
        return True, None
    return False, bad
