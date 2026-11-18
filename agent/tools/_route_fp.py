"""Shared route-fingerprint helpers.

Extracted from ``tools/scratchpad.py`` so both the scratchpad dedup logic
and the ``score_route`` verifier can compute the *same* structural id for a
route. score_route stamps this fingerprint into its result payload so that
``save_best_route`` can prove the cited ``score_route_result_id`` actually
scored the route being saved — not some other higher-scoring route whose id
was reused (the score-gaming gap closed in May 2026).

No heavy dependencies: RDKit is imported lazily and falls back gracefully.
"""

from __future__ import annotations

import hashlib
import json
from typing import List


def canonicalize_smiles(smiles: str) -> str:
    """Best-effort canonicalisation. Falls back to the raw string if
    RDKit can't parse it."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol)
    except Exception:
        return smiles


def route_fingerprint(steps: List[str]) -> str:
    """Order-insensitive, canonical-SMILES-aware structural id.

    - Each step is normalised to ``canon(product)>>canon(R1).canon(R2)…``
      with reactants sorted alphabetically.
    - Steps themselves are sorted before hashing so the fingerprint is
      independent of the order steps were saved.
    - Trailing ``(reaction: <name>)`` annotations are stripped (the
      leading whitespace is required so SMILES like
      ``c1ccccc1B(O)O`` are not truncated).

    Strings that can't be parsed (no ``>>``) are skipped — the
    fingerprint only reflects the chemistry, not free-text labels.

    The annotation strip anchors on a leading space ``" ("`` so legitimate
    SMILES parens (e.g. ``c1ccccc1B(O)O``) are preserved.
    """
    norm: list[str] = []
    for step in steps or []:
        if ">>" not in step:
            continue
        prod, _, react = step.partition(">>")
        # Strip a trailing " (reaction: …)" annotation only — splitting
        # on the bare "(" would chop SMILES like "c1ccccc1B(O)O".
        react = react.split(" (", 1)[0]
        prod_c = canonicalize_smiles(prod.strip())
        rs_c = sorted(
            canonicalize_smiles(r.strip())
            for r in react.split(".") if r.strip()
        )
        norm.append(f"{prod_c}>>{'.'.join(rs_c)}")
    if not norm:
        # Distinct malformed inputs would otherwise collapse onto md5("")
        # and dedup against each other (the first malformed save wins;
        # every later malformed save returns ``status="duplicate"`` even
        # though their step lists differ). Fall back to hashing the raw
        # JSON so each malformed input keeps a unique fingerprint.
        return hashlib.md5(
            ("malformed:" + json.dumps(list(steps or []), sort_keys=True, default=str)).encode()
        ).hexdigest()
    return hashlib.md5("|".join(sorted(norm)).encode()).hexdigest()
