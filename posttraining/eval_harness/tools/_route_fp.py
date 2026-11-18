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
import re
from typing import List, Optional, Tuple

# Component separators tolerated on either side of ``>>``. The canonical
# SMILES mixture separator is ``.``; LLM students also write ``,`` and
# ``A + B``. ``+`` is only treated as a separator when surrounded by
# whitespace so charged atoms such as ``[NH3+]`` / ``[Na+]`` are untouched.
_SEP_RE = re.compile(r"\s*[.,]\s*|\s+\+\s+")


def split_components(side: str) -> List[str]:
    """Split one side of a reaction string into molecule tokens."""
    return [t for t in (p.strip() for p in _SEP_RE.split(side or "")) if t]


def normalize_step(step: str) -> Optional[Tuple[str, List[str], str]]:
    """Parse a free-form step string into ``(product, reactants, note)``.

    Accepts the documented retro form ``product>>r1.r2 (reaction: name)`` and
    the observed student variants: comma / `` + `` separators and the forward
    form ``r1 + r2 >> product``. A step is treated as forward only when the
    left side has several components and the right side exactly one, so a
    genuine single-to-single retro step is never flipped. Returns ``None``
    when the string has no ``>>``.

    Only the spelling is normalised; the chemistry (which molecules sit on
    which side) is what the fingerprint, validators and leaf derivation see.
    """
    if not isinstance(step, str) or ">>" not in step:
        return None
    left, _, right = step.partition(">>")
    note = ""
    if " (" in right:
        right, _, tail = right.partition(" (")
        tail = tail.rstrip()
        if tail.endswith(")"):
            tail = tail[:-1]
        if tail.lower().startswith("reaction:"):
            tail = tail[len("reaction:"):]
        note = tail.strip()
    lhs = split_components(left)
    rhs = split_components(right)
    if len(lhs) > 1 and len(rhs) == 1:
        # forward notation ``A + B >> P`` → retro ``P>>A.B``
        return rhs[0], lhs, note
    product = lhs[0] if lhs else ""
    return product, rhs, note


def render_step(product: str, reactants: List[str], note: str = "") -> str:
    """Inverse of :func:`normalize_step` in the canonical dot-separated form."""
    body = f"{product}>>{'.'.join(reactants)}"
    return f"{body} (reaction: {note})" if note else body


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
        parsed = normalize_step(step)
        if parsed is None:
            continue
        prod, reactants, _ = parsed
        prod_c = canonicalize_smiles(prod.strip())
        rs_c = sorted(canonicalize_smiles(r) for r in reactants)
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
