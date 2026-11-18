"""Curated hazardous-reagent SMARTS table.

Each entry is ``{name, smarts, severity}`` where severity is one of
``low | moderate | high``. Severity → score penalty applied per match:
  low      -> -0.10
  moderate -> -0.20
  high     -> -0.40

Greenness reagent score = max(0.0, 1.0 + sum(penalties_for_unique_matched_classes)).

HCN gets exact-canonical-SMILES handling (`C#N` / `[C-]#N`) so it does not collide
with acetonitrile (`CC#N`). All other entries use SMARTS fragment match.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# (name, smarts, severity, penalty)
HAZARDOUS_REAGENTS: List[Dict[str, Any]] = [
    # High-severity carcinogens / acute toxics
    {"name": "phosgene",            "smarts": "ClC(=O)Cl",      "severity": "high",     "penalty": -0.40},
    {"name": "diphosgene",          "smarts": "ClC(=O)OC(Cl)Cl", "severity": "high",    "penalty": -0.40},
    {"name": "triphosgene",         "smarts": "ClC(Cl)(Cl)OC(=O)OC(Cl)(Cl)Cl", "severity": "high", "penalty": -0.40},
    {"name": "dimethyl sulfate",    "smarts": "COS(=O)(=O)OC",  "severity": "high",     "penalty": -0.40},
    {"name": "diethyl sulfate",     "smarts": "CCOS(=O)(=O)OCC","severity": "high",     "penalty": -0.40},
    {"name": "methyl iodide",       "smarts": "[CH3]I",         "severity": "high",     "penalty": -0.40},
    {"name": "diazomethane",        "smarts": "[CH2]=[N+]=[N-]","severity": "high",     "penalty": -0.40},
    {"name": "osmium tetroxide",    "smarts": "O=[Os](=O)(=O)=O","severity": "high",    "penalty": -0.40},
    {"name": "mercury(II)",         "smarts": "[Hg+2]",         "severity": "high",     "penalty": -0.40},
    {"name": "thallium",            "smarts": "[Tl]",           "severity": "high",     "penalty": -0.40},
    {"name": "azide salt",          "smarts": "[N-]=[N+]=[N-]", "severity": "high",     "penalty": -0.40},
    # Moderate-severity reactives / toxics
    {"name": "thionyl chloride",    "smarts": "ClS(Cl)=O",      "severity": "moderate", "penalty": -0.20},
    {"name": "oxalyl chloride",     "smarts": "ClC(=O)C(=O)Cl", "severity": "moderate", "penalty": -0.20},
    {"name": "phosphorus oxychloride","smarts": "ClP(Cl)(Cl)=O","severity": "moderate", "penalty": -0.20},
    {"name": "tributyltin hydride", "smarts": "[Sn]([H])(CCCC)(CCCC)CCCC","severity":"moderate","penalty":-0.20},
    {"name": "cyanide salt",        "smarts": "[C-]#N",         "severity": "moderate", "penalty": -0.20},
    {"name": "chromium(VI)",        "smarts": "[Cr+6]",         "severity": "moderate", "penalty": -0.20},
    {"name": "selenium dioxide",    "smarts": "O=[Se]=O",       "severity": "moderate", "penalty": -0.20},
    {"name": "lead",                "smarts": "[Pb]",           "severity": "moderate", "penalty": -0.20},
    {"name": "boron tribromide",    "smarts": "BrB(Br)Br",      "severity": "moderate", "penalty": -0.20},
    # Low-severity (use-with-care)
    {"name": "lithium aluminium hydride", "smarts": "[Li+].[Al-H4]", "severity": "low", "penalty": -0.10},
    {"name": "sodium hydride",      "smarts": "[Na+].[H-]",     "severity": "low",      "penalty": -0.10},
    {"name": "n-butyllithium",      "smarts": "[Li]CCCC",       "severity": "low",      "penalty": -0.10},
    {"name": "trifluoroacetic acid","smarts": "OC(=O)C(F)(F)F", "severity": "low",      "penalty": -0.10},
]

# Reagents identified by exact canonical SMILES rather than substructure.
# HCN must not match acetonitrile (CC#N) so we can't use a substructure SMARTS.
EXACT_HAZARDOUS_REAGENTS: List[Tuple[str, List[str], str, float]] = [
    # (name, list of canonical SMILES, severity, penalty)
    ("hydrogen cyanide", ["C#N", "[C-]#N", "[CH]#N", "N#C"], "high", -0.40),
]


def _canonicalize(s: str) -> str:
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(s)
        if m is None:
            return s
        return Chem.MolToSmiles(m)
    except Exception:
        return s


def scan_reagent(reagent_smiles: str) -> List[Dict[str, Any]]:
    """Return list of {name, severity, penalty} matches for a single reagent SMILES.

    De-duplicated by name so a reagent that triggers two equivalent classes only
    counts once.
    """
    matches: Dict[str, Dict[str, Any]] = {}
    if not reagent_smiles or reagent_smiles.lower() in {"none", "unknown", "", "n/a", "nan"}:
        return []

    try:
        from rdkit import Chem
    except Exception:
        return []

    # Exact match path (HCN etc.)
    canonical = _canonicalize(reagent_smiles)
    canonical_set = {canonical}
    for name, exacts, severity, penalty in EXACT_HAZARDOUS_REAGENTS:
        if any(_canonicalize(e) in canonical_set for e in exacts):
            matches[name] = {"name": name, "severity": severity, "penalty": penalty}

    # SMARTS substructure path
    mol = Chem.MolFromSmiles(reagent_smiles)
    if mol is None:
        return list(matches.values())
    for entry in HAZARDOUS_REAGENTS:
        try:
            patt = Chem.MolFromSmarts(entry["smarts"])
            if patt is None:
                continue
            if mol.HasSubstructMatch(patt):
                matches[entry["name"]] = {
                    "name": entry["name"],
                    "severity": entry["severity"],
                    "penalty": entry["penalty"],
                }
        except Exception:
            continue
    return list(matches.values())


def reagent_score(reagent_strings: List[str]) -> Tuple[float, List[Dict[str, Any]]]:
    """Score a list of reagent SMILES (or names; non-SMILES are ignored).

    Returns (score in [0,1], list of unique matches across all reagents).
    """
    seen: Dict[str, Dict[str, Any]] = {}
    for r in reagent_strings or []:
        for m in scan_reagent(r):
            seen[m["name"]] = m
    penalty_total = sum(m["penalty"] for m in seen.values())
    return max(0.0, 1.0 + penalty_total), list(seen.values())
