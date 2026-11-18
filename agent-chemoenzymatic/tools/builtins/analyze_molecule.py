"""Analyze a molecule's structure: functional groups, rings, complexity.

This is the retrosynthesis agent's 'eyes' — letting the LLM see molecular structure
instead of guessing from SMILES strings.
"""

import json
import os
from tools.base import Tool

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments
from rdkit.Chem.Scaffolds import MurckoScaffold

# Functional group SMARTS patterns (retrosynthetically relevant)
_FG_SMARTS = {
    "hydroxyl": "[OX2H]",
    "primary_amine": "[NX3H2;!$([NX3H2](=O))]",
    "secondary_amine": "[NX3H1;!$([NX3H1](=O))]",
    "carboxylic_acid": "[CX3](=O)[OX2H]",
    "ester": "[CX3](=O)[OX2][#6]",
    "amide": "[CX3](=O)[NX3]",
    "aldehyde": "[CX3H1](=O)",
    "ketone": "[#6][CX3](=O)[#6]",
    "nitrile": "[CX2]#[NX1]",
    "nitro": "[$([NX3](=O)=O)]",
    "halide": "[F,Cl,Br,I]",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    "aromatic_ring": "c1ccccc1",
    "epoxide": "C1OC1",
    "sulfonamide": "[SX4](=[OX1])(=[OX1])([NX3])",
    "phosphate": "[PX4](=[OX1])([OX2])([OX2])",
    "thiol": "[SX2H]",
    "thioether": "[#6][SX2][#6]",
    "boronic_acid": "[BX3](O)(O)",
}

# Lazy-load SA scorer
_sa_scorer = None


def _get_sa_score(mol):
    global _sa_scorer
    if _sa_scorer is None:
        try:
            from rdkit.Contrib.SA_Score import sascorer
            _sa_scorer = sascorer
        except ImportError:
            return None
    try:
        return round(_sa_scorer.calculateScore(mol), 2)
    except Exception:
        return None


class AnalyzeMoleculeTool(Tool):
    name = "analyze_molecule"
    description = (
        "Report a molecule's structure: functional groups, rings, chiral "
        "centers, complexity (with `detail_level='full'` adding scaffold, "
        "SA score, rotatable bonds, atom counts). Call once on the target at "
        "session start to anchor the disconnection plan, and on any "
        "intermediate whose structure you need to reason about explicitly. "
        "Cost: <100ms, free (does not consume the iteration budget). Pairs "
        "with `identify_bonds` (analyze first, then enumerate cut sites)."
    )
    tier = 1
    parameters = {
        "type": "object",
        "properties": {
            "smiles": {"type": "string", "description": "SMILES of the molecule"},
            "detail_level": {
                "type": "string", "enum": ["brief", "full"], "default": "brief",
                "description": "'brief': functional groups, ring count, chiral centers, complexity. "
                               "'full': adds scaffold, SA score, rotatable bonds, full atom counts."
            },
        },
        "required": ["smiles"]
    }

    def execute(self, smiles: str, detail_level: str = "brief") -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return json.dumps({"error": f"Invalid SMILES: {smiles}"})

        # Functional groups (brief)
        fg_found = []
        for name, smarts in _FG_SMARTS.items():
            pat = Chem.MolFromSmarts(smarts)
            if pat and mol.HasSubstructMatch(pat):
                count = len(mol.GetSubstructMatches(pat))
                fg_found.append({"name": name, "count": count})

        # Ring info
        ring_info = mol.GetRingInfo()
        ring_count = ring_info.NumRings()
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

        # Chiral centers
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

        # Bertz complexity
        complexity = round(Descriptors.BertzCT(mol), 1)

        result = {
            "smiles": Chem.MolToSmiles(mol),
            "functional_groups": fg_found,
            "ring_count": ring_count,
            "aromatic_rings": aromatic_rings,
            "chiral_centers": len(chiral_centers),
            "complexity": complexity,
        }

        if detail_level == "full":
            # Scaffold
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                result["scaffold"] = Chem.MolToSmiles(scaffold)
            except Exception:
                result["scaffold"] = None

            # SA score
            result["sa_score"] = _get_sa_score(mol)

            # Additional descriptors
            result["mw"] = round(Descriptors.ExactMolWt(mol), 2)
            result["rotatable_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
            result["hbd"] = rdMolDescriptors.CalcNumHBD(mol)
            result["hba"] = rdMolDescriptors.CalcNumHBA(mol)
            result["logp"] = round(Descriptors.MolLogP(mol), 2)
            result["tpsa"] = round(Descriptors.TPSA(mol), 1)

            # Atom counts by element
            atom_counts = {}
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                atom_counts[sym] = atom_counts.get(sym, 0) + 1
            result["atom_counts"] = atom_counts

            # Ring sizes
            result["ring_sizes"] = [len(r) for r in ring_info.AtomRings()]

            # Chiral center details
            result["chiral_details"] = [
                {"atom_idx": idx, "type": ctype} for idx, ctype in chiral_centers
            ]

        return json.dumps(result)
