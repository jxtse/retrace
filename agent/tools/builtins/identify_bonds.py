"""Identify retrosynthetically meaningful bond disconnection sites."""

import json
import logging
from tools.base import Tool

from rdkit import Chem
from rdkit.Chem import BRICS

logger = logging.getLogger(__name__)


class IdentifyBondsTool(Tool):
    name = "identify_bonds"
    description = (
        "Enumerate retrosynthetically meaningful bond disconnection sites in "
        "a molecule (BRICS strategic bonds with the resulting fragments; "
        "`detail_level='full'` widens to all non-ring single bonds). Call "
        "after `analyze_molecule` when you want hand-pickable disconnection "
        "candidates before committing to a template or neural prediction. "
        "Cost: <100ms, free. Pairs with `apply_template` (this names the "
        "bond, the template provides the chemistry)."
    )
    tier = 1
    parameters = {
        "type": "object",
        "properties": {
            "smiles": {"type": "string", "description": "SMILES of the molecule"},
            "detail_level": {
                "type": "string", "enum": ["brief", "full"], "default": "brief",
                "description": "'brief': BRICS strategic bonds only. 'full': all non-ring single bonds."
            },
        },
        "required": ["smiles"]
    }

    def execute(self, smiles: str, detail_level: str = "brief") -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return json.dumps({"error": f"Invalid SMILES: {smiles}"})

        results = []

        # BRICS bonds (always — these are the synthetically strategic ones)
        try:
            brics_bonds = list(BRICS.FindBRICSBonds(mol))
            for (i, j), (label_i, label_j) in brics_bonds:
                atom_i = mol.GetAtomWithIdx(i)
                atom_j = mol.GetAtomWithIdx(j)
                bond = mol.GetBondBetweenAtoms(i, j)
                results.append({
                    "atom_indices": [i, j],
                    "bond": f"{atom_i.GetSymbol()}-{atom_j.GetSymbol()}",
                    "bond_type": str(bond.GetBondType()).split('.')[-1],
                    "brics_labels": [label_i, label_j],
                    "in_ring": bond.IsInRing(),
                    "source": "brics",
                })
        except Exception as exc:
            # BRICS is the strategic-bond core of this tool; a total failure
            # here silently returns a result with no strategic bonds, which
            # looks identical to "molecule has none". Surface it.
            logger.warning(
                "identify_bonds: BRICS bond detection failed for %s (%s: %s); "
                "returning without strategic bonds.",
                smiles, type(exc).__name__, exc,
            )

        if detail_level == "full":
            # Add all non-ring single bonds not already captured
            brics_pairs = {(b["atom_indices"][0], b["atom_indices"][1]) for b in results}
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if (i, j) in brics_pairs or (j, i) in brics_pairs:
                    continue
                atom_i = mol.GetAtomWithIdx(i)
                atom_j = mol.GetAtomWithIdx(j)
                results.append({
                    "atom_indices": [i, j],
                    "bond": f"{atom_i.GetSymbol()}-{atom_j.GetSymbol()}",
                    "bond_type": str(bond.GetBondType()).split('.')[-1],
                    "in_ring": bond.IsInRing(),
                    "source": "all",
                })

        return json.dumps({
            "smiles": Chem.MolToSmiles(mol),
            "strategic_bonds": len([b for b in results if b.get("source") == "brics"]),
            "bonds": results,
        })
