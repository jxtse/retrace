"""Compare two molecules: similarity, substructure relationship."""

import json
from tools.base import Tool

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFMCS


class CompareMoleculesTool(Tool):
    name = "compare_molecules"
    description = (
        "Compare two molecules: Tanimoto similarity and substructure "
        "containment (`detail_level='full'` adds the maximum common "
        "substructure). Use to verify a proposed precursor isn't structurally "
        "the answer in disguise — Tanimoto > 0.85 against the product almost "
        "always means the disconnection collapsed to a no-op (a cheat). "
        "Cost: <100ms, free. Pairs with `validate_smiles` (validate the "
        "precursor first, then compare)."
    )
    tier = 1
    parameters = {
        "type": "object",
        "properties": {
            "smiles_a": {"type": "string", "description": "SMILES of first molecule"},
            "smiles_b": {"type": "string", "description": "SMILES of second molecule"},
            "detail_level": {
                "type": "string", "enum": ["brief", "full"], "default": "brief",
                "description": "'brief': similarity + substructure check. 'full': adds MCS."
            },
        },
        "required": ["smiles_a", "smiles_b"]
    }

    def execute(self, smiles_a: str, smiles_b: str, detail_level: str = "brief") -> str:
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        if mol_a is None or mol_a.GetNumAtoms() == 0:
            return json.dumps({"error": f"Invalid SMILES: {smiles_a}"})
        if mol_b is None or mol_b.GetNumAtoms() == 0:
            return json.dumps({"error": f"Invalid SMILES: {smiles_b}"})

        # Tanimoto on Morgan FP
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
        similarity = round(DataStructs.TanimotoSimilarity(fp_a, fp_b), 3)

        # Substructure check
        a_in_b = mol_b.HasSubstructMatch(mol_a)
        b_in_a = mol_a.HasSubstructMatch(mol_b)

        result = {
            "similarity": similarity,
            "a_is_substructure_of_b": a_in_b,
            "b_is_substructure_of_a": b_in_a,
        }

        if detail_level == "full":
            # MCS (with timeout to avoid long runs on large molecules)
            try:
                mcs = rdFMCS.FindMCS(
                    [mol_a, mol_b],
                    timeout=5,
                    matchValences=False,
                    ringMatchesRingOnly=True,
                )
                result["mcs_smarts"] = mcs.smartsString if mcs.smartsString else None
                result["mcs_num_atoms"] = mcs.numAtoms
                result["mcs_num_bonds"] = mcs.numBonds
            except Exception:
                result["mcs_smarts"] = None

        return json.dumps(result)
