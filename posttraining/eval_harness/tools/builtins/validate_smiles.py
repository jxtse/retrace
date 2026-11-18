"""Enhanced SMILES validation and canonicalization tool."""

import json
from tools.base import Tool

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, inchi


class ValidateSmilesTool(Tool):
    name = "validate_smiles"
    description = (
        "Canonicalize and sanity-check a SMILES string ("
        "`detail_level='full'` adds formula, MW, InChIKey). Call after "
        "generating a SMILES from `apply_template`, a neural prediction, or "
        "any string-manipulated reactant before passing it to a downstream "
        "tool — invalid SMILES poison every later step. Cost: <50ms, free. "
        "Pairs with `compare_molecules` (validate, then compare against the "
        "target to rule out trivial echoes)."
    )
    tier = 1
    parameters = {
        "type": "object",
        "properties": {
            "smiles": {"type": "string", "description": "The SMILES string to validate"},
            "detail_level": {
                "type": "string", "enum": ["brief", "full"], "default": "brief",
                "description": "'brief': canonical SMILES only. 'full': adds formula, MW, InChIKey."
            },
        },
        "required": ["smiles"]
    }

    def execute(self, smiles: str, detail_level: str = "brief") -> str:
        # Reject empty / whitespace-only input explicitly: RDKit's
        # MolFromSmiles("") returns a non-None zero-atom Mol (not None), so a
        # plain `if mol is None` guard would report `{"valid": true,
        # "canonical": ""}` for an empty string.
        if not smiles or not smiles.strip():
            return json.dumps({"valid": False, "error": "Empty SMILES"})
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return json.dumps({"valid": False, "error": f"Invalid SMILES: {smiles}"})

        canonical = Chem.MolToSmiles(mol)
        result = {"valid": True, "canonical": canonical}

        if detail_level == "full":
            result["formula"] = rdMolDescriptors.CalcMolFormula(mol)
            result["mw"] = round(Descriptors.ExactMolWt(mol), 2)
            try:
                result["inchikey"] = inchi.MolToInchiKey(mol)
            except Exception:
                result["inchikey"] = None
            result["num_atoms"] = mol.GetNumHeavyAtoms()
            result["num_rings"] = len(Chem.GetSSSR(mol))

        return json.dumps(result)
