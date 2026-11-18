"""Retrosynthetic decomposition using BRICS and Recap algorithms."""

import json
from tools.base import Tool

from rdkit import Chem
from rdkit.Chem import BRICS, Recap


class RetroDecomposeTool(Tool):
    name = "retro_decompose"
    description = (
        "Decompose a molecule into retrosynthetic fragments using BRICS "
        "(rule-based strategic bond breaking) or Recap (hierarchical, finer-"
        "grained). Use as a fallback when `suggest_templates` and "
        "`neural_retro` both come up empty or give low-support candidates — "
        "this gives a coarse structural skeleton to reason about, not a "
        "real precursor set. Cost: <200ms, free. Pairs with `apply_template` "
        "(use a fragment as the SMARTS-target after decomposition)."
    )
    tier = 2
    parameters = {
        "type": "object",
        "properties": {
            "smiles": {"type": "string", "description": "SMILES of the molecule to decompose"},
            "strategy": {
                "type": "string", "enum": ["brics", "recap"], "default": "brics",
                "description": "'brics': BRICS strategic decomposition. 'recap': Recap hierarchical."
            },
            "detail_level": {
                "type": "string", "enum": ["brief", "full"], "default": "brief",
                "description": "'brief': top fragments only. 'full': all fragments + hierarchy."
            },
        },
        "required": ["smiles"]
    }

    def execute(self, smiles: str, strategy: str = "brics", detail_level: str = "brief") -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return json.dumps({"error": f"Invalid SMILES: {smiles}"})

        if strategy == "brics":
            return self._brics_decompose(mol, smiles, detail_level)
        elif strategy == "recap":
            return self._recap_decompose(mol, smiles, detail_level)
        else:
            return json.dumps({"error": f"Unknown strategy: {strategy}"})

    def _brics_decompose(self, mol, smiles: str, detail_level: str) -> str:
        try:
            fragments = list(BRICS.BRICSDecompose(mol))
        except Exception as e:
            return json.dumps({"error": str(e), "smiles": smiles})

        # Clean dummy atoms from BRICS fragments for readability
        clean_frags = []
        for frag in fragments:
            # Remove [1*], [2*], etc. dummy atom labels — keep as-is for accuracy
            clean_frags.append(frag)

        result = {
            "smiles": Chem.MolToSmiles(mol),
            "strategy": "brics",
            "num_fragments": len(clean_frags),
            "fragments": clean_frags[:10] if detail_level == "brief" else clean_frags,
        }
        return json.dumps(result)

    def _recap_decompose(self, mol, smiles: str, detail_level: str) -> str:
        try:
            tree = Recap.RecapDecompose(mol)
        except Exception as e:
            return json.dumps({"error": str(e), "smiles": smiles})

        leaves = tree.GetLeaves()
        leaf_smiles = list(leaves.keys())

        result = {
            "smiles": Chem.MolToSmiles(mol),
            "strategy": "recap",
            "num_fragments": len(leaf_smiles),
            "fragments": leaf_smiles[:10] if detail_level == "brief" else leaf_smiles,
        }

        if detail_level == "full":
            # Include intermediate nodes. RecapHierarchyNode exposes the
            # fragment SMILES and children via attributes (`.smiles`, `.children`),
            # not bound methods.
            children = {}
            def _traverse(node, depth=0):
                smi = node.smiles
                kids = node.children
                if kids:
                    children[smi] = list(kids.keys())
                for child_node in kids.values():
                    _traverse(child_node, depth + 1)
            _traverse(tree)
            result["hierarchy"] = children

        return json.dumps(result)
