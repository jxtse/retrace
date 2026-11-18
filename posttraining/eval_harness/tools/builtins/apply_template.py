"""Apply a retrosynthetic template (reaction SMARTS) to a product."""

import json
import logging

logger = logging.getLogger(__name__)
from tools.base import Tool

from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from rdchiral.main import rdchiralRunText
    _HAS_RDCHIRAL = True
except ImportError:
    _HAS_RDCHIRAL = False


class ApplyTemplateTool(Tool):
    name = "apply_template"
    description = (
        "Run a single retrosynthetic SMARTS template (product>>reactants) "
        "against a product to enumerate the resulting reactant sets. Use "
        "to verify a hand-written or copy-pasted template works on a target "
        "before committing — most callers should call `suggest_templates` "
        "first and use its proposed reactants directly rather than "
        "re-deriving them via SMARTS. Cost: <500ms, free. Pairs with "
        "`suggest_templates` (suggest gives a template, this re-runs it)."
    )
    tier = 2
    parameters = {
        "type": "object",
        "properties": {
            "product_smiles": {"type": "string", "description": "SMILES of the product"},
            "template_smarts": {
                "type": "string",
                "description": "Reaction SMARTS in retro direction (product>>reactants)"
            },
        },
        "required": ["product_smiles", "template_smarts"]
    }

    def execute(self, product_smiles: str, template_smarts: str, **kwargs) -> str:
        mol = Chem.MolFromSmiles(product_smiles)
        if mol is None:
            return json.dumps({"error": f"Invalid product SMILES: {product_smiles}"})

        results = []

        # Try rdchiral first (better stereochemistry handling)
        if _HAS_RDCHIRAL:
            try:
                outcomes = rdchiralRunText(template_smarts, product_smiles)
                for outcome in outcomes:
                    results.append(outcome)
            except Exception as exc:
                logger.debug("rdchiral failed for template %s on %s: %s", template_smarts[:50], product_smiles, exc, exc_info=True)

        # Fallback to RDKit ReactionFromSmarts
        if not results:
            try:
                rxn = AllChem.ReactionFromSmarts(template_smarts)
                if rxn is None:
                    return json.dumps({"error": f"Invalid template SMARTS: {template_smarts}"})
                products_as_reactants = [mol]
                outcome_sets = rxn.RunReactants(products_as_reactants)
                for outcome_set in outcome_sets[:10]:
                    smiles_set = []
                    for m in outcome_set:
                        try:
                            Chem.SanitizeMol(m)
                            smiles_set.append(Chem.MolToSmiles(m))
                        except Exception as exc:
                            logger.debug("Sanitization failed for outcome: %s", exc, exc_info=True)
                    if smiles_set:
                        results.append(".".join(smiles_set))
            except Exception as e:
                return json.dumps({"error": str(e)})

        # Deduplicate
        unique_results = list(dict.fromkeys(results))

        return json.dumps({
            "product": product_smiles,
            "template": template_smarts,
            "num_outcomes": len(unique_results),
            "reactant_sets": unique_results[:5],
        })
