"""Explicit enzymatic-only access to the curated template library."""

from __future__ import annotations

from tools.base import Tool
from tools.builtins.suggest_templates import SuggestTemplatesTool


class EnzymaticRetroTool(Tool):
    """Narrow wrapper that always requests enzymatic templates."""

    name = "enzymatic_retro"
    description = (
        "Coordinator: call this explicit enzymatic single-step "
        "retrosynthesis tool for chiral amines or alcohols, reductions, "
        "transaminations, and aldol, sugar, or terpene cases. It searches "
        "only the existing enzymatic template library and returns template "
        "source, template kind, and reactants; it does not infer or invent "
        "an enzyme identity. Mixed routes often require organic "
        "retrosynthesis on the target first, then enzymatic_retro on a "
        "plausible upstream non-leaf intermediate rather than only on the "
        "final target. Treat ranked suggestions as sourced candidates that "
        "still require chemical validation."
    )
    tier = 2
    parameters = {
        "type": "object",
        "properties": {
            "product_smiles": {
                "type": "string",
                "description": "Product SMILES to decompose enzymatically.",
            },
            "top_k": {
                "type": "integer",
                "default": 15,
                "description": "Number of ranked suggestions to return.",
            },
            "max_templates_to_try": {
                "type": "integer",
                "default": 4000,
                "description": (
                    "Cap on enzymatic templates executed after pre-filtering."
                ),
            },
        },
        "required": ["product_smiles"],
    }

    def __init__(self, suggest_templates: SuggestTemplatesTool | None = None):
        self._suggest_templates = suggest_templates or SuggestTemplatesTool()

    def execute(
        self,
        product_smiles: str,
        top_k: int = 15,
        max_templates_to_try: int = 4000,
        **kwargs,
    ) -> str:
        return self._suggest_templates.execute(
            product_smiles=product_smiles,
            kind="enzymatic",
            top_k=top_k,
            max_templates_to_try=max_templates_to_try,
        )


__all__ = ["EnzymaticRetroTool"]
