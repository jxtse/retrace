"""Built-in tools (Tier 1 + 2): pure RDKit, no external dependencies."""

from tools.builtins.validate_smiles import ValidateSmilesTool
from tools.builtins.analyze_molecule import AnalyzeMoleculeTool
from tools.builtins.compare_molecules import CompareMoleculesTool
from tools.builtins.identify_bonds import IdentifyBondsTool
from tools.builtins.retro_decompose import RetroDecomposeTool
from tools.builtins.apply_template import ApplyTemplateTool
from tools.builtins.suggest_templates import SuggestTemplatesTool
from tools.builtins.enzymatic_retro import EnzymaticRetroTool
from tools.builtins.recall_similar_routes import RecallSimilarRoutesTool


ALL_BUILTINS = [
    ValidateSmilesTool,
    AnalyzeMoleculeTool,
    CompareMoleculesTool,
    IdentifyBondsTool,
    RetroDecomposeTool,
    ApplyTemplateTool,
    SuggestTemplatesTool,
]
# NOTE: RecallSimilarRoutesTool is NOT in ALL_BUILTINS by design — it
# requires a ``PrecedentDB`` instance which is constructed in build_agent
# and threaded in via the ``precedent_db`` kwarg below. Same pattern we'd
# use for any future tool whose constructor depends on a heavyweight
# external resource.


def register_all_builtins(registry, config=None, precedent_db=None):
    """Register all built-in tools into a ToolRegistry.

    ``config`` is the runtime config dict (or None). Used to
    pass per-tool kwargs to the few builtins that take config. Default
    None keeps the no-arg call sites (e.g. tests) working.

    ``precedent_db`` is an optional ``services.precedent_db.PrecedentDB``
    instance. When supplied, the ``recall_similar_routes`` tool is also
    registered. None (the default) silently omits the tool — keeps
    existing test call sites working.
    """
    config = config or {}
    sugg_cfg = config.get("suggest_templates", {}) or {}
    enzymatic_cfg = config.get("enzymatic_retro", {}) or {}
    suggest_templates = None
    for tool_cls in ALL_BUILTINS:
        if tool_cls is SuggestTemplatesTool:
            suggest_templates = tool_cls(
                enzymatic_max_in_any=sugg_cfg.get(
                    "enzymatic_max_in_any",
                    SuggestTemplatesTool.DEFAULT_ENZYMATIC_MAX_IN_ANY,
                ),
            )
            registry.register(suggest_templates)
        else:
            registry.register(tool_cls())
    if enzymatic_cfg.get("enabled", True):
        registry.register(EnzymaticRetroTool(suggest_templates))
    if precedent_db is not None:
        registry.register(RecallSimilarRoutesTool(precedent_db))
