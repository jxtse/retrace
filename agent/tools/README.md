# `tools/`

Everything the model can call. Each tool subclasses `tools.base.Tool` and declares
`name`, `description`, `parameters` (JSON schema) and `run()`.

| File / dir | Tool(s) |
|---|---|
| `builtins/` | `validate_smiles`, `analyze_molecule`, `compare_molecules`, `identify_bonds`, `template_library`, `suggest_templates`, `apply_template`, `retro_decompose`, `explore_branches`, `rank_routes`, `recall_similar_routes`, and in the chemoenzymatic harnesses `enzymatic_retro` |
| `neural_retro.py` | `neural_retro`: template-free single-step prediction through the retroplanner HTTP service |
| `check_buyability.py` | `check_buyability`: leaf purchasability (price vendors, buyability DB, ZINC bloom filter) |
| `verifier/score_route.py` | `score_route`: feasibility / cost / greenness pillars |
| `verifier/pricing.py`, `buyability.py`, `scscore.py`, `smiles_sanity.py`, `carbon_skeleton.py`, `stereo_check.py`, `hazardous_reagents.py`, `condition_prediction.py` | sub-scores used by `score_route` |
| `verifier/llm_judge.py` | optional LLM feasibility judge |
| `verifier/_reaction_rater_bridge.py` | client for the per-step reaction rater |
| `_retroplanner_http.py`, `_ensemble_utils.py`, `_precursor_sanity.py`, `_route_fp.py` | engine client, ensemble merging, precursor sanity check, route fingerprinting |
| `scratchpad.py`, `todo.py`, `read_doc.py`, `load_result.py` | working memory, task list, document reader, result loader |

Optional resources are reported, not faked: when the reaction rater, the
condition model or the solvent table is missing, `score_route` names the missing
component in its summary and scores around it.
