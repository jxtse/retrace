## Precedent recall

Before expanding any non-trivial intermediate (≥3 rings or ≥2
stereocenters), consider `recall_similar_routes(query_smiles=…)` for a
strategic anchor from the precedent database. See
`tools/recall_similar_routes` for the trigger checklist. Returned
exemplars are strategic anchors, not answers — always cross-check with
`suggest_templates` + `neural_retro` for your actual molecule. The
target itself is prefetched for you at session start; you only need to
fire this on sub-intermediates.
