# `scripts/`

Evaluation helpers and one-off tooling, not part of the agent's runtime path.

| File | Purpose |
|---|---|
| `run_hard_mol_batch.py` | batch driver wrapper used for the hard-molecule set |
| `build_precedent_index.py` | builds the precedent-route index `recall_similar_routes` reads |
| `trace_route.py` | prints a route step by step |
| `visualize_transcripts.py` | renders transcripts to HTML |
| `_audit_*.py`, `_classify_cases.py` | route export and case classification for audits |
| `_molecode_*.py` | stop-reason and summary helpers for a batch run |
| `show_smiles.py`, `UI_single_step_cn.py` | small utilities |
| `audit_workflow.mjs` | review workflow helper |

Some of these expect files from a run tree (audit exports, batch results); they
fail loudly when the input is missing rather than inventing one.
