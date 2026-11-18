# `sampling/`

Rejection-sampling drivers for the base model, plus row finalisation.

| File | Purpose |
|---|---|
| `rs_gen.py`, `rs_gen_facts.py` | sample the base model on the student prompt (the second variant also supplies RDKit facts) |
| `finalize_rows.py` | turns a raw sample into a training row (reference-quote and instruction filters) |
| `export_rs_for_audit.py` | exports samples for the chemistry review |
| `chain_rs_pilot.sh` | pilot chain: two vLLM replicas, 32-way concurrency, two shards |

`rs_gen*.py` depend on internal packages and are documentation-only; `NOTE.md`
records the gate contract they enforced.
