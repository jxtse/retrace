# `screening/`

| File | Purpose |
|---|---|
| `apply_audit.py` | applies the mechanical leak-span checks and re-runs the gates on a candidate batch |
| `chem_audit.py` | chemistry/consistency reviewer (OpenAI-compatible endpoint from `MINTCN_BASE_URL` / `MINTCN_DSV4_FLASH_API_KEY`) |
| `export_pilot.py`, `export_chem.py` | export batches for review |

Output is a per-row verdict in `{pass, minor, reject}` with reasons; only
pass/minor rows can enter training.
