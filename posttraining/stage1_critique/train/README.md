# `train/`

| File | Purpose |
|---|---|
| `build_s1v2.py` | selects rows (at most 2 per question), checks token/loss alignment and leakage, writes the ms-swift file |
| `convert_s1.py` | converts an earlier stage-1 row set to the same format |
| `run_s1v2.sh` | ms-swift SFT: LoRA r8/α16/d0.05, LR 2e-5 constant, global batch 4, one epoch |
| `chain_s1v2_a1v2.sh` | runs S1 then stage 2 and checks the adapters |

Requires the ms-swift environment (`venv-swift`, ms-swift 4.6.0) and the stage-1
row files.
