# `metrics/`

| File | Purpose |
|---|---|
| `compute_metrics.py` | reads snapshots and audit campaigns, writes `metrics.json` and `per_target_metrics.csv` (stdlib only, deterministic) |
| `METRICS_DEFINITIONS.md` | every field's definition, the denominators, the defect-family mapping and what the data cannot answer |

```bash
RETRACE_SNAP=/path/to/archive RETRACE_OUTPUTS=/path/to/outputs python compute_metrics.py
```

Edit the `ARMS` table at the top of the script to point at your snapshot
directories.
