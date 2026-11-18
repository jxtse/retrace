# Evaluation launchers

Per-arm launchers for the held-out evaluation set (`test40`, medium reasoning
effort). Each arm follows the same protocol: temperature 0.7, 32,768-token
generation cap, 131,072-token context, 40 rounds, 14,400 s per target.

Three arms have launchers here: **base**, **s1v2** (stage-1 adapter only) and
**teachersft** (direct teacher SFT). The stage-2 and stage-3 adapters are
evaluated by `../stage3_self_refinement/runtime/src/eval_s3.py`, and the OPSD
driver's final phase evaluates its exported adapter the same way.

Each arm directory has the same five scripts:

| Script | Purpose |
|---|---|
| `launch_*.py` | `plan` prints the exact plan; `execute` rewrites the arm's `eval_config.yaml`, copies the inherited harness argv/env, swaps the served LoRA alias, waits for free GPUs, and launches two shards |
| `launch_*_rerun.py` | re-runs selected indices after an infrastructure failure (`TAG idx,idx,...`), refusing to touch targets that already produced an accepted result |
| `*_scan.py` | read-only status scan across the arm's run directories |
| `snapshot_*.py` | accepted-final snapshot: latest attempt per index (the archive `../metrics/compute_metrics.py` reads) |
| `stop_*_serving.py` | stops the two vLLM replicas by PGID |

Arms differ only in the model alias and adapter path (and
`CHEMPRICE_DISABLE_MOLPORT=1` for `base`, `s1v2` and `teachersft`, i.e. no external
stock lookup; the stage-2 run kept it enabled). Adapter paths are verified by
SHA-256 before launch.

## Layout assumptions

The launchers inherit argv, environment, working directory and the shard split
from the harness runtime manifests under `RETRACE_SHARE`
(`<runtime>/test40-<arm>-<half>-001/launch_manifest.json`,
`<runtime>/serving-test40-v1/<half>/OWNER.json`) and assert that nothing drifted.
They also read the frozen `eval_config.yaml` and need the retroplanner and rater
endpoints. That is why they document the launch protocol rather than act as a
standalone evaluation runner: point `RETRACE_SHARE` at a run tree with that
layout, or adapt the manifest paths at the top of each script.

Ports and GPUs come from `RETRACE_H0_PORT` / `RETRACE_H1_PORT` and
`RETRACE_H0_CUDA` / `RETRACE_H1_CUDA` (defaults 8000 / 8002 and `0,1` / `2,3`); the
tool endpoint is `RETROPLANNER_HTTP_URL`.
