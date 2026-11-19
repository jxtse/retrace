# OPSD (on-policy self-distillation) — Megatron-SWIFT GKD

Recipe for the on-policy self-distillation (OPSD) arm: steps of 16 targets, a fresh LoRA from
base, a frozen base-model teacher whose prompt gets a privileged audited route
appended, GJS loss (β = 0.5, T = 1.0, top-64 teacher logits), and supervision on
visible assistant tokens and tool calls only (reasoning excluded).

## Driver

`driver/opsd_driver.py` is a resumable state machine. Per step:

```
serve → rollout → infra → rows → parity → stop_serve → train → export
```

Phases are idempotent; progress lives in `state.json` (atomic write) and
`events.jsonl` (append). A restart resumes at the first phase without `ok: true`;
`run/opsd_launch.sh` runs it detached under `setsid nohup` with an `flock`, so a
second launch exits immediately.

Two deliberate deviations from the naive order (also stated in the driver
docstring): `rows` and `parity` run while the step's vLLM replicas are still up,
because the `/tokenize` parity check needs the served tokenizer; and the last
phase runs the frozen test40@medium evaluation on the adapter exported at the final
step, so the loop ends with an evaluation instead of a dangling adapter.

```
python opsd_driver.py run [--from-step N]      # full loop
python opsd_driver.py phase <step> <phase>     # single phase, for debugging
bash run/opsd_launch.sh [from_step]            # detached
```

## Pieces

| Path | Role |
|---|---|
| `build/s0_manifest.py`, `s0_build.py`, `s0_privileged.py`, `s0_rows_v2.py`, `s0_tito.py` | target manifest, rollout rows, privileged-context teacher view, row build |
| `build/s2_rows.py`, `s2b_build.py`, `s2b_template.py`, `s2b_project.py` | row builders plus the registered `retrace_opsd_ptok_v1` template (loss mask) |
| `build/s2b_parity.py`, `s2b_tokenize_parity.py` | CPU-side and served-tokenizer parity checks (fail closed on token drift) |
| `lib/opsd_rows.py`, `opsd_infra.py`, `opsd_eval_drycheck.py` | row helpers, rollout/infra scanning, evaluation dry-check |
| `run/s2_run_gkd.sh`, `run/s2b_run.sh`, `run/run_gkd_*.sh` | diagnostics: GKD at several SP/TP layouts, top-k sweeps |
| `run/opsd_run_train.sh` | the step launcher: `megatron rlhf --rlhf_type gkd`, TP4 + SP, `--global_batch_size 16`, `--gkd_logits_topk 64`, `--lmbda 0 --beta 0.5 --sft_alpha 0`, `--max_completion_length 8`, `--save_steps 1`, resume via `--mcore_adapter … --no_finetune` |
| `run/opsd_export.sh` | Megatron adapter export |
| `run/opsd_serve.py`, `opsd_snapshot.py`, `opsd_stop_serve.py`, `opsd_keys.py` | two-shard vLLM serving, serving snapshot, key-set diff over the 192 mcore LoRA module names |
| `wire/s0_wire_check.py` | wire-format sanity check (no model loads) |

## Environment

`venv-mcore`: ms-swift 4.6.1, Megatron-Core 0.20.0, mcore-bridge 1.7.0,
DeepSpeed 0.20.0, torch 2.9.1, Transformer Engine 2.10.0, TRL 0.29.1. The SFT
stages use a different environment (`venv-swift`, ms-swift 4.6.0); do not mix
them.

## Inputs required

Collected rollouts (wire logs) from the harness, the audited stage-2 routes used as
the privileged block, and two vLLM replicas on `RETRACE_H0_PORT` / `RETRACE_H1_PORT`
bound to `RETRACE_H0_CUDA` / `RETRACE_H1_CUDA`. The driver also needs a single-step
retroplanner tool at `RETROPLANNER_HTTP_URL`. None of the rollout data is
distributed here.
