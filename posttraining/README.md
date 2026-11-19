# ReTRACE post-training pipeline

Code for the post-training stage of ReTRACE (Qwen3.8-27B plus a LoRA adapter):
the three-stage curriculum (reference-guided critique → teacher-trajectory SFT →
student self-refinement), the direct teacher-SFT comparison arm, the on-policy
self-distillation (OPSD) and GRPO recipes, the evaluation launchers, the route
audit adapters and the metric computation.

This directory holds code, configs, prompt/spec documents and a few small test
fixtures. It contains no trajectories, no teacher outputs, no audit verdicts, no
checkpoints or adapter weights, no logs and no run directories. The datasets
these recipes consume are described in [Inputs](#inputs-you-must-supply); they
are not distributed.

## Layout

```
posttraining/
├── stage1_critique/             stage 1: reference-guided chemical critique
│   ├── sdft/                    reference-conditioned self-distillation data adapters (+ tests)
│   ├── sampling/                rejection-sampling drivers, row finalisation
│   ├── screening/               leak/consistency audit and chemistry review export
│   └── train/                   S1 dataset build + ms-swift SFT launcher
├── stage2_teacher_trajectories/ stage 2: route admission and turn-level loss masking
│   ├── collection/              teacher collection controller (replicates × train targets)
│   ├── split/                   target-disjoint split freeze (130 / 20 / 40)
│   ├── audit/                   frozen route audit driver + spec (VALID/SUSPECT/FAKE)
│   ├── selector/                turn-level selector driver + frozen selector config
│   ├── assemble/                trajectory rendering, loss-mask construction, assembly
│   └── train/                   medium-effort re-render, token/loss parity, S2 launcher
├── stage3_self_refinement/      stage 3: offline student self-refinement
│   ├── collect/                 student self-sampling launcher
│   ├── screen/                  admission screen validator (+ tests) and rubric
│   ├── admission/               authorization chain / bridge
│   ├── runtime/src/             gated trainer, contract + evidence gates, registered template
│   ├── runtime/tests/           offline boundary tests (+ fixture)
│   └── data_prep/               native cached-dataset reader (+ tests)
├── teacher_sft/                 direct teacher-SFT comparison arm
├── online_opsd/                 online on-policy self-distillation (Megatron-SWIFT GKD)
│   ├── driver/                  resumable serve→rollout→rows→train→export driver
│   ├── build/                   privileged-context row builders, template plugin, parity checks
│   ├── lib/                     rollout/infra scanning, row helpers, eval dry-check
│   ├── run/                     GKD launchers
│   └── wire/                    wire-format sanity check
├── online_grpo/                 GRPO replay trainer (ms-swift, build only)
├── evaluation_test40/           per-arm evaluation launchers (base, s1v2, teachersft)
├── chemistry_audit/             route audit adapter, validators, RDKit grounding, paired comparison
├── metrics/                     metric computation + definitions
├── configs/                     frozen split, stage-3 training-chain config
└── eval_harness/                harness variant used to evaluate the trained models
```

## Methods → code

| Step | Implementation |
|---|---|
| **Split.** Retro\*-190 partitioned by canonical target SMILES into 130 / 20 / 40, verified target-disjoint from the 40 evaluation targets. | `stage2_teacher_trajectories/split/freeze_split.py` (Butina clustering on Morgan r=2, 2048-bit, Tanimoto 0.6, seed 42) → `split/SPLIT_STAGE2.json`, `configs/split_retro190.json`. |
| **Stage 1 generation.** The base model produces reasoning while consulting a reference explanation that is removed from the student prompt. | `stage1_critique/sdft/sdft_build_pairs.py` (prompt / teacher_prompt pairs), `render_sdft_pairs.py` (native-template rendering, tool-argument normalisation), `retrace_teacher_context.py` (demonstration-context injection), `live_training_inputs.py` + `admit_live_batch.py` (per-turn supervision, wire-level admission), `sdft_offload.py`. The training loop and loss are upstream (see [Frameworks](#frameworks)). |
| **Stage 1 sampling.** On-policy sampling, suffix resampling, student-prefix log-prob acceptance, automatic gates on the final candidate. | `stage1_critique/sampling/rs_gen.py`, `rs_gen_facts.py` (rejection sampling over the base model), `finalize_rows.py` (row normalisation), `export_rs_for_audit.py`. The gate contract is documented in `sampling/NOTE.md`. |
| **Stage 1 screening.** Screen for reliance on the hidden reference, label consistency and parseability of new SMILES; a separate chemistry review admits pass/minor. | `stage1_critique/screening/apply_audit.py` (mechanical leak-span application + re-gate), `chem_audit.py` (chemistry/consistency reviewer), `export_pilot.py`, `export_chem.py`. |
| **Stage 1 dataset and training.** 819 rows over 804 questions (788 self-distillation + 31 rejection sampling), at most 2 per question; ms-swift SFT, fresh LoRA. | `stage1_critique/train/build_s1v2.py` (selection + token/loss/leak checks), `convert_s1.py`, `run_s1v2.sh`, `chain_s1v2_a1v2.sh` (S1 → S2 chain + adapter checks). |
| **Stage 2 collection.** Teacher model, 3 trajectories per training target. | `stage2_teacher_trajectories/collection/run_remote.py`, `append_rep3.py`, `prepare_verify_remote.py`. The planner itself is `../agent/`. |
| **Stage 2 audit.** Only trajectories whose final route is strictly VALID are admitted. | `stage2_teacher_trajectories/audit/` — `AUDIT_SPEC.md`, `audit_driver.py` (two-stage deep read + adversarial FAKE defence), `run_audit.py`, `grounding_reference.py` (deterministic RDKit per-step facts). |
| **Turn-level selector.** Two evaluations per trajectory; mean ≥ 7/10, risk veto ≥ 3, deterministic exclusions. | `stage2_teacher_trajectories/selector/seg_select.py` (per-turn scoring, two samples), `run_selection.py`, `selector/FROZEN_v4.json` (frozen prompt and mechanism-flag digests plus thresholds). |
| **Turn-level loss mask.** Supervision only on visible text, tool calls and the end-of-message token of selected turns; reasoning prefixes and other turns are masked but kept as context. | `stage2_teacher_trajectories/assemble/build_full_v1.py` (renders each trajectory as the deployed student sees it), `assemble_a2_v1.py` (VALID-only assembly, selector-kept spans, `A2_VALUE_TH=7`), `compact_traj.py`, `mech_flags_v1.py`. Parity: `train/convert_and_check.py`, `check_equiv.py`; medium-effort re-render: `train/convert_a1_medium.py`. |
| **Stage 2 training.** Continued from the S1 adapter. | `stage2_teacher_trajectories/train/run_a1v2.sh` (ms-swift, ZeRO-3, sequence parallelism 4). |
| **Stage 3 collection.** The stage-2 student samples its own trajectories on train targets. | `stage3_self_refinement/collect/launch_collect.py`, `prepare_isolated.py`, `chain_collect.sh`, `canary_admit_v1.py`. |
| **Stage 3 admission.** Completion to purchasable stock, clean infrastructure, turn-level action screening, VALID audit. | `stage3_self_refinement/screen/validate_screen.py` (+ `test_validate_screen.py`, `RUBRIC.md`), `runtime/src/admission_candidate.py`, `evidence_gates.py`, `contract.py`. |
| **Stage 3 training.** From the stage-2 checkpoint with a fresh optimizer, supervising visible actions and tool calls only. | `admission/run_authorized_training_chain.py`, `bridge_owner_v2.py`, `configs/stage3_train_chain_v2.json`, `runtime/src/train_s3.py`, `runtime/src/pretokenized_v3.py`, `data_prep/pretokenized.py`, `runtime/tests/`. |
| **Online arm — OPSD.** Steps of serve → rollout → row build → train → export against a frozen copy of the base model as teacher, given a privileged audited route; GJS β=0.5, T=1.0, top-64 teacher logits, visible tokens only, Megatron-SWIFT GKD, TP4 + SP, AdamW, LR 2e-5. | `online_opsd/driver/opsd_driver.py`, `build/s0_privileged.py`, `s0_build.py`, `s2_rows.py`, `s2b_build.py`, `build/s2b_template.py` (loss-mask template plugin), `run/opsd_run_train.sh`, `run/opsd_serve.py`, `lib/*`. |
| **Online arm — GRPO.** ms-swift GRPO with a verdict-shaped reward. Build and replay seam only. | `online_grpo/retrace_reward.py` (reward ORM plugin), `retrace_template.py`, `retrace_replay.py` (wire log → training row + teacher view), `replay_grpo_trainer.py`, `run_grpo_replay.py`, `test_spans.py`. |
| **Optimisation.** LoRA r8 α16 dropout 0.05 on attention, linear-attention and MLP projections, bf16, ms-swift + ZeRO-3, LR 2e-5 constant, one epoch, global batch 4, seed 42, max length 16,384 (stage 1) and 131,072 with sequence parallelism 4 (stages 2–3). | `stage1_critique/train/run_s1v2.sh`, `stage2_teacher_trajectories/train/run_a1v2.sh`, `teacher_sft/run_teacher_sft_base.sh`, `configs/stage3_train_chain_v2.json`, `runtime/src/train_s3.py`. |
| **Evaluation protocol.** Temperature 0.7, 32,768-token generation cap, 131,072 context, 40 rounds, 14,400 s per target, one archived result per target. | `evaluation_test40/{base,s1v2,teachersft}/launch_*.py`, `*_scan.py`, `snapshot_*.py`, `stop_*_serving.py`; harness: `eval_harness/`. |
| **Route audit and metrics.** Rank-1 route audit; completion and audit kept separate; McNemar, bootstrap and Clopper–Pearson intervals. | `chemistry_audit/adapter.py`, `audit_driver.py`, `validators.py`, `grounding.py`, `compare_s3_base.py`; `metrics/compute_metrics.py`, `metrics/METRICS_DEFINITIONS.md`. |

## Checkpoint lineage

| Arm | Recipe | Steps |
|---|---|---|
| S1 | ms-swift SFT, 819 rows, fresh LoRA from base | 204 (819 rows / batch 4) |
| S2 | ms-swift SFT, 167 rows, `--adapters` = S1 | 42 (167 / 4) |
| S3 | gated ms-swift SFT, fresh optimizer from S2 | 7 |
| Teacher SFT | same 167 rows, fresh LoRA from base | 42 |
| OPSD | serve/rollout/train/export per step; the driver exports an adapter after each step | recipe only |
| GRPO | ms-swift GRPO, replay from wire logs | build only |

No adapters or checkpoints are distributed. The scripts expect them under a run
tree you control; `RETRACE_SHARE` is the root that the launchers use.

## Frameworks

These are installed, not vendored.

| Component | Version |
|---|---|
| ms-swift (`swift sft`; stages 1–3, teacher SFT) | 4.6.0 |
| DeepSpeed (ZeRO-3) | 0.18.9 |
| TRL / transformers / peft | 0.24.0 / 5.14.1 / 0.19.1 |
| ms-swift (`megatron rlhf`; OPSD and GRPO) | 4.6.1 |
| Megatron-Core / mcore-bridge | 0.20.0 / 1.7.0 |
| torch / Transformer Engine | 2.9.1 / 2.10.0 |
| Self-Distillation (stage-1 trainer) | commit `d7757321`, see `stage1_critique/sdft/UPSTREAM.md` |
| Model | Qwen3.8-27B, bf16 |

The stage-1 training loop comes from the upstream Self-Distillation repository
with exactly one local change, a three-line `warnings_issued` compatibility guard
for transformers v5 in `distil_trainer.py`; `stage1_critique/sdft/UPSTREAM.md`
records it. ms-swift, Megatron-Core and DeepSpeed are used unmodified. The only
seams we add are our own files: a registered template
(`online_opsd/build/s2b_template.py`, `stage3_self_refinement/runtime/src/pretokenized_v3.py`)
and a trainer override (`online_grpo/replay_grpo_trainer.py`).

## Inputs you must supply

| Artefact | Produced by | Shape |
|---|---|---|
| Split | `split/freeze_split.py` | 130 / 20 / 40 target lists (`configs/split_retro190.json`) |
| Stage-1 candidate rows | stage-1 sampling | `reference_generation_candidates.jsonl` (question + reference + RDKit facts) |
| Stage-1 self-distillation rows | upstream SDFT run | `train_s1_swift.jsonl` (input to `build_s1v2.py`) |
| Teacher trajectories | the harness + `collection/run_remote.py` | 3 × 130 wire transcripts |
| Audit verdicts | `stage2_teacher_trajectories/audit/` | `verdicts.jsonl` (VALID/SUSPECT/FAKE + reasons) |
| Selector outputs | `selector/run_selection.py` | two evaluations per turn |
| Stage-2 rows | `assemble/` + `train/convert_a1_medium.py` | `train_a1_medium.jsonl` (167 rows; also the teacher-SFT input) |
| Stage-3 trajectories and screen verdicts | stage-3 collect/screen | trajectories admitted by the screen |
| Evaluation snapshots | the harness | `snapshots/<idx>/{result.json,transcript.json}` |
| Audit campaign records | `chemistry_audit/` | `campaign-<arm>/campaign.json` |

Where a step needs one of these, the script reads it from a path you set; no
script invents data. The planner, its tools and the streaming runtime are in
[`../agent`](../agent) (or [`eval_harness/`](eval_harness/) for the evaluation
variant) — this directory does not duplicate them.

## Running a recipe

Nothing here runs end to end without a GPU host, the training environment and the
inputs above. The order is:

1. Freeze the split: `python stage2_teacher_trajectories/split/freeze_split.py`.
2. Build stage-1 rows (sampling + screening), then train S1 with
   `stage1_critique/train/run_s1v2.sh`.
3. Collect teacher trajectories, audit them, run the selector, assemble rows, then
   `stage2_teacher_trajectories/train/run_a1v2.sh`.
4. Collect student trajectories, screen and admit them, then run the gated
   stage-3 chain (`configs/stage3_train_chain_v2.json`).
5. Evaluate with `evaluation_test40/<arm>/launch_*.py`, audit with
   `chemistry_audit/adapter.py`, score with `metrics/compute_metrics.py`.

Shell launchers read their paths and endpoints from the environment:

| Variable | Meaning | Used in |
|---|---|---|
| `RETRACE_SHARE` | root of the run/share tree (`models/`, run dirs) | most Python and shell scripts |
| `RETRACE_DATA_ROOT` | external data root (route exports) | `grounding_reference.py`, audit inputs |
| `RETRACE_OUTPUTS` | root of audit/campaign/metrics outputs | audit, metrics, screen freeze |
| `RETRACE_SNAP` | archived evaluation snapshot root | `metrics/compute_metrics.py` |
| `RETRACE_HOST` | SSH alias of the GPU host | OPSD driver, controller scripts |
| `RETRACE_PYTHON312` | Python 3.12.14 interpreter used to build isolated stage-3 runtime wrappers (default `python3.12`) | `stage3_self_refinement/runtime/src/repair_envs.py` |
| `MINTCN_BASE_URL`, `MINTCN_DSV4_FLASH_API_KEY` | OpenAI-compatible endpoint and key for the stage-1 chemistry reviewer | `chem_audit.py` |
| `JUDGE_BASE_URL` | audit judge endpoint | `chemistry_audit/*` |
| `ASTRA_BASE_URL` | teacher endpoint (`/v1/responses`) | `selector/seg_select.py` |
| `RETROPLANNER_HTTP_URL` | single-step retroplanner tool endpoint | OPSD driver, evaluation launchers |
| `RETRACE_H0_PORT` / `RETRACE_H1_PORT`, `RETRACE_H0_CUDA` / `RETRACE_H1_CUDA` | two-shard serving ports and GPUs | OPSD driver, test40 launchers |

Defaults are neutral placeholders (`/path/to/...`, `gpu-host`), so set these
before running anything. Keys are read from the environment only; nothing is
committed.

## Offline tests

```bash
python -m compileall -q stage1_critique stage2_teacher_trajectories stage3_self_refinement \
    teacher_sft online_opsd online_grpo chemistry_audit metrics
bash -n stage1_critique/sampling/*.sh stage1_critique/train/*.sh \
    stage2_teacher_trajectories/train/*.sh stage3_self_refinement/collect/*.sh

PYTHONPATH=stage1_critique/sdft python -m pytest -q stage1_critique/sdft/tests
PYTHONPATH=stage2_teacher_trajectories/collection python -m pytest -q \
    stage2_teacher_trajectories/collection/test_append_rep3.py
python -m pytest -q stage3_self_refinement/screen/test_validate_screen.py
PYTHONPATH=stage3_self_refinement/runtime/src python -m pytest -q stage3_self_refinement/runtime/tests
```

All of these are offline: no GPU, no model inference, no network. Tests that need
a tokenizer download, `swift`/Megatron/vLLM or a live endpoint are not in that
set (`test_spans.py`, `s2b_parity.py`, `s2b_tokenize_parity.py`,
`test_native.py`, `test_legacy_defects.py`, `check_equiv.py`,
`convert_and_check.py`, `build_s1v2.py`, `convert_s1.py`).

## What is documentation-only

- **Stage-1 sampling drivers.** `rs_gen*.py` import `thinking_v2` (the token-exact
  renderer) and `rewrite_v3` from an internal package tree that is not published,
  so the sampling step is documented rather than runnable. The gate contract they
  enforced is in `sampling/NOTE.md`, and `screening/apply_audit.py` reuses the
  same checks.
- **Stage-2 replication controller and stage-3 admission chain.** Both encode the
  run layout and per-run receipts of the machines they drove; they ship as-is.
- **Online arms.** GRPO ships build-time code and a replay seam only. OPSD ships
  the driver, loss construction and row builders.

## License

Files written for ReTRACE here are MIT, covered by the repository-root
[`LICENSE`](../LICENSE). Third-party code that these files import or invoke keeps
its own license and is not relicensed: ms-swift, Megatron-Core, DeepSpeed, TRL,
transformers, peft and vLLM are Apache-2.0, RDKit is BSD-3-Clause, and the
Self-Distillation trainer is referenced rather than vendored. See
[`../LICENSES.md`](../LICENSES.md).
