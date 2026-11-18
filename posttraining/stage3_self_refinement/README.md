# Stage 3 — offline student self-refinement

Two halves: collection and screening of the student's own trajectories, and the
gated training chain that produces the stage-3 adapter.

```
collect/          launch_collect.py <c0|c1> <rep1|rep2> <workers>
                  canary_admit_v1.py                              # supplemental-run canary admission
                  prepare_isolated.py / chain_collect.sh          # isolated copy + chain driver
screen/           validate_screen.py                                # rubric validator (+ tests)
                  RUBRIC.md                                       # admission rubric (ADMIT_CANDIDATE/EXCLUDE/HOLD)
                  freeze_supplement.py                            # freeze supplemental records, read-only
admission/        run_authorized_training_chain.py                 # owner-authorized chain driver
                  bridge_owner_v2.py                              # bridge interface to the runtime
runtime/src/      train_s3.py                                      # gated ms-swift SFT launcher (plan by default)
                  contract.py, evidence_gates.py, training_trace.py
                  pretokenized_v3.py                              # registered template retrace_s3_final_selected_v3
                  admission_candidate.py, export_adapter.py, tool_preflight.py
                  eval_s3.py, aggregate_test40.py                # stage-3 evaluation + aggregation
runtime/tests/    offline boundary tests (fixture selector_no_claim.json)
data_prep/        pretokenized.py                                 # hash-bound native cached-dataset reader
                  finalize.py, test_native.py, test_legacy_defects.py
```

## How admission works

Two layers, both fail-closed:

- **Scientific admission** (`screen/validate_screen.py` + `RUBRIC.md`): an admitted
  trajectory must reach purchasable stock, have clean infrastructure execution,
  pass turn-level action screening and end in a strictly VALID audit verdict. The
  validator checks that quoted evidence is literal, that every final step was
  read, and that verdicts are in `{ADMIT_CANDIDATE, EXCLUDE, HOLD}`.
- **Runtime gates** (`runtime/src/contract.py`, `evidence_gates.py`): the trainer
  refuses to run unless native-reader acceptance, tokenizer/label parity, optimizer
  evidence and predecessor pins all match. `train_s3.py` defaults to a read-only
  plan; executing it writes evidence under `runtime/evidence/` before spawning
  `swift sft`.

## Training recipe

`configs/stage3_train_chain_v2.json` plus `train_s3.py`: ms-swift 4.6.0 with
`--template retrace_s3_final_selected_v3 --external_plugins pretokenized_v3.py`,
a cached dataset, LoRA r8/α16/dropout 0.05, LR 2e-5 constant, global batch 4,
fresh optimizer from the stage-2 adapter.

## Tests

The tests use `runtime/evidence/fix` as their scratch root and expect it to exist
(it ships with a `.gitkeep`):

```
PYTHONPATH=runtime/src python -m pytest runtime/tests
PYTHONPATH=screen    python -m pytest screen/test_validate_screen.py
```

## What needs a specific run layout

`collect/` and `admission/` are bound to the run layout they drove (owner receipts,
`test40-s3-h0-001/...` manifests, `serving-test40-v1/<half>/OWNER.json`). They are
included as the exact drivers used, not as a portable pipeline; point
`RETRACE_SHARE` at a tree with that layout or adapt the paths.
`data_prep/test_native.py` and `test_legacy_defects.py` need a `swift` environment
and the Qwen3.8 tokenizer.
