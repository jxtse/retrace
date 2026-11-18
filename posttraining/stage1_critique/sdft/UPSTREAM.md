# Upstream: Self-Distillation (stage-1 trainer)

The stage-1 reference-conditioned self-distillation **training loop** is the upstream
`Self-Distillation` repository. It is not vendored here.

| | |
|---|---|
| Repository | https://github.com/idanshen/Self-Distillation (README points at https://github.com/Continual-Intelligence/Self-Distillation) |
| Commit used | `d77573212fa0a3ae2eeb64b9b44db1c251f75e3e` ("Update README.md", 2026-04-07) |
| Papers | arXiv 2601.19897, "Self-Distillation Enables Continual Learning" |
| Files that matter | `main.py` (dataset loaders, demonstration-conditioning template), `distil_trainer.py` / `distil_config.py` (trainer + loss), `eval_science.py` |
| Environment | TRL 0.24.0, transformers 5.14.1, peft 0.19.1 (single-GPU H200-class recipe upstream; we ran it on 4×A800) |

## Local modification (only one)

`distil_trainer.py`, 3 added lines inside the token-estimation warning path (~line 377):

```python
# transformers v5: PreTrainedModel no longer defines warnings_issued; guard for compat.
if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}
```

Without it the trainer crashes on `transformers==5.14.1` because `warnings_issued` is no longer
a `PreTrainedModel` attribute. Apply the same guard if you reproduce with transformers v5;
upstream at the pinned commit assumes an attribute that exists.

Nothing else in the clone is modified (`git status --porcelain` shows exactly `M distil_trainer.py`).

## What this directory contributes instead

`sdft_build_pairs.py`, `render_sdft_pairs.py`, `retrace_teacher_context.py`,
`live_training_inputs.py`, `admit_live_batch.py`, `capture_adapter.py`, `sdft_offload.py` are
data-format adapters around the upstream loop: they build `(prompt, teacher_prompt)` pairs from
ReTRACE teacher traces, render them through the native Qwen3.8 template, and enforce per-turn
supervision. The upstream `main.py` loader consumes the `prompt` / `teacher_prompt` JSONL this
directory produces; no training logic is duplicated here.

After the SDFT round, the admitted responses were re-serialised and the **reported** stage-1
adapter (`S1-v2`, checkpoint-204) was trained with ms-swift on 819 audited rows — see
`../train/`.
