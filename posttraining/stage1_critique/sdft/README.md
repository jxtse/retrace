# `sdft/`

Data adapters around the upstream Self-Distillation trainer (not vendored; see
`UPSTREAM.md`).

| File | Purpose |
|---|---|
| `sdft_build_pairs.py` | builds `(prompt, teacher_prompt)` pairs from teacher traces |
| `retrace_teacher_context.py` | injects the demonstration context into the teacher prompt |
| `render_sdft_pairs.py` | renders pairs through the native Qwen template, normalising tool arguments |
| `live_training_inputs.py`, `admit_live_batch.py` | per-turn supervision and wire-level admission |
| `raw_prompt_dataset.py`, `capture_adapter.py`, `sdft_offload.py` | dataset plumbing and adapter capture |
| `tests/` | offline tests for the adapters |

The upstream loader consumes the `prompt` / `teacher_prompt` JSONL this directory
produces; no training logic is duplicated here.
