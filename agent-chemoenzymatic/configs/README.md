# `configs/`

YAML overlays layered on the built-in preset (`main.py run ... --config
configs/<file>.yaml`, or `AGENT_CONFIG=...` for the benchmark drivers).

| File | Purpose |
|---|---|
| `qwen36_vllm.yaml` | serve a Qwen3.6-35B-A3B base model from a local vLLM: 65k context, 8k output budget, relaxed tool timeout, tuned compaction |
| `retro190_gpt56_fast.yaml` | fast search budget with `reasoning_effort: low` |
| `retro190_gpt56_standard.yaml` | standard search budget with `reasoning_effort: medium` |
| `retro190_gpt56_thorough.yaml` | thorough search budget with `reasoning_effort: xhigh` |
| `teacher_full_no_compact.yaml` | teacher-trajectory collection: postprocess and compaction off, full history preserved |

These files select experiment settings. Copy a preset before editing its deployment fields. The shared [deployment example](../../deployment.example.yaml) documents those fields. Run `uv run --locked python main.py check-config --config PATH` from the harness directory before planning.
