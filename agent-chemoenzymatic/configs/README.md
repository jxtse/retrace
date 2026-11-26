# `configs/`

YAML overlays layered on the built-in preset (`main.py run ... --config
configs/<file>.yaml`, or `AGENT_CONFIG=...` for the benchmark drivers).

| File | Purpose |
|---|---|
| `qwen36_vllm.yaml` | serve a Qwen3.6-35B-A3B base model from a local vLLM: 65k context, 8k output budget, relaxed tool timeout, tuned compaction |
| `retro190_gpt56_fast.yaml` | pair with `--thinking fast` for the fast search budget and `low` reasoning effort |
| `retro190_gpt56_standard.yaml` | pair with `--thinking standard` for the standard search budget and `medium` reasoning effort |
| `retro190_gpt56_thorough.yaml` | pair with `--thinking thorough` for the thorough search budget and `xhigh` reasoning effort |
| `teacher_full_no_compact.yaml` | teacher-trajectory collection: postprocess and compaction off, full history preserved |

For Retro-190, `run`, `evaluate`, and `check-config` default to `--thinking fast`. The overlay sets reasoning effort only. When invoking a benchmark driver directly, set `AGENT_THINKING` to the matching level alongside `AGENT_CONFIG`. Copy a preset before editing its deployment fields. The shared [deployment example](../../deployment.example.yaml) documents those fields. Run `uv run --locked python main.py check-config --thinking LEVEL --config PATH` from the harness directory with the same level and YAML you will use for planning.
