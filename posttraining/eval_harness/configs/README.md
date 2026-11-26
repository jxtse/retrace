# `configs/`

YAML overlays layered on the built-in preset (`main.py run ... --config
configs/<file>.yaml`).

| File | Purpose |
|---|---|
| `qwen36_vllm.yaml` | serve a Qwen3.6-35B-A3B base model from a local vLLM |
| `retro190_gpt56_fast.yaml`, `retro190_gpt56_standard.yaml`, `retro190_gpt56_thorough.yaml` | same budgets with low / medium / xhigh reasoning effort |
| `teacher_full_no_compact.yaml` | teacher collection: postprocess and compaction off, full history preserved |
| `student_27b_vllm.yaml` | student rollout config: vLLM on localhost, 65k context, 32k output |
| `student_27b_vllm_lean.yaml` | same with the lean prompt and `fixed_output_budget: true` |
| `test40_eval_config.yaml` | frozen evaluation config: temperature 0.7, 32,768-token output cap, 131,072 context, four shards of ten targets |

These files select experiment settings. Copy a preset before editing its deployment fields. The shared [deployment example](../../../deployment.example.yaml) documents those fields. Run `uv run --locked python main.py check-config --config PATH` from the harness directory before planning.
