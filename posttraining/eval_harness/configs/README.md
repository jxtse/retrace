# `configs/`

YAML overlays layered on the built-in preset (`main.py run ... --config
configs/<file>.yaml`).

| File | Purpose |
|---|---|
| `qwen36_vllm.yaml` | serve a Qwen3.6-35B-A3B base model from a local vLLM |
| `retro190_gpt56_fast.yaml` | pair with `--thinking fast` for the fast search budget and `low` reasoning effort |
| `retro190_gpt56_standard.yaml` | pair with `--thinking standard` for the standard search budget and `medium` reasoning effort |
| `retro190_gpt56_thorough.yaml` | pair with `--thinking thorough` for the thorough search budget and `xhigh` reasoning effort |
| `teacher_full_no_compact.yaml` | teacher collection: postprocess and compaction off, full history preserved |
| `student_27b_vllm.yaml` | student rollout config: vLLM on localhost, 65k context, 32k output |
| `student_27b_vllm_lean.yaml` | same with the lean prompt and `fixed_output_budget: true` |
| `test40_eval_config.yaml` | frozen evaluation config: temperature 0.7, 32,768-token output cap, 131,072 context, four shards of ten targets |

For Retro-190, `run`, `evaluate`, and `check-config` default to `--thinking fast`. The overlay sets reasoning effort only. When invoking a benchmark driver directly, set `AGENT_THINKING` to the matching level alongside `AGENT_CONFIG`. Copy a preset before editing its deployment fields. The shared [deployment example](../../../deployment.example.yaml) documents those fields. Run `uv run --locked python main.py check-config --thinking LEVEL --config PATH` from the harness directory with the same level and YAML you will use for planning.
