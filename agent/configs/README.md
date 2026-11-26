# `configs/`

YAML overlays layered on the built-in preset (`main.py run ... --config
configs/<file>.yaml`, or `AGENT_CONFIG=...` for the benchmark drivers).

| File | Purpose |
|---|---|
| `qwen36_vllm.yaml` | serve a Qwen3.6-35B-A3B base model from a local vLLM: 65k context, 8k output budget, relaxed tool timeout, tuned compaction |

These files select experiment settings. Copy a preset before editing its deployment fields. The shared [deployment example](../../deployment.example.yaml) documents those fields. Run `uv run --locked python main.py check-config --config PATH` from the harness directory before planning.
