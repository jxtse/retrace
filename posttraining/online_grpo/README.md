# `online_grpo/`

GRPO on replayed wire logs (ms-swift). Build-time code and a replay seam; no
training run is included.

| File | Purpose |
|---|---|
| `retrace_replay.py` | turns a wire log into a training row plus the privileged teacher view |
| `retrace_reward.py` | verdict-shaped reward plugin |
| `retrace_template.py` | replay template with manifest-checked loss masks |
| `replay_grpo_trainer.py` | trainer override: replay `_rollout_samples` |
| `run_grpo_replay.py`, `run_grpo_replay.sh`, `run_grpo_replay_v2.sh` | launchers |
| `test_spans.py` | span-mask test (needs the tokenizer) |
