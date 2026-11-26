# Evaluation harness for post-trained checkpoints

The harness variant used to evaluate the post-trained models on the held-out
target set. It is the chemoenzymatic harness (`../../agent-chemoenzymatic/`) with
one addition: a **lean prompt profile** for student models, plus the configs used
to serve them.

What "lean" means here:

- `prompts/core_lean.md` replaces `prompts/core.md` — same tools and rules,
  written for a model that gets a fixed per-turn output budget.
- The opt-in `agent` options (`tools_allowlist`, `target_prefetch`,
  `mini_status_every`, `iteration_mode`, `user_nudges_enabled`,
  `complete_grace_turns`) are validated separately from the frozen services
  schema, so an evaluation config can change loop behaviour without touching the
  tool layer.
- `configs/student_27b_vllm*.yaml` serve a 27B student through vLLM with an
  explicit `fixed_output_budget`, a 65k context and the reaction rater on
  localhost. `configs/test40_eval_config.yaml` is the frozen evaluation config:
  temperature 0.7, `max_tokens` 32768, 131k context, `reasoning_effort: medium`,
  four shards of ten targets.

## Run

Follow the [deployment guide](../../README.md#configure-a-deployment) to install the locked environment, copy the shared example and supply service URLs, credentials and data paths. From this harness directory:

```bash
export AGENT_CONFIG=deployment.local.yaml
uv run --locked python main.py check-config
# After configuring authorized services:
uv run --locked python main.py run "CC(=O)Oc1ccccc1C(=O)O" --trace
```

The harness is a client: the LLM endpoint (the served student, or any
OpenAI-compatible endpoint) and the retroplanner service are not part of this
repository. The student and test40 presets select explicit local deployments.

## Layout

Same layout as `../../agent/` (without the `webapp/` review UI), plus:

| Path | Purpose |
|---|---|
| `prompts/core_lean.md` | lean system prompt for student models |
| `configs/student_27b_vllm.yaml` | student rollout config (single-target smoke) |
| `configs/student_27b_vllm_lean.yaml` | same, with the lean prompt and fixed output budget |
| `configs/test40_eval_config.yaml` | frozen evaluation config (temperature, budgets, shards) |

The launchers that drive many targets per arm live in `../evaluation_test40/`;
the audit and metric code lives in `../chemistry_audit/` and `../metrics/`.

## Tests

```bash
uv run pytest -q
```

Offline only. `tests/test_lean_profile.py` compares the lean config against the
base config and pins exactly which keys may differ; `fixed_output_budget` is one
of them (the lean profile must hold the output budget constant, so every attempt
sends an identical sampling request).

## License

Contributors' code and docs are MIT (repository root `LICENSE`). Third-party
components keep their own licenses — see
[`../../LICENSES.md`](../../LICENSES.md), including the bundled
`tools/verifier/scscore_coley/LICENSE` (MIT) and
`tools/verifier/sa_score_vendor/LICENSE` (BSD-3-Clause).
