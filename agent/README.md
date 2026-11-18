# ReTRACE agent harness

The LLM-coordinated retrosynthesis agent: a ReACT loop that plans a route by
calling chemistry tools, scores saved routes on three criteria, and stops when a
route is complete and its leaves are buyable. This is the harness used for the
main evaluations (Retro-190, JMC, the FDA small-molecule set, PKU-8 and the
experimental-validation cases).

The harness is a client. It calls an OpenAI-compatible LLM endpoint, a
retroplanner HTTP service (template-free single-step engines and the per-step
reaction rater), optionally a remote forward-reaction model, and optionally
vendor price APIs. None of those services ship here; the shipped config points
at `*.example.org` placeholders. Live vendor pricing also needs a separate `ChemPriceClient` package that is not included; without it the cost pillar falls back to a built-in price heuristic (see the root README, "Vendor price lookup").

## Run

```bash
uv sync                                         # Python >= 3.11
cp .env.example .env                            # credentials + endpoints
uv run python main.py run "CC(=O)Oc1ccccc1C(=O)O"            # add --trace for a trace
uv run python main.py run "<SMILES>" --thinking thorough --model <model>
uv run python main.py evaluate retro190 --indices 0-4 --dry-run
uv run python main.py repl
```

`evaluate` needs benchmark inputs; see `dataset/README.md`.

## Tools the agent can call

| Group | Tools |
|---|---|
| RDKit operations | `validate_smiles`, `analyze_molecule`, `compare_molecules`, `identify_bonds` |
| Template-based single step | `template_library`, `suggest_templates`, `apply_template` (rdchiral over a curated USPTO template set) |
| Template-free single step | `neural_retro`, `retro_decompose` (GraphFP / RxNGormer / ChemBART / ring-breaker through the retroplanner service) |
| Purchasability | `check_buyability` |
| Route memory | `recall_similar_routes` (precedent database), `load_result` |
| Ranking and scoring | `rank_routes`, `score_route` (feasibility, cost, greenness) |
| Working memory | `scratchpad`, `todo`, `read_doc` |

## Thinking presets

`--thinking fast|standard|thorough`, built into `core/config.py`.

| Preset | reasoning | max_iter | stop_after | auto_finish | wallclock | scratchpad |
|---|---|---:|---:|---:|---:|---:|
| `fast` | medium | 50 | 10 | 20 | 4 h | 80 |
| `standard` | high | 100 | 10 | 30 | 6 h | 120 |
| `thorough` | xhigh | 200 | 20 | 40 | 10 h | 200 |

Small YAML overlays via `--config path` are layered on top of the preset; unknown
keys are rejected at startup.

## Python API

```python
from core import build_agent
from core.config import load_config

config = load_config(thinking="fast")
agent = build_agent(config)
result = agent.run("CC(=O)Oc1ccccc1C(=O)O")
print(result["success"], result["best_route"])
```

## Layout

| Path | Purpose |
|---|---|
| `main.py` | CLI entry: `run`, `evaluate <suite>`, `repl` |
| `core/agent.py` | `RetrosynthesisAgent` ReACT loop |
| `core/agent_factory.py` | `build_agent()` and tool wiring |
| `core/config.py` | Base config, thinking presets, overlay validation |
| `core/evaluation.py` | Benchmark suite registry and driver delegation |
| `core/llm_client.py` | OpenAI-compatible client (chat-completions and messages protocols, retries, fallback) |
| `tools/` | LLM-callable chemistry tools and the multi-criterion `score_route` verifier |
| `services/` | Concurrency, context compaction, result store, transcript writer, streaming |
| `prompts/` | System prompt, JIT-loaded sections and nudge messages |
| `dataset/` | Benchmark drivers and documentation (inputs are not distributed) |
| `scripts/` | Evaluation helpers, route export, transcript visualisation |
| `tests/` | Offline unit tests |
| `docs/` | API guide and tool-description conventions |
| `webapp/` | Local route-review UI (Ketcher-based) |

## Scoring and optional assets

`score_route` combines three pillars: feasibility (reaction rater + SMILES
sanity + carbon-skeleton and stereo checks), cost (step efficiency, leaf
economics, synthetic complexity) and greenness (atom economy, hazardous-reagent
scan, optional solvent class and predicted temperature). Feasibility requires
`REACTION_RATER_URL`. The reaction-condition model and the CHEM21 solvent table
are optional and are not included: when absent, the pillar reports the missing
component in its summary and scores around it rather than substituting a value.
SCScore weights and the RDKit `SA_Score` data are bundled so cost and complexity
scoring work offline.

## Tests

```bash
uv run pytest -q
```

No network, no GPU, no model inference: HTTP paths are exercised with fake
clients, and any test that would need a live service is skipped or mocked.

## License

Contributors' code and docs are MIT (repository root `LICENSE`). Third-party
components keep their own licenses: `tools/verifier/scscore_coley/LICENSE` (MIT)
and `tools/verifier/sa_score_vendor/LICENSE` (BSD-3-Clause). See
[`../LICENSES.md`](../LICENSES.md).

## Sister directories

`../agent-chemoenzymatic/` adds the enzymatic extension to this harness, and
`../posttraining/eval_harness/` is the variant used to evaluate post-trained
checkpoints.
