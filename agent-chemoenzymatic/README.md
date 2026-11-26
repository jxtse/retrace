# ReTRACE agent harness — chemoenzymatic extension

`agent/` plus the chemo-enzymatic extension. If you want mixed organic/enzymatic
routes, use this harness; if you only plan organic disconnections, `agent/`
already does that and this directory adds tools you will not call.

What the extension adds on top of `agent/`:

- **`enzymatic_retro`** — an explicit enzymatic single-step tool that narrows
  `suggest_templates` to the enzymatic part of the template library and returns
  the template source, template kind and reactants. It searches the curated
  library; it does not invent an enzyme identity.
- **Enzymatic branch nudges** — three prompt messages
  (`nudge_enzymatic_branch.md`, `nudge_enzymatic_target_branch.md`,
  `nudge_enzymatic_route_branch.md`) that the loop injects when the target or a
  saved route contains a plausible upstream intermediate that has not been
  queried yet, so enzyme search is not limited to the final target.
- **Per-step template provenance** — each saved step records whether it came
  from an organic or enzymatic template and which template produced it, so a
  mixed route keeps an auditable trail.

The harness is a client: it calls an OpenAI-compatible LLM endpoint and a
retroplanner HTTP service. Supply both services for your deployment.

## Run

Follow the [deployment guide](../README.md#configure-a-deployment) to install the locked environment, copy the shared example and supply service URLs, credentials and data paths. From this harness directory:

```bash
export AGENT_CONFIG=deployment.local.yaml
uv run --locked python main.py check-config
# After configuring authorized services:
uv run --locked python main.py run "CC(=O)Oc1ccccc1C(=O)O" --trace
```

The enzymatic library needs a local template set (`RETRO_TEMPLATE_DATASET_DIR`);
without it the enzymatic tool reports the library as unavailable, and the
organic tools still work.

## Thinking presets

`--thinking fast|standard|thorough`, built into `core/config.py`.

| Preset | reasoning | max_iter | stop_after | auto_finish | wallclock | scratchpad |
|---|---|---:|---:|---:|---:|---:|
| `fast` | medium | 50 | 10 | 20 | 4 h | 80 |
| `standard` | high | 100 | 10 | 30 | 6 h | 120 |
| `thorough` | xhigh | 200 | 20 | 40 | 10 h | 200 |

YAML overlays via `--config path` are layered on the preset; unknown keys are
rejected at startup.

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

Same as `agent/` (`main.py`, `core/`, `tools/`, `services/`, `prompts/`,
`dataset/`, `scripts/`, `tests/`, `docs/`, `webapp/`), with `tools/builtins/
enzymatic_retro.py` and the enzymatic prompt messages added. `core/agent.py`
carries the branch-checkpoint logic that decides when to nudge.

## Tests

```bash
uv run pytest -q
```

Offline only: no network, no GPU, no model inference.

## License

Contributors' code and docs are MIT (repository root `LICENSE`). Third-party
components keep their own licenses — see
[`../LICENSES.md`](../LICENSES.md) and the bundled
`tools/verifier/scscore_coley/LICENSE` (MIT) and
`tools/verifier/sa_score_vendor/LICENSE` (BSD-3-Clause).
