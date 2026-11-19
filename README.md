# ReTRACE

[Paper (arXiv:2610.24744)](https://arxiv.org/abs/2610.24744) · [Model](https://huggingface.co/jxtse/ReTRACE-27B-latest) · [Training data](https://huggingface.co/datasets/jxtse/ReTRACE-train-data)

ReTRACE is an LLM-coordinated retrosynthesis agent with a chemistry tool layer around it. A coordinator model plans in a ReACT loop and calls tools instead of answering from memory: RDKit operations (parse/validate, analyse, compare, identify disconnections), template-based single-step expansion over a curated USPTO reaction-template library, template-free single-step prediction through an external retroplanner HTTP service (for example GraphFP, RXNGraphormer, ChemBART and a ring-breaker model), purchasability lookup for proposed leaves, a route memory of precedent reactions, and a multi-criterion route scorer that rates feasibility, cost and greenness. Saved routes can be checked with the separate route-audit pipeline in `posttraining/chemistry_audit/`. The repository also contains the post-training code used to train the coordinator: critique-oriented SFT, teacher trajectories, self-refinement, teacher SFT, and online self-distillation / GRPO recipes, plus the evaluation harness and metric code used to compare them.

## Directories

| Path | What it is | Use it when |
|---|---|---|
| `agent/` | The agent harness used for the paper's main experiments: CLI, ReACT loop, chemistry tools, route scorer, prompt set, local review UI. | You want to run the planner end-to-end or read/cite the plain harness. |
| `agent-chemoenzymatic/` | The same harness with the chemo-enzymatic extension: an `enzymatic_retro` tool, enzyme-branch nudges, per-step template provenance. | You want mixed organic/enzymatic routes, or you want to see what the extension adds. |
| `posttraining/eval_harness/` | Harness variant used to evaluate post-trained checkpoints: lean prompt profile, student vLLM configs, same tools and scorer. | You want to reproduce the evaluation protocol used for the trained models. |
| `posttraining/stage1_critique/` | Stage 1: critique/rejection-sampling data build, chemistry screening, self-distillation (SDFT) adapters, training scripts. | You are building or retraining the first post-training stage. |
| `posttraining/stage2_teacher_trajectories/` | Stage 2: teacher trajectory collection, segment selection, assembly, and the teacher-trajectory SFT recipe. | You are building the teacher-trajectory stage. |
| `posttraining/stage3_self_refinement/` | Stage 3: student self-sampling on train targets, admission, screening, runtime contracts, training and adapter export. | You are running the self-refinement stage. |
| `posttraining/teacher_sft/` | Direct teacher-trajectory SFT baseline (no stage-1 adapter). | You need the teacher-SFT comparison arm. |
| `posttraining/online_opsd/` | Online on-policy self-distillation with a frozen teacher. | You want the online distillation objective. |
| `posttraining/online_grpo/` | GRPO replay trainer: replay records, reward, templates, span tests. | You want the RL arm. |
| `posttraining/evaluation_test40/` | Per-arm evaluation launchers (plan/execute/scan/snapshot/stop) for the held-out target set. | You are launching or auditing an evaluation run. |
| `posttraining/chemistry_audit/` | The route audit pipeline: deterministic RDKit checks, LLM deep read, adversarial defence, validators, grounding. | You want to audit saved routes or reproduce route verdicts. |
| `posttraining/metrics/` | Metric computation: audit@k, completion, McNemar, bootstrap and Clopper–Pearson intervals. | You have run archives and want the metric table. |

Each directory in the table has its own `README.md` with the details.

The repository contains code only: no benchmark inputs, reference routes, run outputs or result tables. `dataset/` ships the benchmark drivers and says where to obtain the inputs; `LICENSES.md` lists what is deliberately not redistributed.

## Installation

Python ≥ 3.11. Each harness is a self-contained `uv` project:

```bash
cd agent
uv sync                 # creates .venv from uv.lock
cp .env.example .env    # fill in the endpoints you have
```

`agent-chemoenzymatic/` and `posttraining/eval_harness/` are separate projects with the same layout; `uv sync` in each. `posttraining/` recipes are shell/Python drivers that expect their own environment (ms-swift, TRL or vLLM on the training host) — see `posttraining/README.md`.

## Configuration

Credentials and service endpoints come from the environment (optionally a `.env` file next to the harness):

| Variable | Needed for |
|---|---|
| `LLM_API_KEY` (+ optional `OPENROUTER_API_KEY` fallback) | Planning. The base URL and model live in the agent config (`configs/*.yaml` or `AGENT_CONFIG`). |
| `RETROPLANNER_HTTP_URL` | Template-free single-step engines. Overrides the `*.example.org` placeholders in the shipped config. |
| `CHEMBART_API_URL`, `RINGBREAKER_API_URL` | Per-engine overrides when the engines are deployed separately. |
| `REACTION_RATER_URL` | `score_route`; the feasibility pillar refuses to construct without it. |
| `RETRO_TEMPLATE_DATASET_DIR`, `USPTO_ROUTES_PKL`/`USPTO_ROUTES_HDF5` | Template library and precedent route memory, if you have those datasets. |
| `MCULE_API_KEY`, `MOLPORT_API_KEY`, `CHEMSPACE_API_KEY` | Optional vendor price lookups for the cost pillar. These also require a separate price-lookup client that is not included in this repository (see below). |

None of these services is part of the repository. Without them the package imports and the offline tests pass, but planning cannot run.

**Vendor price lookup.** `tools/verifier/pricing.py` imports `backend.clients.chemprice_client.ChemPriceClient` from a separate verifier package located by `scoring.cost.chemprice.verifier_repo` in the agent config. That package is not released here. Without it, the vendor keys have no effect: the cost pillar logs `ChemPriceClient init failed` and uses the built-in molecular-weight/complexity price heuristic for every leaf. To use live vendor prices, provide a module exposing a `ChemPriceClient` with the same `status()` and lookup interface and point `verifier_repo` at it.

## Usage

```bash
cd agent
uv run python main.py run "CC(=O)Oc1ccccc1C(=O)O"            # one target
uv run python main.py run "<SMILES>" --thinking thorough    # longer search budget
uv run python main.py run "<SMILES>" --trace                # print the step trace
uv run python main.py repl                                  # interactive
```

Python API:

```python
from core import build_agent
from core.config import load_config

agent = build_agent(load_config(thinking="fast"))
result = agent.run("CC(=O)Oc1ccccc1C(=O)O")
print(result["best_route"])
```

Benchmark drivers live in `dataset/benchmark/<suite>/run_benchmark.py` and read a `cases.csv` you supply; `main.py evaluate <suite>` dispatches to them.

## Offline tests

```bash
cd agent && uv run pytest -q
cd ../agent-chemoenzymatic && uv run pytest -q
cd ../posttraining/eval_harness && uv run pytest -q
```

These tests use no network, no GPU and no model inference. Anything needing a live endpoint is exercised through fake HTTP clients.

## Post-training

The stages map to directories as follows:

| Stage | Code | Output |
|---|---|---|
| 1. critique / rejection sampling | `posttraining/stage1_critique/` | an S1 adapter (LoRA) |
| 2. teacher trajectories | `posttraining/stage2_teacher_trajectories/` | filtered teacher-trajectory SFT rows and a stage-2 adapter |
| 3. self-refinement | `posttraining/stage3_self_refinement/` | student self-sampled rows, admission, S3 adapter |
| comparison arms | `posttraining/teacher_sft/`, `posttraining/online_opsd/`, `posttraining/online_grpo/` | teacher-SFT, online distillation and GRPO adapters |
| evaluation | `posttraining/evaluation_test40/`, `posttraining/chemistry_audit/`, `posttraining/metrics/` | run archives, route verdicts, metric tables |

Entry points are documented in `posttraining/README.md`.

## Model and training data

| Resource | Hugging Face | Notes |
|---|---|---|
| Trained adapter | [`jxtse/ReTRACE-27B-latest`](https://huggingface.co/jxtse/ReTRACE-27B-latest) | LoRA adapter for [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), Apache-2.0. The version evaluated in the paper is tag [`s3-v1`](https://huggingface.co/jxtse/ReTRACE-27B-latest/tree/s3-v1); `main` tracks later versions. |
| Post-training data | [`jxtse/ReTRACE-train-data`](https://huggingface.co/datasets/jxtse/ReTRACE-train-data) | The three curriculum stages used to train the adapter: `stage1_critique` (819 rows), `stage2_teacher_trajectories` (167 rows, also used for the teacher-SFT arm) and `stage3_self_refinement` (26 rows, pretokenized). Access is currently restricted while third-party terms for the teacher-generated content are reviewed. |

The adapter is not a standalone model; load it on top of the base model. The training data are consumed by the stage scripts in `posttraining/`.

## Citation

If you use ReTRACE, please cite:

```bibtex
@misc{xie2026retrace,
  title         = {ReTRACE: Feedback-Guided LLM Agent in a Chemistry Harness for Retrosynthesis Planning},
  author        = {Jinxiang Xie and Yijian Zhang and Qingyuan Yi and Jianyi Wang and Hao Dong},
  year          = {2026},
  eprint        = {2610.24744},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2610.24744}
}
```

## License

The repository-root `LICENSE` (MIT, Copyright (c) 2026 ReTRACE contributors) covers the contributors' own code, prompts and documentation. Third-party components keep their own licenses: SCScore (MIT) and RDKit `SA_Score` (BSD-3-Clause) ship with their upstream license files, and the Self-Distillation training loop is referenced rather than vendored. See `LICENSES.md` for the full notice list, including material that is deliberately not redistributed.
