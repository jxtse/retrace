# ReTRACE

[Paper (arXiv:2610.24744)](https://arxiv.org/abs/2610.24744) · [Model](https://huggingface.co/jxtse/ReTRACE-27B-latest) · [Training data](https://huggingface.co/datasets/jxtse/ReTRACE-train-data)

ReTRACE is an LLM-coordinated retrosynthesis agent with a chemistry tool layer around it. A coordinator model plans in a ReACT loop and calls tools instead of answering from memory: RDKit operations (parse/validate, analyse, compare, identify disconnections), template-based single-step expansion over a curated USPTO reaction-template library, template-free single-step prediction through an external retroplanner HTTP service (for example GraphFP, RXNGraphormer, ChemBART and a ring-breaker model), purchasability lookup for proposed leaves, a route memory of precedent reactions, and a multi-criterion route scorer that rates feasibility, cost and greenness. Saved routes can be checked with the separate route-audit pipeline in `posttraining/chemistry_audit/`. The repository also contains the post-training code used to train the coordinator: critique-oriented SFT, teacher trajectories, self-refinement, teacher SFT, and on-policy self-distillation (OPSD) and GRPO recipes, plus the evaluation harness and metric code used to compare them.

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
| `posttraining/online_opsd/` | On-policy self-distillation (OPSD): the frozen base model, given an audited route in context, teaches the current student on the student's own rollouts. | You want the OPSD comparison arm. |
| `posttraining/online_grpo/` | GRPO replay trainer: replay records, reward, templates, span tests. | You want the RL arm. |
| `posttraining/evaluation_test40/` | Per-arm evaluation launchers (plan/execute/scan/snapshot/stop) for the held-out target set. | You are launching or auditing an evaluation run. |
| `posttraining/chemistry_audit/` | The route audit pipeline: deterministic RDKit checks, LLM deep read, adversarial defence, validators, grounding. | You want to audit saved routes or reproduce route verdicts. |
| `posttraining/metrics/` | Metric computation: audit@k, completion, McNemar, bootstrap and Clopper–Pearson intervals. | You have run archives and want the metric table. |

Each directory in the table has its own `README.md` with the details.

The repository contains code only: no benchmark inputs, reference routes, run outputs or result tables. `dataset/` ships the benchmark drivers and says where to obtain the inputs; `LICENSES.md` lists what is deliberately not redistributed.

## Configure a deployment

Use Python 3.11 or later and uv. Each harness has its own lockfile. From a clean checkout:

```bash
git clone https://github.com/jxtse/retrace.git
cd retrace/agent
uv sync --locked
cp ../deployment.example.yaml deployment.local.yaml
export AGENT_CONFIG=deployment.local.yaml
```

Edit the copy. Set `llm.base_url` to your authorized gateway's API base and `llm.model` to its served model ID. Supply full POST URLs for the selected single-step engines and the Reaction Rater. The built-in reviewer and feasibility judge request their own configured models at the coordinator endpoint unless you set their respective `base_url`. Use endpoint URLs without embedded credentials. Inject `LLM_API_KEY` from your secret manager if your gateway needs authentication. Alternatively, copy `.env.example` to `.env` in the harness working directory and fill it locally. Process environment values take precedence over that file.

```bash
uv run --locked python main.py check-config --config deployment.local.yaml
```

The unfilled example exits with status 2 and lists missing fields. The check validates configuration only. It does not open a service connection, load stock files, or establish model access.

| Integration | What ships | What to supply |
|---|---|---|
| Coordinator, reviewer and judges | Client, prompts and experiment presets | Authorized model serving. The adapter and its required base model are listed below. |
| GraphFP, RXNGraphormer, ChemBART and ring-breaker | Single-step HTTP clients | Compatible service deployment and its model assets. GraphFP and RXNGraphormer use the shared URL list. |
| Reaxys template relevance | Optional HTTP client | Select `template_relevance` in `neural_retro.engines` and set `template_relevance_url` to an authorized service. Arrange access to its data separately. |
| Feasibility | Reaction Rater client and scoring code | `verifier.feasibility.rater.http_url` (one URL or an ordered list) for the enabled verifier. |
| Stock and cost | Buyability readers, local cost heuristic, SCScore and SA_Score | Your BioNav and ZINC stock CSVs with a SMILES column. Optional Enamine index, buyability database and legacy CSV directory are under `verifier.cost`. |
| Live vendor prices | Pricing adapter | A compatible `backend.clients.chemprice_client.ChemPriceClient` package in `verifier.cost.chemprice.verifier_repo` and authorized MolPort or ChemSpace credentials (`MOLPORT_API_KEY`, `CHEMSPACE_API_KEY`). The shipped pricing adapter disables MCule. Without the client the cost pillar uses its local fallback. |
| Templates and precedents | Template tools and precedent-index builder | Your compatible curated template library and, for recall, a prepared precedent database. `USPTO_ROUTES_PKL` / `USPTO_ROUTES_HDF5` select input to the index builder. |
| Greenness | Scoring code | ASKCOS condition predictor package/weights and CHEM21 solvent classes where available. Missing inputs reduce scoring coverage. |
| Benchmarks | Drivers and metric code | Target lists and reference materials obtained under the source terms. See [input layouts](agent/dataset/README.md) and [third-party notices](LICENSES.md). |

**Resolution rules.** A relative YAML filename and relative stock or asset paths resolve from the selected harness directory. Absolute stock paths work directly. For older stock overlays only, an existing file under `vendor/legacy_retro_planner/` is used when the harness-relative file is absent.

- `--config` selects the YAML, otherwise `AGENT_CONFIG` does. The loader merges built-in settings, the selected `--thinking` budget, the YAML and then `--model`. Benchmark workers accept `AGENT_MODEL`; the review UI applies its selected model-family preset.
- For shared single-step clients the order is `RETROPLANNER_HTTP_URLS`, `RETROPLANNER_HTTP_URL`, YAML `retroplanner_http_urls`, then `retroplanner_http_url`. Environment lists accept commas or whitespace. YAML lists retain order and may specify parallel first-attempt weights. Environment overrides use equal first-attempt weights.
- For ChemBART and ring-breaker, `CHEMBART_API_URL` and `RINGBREAKER_API_URL` override their dedicated YAML URLs. Each uses the shared service if its dedicated URL is unset. `REACTION_RATER_URL` overrides the rater configuration with a single URL. `RETRO_TEMPLATE_DATASET_DIR` overrides the template directory. An enabled precursor sanity judge honors `SANITY_JUDGE_BASE_URL`, `SANITY_JUDGE_MODEL` and `SANITY_JUDGE_EFFORT` before its YAML options.
- `LLM_API_KEY` authenticates configured LLM clients. Without it, `OPENROUTER_API_KEY` and `OPENAI_API_KEY` apply only to their respective public HTTPS API origins. Configure the final API base URL for each LLM endpoint. LLM clients reject redirects. The optional Anthropic-protocol fallback is disabled by default. Enabling it selects OpenRouter unless you configure a different fallback URL. Use `OPENROUTER_API_KEY` for OpenRouter or `LLM_FALLBACK_API_KEY` for your custom fallback.
- Disable a whole integration with `neural_retro.enabled: false` or `verifier.enabled: false` only when you intend to omit those tools. Use YAML booleans. For a study, copy the actual experiment preset and edit its deployment fields. Preserve its model IDs, tool set, thresholds, weights and budgets.

## Capability smoke test

After the check passes and the configured services and datasets are authorized and available, try the aspirin example from this harness:

```bash
uv run --locked python main.py run "CC(=O)Oc1ccccc1C(=O)O" --config deployment.local.yaml --trace
```

This single-target run exercises your deployment. Paper comparisons also depend on the exact preset, checkpoint, service versions, stocks, templates, scoring assets, benchmark inputs and evaluation protocol. Record any disabled tool or missing asset alongside results.

The same YAML fields work in `agent-chemoenzymatic/` and `posttraining/eval_harness/`. Set `AGENT_CONFIG` for benchmark workers and the two review UIs. For distributed evaluation, the coordinator validates the selected YAML and its schema, then pins the resolved model in the plan and each worker command. Model precedence is `--model`, then the coordinator's `AGENT_MODEL`, then the YAML model. Each worker reads the YAML at the same absolute path and performs full deployment preflight using its own environment before running cases. Supply credentials and service overrides such as `REACTION_RATER_URL` on the workers. Distributed evaluation requires explicit `--hosts`; use `--local` for a local benchmark. The post-training collection, cluster launch and held-out evaluation scripts outside these harnesses encode experiment-specific provisioning and model choices. They require separate site configuration before use.

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
| comparison arms | `posttraining/teacher_sft/`, `posttraining/online_opsd/`, `posttraining/online_grpo/` | teacher-SFT, OPSD and GRPO adapters |
| evaluation | `posttraining/evaluation_test40/`, `posttraining/chemistry_audit/`, `posttraining/metrics/` | run archives, route verdicts, metric tables |

Entry points are documented in `posttraining/README.md`.

## Model and training data

| Resource | Hugging Face | Notes |
|---|---|---|
| Trained adapter | [`jxtse/ReTRACE-27B-latest`](https://huggingface.co/jxtse/ReTRACE-27B-latest) | LoRA adapter for [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), Apache-2.0. The version evaluated in the paper is tag [`s3-v1`](https://huggingface.co/jxtse/ReTRACE-27B-latest/tree/s3-v1); `main` tracks later versions. |
| Post-training data | [`jxtse/ReTRACE-train-data`](https://huggingface.co/datasets/jxtse/ReTRACE-train-data) | The three curriculum stages used to train the adapter: `stage1_critique` (819 rows), `stage2_teacher_trajectories` (167 rows, also used for the teacher-SFT arm) and `stage3_self_refinement` (26 rows, pretokenized). Released for non-commercial research use; teacher-generated content remains subject to its providers' terms (see the dataset card). |

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
