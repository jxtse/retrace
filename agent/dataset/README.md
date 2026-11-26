# Benchmark drivers

This directory contains the benchmark **drivers**, not the benchmarks. Target
lists, reference routes and worked answer keys are not distributed with this
repository: they are other people's published material or unpublished cases, and
shipping them would hand out the answers to the tasks the harness is meant to
plan.

Supply your own inputs and the drivers work unchanged.

## Drivers and the input each one expects

| Driver | Input file it reads | Columns used |
|---|---|---|
| `benchmark/retro-190_dataset/run_benchmark.py` | `benchmark/retro-190_dataset/cases.csv` | `idx`, `smiles`, `n_steps` |
| `benchmark/hard_mol/run_benchmark.py` | `benchmark/hard_mol/cases.csv` | `idx`, `smiles`, `ref_step`, `name`, `note`, `source` |
| `benchmark/JMC/run_benchmark.py` | `benchmark/JMC/cases.csv` | `idx`, `smiles`, `step`, `name`, `paper_id` |
| `benchmark/FDA_small_mol/run_benchmark.py` | `benchmark/FDA_small_mol/2025FDA小分子汇总.csv` (GBK) | `idx`, `Name`, `SMILES`, `SMILES_canonical`, `Step`, … |
| `retro-190_dataset/run_benchmark.py` | `retro-190_dataset/cases.csv` | legacy copy of the Retro-190 runner, kept for older scripts |

Each driver writes `runs/{idx:03d}.json`, `transcripts/{idx:03d}.txt` and
`logs/{idx:03d}.log` under its own directory (paths are relative to the driver, so
the tree stays portable). When an input file is missing the driver stops with a
`FileNotFoundError` naming the path; nothing is substituted.

`main.py evaluate <suite>` dispatches to these drivers:

```bash
uv run python main.py evaluate retro190 --indices 0-4 --local
uv run python main.py evaluate fda --indices 0-4 --local --variants stereo,canonical
uv run python main.py evaluate jmc --local
uv run python main.py evaluate hard-mol --local
```

For distributed evaluation, supply `--hosts`. The YAML selected by `--config` or `AGENT_CONFIG` must be readable at the same absolute path on every worker.
The distributor may probe those hosts even with `--dry-run`. Use `main.py check-config` for offline validation.

## Where the inputs come from

- **Retro\*-190** — Chen et al., *Scientific Data* 7, 127 (2020), MIT. Take the
  hard test split (`routes_possible_test_hard`) from the original release; the
  driver's `cases.csv` is a flat index over it (`idx`, `smiles`, `n_steps`).
- **JMC** — the target set and route-evaluation workbook come from the JMC
  publication; the driver expects a CSV export of the workbook's per-target
  sheet.
- **FDA small molecules** — a table of approved small-molecule drugs; the driver
  expects the CSV layout above. Compile it from FDA approval documents or your
  own source.
- **PKU-8 / experimental validation / chemoenzymatic cases** — obtain these from
  the corresponding publications and their authors.
- **USPTO template library and precedent routes** — the template-based tools read
  `RETRO_TEMPLATE_DATASET_DIR`; the precedent database reads
  `USPTO_ROUTES_PKL` / `USPTO_ROUTES_HDF5`. Both are optional: without them the
  affected tools report the dataset as unavailable.

## What still works without any of this

- Importing the package and running the offline tests.
- Anything that takes an explicit SMILES or route as an argument (`main.py run`,
  the REPL, individual tools).
- Route scoring, as long as the reaction rater endpoint is reachable.
