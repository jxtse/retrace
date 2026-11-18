# Third-party notices

The repository-root [`LICENSE`](LICENSE) (MIT, Copyright (c) 2026 ReTRACE
contributors) covers the contributors' own code, prompts and documentation. It
does **not** relicense third-party material. This file lists every third-party
component this repository ships, references, or deliberately leaves out.

## Bundled, with the upstream license in place

| Component | Path | Origin | License |
|---|---|---|---|
| Coley SCScore (1024-bit Morgan fingerprint + MLP, numpy export) | `agent/tools/verifier/scscore_coley/` (also under `agent-chemoenzymatic/tools/verifier/` and `posttraining/eval_harness/tools/verifier/`) | https://github.com/connorcoley/scscore, weights from `model.ckpt-10654` | **MIT**, Copyright (c) 2017 Connor Coley — license text in `scscore_coley/LICENSE` |
| RDKit `SA_Score` (Ertl & Schuffenhauer 2009) | `agent/tools/verifier/sa_score_vendor/` (same three copies) | https://github.com/rdkit/rdkit, `Contrib/SA_Score` | **BSD-3-Clause**, Rational Discovery LLC / Greg Landrum and others — license text in `sa_score_vendor/LICENSE` |

Both are used only for offline scoring. The SCScore weights (`weights.json.gz`,
6 MB) and the SA_Score fragment table (`fpscores.pkl.gz`, 4 MB) are the largest
files in the repository and are included so route scoring works without a
download.

## Referenced, not vendored

| Component | How it is used | License |
|---|---|---|
| **Self-Distillation** training loop (https://github.com/idanshen/Self-Distillation) | `posttraining/stage1_critique/sdft/` provides data-format adapters around the upstream trainer; the trainer itself is not in this repository. See `posttraining/stage1_critique/sdft/UPSTREAM.md` for the commit and the one compatibility patch. | see the upstream repository; nothing from it is redistributed here |
| **Retro\*-190** benchmark (Chen et al., *Sci. Data* 2020) | The 190-target test split is what `agent/dataset/benchmark/retro-190_dataset/` evaluates. The target list and reference routes are not redistributed; obtain them from the original publications. | MIT (per the original release) |
| **ChemEnzyRetroPlanner** (Wang et al., *Nat. Commun.* 2025, doi:10.1038/s41467-025-65898-3) | The agent calls retroplanner engines over HTTP. No copy of the planner tree is shipped. | MIT |
| **ASKCOS reaction-condition model** (Gao et al. 2018) | Optional input to the greenness pillar of `score_route`. **Not included** in this repository — no code, no weights. When it is absent the pillar reports the component as unavailable and degrades instead of substituting a value. Expect MPL-2.0 (file-level copyleft) if you add it back yourself. | MPL-2.0 (if added separately) |
| **CHEM21 solvent classes** (Prat et al., *Green Chem.* 18, 288, 2016) | The solvent-class sub-score reads a local table if one is present. The published table is not redistributed. | no explicit redistribution terms |
| **Ketcher** editor and npm packages | `*/webapp/client/` is a Vite/React app; `package-lock.json` is included, `node_modules` is not. Install from npm to build it. | Apache-2.0 (Ketcher), MIT/BSD-3 for the rest |

## Data that is not distributed

Benchmark inputs, reference routes, worked answer keys, transcripts and run
outputs are not part of this repository. Dataset directories ship the drivers
and their documentation only. If you want to run a benchmark, supply the inputs
yourself:

- **Retro\*-190** — Chen et al., *Scientific Data* 7, 127 (2020), MIT; take the
  hard test split and its reference routes from the original release.
- **JMC**, **FDA small molecules**, **PKU-8**, the experimental-validation
  cases and the chemoenzymatic benchmark — obtain these from the corresponding
  publications and their authors. Route facts in published papers are the
  authors'; any compilation of them is not redistributed here.

Each `dataset/README.md` in this repository names the input format the drivers
expect (`dataset/benchmark/<suite>/cases.csv`).

## Python and Node dependencies

Dependencies are installed from PyPI/npm, not vendored. `uv.lock` in each
harness pins the Python set (RDKit BSD-3, pandas/numpy BSD-3, rdchiral MIT,
transformers Apache-2.0, pydantic MIT, molbloom Apache-2.0, pytest MIT, and so
on); consult each package for its own license.
