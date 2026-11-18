# Third-party notices for this harness

The repository-root [`LICENSE`](../../LICENSE) (MIT) covers the
contributors' own code, prompts and documentation in this directory. It does not
relicense third-party material.

## Bundled

| Component | Path | License |
|---|---|---|
| Coley SCScore (numpy export of `model.ckpt-10654`) | `tools/verifier/scscore_coley/` | MIT, Copyright (c) 2017 Connor Coley — `scscore_coley/LICENSE` |
| RDKit `SA_Score` (Ertl & Schuffenhauer 2009) | `tools/verifier/sa_score_vendor/` | BSD-3-Clause — `sa_score_vendor/LICENSE` |

Both are used only by `score_route` for the cost and complexity sub-scores.

## Optional, not included

The reaction-condition model (ASKCOS-derived, MPL-2.0 if you add it) and the
CHEM21 solvent-class table (no explicit redistribution terms) are not shipped.
`score_route` reports them as unavailable and degrades rather than substituting
a value. `tools/condition_predictor/` is the expected location if you add the
condition model yourself.

## Not shipped

Benchmark inputs, reference routes and run outputs are not distributed here. See
the repository-root [`LICENSES.md`](../../LICENSES.md) for the full
notice list.
