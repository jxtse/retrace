# Chemoenzymatic retrosynthesis benchmark

A description of the chemoenzymatic benchmark the agent is evaluated on: each
target is paired with a literature-supported route that mixes chemical and
enzymatic transformations. It is compiled from published papers, and the
compiled files are **not distributed here** — they are worked answer keys. This
page records the composition, the schema and the source DOIs so the benchmark
stays citable.

## Files (not distributed with this repository)

| File | Contents |
|---|---|
| `cases_verified.json` | one object per case: target, source paper, ground-truth route, difficulty, tags |
| `targets.csv` | flat per-target index (id, name, SMILES, step counts, difficulty, tags, DOI) |
| `ground_truth_routes.jsonl` | one JSON line per case: id, target SMILES, steps, starting materials |

## Curation

1. Manual extraction from seven source papers, LLM-assisted.
2. SMILES validation with RDKit (canonicalisation plus structure checks).
3. Hallucination audit against the source text: DOI present in the references,
   target name appears in the paper, EC numbers syntactically valid, reported
   step count matches the route.
4. Only cases passing all four checks are labelled VERIFIED; the rest are
   labelled PLAUSIBLE and are not part of the evaluation set.

## Schema

### Case object

```json
{
  "id": "CE-001",
  "target_molecule": {"name": "...", "smiles": "..."},
  "source_paper": {"doi": "...", "first_author": "...", "year": 2022, "figure_or_table": "..."},
  "ground_truth_route": {
    "n_steps": 4,
    "n_enzymatic_steps": 1,
    "n_chemical_steps": 3,
    "n_transitions": 1,
    "steps": [
      {
        "step": 1,
        "type": "chemical",
        "product_smiles": "...",
        "reactant_smiles": ["...", "..."],
        "conditions": "...",
        "ec_number": null,
        "enzyme_name": null
      }
    ],
    "starting_materials": [
      {"smiles": "...", "commercially_available": true}
    ]
  },
  "difficulty": "easy",
  "tags": ["pharmaceutical", "cyclization"]
}
```

`type` is `chemical` or `enzymatic`; enzymatic steps carry `ec_number` and
`enzyme_name`. `n_transitions` counts chemical↔enzymatic boundary crossings.

### `targets.csv` columns

`id, target_name, target_smiles, n_steps, n_enzymatic_steps, n_chemical_steps, n_transitions, difficulty, tags, source_doi`

## Composition

73 VERIFIED cases.

| Difficulty | Cases |
|---|---:|
| easy | 25 |
| medium | 39 |
| hard | 9 |

| `n_enzymatic_steps` | Cases |
|---|---:|
| 0 | 6 |
| 1 | 40 |
| 2 | 9 |
| 3 | 10 |
| 4 | 1 |
| 5 | 4 |
| 6 | 1 |
| 7 | 1 |
| 9 | 1 |

## Suggested metrics

- **Solved rate** — fraction of targets where the agent returns at least one
  route reaching commercially available starting materials.
- **Route similarity** — tree-edit distance or Jaccard overlap of canonical
  reaction SMILES against the reference route.
- **Enzymatic step coverage** — fraction of reference enzymatic steps (matched by
  EC number or enzyme name) that appear in the predicted route.
- **Step-count error** — absolute difference in step counts.
- **Transition fidelity** — whether the predicted route crosses the
  chemical↔enzymatic boundary where the reference does.
- Report all of these stratified by difficulty and source paper.

## Source papers

| Short name | DOI |
|---|---|
| Levin2022 (cannabinoid chemoenzymatic syntheses) | 10.1038/s41467-022-35422-y |
| minChemBio2025 (min-cut chemo/bio retrosynthesis) | 10.1039/d3sc01355c |
| DORAnet2025 (pathway design) | 10.1039/D4SC02408G |
| TTLAB2024 (chemoenzymatic dataset) | 10.1039/d5dd00229j |
| Sankar2023 (ACS Synth. Biol.) | 10.1021/acssynbio.4c00692 |
| NatComm2025 (chemoenzymatic routes) | 10.1038/s41467-025-65898-3 |
| RetroBioCat2021 (enzymatic cascades) | 10.1038/s41929-020-00556-z |
| Other single-target cases | 10.1126/science.aay8484, 10.1126/science.1188934, 10.1038/nature06450, and others |

Each case carries its own `source_paper.doi`; cite the underlying papers when you
use this benchmark.
