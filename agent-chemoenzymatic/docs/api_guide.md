# Retroplanner service API

The harness is a client of a retroplanner HTTP service. This page documents the
endpoints it calls, their request shapes and responses. The service itself is not
part of this repository; the base URL in the shipped config is a placeholder
(`http://retro-fallback.example.org:8001`), overridden by
`RETROPLANNER_HTTP_URL`, `CHEMBART_API_URL` or `RINGBREAKER_API_URL`.

The service implements the model set of ChemEnzyRetroPlanner (Wang et al.,
*Nat. Commun.* 16, 10929, 2025; MIT).

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/retroplanner/api/single_step` | POST | one-step retrosynthesis prediction |
| `/retroplanner/api/retroplanner` | POST | multi-step planning (async job) |
| `/retroplanner/api/retroplanner_results` | POST | poll a multi-step job |
| `/retroplanner/api/condition_predictor` | POST | reaction conditions (catalyst, solvent, temperature) |
| `/retroplanner/api/reaction_rater` | POST | forward-reaction feasibility score |
| `/retroplanner/api/enzymatic_rxn_identifier` | POST | is this reaction enzymatic? |
| `/retroplanner/api/enzyme_recommender` | POST | suggest EC numbers |
| `/retroplanner/api/easifa` | POST | enzyme active-site identification |

## 1. `single_step`

### Request

```json
{
  "smiles": "O=C(c1ccccc1)c1ccccc1",
  "savedOptions": {"topk": 10, "oneStepModel": ["RxNGormer"]}
}
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `smiles` | string | yes | target molecule (a single molecule) |
| `savedOptions.topk` | int | no | results per model (default 50) |
| `savedOptions.oneStepModel` | string[] | no | which models to use; omitted means all |

### Response

```json
{
  "status": "success",
  "message": "Data received",
  "one_step_results": {
    "reactants": ["OC(c1ccccc1)c1ccccc1", "O=C(Cl)c1ccccc1.c1ccccc1", "O.O=C(c1ccccc1)c1ccccc1"],
    "scores": [-0.4507, -1.0150, -9.5235],
    "costs": [6.9078, 6.9078, 6.9078],
    "template": ["[RxNGormer]", "[RxNGormer]", "[RxNGormer]"],
    "model_full_name": ["rxngraphormer_models", "rxngraphormer_models", "rxngraphormer_models"],
    "weight": [1.0, 1.0, 1.0]
  }
}
```

| Field | Meaning |
|---|---|
| `one_step_results.reactants` | predicted precursors (dot-separated SMILES for multi-component reactions) |
| `one_step_results.scores` | model confidence, higher is better |
| `one_step_results.costs` | search cost, lower is better |
| `one_step_results.template` | template or model tag that produced the prediction |
| `one_step_results.model_full_name` | internal model identifier |
| `one_step_results.weight` | ensemble weight (currently 1.0) |

### Models

| Display name | Internal id | Type | Trained on |
|---|---|---|---|
| `RxNGormer` | `rxngraphormer_models` | graph Transformer | USPTO-50k |
| `GraphFP (USPTO-Full)` | `graphfp_models.USPTO-full_remapped` | graph neural network (DMPNN) | USPTO-Full |
| `ChemBart (Full)` | `chembart_models.chembart_full` | BART chemical language model | USPTO-Full |
| `Transformer (USPTO-NPL+BioChem)` | `onmt_models.bionav_one_step` | Transformer seq2seq | USPTO-NPL + BioChem |
| `Pistachio` | `template_relevance.pistachio` | template matching | Pistachio patents |
| `Pistachio Ringbreaker` | `template_relevance.pistachio_ringbreaker` | template matching | Pistachio ring-opening reactions |
| `Reaxys` | `template_relevance.reaxys` | template matching | Reaxys |
| `Reaxys Biocatalysis` | `template_relevance.reaxys_biocatalysis` | template matching | Reaxys biocatalysis |
| `BKMS Metabolic` | `template_relevance.bkms_metabolic` | template matching | BKMS metabolite reactions |

Notes: `RxNGormer` loads a ~218 MB checkpoint on first call and runs on GPU
(2–3 s per molecule). The template-matching models need no GPU and are fast, but
only cover known templates. `ChemBart` may be deployed as a separate service; if
so, point `CHEMBART_API_URL` at it.

## 2. Multi-step planning

### Submit

```json
{
  "smiles": "O=C(c1ccccc1)c1ccccc1",
  "savedOptions": {
    "topk": 10,
    "iterations": 10,
    "max_depth": 6,
    "expansion_topk": 50,
    "oneStepModel": ["GraphFP (USPTO-Full)", "RxNGormer", "ChemBart (Full)"],
    "stocks": ["Zinc_Fix-stock"],
    "condition": ["rcr"],
    "enzymatic": true
  }
}
```

| Field | Type | Meaning |
|---|---|---|
| `savedOptions.iterations` | int | MCTS iterations (default 10) |
| `savedOptions.max_depth` | int | maximum search depth (default 6) |
| `savedOptions.expansion_topk` | int | expansions per step (default 50) |
| `savedOptions.stocks` | string[] | building-block stocks to use |
| `savedOptions.condition` | string[] | condition model, e.g. `["rcr"]` |
| `savedOptions.enzymatic` | bool | allow enzymatic steps (default true) |
| `savedOptions.reaction_feasibility_filter` | bool | filter by the reaction rater (default true) |
| `savedOptions.enzyme_assign` | bool | assign EC numbers |
| `savedOptions.rxn_classification` | bool | classify reactions |

Response:

```json
{"status": "success", "message": "Data received", "results_id": "uuid-task-id-here"}
```

### Poll

```json
{"results_id": "uuid-task-id-here"}
```

Available stocks: `Zinc_Fix-stock` (ZINC purchasable, about 10.3 M compounds),
`RetroStar-stock`, `PaRotes_n1-stock`, `PaRotes_n5-stock`, `BioNav-stock`. Search
is MCTS_STAR (MCTS with a Retro*-style value heuristic).

The agent does not use this endpoint by default: it plans step by step through
the tools, so route construction stays inspectable. It is documented here because
the same service exposes it.

## 3. `condition_predictor`

Request: `{"reaction": "O=C(c1ccccc1)c1ccccc1.O>>OC(c1ccccc1)c1ccccc1"}`

```json
{
  "status": "success",
  "results": {
    "conditions": {
      "Temperature": [25.0, 80.0],
      "Solvent": ["THF", "DCM"],
      "Reagent": ["NaBH4", "LiAlH4"],
      "Catalyst": ["", "Pd/C"],
      "Score": ["0.85", "0.62"]
    }
  }
}
```

Condition models: `rcr` (Reaction Condition Recommendation) and `parrot` (if
deployed as a separate container). `score_route`'s greenness pillar uses this
endpoint only when a condition model is configured; otherwise it reports the
component as unavailable.

## 4. `reaction_rater`

Request: `{"reaction": "O=C(c1ccccc1)c1ccccc1.O>>OC(c1ccccc1)c1ccccc1"}`

```json
{"status": "success", "results": {"reaction_is_feasible": true, "confidence": 0.9234}}
```

`score_route` requires this endpoint for its feasibility pillar; without
`REACTION_RATER_URL` it refuses to construct.

## 5. Enzymatic endpoints

### `enzymatic_rxn_identifier`

Request: `{"reaction": "O=C(c1ccccc1)c1ccccc1>>OC(c1ccccc1)c1ccccc1"}`

```json
{"status": "success", "results": {"reaction type": "enzymatic", "confidence": 0.9612}}
```

### `enzyme_recommender`

Same request; response:

```json
{
  "status": "success",
  "results": {
    "recommended enzyme type": [["EC 1.1.1.1", "EC 1.1.1.2", "EC 1.1.1.3"]],
    "confidence": [[0.85, 0.72, 0.61]]
  }
}
```

### `easifa`

Request: `{"reaction": "...", "EC number": "1.1.1.1"}` — returns `uniprot_info`
and `enzyme_site_analysis` for the matching enzymes and their active sites.

## Tool-level contract

Whatever transport is used, the tools return a dict with a `success` flag, the
parsed payload, and on failure an `error` string. The agent loop treats a
non-empty `error` as a rejected tool call and retries or changes plan; it never
receives a silently empty result.

## References

- Wang, X., Yin, X., Zhang, X. et al. *A virtual platform for automated hybrid
  organic-enzymatic synthesis planning.* Nat. Commun. 16, 10929 (2025).
  doi:10.1038/s41467-025-65898-3
- GraphFP: Yang, K. et al. *Analyzing Learned Molecular Representations for
  Property Prediction.* J. Chem. Inf. Model. 59(8), 3370–3388 (2019); Chen, B. et
  al. *Retro\*: Learning Retrosynthetic Planning with Neural Guided A\* Search.*
  ICML 2020, arXiv:2006.15820.
- RxNGormer: Shi, Y. et al. *Masked Label Prediction: Unified Message Passing
  Model for Semi-Supervised Classification*, arXiv:2009.03509; Vaswani, A. et al.
  *Attention Is All You Need*, NeurIPS 2017.
- ChemBart: Lewis, M. et al. *BART: Denoising Sequence-to-Sequence Pre-training*,
  ACL 2020, doi:10.18653/v1/2020.acl-main.703.
