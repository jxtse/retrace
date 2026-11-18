# neural_retro

Unified neural one-step retro engine. **Default = ensemble**: a single call
fans out concurrently across every configured engine (`graph_retrosyn`,
`rxngraphormer`, `pistachio_ringbreaker`, `chembart`, optional
`template_relevance`), deduplicates the merged candidate pool on canonical
reactant multisets, runs an LLM-judge precursor sanity filter, clusters
the survivors with Morgan2/Butina (`distance_threshold=0.65`,
`k_per_cluster=2`), and returns up to ~15 *diverse* precursor sets.

Ring-opening picks from `pistachio_ringbreaker` now appear inside the
default ensemble output — look for entries with
`support: ["pistachio_ringbreaker"]` (or that engine listed alongside
others when the general engines agree). No separate alias tool is
needed; ringbreaker is just another engine in the default fan-out.

## Output schema (projected stub)

```
{
  "tool": "neural_retro",
  "engine": "ensemble"|"<single>",
  "n_unique": int,            # candidates after canonical dedup
  "n_dropped_by_sanity": int, # dropped by the LLM-judge precursor filter
  "n_kept_after_sanity": int, # = n_unique - n_dropped_by_sanity
  "n_clusters": int,          # Butina clusters formed (also = len(predictions))
  "engine_errors": {engine: msg},   # only present when an engine failed
  "predictions": [
    {"rank": 1, "reactants": "R1.R2", "support": ["graph_retrosyn", "chembart"]},
    ...
  ],
  "_result_id": "<id>"        # load_result(_result_id) for full per-engine probs
}
```

`support` lists the engines that proposed this candidate — entries with
`len(support) >= 2` are the high-confidence cross-engine picks.

## How to read the funnel

The three `n_*` fields tell you why the stub has the count it has:

```
n_unique  →  n_kept_after_sanity  →  n_clusters (= len(predictions))
```

- If `len(predictions)` is much smaller than `n_kept_after_sanity`, the
  targets are structurally similar (often same scaffold, different
  leaving groups); consider single-engine queries to find chemically-
  different alternatives.
- If `n_dropped_by_sanity` is a large fraction of `n_unique`, the LLM
  judge is dropping fragments — usually fine, but inspect a couple via
  `load_result` if you suspect over-filtering.
- If `engine_errors` is present, one or more engines failed (network /
  timeout); the surviving engines still produced the predictions you
  see. Retry once if the missing engine is critical (e.g. only
  `chembart` errored on an enzyme-rich target).

## When / how

- Default: omit `engine=` to get the ensemble.
- For harder targets, raise `per_engine_topk` (default 50) before raising
  `return_topk` (default 15).
- Pass `engine="rxngraphormer"` (or another single engine) only when
  diagnosing why a particular engine missed something.
- Use `load_result(result_id, fields=["predictions"])` to retrieve the
  full pre-stub list when the projected stub looks too short.

## Pairing

Always call **alongside** `suggest_templates(top_k=15)` in the same assistant
turn. The two are complementary: templates cover documented chemistry; the
ensemble covers novel scaffolds, protecting groups, enzymatic alternatives,
and ring-opening disconnections (the latter via `pistachio_ringbreaker`,
which is now part of the default ensemble — no separate call needed).
Then merge the top picks of each into a single `explore_branches` call.
