# explore_branches

Cross-source consensus ranker for K candidate first-step disconnections.
Ranks them by the **same three pillars** `score_route` uses (feasibility
0.70 / cost 0.15 / greenness 0.15) so the pre-filter and the final
verifier optimise the same axes.

## When

After parallel `suggest_templates` + `neural_retro` on the same product
(the default `neural_retro` ensemble now includes the ring-opening
specialist, so a separate ringbreaker call is no longer needed on
ring-rich targets), fold their top entries (plus 1-2 chemistry-intuition
branches you came up with) into a single `explore_branches` call. The
tool reverse-looks-up those prior calls in `ResultStore` and credits
each candidate with a `support` score wherever its reactant set matches
what an engine / template / ringbreaker already proposed — that is the
cheap proxy for the expensive ReactionT5 feasibility check `score_route`
would otherwise run.

DO NOT use this on a product you have not yet probed with
`suggest_templates` and `neural_retro` — `support_signal` will be 0 and
ranking degrades to pure carbon-conservation + buyability + atom economy.

## Iteration cost

Free. `explore_branches` is NOT in `agent.iteration_billable_tools` —
only `suggest_templates` and `neural_retro` consume the search budget.

## Args

- `product_smiles` (str, required): the product being disconnected.
- `branches` (list, required): each item
  `{"label": str, "reactants": [SMILES, ...], "notes": str}`.
  Min 1, max 12.
- `top_k` (int, default 3): how many top-ranked branches to return.
- `max_workers` (int, default 8): parallelism cap.

## Scoring

```
score = 0.70 * feasibility + 0.15 * cost + 0.15 * greenness

feasibility = 0.60 * support_signal      # cross-source agreement
            + 0.30 * carbon_conservation # no fabricated skeletal carbon
            + 0.10 * stereo_ok           # warning-free product stereo

cost        = 0.60 * leaf_buyability     # 4-tier BuyabilityChain
            + 0.40 * sa_decrease         # SCScore (target → leaves)

greenness   = atom_economy               # Trost MW(p)/Σ MW(reactants)
```

Within `support_signal`: each strict cross-source match contributes
0.40, each partial (subset/superset) match contributes 0.20. Multiple
matches from the same source key count once. Capped at 1.0. Two strict
matches across different engines → 0.80; lone template hit → 0.20; no
matches → 0.0.

## Output schema (projected)

```
{
  "tool": "explore_branches",
  "ranked": [
    {
      "rank": 1,
      "score": 0.81,
      "feasibility": 0.84,
      "cost": 0.78,
      "greenness": 0.71,
      "reactants": [...],
      "support_count_strict": 2,
      "support_count_partial": 0
    },
    ...
  ],
  "_result_id": "<id>"
}
```

Per-branch `sub_scores` (each axis broken out) and the typed `support`
list are dropped from the projection — fetch them with
`load_result(result_id=...)` if you need to debug a ranking decision.

## Recommended workflow

1. `suggest_templates` + `neural_retro` in parallel on the product.
2. Hand the top template results + top neural-cluster representatives
   (favour entries with `len(support) >= 2`) plus 1-2 of your own
   chemistry-driven ideas to a single `explore_branches` call.
3. Expand the winner with another `neural_retro` / `suggest_templates`
   pair on its biggest intermediate. Keep the runner-up as a fallback.
4. Call `score_route` only on a finished candidate route — never on
   per-step branches.
