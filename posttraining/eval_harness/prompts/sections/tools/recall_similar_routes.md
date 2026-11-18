# recall_similar_routes

Adaptive RAG over a precedent route database (USPTO + RetroBioCat / BKMS
enzymatic). Returns up to K full N-step routes whose **target molecule**
is Morgan-Tanimoto-similar to your query, selected with MMR for diversity.

The model is CPU-only and cheap — it does NOT consume `max_iterations`
and runs in the same free pool as `analyze_molecule` / `check_buyability`.

The target's precedent recall is **already prefetched for you** at session
start (Mode C). You only need to call this tool yourself for
sub-intermediates.

## Args

- `query_smiles`: required. Canonicalised internally.
- `k`: 1-5, default 3. >5 wastes context; clamped server-side.
- `source`: `"any" | "chemo" | "enzyme" | "mixed"` (default `"mixed"`).
  - `"any"` — top-k by similarity, no source filter.
  - `"chemo"` — USPTO only.
  - `"enzyme"` — RetroBioCat / BKMS only.
  - `"mixed"` — guarantees ≥1 enzyme exemplar if the query is enzyme-friendly
    (free OH on aromatic / α-keto / amine on chiral carbon / etc.); else
    falls through to `"any"`.
- `min_similarity`: default 0.2. Hard floor — returns fewer (or zero)
  results if no candidate clears it.

## When to call

- You are about to expand a non-trivial intermediate (≥3 rings,
  ≥2 stereocenters, or unfamiliar scaffold class) AND
- You have not already retrieved precedents for a structurally similar
  molecule this run.

## When NOT to call

- The molecule is small / simple (`analyze_molecule` shows MW < 250 and
  ≤ 2 rings).
- `suggest_templates` has already returned a high-consensus top-1.
- The molecule is buyable — you wouldn't expand it anyway.

## Output schema (projected)

```
{
  "tool": "recall_similar_routes",
  "confidence": "high|medium|low|none",
  "max_similarity": float,
  "examples": [
    {"similarity": 0.62, "source_target": "...",
     "n_steps": 4, "is_enzymatic": false,
     "rxn_classes": [...],
     "steps": [
       {"product": "...", "reactants": ["...", "..."],
        "reaction_smarts": "...", "rxn_class": "..."},
       ...
     ]},
    ...
  ],
  "_result_id": "<id>"
}
```

(`query_canonical` only appears when canonicalization actually rewrote
your input SMILES. `n_examples` is just `len(examples)`.)

Each example carries the full per-step chain (product / reactants /
reaction_smarts / rxn_class) so you can read the precedent strategy
directly — no follow-up `load_result` needed for the common case.
The on-disk payload additionally has per-step extras
(`conditions`, `yield`, `doi`, `notes`) and the source `route_id`;
call `load_result(result_id=<the stub's _result_id>)` if you need any
of those for post-mortem or citation.

## How to use the result

Returned routes are **strategic anchors, not answers.** The expert who
solved the analog made specific choices for that analog's functional
groups; transfer the *strategy* (disconnection pattern, ordering of
rings vs side-chain installation, enzymatic vs chemo choice), not the
specific reagents. Always cross-check with `suggest_templates` +
`neural_retro` for your actual molecule.

Read each exemplar's first step (`steps[0]`) carefully — that
disconnection is what defines the strategic class. Later steps tell
you whether the analog used a convergent or linear synthesis, and
which functional-group transformations it leaned on. Whole-route
agreement across two or more high-similarity exemplars is a strong
signal for that strategy.

`confidence`:
- `high` (sim ≥ 0.7) — strong analog; favour its disconnection class.
- `medium` (0.4-0.7) — partial analog; use as a hint, not a constraint.
- `low` (0.2-0.4) — only the broad strategy is portable.
- `none` (< 0.2) — no useful precedent; ignore.
