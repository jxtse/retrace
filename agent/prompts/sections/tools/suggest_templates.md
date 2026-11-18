# suggest_templates

Substructure-prefiltered top-K match against the curated 490k-template library:
USPTO + DORAnet organic AND enzymatic (RetroBioCat / BRENDA / RetroRules / BKMS
/ DORAnet JN1224MIN + JN3604IMT).

**TEMPLATE-FIRST POLICY.** Call this on every disconnection BEFORE invoking
the neural ensemble. Fall back to `neural_retro` only when fewer than 3
template candidates yield buyable leaves.

## Args

- `product_smiles`: required.
- `top_k`: default 10; bump to 15 for normal use, 25 for hard targets.
- `kind`: `"organic"` | `"enzymatic"` | omitted (both). Use `kind="enzymatic"`
  when planning a deliberately enzymatic step.
- `enzymatic_max_in_any`: default 3. When `kind="any"` (the default in the
  prefetch on every target), at most this many enzymatic suggestions appear
  in the final top-K — the rest are organic. The library is dominated by
  enzymatic sources, so without this cap top-K is often crowded out by
  enzymatic hits even on obviously organic targets. Use `kind="enzymatic"`
  to bypass the cap entirely and rank only enzymatic templates; pass a
  larger integer (or a negative value to disable) when you explicitly want
  more enzymatic options in a mixed query.

## Output schema (projected)

```
{
  "tool": "suggest_templates",
  "library_size": int,
  "n": int,
  "enzymatic_in_top_k": int,   # how many enzymatic survived the cap
  "suggestions": [
    {"rank": 1, "kind": "organic"|"enzymatic", "source": "USPTO"|"BRENDA"|...,
     "reactants": "R1.R2"},
    ...
  ],
  "_result_id": "<id>"
}
```

Use `load_result` to inspect the full template SMARTS / mw-balance /
prefilter rank when triaging.
