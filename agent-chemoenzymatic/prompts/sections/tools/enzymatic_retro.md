# enzymatic_retro

Explicit enzymatic-only access to the curated template library. It returns
ranked, sourced reactant suggestions and does not infer an enzyme identity or
guarantee that a transformation is valid.

## When

Use it for a plausible enzymatic step such as a reduction, transamination,
carboligation/aldol reaction, hydrocyanation, or transformations of chiral
amines, alcohols, sugars, and terpenes. In a mixed chemoenzymatic route, the
correct sequence is often organic retrosynthesis on the final target followed
by `enzymatic_retro` on an upstream non-leaf intermediate. A target-only query
does not cover those opportunities.

## How to use a result

Do not force rank 1. Select a sourced suggestion only when it is chemically
plausible, validate the reactants and step, continue that branch until all
leaves are confirmed buyable, then call `score_route` and `save_best_route`.
Pass the same candidate to `explore_branches` when comparing it with organic
or neural disconnections; the explicit enzyme result contributes source/rank
support from `ResultStore`.
