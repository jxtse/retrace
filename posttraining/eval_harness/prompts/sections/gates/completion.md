# Completion is system-driven with an LLM-controlled stop signal

The harness watches two HARD upper bounds and one SOFT threshold; the
loop terminates when ANY trips:

1. `iteration >= max_iterations` (billable retro-search budget exhausted)
2. `wallclock_s` exceeded
3. After `distinct_routes_saved >= stop_after_distinct_routes` (default
   10, configured under `agent.stop_after_distinct_routes`), the harness
   pushes a one-shot reminder telling you the soft threshold has been
   reached. From that turn onward, emitting a single line
   `COMPLETE ROUTE FOUND:` (no tool call in that turn) ends the loop;
   the harness then runs `rank_routes` and the system verifier picks
   the rank-1 saved route.

Before the soft threshold fires, `COMPLETE ROUTE FOUND` is logged
(`llm_self_complete_ignored`), refunded, and a
`nudge_self_complete_ignored` reminder is sent — there is no shortcut
out of the loop until the threshold reminder has fired.

## What you must do every turn

- Save each strategically-distinct candidate via `save_best_route` as
  soon as you have one (`check_buyability` must pass on every leaf
  first). The scratchpad dedups structural duplicates automatically.
- Distinct = different first-step disconnection OR different chemistry
  class — not a different leaf substitution.
- Keep `suggest_templates` + `neural_retro` paired (same turn) on every
  non-leaf molecule you commit to expanding.
- A separate soft reminder fires once when the saved-route count
  crosses `auto_finish_after_saved` — that is permission to stop
  proposing new strategies, not (yet) permission to end the loop.

## After the soft-threshold reminder fires

You may emit `COMPLETE ROUTE FOUND:` whenever you judge that further
iteration will not yield a meaningfully better disconnection. You do
not need to pick a winner; the system's verifier handles ranking. If
you still see a meaningfully different strategy worth exploring, keep
calling tools and saving routes — the iteration budget is still live.

## Quality, not just quantity

Use the per-dimension breakdown from `score_route` (and `rank_routes`
during finalisation) to self-critique while you still have budget:
identify the lowest sub-score and try a disconnection that plausibly
lifts it.
