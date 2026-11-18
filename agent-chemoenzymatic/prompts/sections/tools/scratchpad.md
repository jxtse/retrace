# Scratchpad — top-N saved routes + lessons

Long-horizon memory for the search. Keeps the top-5 routes by score
(rejection on insert if a new candidate ≤ current worst) and a
lessons-learned log. The agent loop also uses scratchpad checkpoints to
trigger context compaction.

## Tools

- `save_best_route(label, steps, total_score, sub_scores=None, notes=None)`
  Persist a candidate route. **Call after EVERY `score_route`.** Pass
  `total_score` into the `total_score` arg.
- `list_best_routes()` — view all saved candidates ordered by score.
  **Call before emitting `COMPLETE ROUTE FOUND:`** to confirm you're
  returning the top-scoring one.
- `get_best_route(label)` — recover the full steps for a labelled route.
- `backtrack_and_summarize(reason, tanked_subscore=None)` — append a
  lesson, drop the failed branch's tool turns from context (keeps
  target / todos / top-N / lessons), and pivot to a different strategy.

## Compaction milestones

After a successful `save_best_route`, the loop drops older `tool` messages
older than the most recent `score_route` for that branch. After
`backtrack_and_summarize`, the loop rewrites history to keep only system +
target + a synthetic snapshot. You don't need to manage compaction yourself;
just call the scratchpad tools and the loop handles the bookkeeping.

### When `save_best_route` returns `status: "duplicate"`

The scratchpad rejects structurally identical routes (same canonical
SMILES per step, order-insensitive). A `status: "duplicate"` response
is **not** an error — it tells you the route you tried to save is
already on the scratchpad under `duplicate_of: <label>`. Don't retry
with a renamed label; instead, propose a fundamentally different
first-step disconnection (different bond from `identify_bonds`,
different chemistry class via `suggest_templates(kind=...)`, or a
different `engine` for `neural_retro`) and save *that*.

### Provenance fields in saved entries

Every saved route carries a `provenance` tag inferred by the agent
loop from the tools you called in the turns leading up to the save:

- `rag_anchored` — `recall_similar_routes` was in context and its
  payload had exemplars. Call `get_best_route(label=...)` to see the
  precedent `route_id`s in `rag_source_ids`; you can then
  `load_result(result_id=<that recall stub's id>)` to pull the full
  N-step source route back.
- `neural_consensus` — `neural_retro` drove the disconnection.
- `template_suggested` — `suggest_templates` drove it.
- `zero_shot` — none of the above (chemical intuition only).

`list_best_routes` shows the `provenance` tag on every entry but not
the `rag_source_ids` list (use `get_best_route` for that). When the
list shows a mix of provenances, prefer keeping ones tagged
`rag_anchored` or `neural_consensus` as your final answer — they have
an external check behind them. If everything is `zero_shot`, that's a
signal you have not exercised the retro tools enough; generate at
least one more route grounded in a tool call.
