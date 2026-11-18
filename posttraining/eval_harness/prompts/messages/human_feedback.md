A chemist has reviewed the route(s) you found and is asking you to continue the search with their feedback. The previously-saved routes are still in your scratchpad — use `list_best_routes` / `get_best_route` to revisit them.

Chemist feedback:
{feedback}

Act on this feedback directly. If it asks you to refine or replace a step, propose the disconnection, validate it, score the revised route with `score_route`, and `save_best_route` so it can compete with the existing candidates. Keep `check_buyability` on every leaf. When you have a route that satisfies the feedback and all leaves are buyable, finish as usual.
