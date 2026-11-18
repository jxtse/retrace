"""rank_routes — produce a sorted breakdown of every saved route.

Used at the end of a search session (or on demand) to surface the full
exploration result instead of just the top-scoring route. Sub-scores
are exposed so the user can audit which dimension dragged any route
down.
"""

from __future__ import annotations

import json

from tools.base import Tool


class RankRoutesTool(Tool):
    name = "rank_routes"
    description = (
        "Return ALL saved routes, sorted by total_score descending, with "
        "per-dimension sub-scores (feasibility / cost / greenness) surfaced. "
        "No filtering — every route the agent saved appears in the table. "
        "Call once near the end of a search session, after every route in "
        "the scratchpad has been scored."
    )
    parameters = {"type": "object", "properties": {}}

    SCHEMA_VERSION = 1

    def __init__(self, scratchpad_store):
        self._store = scratchpad_store

    def execute(self, **_kwargs) -> str:
        all_routes = list(self._store.all())
        all_routes.sort(key=lambda e: e.total_score, reverse=True)
        payload_routes = []
        for rank, e in enumerate(all_routes, start=1):
            sub = e.sub_scores or {}
            payload_routes.append({
                "rank": rank,
                "label": e.label,
                "total_score": round(float(e.total_score), 4),
                "feasibility": sub.get("feasibility"),
                "cost": sub.get("cost"),
                "greenness": sub.get("greenness"),
                "n_steps": len(e.steps),
                "first_step": e.steps[0] if e.steps else None,
                "starting_materials": e.starting_materials,
                "notes": e.notes,
            })
        return json.dumps({
            "schema_version": self.SCHEMA_VERSION,
            "routes": payload_routes,
        })
