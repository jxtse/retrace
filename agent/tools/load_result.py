"""Companion to ResultStore: load full tool result by id.

The agent loop replaces older heavy tool results with compact projection
stubs that carry an ``_result_id``. When the model wants the full payload,
it calls ``load_result(result_id=..., fields=[...])`` to get either the
entire JSON or a projection over named fields.
"""

from __future__ import annotations

import json

from tools.base import Tool


class LoadResultTool(Tool):
    name = "load_result"
    description = (
        "Fetch the full payload of a previously-summarised tool result from "
        "disk by passing the `_result_id` from a stub (optionally narrow "
        "with `fields` to specific top-level keys). Call when the stub "
        "dropped detail you now need — for example per-engine probability "
        "arrays from a `neural_retro` call, or the full `ranked` list from "
        "an `explore_branches` summary. Cost: under 100ms, free. Pairs "
        "with the heavy tool whose stub you saw (neural_retro, "
        "suggest_templates, explore_branches, score_route)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "result_id": {
                "type": "string",
                "description": "The _result_id from a tool-result stub.",
            },
            "fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of top-level keys to keep.",
            },
        },
        "required": ["result_id"],
    }
    tier = 1

    def __init__(self, store):
        self.store = store

    def execute(self, result_id: str, fields: list[str] | None = None, **_) -> str:  # type: ignore[override]
        try:
            data = self.store.load(result_id, fields=fields)
        except Exception as e:
            data = {"error": str(e)}
        return json.dumps(data, ensure_ascii=False, default=str)
