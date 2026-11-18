"""
TodoWriteTool / TodoReadTool — explicit plan-state tracking for the
retrosynthesis agent, inspired by Claude Code's TodoWriteTool.

Retrosynthesis is naturally tree-shaped: each non-buyable precursor opens a
new sub-goal. Without explicit tracking the LLM tends to:
  - forget unfinished leaves ("consecutive_no_action" heuristic kicks in late)
  - re-decompose the same molecule across refinement rounds
  - declare COMPLETE ROUTE while leaves remain unresolved

These two tools give the LLM a stable, readable scratchpad for its plan:

  todo_write(todos=[
      {"id": "1", "content": "decompose target X", "status": "in_progress"},
      {"id": "2", "content": "check buyability of Y", "status": "pending"},
      ...
  ])
  todo_read()

State is held per-agent-run in a lightweight, in-memory store.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Iterable

from tools.base import Tool

logger = logging.getLogger(__name__)


ALLOWED_STATUSES = {"pending", "in_progress", "completed", "cancelled"}


@dataclass
class TodoItem:
    id: str
    content: str
    status: str = "pending"
    active_form: str = ""      # present-continuous form for the spinner
    smiles: str = ""           # optional SMILES this todo is about
    note: str = ""             # free-form notes / evidence

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "status": self.status,
            "activeForm": self.active_form,
            "smiles": self.smiles,
            "note": self.note,
        }


class TodoStore:
    """Thread-safe todo list, scoped to one agent run."""

    def __init__(self):
        self._lock = threading.Lock()
        self._items: list[TodoItem] = []

    def replace(self, todos: list[TodoItem]) -> tuple[list[TodoItem], dict]:
        """Replace items AND compute the summary atomically under the same lock."""
        with self._lock:
            self._items = list(todos)
            counts = {s: 0 for s in ALLOWED_STATUSES}
            for t in self._items:
                counts[t.status] = counts.get(t.status, 0) + 1
            summary = {
                "total": len(self._items),
                "by_status": counts,
                "has_pending": counts.get("pending", 0) + counts.get("in_progress", 0) > 0,
            }
            return list(self._items), summary

    def read(self) -> list[TodoItem]:
        with self._lock:
            return list(self._items)

    def summary(self) -> dict:
        with self._lock:
            counts = {s: 0 for s in ALLOWED_STATUSES}
            for t in self._items:
                counts[t.status] = counts.get(t.status, 0) + 1
            return {
                "total": len(self._items),
                "by_status": counts,
                "has_pending": counts.get("pending", 0) + counts.get("in_progress", 0) > 0,
            }

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def restore(self, items: Iterable) -> dict:
        """Rehydrate raw todo dicts (e.g. from ``to_dict()``) into the store.

        Routes through ``_coerce_todos`` so the camelCase ``activeForm`` key
        emitted by ``TodoItem.to_dict()`` maps back onto the ``active_form``
        field — a naive ``TodoItem(**d)`` would raise on that mismatch. Used
        to repopulate plan state on a continuation run. Returns the summary.
        """
        coerced, _warnings = _coerce_todos(items)
        _items, summary = self.replace(coerced)
        return summary


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _coerce_todos(raw: Iterable) -> tuple[list[TodoItem], list[str]]:
    """
    Validate incoming todos. Returns (items, warnings).
    Does NOT raise — warnings are surfaced back to the LLM so it can fix them.
    """
    items: list[TodoItem] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    in_progress_count = 0

    for i, t in enumerate(raw or []):
        if not isinstance(t, dict):
            warnings.append(f"todo #{i}: expected object, got {type(t).__name__}; skipped")
            continue
        tid = str(t.get("id") or f"auto-{i+1}").strip()
        content = str(t.get("content") or "").strip()
        if not content:
            warnings.append(f"todo #{i}: empty 'content'; skipped")
            continue
        if tid in seen_ids:
            warnings.append(f"todo #{i}: duplicate id '{tid}'; renamed to '{tid}-dup{i}'")
            tid = f"{tid}-dup{i}"
        seen_ids.add(tid)

        status = str(t.get("status") or "pending").strip().lower()
        if status not in ALLOWED_STATUSES:
            warnings.append(
                f"todo #{i}: unknown status '{status}'; coerced to 'pending'. "
                f"Allowed: {sorted(ALLOWED_STATUSES)}"
            )
            status = "pending"
        if status == "in_progress":
            in_progress_count += 1

        items.append(TodoItem(
            id=tid,
            content=content,
            status=status,
            active_form=str(t.get("activeForm") or t.get("active_form") or "").strip(),
            smiles=str(t.get("smiles") or "").strip(),
            note=str(t.get("note") or "").strip(),
        ))

    if in_progress_count > 1:
        warnings.append(
            f"Only one todo should be 'in_progress' at a time (found {in_progress_count}). "
            "Keep exactly one active focus."
        )

    return items, warnings


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class TodoWriteTool(Tool):
    """
    Replace the current todo list with a new version.

    The LLM should call this:
      - After analyzing the target, to register initial sub-goals
      - Whenever a leaf is confirmed buyable (mark its todo 'completed')
      - Whenever a leaf requires further decomposition (add new todos)
      - When switching focus (flip prior 'in_progress' back to 'pending')
    """
    name = "todo_write"
    description = (
        "Overwrite the agent's plan (todo list) with a new version. Each "
        "todo carries id, content, status (pending / in_progress / "
        "completed / cancelled), and optional activeForm / smiles / note "
        "fields; keep exactly ONE item in_progress at a time. Use on long "
        "branchy searches to track sub-goals (non-buyable precursors, "
        "verification steps, alternative disconnections); skip on short "
        "routes (≤ 2 steps) where the plan fits in one turn. Cost: under "
        "50ms, free. Pairs with `todo_read` (read at turn start to resume)."
    )
    tier = 1
    parameters = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "Full new todo list. Overwrites prior state.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id":         {"type": "string"},
                        "content":    {"type": "string"},
                        "status":     {
                            "type": "string",
                            "enum": sorted(list(ALLOWED_STATUSES)),
                        },
                        "activeForm": {"type": "string"},
                        "smiles":     {"type": "string"},
                        "note":       {"type": "string"},
                    },
                    "required": ["content"],
                },
            },
        },
        "required": ["todos"],
    }

    def __init__(self, store: TodoStore):
        self._store = store

    def execute(self, todos=None, **kwargs) -> str:
        if todos is None:
            # Tolerate LLMs that pass a stringified JSON
            todos = kwargs.get("items") or []
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except json.JSONDecodeError as e:
                return json.dumps({
                    "error": f"Could not parse 'todos' as JSON array: {e}",
                    "hint": "Pass an array of objects, not a string.",
                })
        if not isinstance(todos, list):
            return json.dumps({
                "error": f"'todos' must be an array, got {type(todos).__name__}",
            })

        items, warnings = _coerce_todos(todos)
        if todos and not items:
            return json.dumps({
                "ok": False,
                "error": "No valid todos parsed from input.",
                "warnings": warnings,
            })
        _, summary = self._store.replace(items)

        logger.info(
            "todo_write: total=%d pending=%d in_progress=%d completed=%d cancelled=%d",
            summary["total"],
            summary["by_status"].get("pending", 0),
            summary["by_status"].get("in_progress", 0),
            summary["by_status"].get("completed", 0),
            summary["by_status"].get("cancelled", 0),
        )

        return json.dumps({
            "ok": True,
            "summary": summary,
            "todos": [it.to_dict() for it in items],
            "warnings": warnings,
        })


class TodoReadTool(Tool):
    """Return the current todo list."""
    name = "todo_read"
    description = (
        "Return the current todo list. Call at the start of a turn on long "
        "branchy searches to remind yourself what's pending, and once before "
        "committing to `COMPLETE ROUTE FOUND` to verify no pending leaves "
        "remain. Cost: <50ms, free. Pairs with `todo_write` (read state, "
        "decide next move, write the updated plan)."
    )
    tier = 1
    parameters = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, store: TodoStore):
        self._store = store

    def execute(self, **kwargs) -> str:
        items = self._store.read()
        summary = self._store.summary()
        return json.dumps({
            "summary": summary,
            "todos": [it.to_dict() for it in items],
        })


# ---------------------------------------------------------------------------
# Helper: build & register both tools on a ToolRegistry
# ---------------------------------------------------------------------------

def register_todo_tools(registry) -> TodoStore:
    """
    Create a fresh TodoStore and register TodoWriteTool + TodoReadTool on
    `registry`. Returns the store so the agent can inspect it (e.g. to refuse
    a COMPLETE ROUTE while items are still pending).
    """
    store = TodoStore()
    registry.register(TodoWriteTool(store))
    registry.register(TodoReadTool(store))
    return store
