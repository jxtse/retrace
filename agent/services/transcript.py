"""Live transcript recorder for retrosynthesis-agent runs.

Hooks into the agent's existing ``trace`` callback (every ``iteration``,
``thinking``, ``tool_call``, ``tool_result``, ``complete``, ``error``,
``checkpoint_compacted`` and ``backtrack_compacted`` event already fires
through it with verbatim payloads) and writes two artefacts under
``runs/_session_<TS>/``:

* ``transcript.jsonl`` — one event per line, written immediately. A crash
  mid-run still leaves an inspectable timeline.
* ``transcript.json`` — final summary written in ``agent.run()``'s finally
  block: target + status + final route_text + ranked scratchpad +
  per-iteration timeline rolled up from the JSONL events.

Tool inputs are captured at the ``tool_call`` event (raw arguments string
the model emitted, parsed when JSON). Tool outputs are captured at the
``tool_result`` event BEFORE ``ResultStore.persist`` rewrites them into
projection stubs — so the transcript holds the full payload even when the
in-context message stream only sees a stub.

The recorder is best-effort and never raises into the agent loop.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Cap any single payload we serialise so a runaway tool result can't
# blow up the JSONL writer (or, worse, the post-run reviewer's editor).
# Full content is still on disk via ResultStore at results/<id>.json.
_MAX_PAYLOAD_CHARS = 200_000
_MAX_TRAINING_TEXT_CHARS = 2_000
_MAX_TRAINING_ITEMS = 8

_TRAINING_TOOL_NAMES = {
    "neural_retro",
    "suggest_templates",
    "score_route",
    "save_best_route",
    "backtrack_and_summarize",
    "check_buyability",
    "list_best_routes",
    "rank_routes",
}

_TRAINING_STOP_EVENTS = {
    "complete",
    "complete_blocked_quality",
    "complete_blocked_refinement",
    "complete_below_target",
    "llm_self_complete_honored",
    "llm_self_complete_ignored",
    "soft_threshold_reached",
    "saved_threshold_nudge",
    "wallclock_exceeded",
    "total_turns_exceeded",
    "native_bailout",
}

_TRAINING_CONTEXT_EVENTS = {
    "iteration",
    "checkpoint_compacted",
    "backtrack_compacted",
    "error",
}


def _truncate(s: str, limit: int = _MAX_PAYLOAD_CHARS) -> tuple[str, bool]:
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= limit:
        return s, False
    return s[:limit] + f"\n…[truncated {len(s) - limit} chars]", True


def _try_json(s: Any) -> Any:
    """Parse JSON if possible; otherwise return the original string."""
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except Exception:
        return s


def _compact_value(value: Any, *, depth: int = 3) -> Any:
    """Return a JSON-safe compact copy for training-oriented summaries."""
    if depth <= 0:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(type(value).__name__)
    if isinstance(value, str):
        text, truncated = _truncate(value, _MAX_TRAINING_TEXT_CHARS)
        return {"text": text, "truncated": True} if truncated else text
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= _MAX_TRAINING_ITEMS:
                out["_truncated_items"] = len(value) - _MAX_TRAINING_ITEMS
                break
            out[str(k)] = _compact_value(v, depth=depth - 1)
        return out
    if isinstance(value, (list, tuple)):
        out = [_compact_value(v, depth=depth - 1)
               for v in list(value)[:_MAX_TRAINING_ITEMS]]
        if len(value) > _MAX_TRAINING_ITEMS:
            out.append({"_truncated_items": len(value) - _MAX_TRAINING_ITEMS})
        return out
    return str(value)


def _pick_compact(d: Dict[str, Any], keys: tuple[str, ...]) -> Dict[str, Any]:
    return {k: _compact_value(d[k]) for k in keys if k in d}


class TranscriptRecorder:
    """Wraps an optional user trace; persists everything to JSONL + JSON."""

    def __init__(self, session_dir: str, target_smiles: str,
                 user_trace: Any = None,
                 scratchpad_store: Any = None,
                 todo_store: Any = None):
        self.session_dir = session_dir
        self.target_smiles = target_smiles
        self.user_trace = user_trace
        # Held so checkpoint() can roll up the live scratchpad / todos without
        # the caller threading them through on every periodic flush.
        self.scratchpad_store = scratchpad_store
        self.todo_store = todo_store
        self.t0 = time.time()
        self.events: List[Dict[str, Any]] = []
        # Pending-tool-call lookup: tool_call_id → {name, args, t_call}.
        # Used at tool_result time so we can stitch input ↔ output into a
        # single timeline entry without re-parsing the message stream.
        self._pending_calls: Dict[str, Dict[str, Any]] = {}
        # Anonymous tool_calls (no id) get matched FIFO per-name. The agent
        # is now native-tools only, so every call should arrive with an id;
        # the FIFO path is kept as a defensive fallback.
        self._fifo_by_name: Dict[str, List[Dict[str, Any]]] = {}
        # A compact, training-oriented view of the same event stream. This is
        # intentionally redundant with `events`: post-training builders should
        # not need to infer every decision from free text or giant tool blobs.
        self.training_trace: List[Dict[str, Any]] = []
        self._training_counter = 0
        self._current_iteration = 0
        self._lock = threading.Lock()
        self.metadata: Dict[str, Any] = {}
        self._manifest_payload: Dict[str, Any] = {}
        self._llm_request_counter = 0

        os.makedirs(self.session_dir, exist_ok=True)
        self.conversation_dir = os.path.join(self.session_dir, "conversation")
        os.makedirs(self.conversation_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.session_dir, "transcript.jsonl")
        try:
            # Open in append mode so partial writes from a crashed prior
            # session aren't blown away (session dir is timestamped though,
            # so this is theoretical safety).
            self._jsonl_fh = open(self.jsonl_path, "a", encoding="utf-8")
        except Exception as exc:
            logger.warning("TranscriptRecorder: cannot open %s: %s",
                           self.jsonl_path, exc)
            self._jsonl_fh = None

        self._emit({
            "ts_rel": 0.0,
            "event": "session_start",
            "target_smiles": target_smiles,
            "session_dir": session_dir,
        })

    def update_metadata(self, **kwargs: Any) -> None:
        """Attach run metadata that should be included in transcript.json."""
        for key, value in kwargs.items():
            if value is not None:
                self.metadata[key] = value

    def write_session_manifest(self, **kwargs: Any) -> str:
        """Persist the resumable/auditable session manifest."""
        self._manifest_payload.update({
            "schema_version": 1,
            "target_smiles": self.target_smiles,
            "session_dir": self.session_dir,
            "updated_at": time.time(),
            **kwargs,
        })
        self._manifest_payload.setdefault("created_at", time.time())
        path = os.path.join(self.session_dir, "session_manifest.json")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(
                    self._manifest_payload,
                    fh,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            self.metadata["session_manifest_path"] = path
        except Exception as exc:
            logger.debug("session_manifest write failed: %s", exc)
        return path

    def write_latest_messages(
        self,
        messages: list[dict],
        *,
        stage: str = "",
        iteration: int | None = None,
    ) -> str:
        """Persist the latest resumable message list."""
        path = os.path.join(self.conversation_dir, "latest_messages.json")
        payload = {
            "schema_version": 1,
            "target_smiles": self.target_smiles,
            "session_dir": self.session_dir,
            "stage": stage,
            "iteration": iteration,
            "updated_at": time.time(),
            "messages": messages,
        }
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
            self.metadata["latest_messages_path"] = path
        except Exception as exc:
            logger.debug("latest_messages write failed: %s", exc)
        return path

    def record_llm_request(
        self,
        *,
        phase: str,
        messages: list[dict],
        model_settings: Dict[str, Any] | None = None,
        tools: list[dict] | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> str:
        """Append one exact agent-bound LLM request snapshot."""
        self._llm_request_counter += 1
        path = os.path.join(self.conversation_dir, "llm_requests.jsonl")
        payload: Dict[str, Any] = {
            "schema_version": 1,
            "request_index": self._llm_request_counter,
            "ts_rel": round(time.time() - self.t0, 3),
            "phase": phase,
            "model_settings": model_settings or {},
            "messages": messages,
        }
        if tools is not None:
            payload["tools"] = tools
        if extra:
            payload["extra"] = extra
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self.metadata["llm_requests_path"] = path
        except Exception as exc:
            logger.debug("llm request snapshot write failed: %s", exc)
        return path

    # ── public callable used as the agent's trace callback ────────────────

    def __call__(self, event: str, **kw):
        try:
            self._record(event, kw)
        except Exception as exc:
            logger.debug("TranscriptRecorder._record swallowed: %s", exc)
        # Always forward to the user-provided trace (e.g. --trace coloring)
        if self.user_trace is not None:
            try:
                self.user_trace(event, **kw)
            except Exception as exc:
                logger.debug("user_trace forward swallowed: %s", exc)

    # ── core: convert an event into a structured timeline entry ──────────

    def _record(self, event: str, kw: Dict[str, Any]) -> None:
        ts_rel = time.time() - self.t0
        entry: Dict[str, Any] = {"ts_rel": round(ts_rel, 3), "event": event}

        if event == "iteration":
            entry["n"] = kw.get("n")

        elif event == "thinking":
            content, truncated = _truncate(kw.get("content", "") or "")
            entry["content"] = content
            if truncated:
                entry["truncated"] = True

        elif event == "reasoning":
            # Model reasoning/thinking trace. Stored in FULL (no truncation):
            # it is the primary artefact for post-run debugging and is written
            # only to transcript.jsonl, not echoed to the model or the live
            # user trace. Reasoning can run to several KB per turn.
            entry["content"] = kw.get("content", "") or ""

        elif event == "tool_call":
            name = kw.get("name") or "?"
            tool_call_id = kw.get("tool_call_id") or ""
            args_raw = kw.get("args", "")
            args_parsed = _try_json(args_raw)
            args_str, args_trunc = _truncate(
                args_raw if isinstance(args_raw, str)
                else json.dumps(args_raw, default=str, ensure_ascii=False)
            )
            entry["name"] = name
            if tool_call_id:
                entry["tool_call_id"] = tool_call_id
            entry["arguments"] = args_parsed if not args_trunc else args_str
            if args_trunc:
                entry["arguments_truncated"] = True
            # Stash so tool_result can resolve back to this call.
            call_info = {
                "name": name,
                "args": args_parsed,
                "t_call": ts_rel,
            }
            with self._lock:
                if tool_call_id:
                    self._pending_calls[tool_call_id] = call_info
                self._fifo_by_name.setdefault(name, []).append(call_info)

        elif event == "tool_result":
            tool_call_id = kw.get("tool_call_id") or ""
            tool_name = kw.get("tool_name") or "?"
            elapsed_ms = kw.get("elapsed_ms")
            content_raw = kw.get("content") or ""
            content_parsed = _try_json(content_raw)
            content_str, content_trunc = _truncate(
                content_raw if isinstance(content_raw, str)
                else json.dumps(content_raw, default=str, ensure_ascii=False)
            )
            # Resolve back to the most recent matching tool_call to recover
            # the input arguments for this row.
            with self._lock:
                call_info = self._pending_calls.pop(tool_call_id, None)
                if call_info is None:
                    queue = self._fifo_by_name.get(tool_name) or []
                    call_info = queue.pop(0) if queue else None
            entry["tool_call_id"] = tool_call_id
            entry["name"] = tool_name
            entry["elapsed_ms"] = elapsed_ms
            if call_info is not None:
                entry["arguments"] = call_info.get("args")
                entry["t_call_rel"] = round(float(call_info.get("t_call", 0.0)), 3)
            entry["result"] = (
                content_parsed if not content_trunc else content_str
            )
            if content_trunc:
                entry["result_truncated"] = True
                entry["result_full_path"] = os.path.join(
                    self.session_dir, "results", f"{tool_call_id}.json"
                )

        elif event in ("complete", "complete_blocked_quality",
                        "complete_blocked_refinement", "complete_below_target",
                        "native_bailout"):
            content, truncated = _truncate(kw.get("content", "") or "")
            entry["content"] = content
            if truncated:
                entry["truncated"] = True
            # Carry the gate's diagnostic fields if present.
            for k in ("best_score", "target", "n_saved",
                      "required_refinements", "iteration", "error"):
                if k in kw:
                    entry[k] = kw[k]

        elif event == "error":
            entry["message"] = str(kw.get("message", ""))[:4000]

        elif event == "prefetch":
            entry["tools"] = kw.get("tools")

        elif event == "llm_call":
            entry["phase"] = kw.get("phase")
            entry["elapsed_ms"] = kw.get("elapsed_ms")
            entry["finish_reason"] = kw.get("finish_reason")
            entry["n_tool_calls"] = kw.get("n_tool_calls")
            entry["content_chars"] = kw.get("content_chars")
            entry["n_messages"] = kw.get("n_messages")
            if kw.get("error"):
                entry["error"] = str(kw.get("error"))[:4000]

        elif event in ("checkpoint_compacted", "backtrack_compacted"):
            entry.update({k: v for k, v in kw.items() if k in (
                "demoted", "dropped", "kept_recent", "kept_head",
            )})

        else:
            # Unknown event — record raw kwargs (truncated) so future event
            # types are captured without code change.
            try:
                blob = json.dumps(kw, default=str, ensure_ascii=False)
            except Exception:
                blob = str(kw)
            entry["raw"], _ = _truncate(blob, 8000)

        self.events.append(entry)
        self._capture_training_event(event, entry)
        self._emit(entry)

    def _emit(self, entry: Dict[str, Any]) -> None:
        if self._jsonl_fh is None:
            return
        try:
            self._jsonl_fh.write(
                json.dumps(entry, default=str, ensure_ascii=False) + "\n"
            )
            self._jsonl_fh.flush()
        except Exception as exc:
            logger.debug("transcript jsonl write failed: %s", exc)

    # ── final roll-up ─────────────────────────────────────────────────────

    def finalize(
        self,
        target_smiles: str,
        result: Optional[Dict[str, Any]],
        scratchpad_store: Any = None,
        todo_store: Any = None,
    ) -> Dict[str, Any]:
        """Write the authoritative ``transcript.json`` (timeline + scratchpad
        + final route) and close the JSONL handle. Called once from
        ``agent.run()``'s finally."""
        payload = self._build_payload(
            target_smiles, result,
            scratchpad_store if scratchpad_store is not None else self.scratchpad_store,
            todo_store if todo_store is not None else self.todo_store,
        )
        self._write_json(payload, final=True)
        if self._jsonl_fh is not None:
            try:
                self._jsonl_fh.close()
            except Exception:
                pass
            self._jsonl_fh = None
        return payload

    # ── training-oriented compact trace ─────────────────────────────────

    def _next_training_id(self) -> str:
        self._training_counter += 1
        return f"tt-{self._training_counter:06d}"

    def _capture_training_event(self, source_event: str, entry: Dict[str, Any]) -> None:
        try:
            if source_event == "iteration":
                self._current_iteration = int(entry.get("n") or 0)
                self._append_training_record(
                    kind="turn_state",
                    action="iteration_start",
                    source_event=source_event,
                    payload={"state": self._state_snapshot()},
                    include_state=False,
                )
                return

            if source_event == "tool_call":
                name = entry.get("name") or "?"
                if name in _TRAINING_TOOL_NAMES:
                    self._append_training_record(
                        kind="tool_call",
                        action=f"call_{name}",
                        source_event=source_event,
                        payload={
                            "tool_call_id": entry.get("tool_call_id"),
                            "arguments": _compact_value(entry.get("arguments")),
                        },
                    )
                return

            if source_event == "tool_result":
                name = entry.get("name") or "?"
                if name in _TRAINING_TOOL_NAMES:
                    self._append_training_record(
                        kind="tool_result",
                        action=f"{name}_result",
                        source_event=source_event,
                        payload={
                            "tool_call_id": entry.get("tool_call_id"),
                            "elapsed_ms": entry.get("elapsed_ms"),
                            "arguments": _compact_value(entry.get("arguments")),
                            "result_summary": self._summarize_tool_result(
                                name, entry.get("result")
                            ),
                        },
                    )
                return

            if source_event in _TRAINING_STOP_EVENTS:
                payload_keys = (
                    "content", "best_score", "target", "n_saved",
                    "required_refinements", "distinct", "target_distinct",
                    "stop_after", "iters_remaining", "plateau_turns",
                    "budget_s", "iterations", "total_turns",
                    "max_total_turns", "iteration", "error",
                )
                self._append_training_record(
                    kind="stop_or_gate",
                    action=source_event,
                    source_event=source_event,
                    payload=_pick_compact(entry, payload_keys),
                )
                return

            if source_event in _TRAINING_CONTEXT_EVENTS:
                payload = _pick_compact(
                    entry,
                    ("message", "demoted", "dropped", "kept_recent",
                     "kept_head", "kept_prefetch", "snapshot"),
                )
                self._append_training_record(
                    kind="context",
                    action=source_event,
                    source_event=source_event,
                    payload=payload,
                )
        except Exception as exc:
            logger.debug("training trace capture failed: %s", exc)

    def _append_training_record(
        self,
        *,
        kind: str,
        action: str,
        source_event: str,
        payload: Dict[str, Any],
        include_state: bool = True,
    ) -> None:
        record: Dict[str, Any] = {
            "schema_version": 1,
            "id": self._next_training_id(),
            "ts_rel": payload.pop("ts_rel", None),
            "iteration": self._current_iteration,
            "kind": kind,
            "action": action,
            "source_event": source_event,
            "payload": payload,
        }
        if record["ts_rel"] is None:
            record.pop("ts_rel")
        if include_state:
            record["state"] = self._state_snapshot()
        self.training_trace.append(record)

    def _state_snapshot(self) -> Dict[str, Any]:
        return {
            "iteration": self._current_iteration,
            "scratchpad": self._scratchpad_snapshot(),
            "todos": self._todo_snapshot(),
        }

    def _scratchpad_snapshot(self, *, keep_best_n: int = 5) -> Dict[str, Any]:
        if self.scratchpad_store is None:
            return {"n_saved": 0, "top_routes": []}
        try:
            routes = list(self.scratchpad_store.list())
        except Exception:
            return {"n_saved": None, "top_routes": []}
        return {
            "n_saved": len(routes),
            "top_routes": [
                self._route_brief(route, rank)
                for rank, route in enumerate(routes[:keep_best_n], start=1)
            ],
        }

    def _todo_snapshot(self) -> List[Dict[str, Any]]:
        if self.todo_store is None:
            return []
        try:
            items = self.todo_store.read()
        except Exception:
            return []
        out: List[Dict[str, Any]] = []
        for item in list(items)[:_MAX_TRAINING_ITEMS]:
            try:
                raw = item.to_dict()
            except Exception:
                raw = {"text": str(item)}
            out.append(_compact_value(raw, depth=2))
        return out

    def _route_brief(self, route: Any, rank: int) -> Dict[str, Any]:
        if isinstance(route, dict):
            label = route.get("label")
            total_score = route.get("total_score")
            sub_scores = route.get("sub_scores") or {}
            steps = route.get("steps") or []
            starting_materials = route.get("starting_materials") or []
        else:
            label = getattr(route, "label", None)
            total_score = getattr(route, "total_score", None)
            sub_scores = getattr(route, "sub_scores", {}) or {}
            steps = getattr(route, "steps", []) or []
            starting_materials = getattr(route, "starting_materials", []) or []
        return {
            "rank": rank,
            "label": label,
            "total_score": round(float(total_score), 4)
            if isinstance(total_score, (int, float)) else total_score,
            "sub_scores": _compact_value(sub_scores, depth=2),
            "num_steps": len(steps) if hasattr(steps, "__len__") else None,
            "num_starting_materials": (
                len(starting_materials)
                if hasattr(starting_materials, "__len__") else None
            ),
            "starting_materials": _compact_value(starting_materials, depth=2),
        }

    def _summarize_tool_result(self, tool_name: str, result: Any) -> Any:
        if not isinstance(result, dict):
            return _compact_value(result)

        common = _pick_compact(result, (
            "ok", "success", "accepted", "saved", "duplicate", "rejected",
            "error", "reason", "result_id", "score_route_result_id",
            "label", "route_label", "total_score", "score", "valid",
            "hard_failure", "buyable", "below_target", "n_routes",
            "n_saved", "rank", "ranked_routes",
        ))
        if "sub_scores" in result:
            common["sub_scores"] = _compact_value(result["sub_scores"], depth=2)
        if "route" in result:
            common["route"] = _compact_value(result["route"], depth=2)
        if "routes" in result:
            routes = result.get("routes") or []
            common["routes_count"] = len(routes) if hasattr(routes, "__len__") else None
            common["routes_preview"] = _compact_value(routes, depth=2)
        if "results" in result:
            rows = result.get("results") or []
            common["results_count"] = len(rows) if hasattr(rows, "__len__") else None
            common["results_preview"] = _compact_value(rows, depth=2)
        if tool_name == "score_route":
            for key in ("feasibility", "greenness", "confidence", "selectivity"):
                if key in result and key not in common:
                    common[key] = _compact_value(result[key])
        return common or _compact_value(result, depth=2)

    def checkpoint(self) -> None:
        """Write an interim ``transcript.json`` rollup mid-run WITHOUT closing
        the JSONL handle. Guards against losing the summary view if the
        process is killed before ``finalize()`` runs (e.g. the agent hangs in
        an LLM retry storm and the benchmark harness SIGKILLs it). Best-effort:
        never raises."""
        try:
            payload = self._build_payload(
                self.target_smiles, None,
                self.scratchpad_store, self.todo_store,
            )
            self._write_json(payload, final=False)
        except Exception as exc:
            logger.debug("transcript checkpoint failed: %s", exc)

    def _write_json(self, payload: Dict[str, Any], *, final: bool) -> None:
        json_path = os.path.join(self.session_dir, "transcript.json")
        try:
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
            if final:
                logger.info(
                    "transcript: wrote %d events / %d iterations / %d ranked routes to %s",
                    payload.get("stats", {}).get("n_events", 0),
                    len(payload.get("timeline", [])),
                    payload.get("scratchpad", {}).get("n_saved", 0),
                    json_path,
                )
        except Exception as exc:
            logger.warning("transcript.json write failed: %s", exc)

    def _build_payload(
        self,
        target_smiles: str,
        result: Optional[Dict[str, Any]],
        scratchpad_store: Any = None,
        todo_store: Any = None,
    ) -> Dict[str, Any]:
        """Build the transcript.json rollup payload from current events +
        live stores. Pure (no IO); shared by finalize() and checkpoint()."""
        # Group events by iteration to make the timeline easy to skim.
        iters: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        for ev in self.events:
            if ev["event"] == "iteration":
                if current is not None:
                    iters.append(current)
                current = {"n": ev.get("n"), "ts_rel": ev["ts_rel"], "events": []}
                continue
            if current is None:
                # Pre-iteration events (session_start, prefetch, …) live in a
                # synthetic bucket so the timeline stays linear.
                current = {"n": 0, "ts_rel": 0.0, "events": []}
            current["events"].append(ev)
        if current is not None:
            iters.append(current)

        ranked: List[Dict[str, Any]] = []
        if scratchpad_store is not None:
            try:
                routes = scratchpad_store.list()
                for rank, r in enumerate(routes, start=1):
                    ranked.append({
                        "rank": rank,
                        "label": r.label,
                        "total_score": round(r.total_score, 4),
                        "sub_scores": r.sub_scores,
                        "num_steps": len(r.steps),
                        "starting_materials": r.starting_materials,
                        "steps": r.steps,
                        "notes_excerpt": (r.notes[:400] + "…") if len(r.notes) > 400 else r.notes,
                    })
            except Exception as exc:
                logger.debug("scratchpad serialise failed: %s", exc)

        lessons: List[Dict[str, Any]] = []
        if scratchpad_store is not None:
            try:
                lessons = [l.to_dict() for l in scratchpad_store.lessons()]
            except Exception:
                pass

        todos: List[Dict[str, Any]] = []
        if todo_store is not None:
            try:
                todos = [it.to_dict() for it in todo_store.read()]
            except Exception:
                pass

        # Counter rollup so the reviewer doesn't have to re-aggregate.
        n_tool_calls = sum(1 for e in self.events if e["event"] == "tool_call")
        n_tool_results = sum(1 for e in self.events if e["event"] == "tool_result")
        llm_calls = [e for e in self.events if e["event"] == "llm_call"]
        tool_results = [e for e in self.events if e["event"] == "tool_result"]
        tool_name_counts: Dict[str, int] = {}
        for e in self.events:
            if e["event"] == "tool_call":
                tool_name_counts[e.get("name", "?")] = tool_name_counts.get(e.get("name", "?"), 0) + 1
        tool_elapsed_by_name: Dict[str, int] = {}
        for e in tool_results:
            elapsed = e.get("elapsed_ms")
            if isinstance(elapsed, (int, float)):
                name = e.get("name", "?")
                tool_elapsed_by_name[name] = tool_elapsed_by_name.get(name, 0) + int(elapsed)

        payload: Dict[str, Any] = {
            "target_smiles": target_smiles,
            "session_dir": self.session_dir,
            "elapsed_seconds": round(time.time() - self.t0, 2),
            "result": {
                "success": (result or {}).get("success"),
                "below_target": (result or {}).get("below_target"),
                "iterations": (result or {}).get("iterations"),
                "route_text": (result or {}).get("route_text"),
            },
            "stats": {
                "n_iterations": (result or {}).get("iterations"),
                "n_events": len(self.events),
                "n_tool_calls": n_tool_calls,
                "n_tool_results": n_tool_results,
                "n_llm_calls": len(llm_calls),
                "llm_elapsed_ms_total": sum(
                    int(e.get("elapsed_ms") or 0) for e in llm_calls
                    if isinstance(e.get("elapsed_ms"), (int, float))
                ),
                "tool_elapsed_ms_total": sum(
                    int(e.get("elapsed_ms") or 0) for e in tool_results
                    if isinstance(e.get("elapsed_ms"), (int, float))
                ),
                "tool_elapsed_ms_by_name": tool_elapsed_by_name,
                "tool_call_counts": tool_name_counts,
            },
            "scratchpad": {
                "n_saved": len(ranked),
                "routes": ranked,
                "lessons": lessons,
            },
            "todos": todos,
            "training_trace": {
                "schema_version": 1,
                "records": self.training_trace,
                "notes": (
                    "Compact post-training view derived from structured "
                    "events; full raw payloads remain under events/timeline."
                ),
            },
            "timeline": iters,
            # The raw event log — same content as transcript.jsonl, kept here
            # for convenience so a single file holds everything.
            "events": self.events,
        }
        for key in (
            "config",
            "model_settings",
            "prompt_artifacts",
            "postprocess",
            "resume_state",
        ):
            if key in self.metadata:
                payload[key] = self.metadata[key]
        payload["session_manifest_path"] = self.metadata.get("session_manifest_path")
        if self.metadata.get("latest_messages_path"):
            payload.setdefault("resume_state", {})
            payload["resume_state"]["latest_messages_path"] = self.metadata.get(
                "latest_messages_path"
            )
        if self.metadata.get("llm_requests_path"):
            payload.setdefault("resume_state", {})
            payload["resume_state"]["llm_requests_path"] = self.metadata.get(
                "llm_requests_path"
            )

        return payload
