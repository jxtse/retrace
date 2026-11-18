"""Filesystem-backed tool-result store with per-tool projection stubs.

Goal: keep the message stream small. The full JSON of every heavy tool result
goes to disk under ``runs/<session>/results/<tool_call_id>.json``; the model
sees only a compact summary stub (≤ ``summary_max_chars``, default 800). The
stub carries an ``_result_id`` so the model can call ``load_result`` to fetch
fields it needs.

Mirrors Claude Code's ``microCompact`` content-replace pattern but applied
**at append-time** rather than reactively (see
``../claude-code-source-code-main/src/services/compact/microCompact.ts``).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import tempfile
import threading
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-tool projection registry
# ---------------------------------------------------------------------------
#
# Each projector takes the parsed JSON payload (any) and returns a plain dict
# with only the fields the model needs to make the next decision. Keep the
# projection AGGRESSIVE — anything not here is recoverable via ``load_result``.

def _proj_neural_retro(payload: Any) -> dict:
    if not isinstance(payload, dict):
        return {"raw": str(payload)[:300]}
    preds = payload.get("predictions") or []
    # Cap mirrors `neural_retro.return_topk` (default 15) so every
    # candidate that survived clustering reaches the LLM. Was 10 before
    # 2026-05-15 — left a 5-entry gap when n_clusters >= 15 because
    # the upstream cap was bumped 12 → 15 without updating the stub.
    out_preds = []
    for i, p in enumerate(preds[:15]):
        # Keep only fields the LLM acts on. `bucket`/`cluster_size` are
        # internal dedup state and intentionally dropped.
        pred = {
            "rank": i + 1,
            "reactants": p.get("reactants"),
            "support": p.get("support"),
        }
        # Per-prediction confidence: the only ranking signal the engines
        # actually compute (e.g. a beam log-prob or template-relevance
        # score, normalised by the retroplanner /single_step service).
        # Lets the LLM tell a high-confidence top beam from a marginal
        # tail candidate.
        if p.get("score") is not None:
            pred["score"] = p.get("score")
        out_preds.append(pred)
    # Surface the full funnel so the LLM can see WHY only N predictions
    # survived when more were available pre-clustering. Order:
    #   per-engine emissions  →  n_unique (after canonical dedup)
    #     →  n_kept_after_sanity (LLM-judge filter)  →  n_clusters
    #     →  n_in_stub (this stub).
    n_unique = payload.get("n_unique_after_dedup")
    n_dropped = ((payload.get("dropped_by_sanity") or {}).get("n_dropped") or 0)
    n_kept_after_sanity = (
        max(0, int(n_unique) - int(n_dropped)) if n_unique is not None else None
    )
    out: dict[str, Any] = {
        "tool": "neural_retro",
        "engine": payload.get("engine"),
        "n_unique": n_unique,
        "n_dropped_by_sanity": n_dropped,
        "n_kept_after_sanity": n_kept_after_sanity,
        "n_clusters": payload.get("n_clusters"),
        "predictions": out_preds,
    }
    # Surface per-engine failure messages when any engine errored out,
    # otherwise the LLM sees a missing engine in `n_unique` with no
    # diagnosis and may waste retries. The success-case per-engine
    # counts are intentionally NOT surfaced — LLM has no action mapping
    # for "engine X returned 5 vs engine Y returned 7".
    engine_errors = payload.get("engine_errors") or {}
    if isinstance(engine_errors, dict) and engine_errors:
        out["engine_errors"] = {
            str(k): str(v)[:200] for k, v in engine_errors.items()
        }
    # A top-level `error` on a neural_retro payload means every engine failed
    # (total outage) — NOT a confirmed dead end. Surface it so the LLM treats
    # the empty prediction list as "retry / engines down", never as a leaf.
    if payload.get("error"):
        out["error"] = str(payload.get("error"))[:300]
        if payload.get("note"):
            out["note"] = str(payload.get("note"))[:200]
    return out


def _proj_suggest_templates(payload: Any) -> dict:
    if not isinstance(payload, dict):
        return {"raw": str(payload)[:300]}
    sugg = payload.get("suggestions") or []
    out = []
    for i, s in enumerate(sugg[:8]):
        out.append({
            "rank": i + 1,
            "kind": s.get("template_kind"),
            "source": s.get("template_source"),
            "reactants": s.get("reactants_smiles") or s.get("reactants"),
        })
    return {
        "tool": "suggest_templates",
        "n": payload.get("num_suggestions"),
        # enzymatic_in_top_k drives the "rerun with kind='enzymatic'?"
        # decision documented in prompts/sections/tools/suggest_templates.md;
        # silently dropping it left projector and prompt out of sync.
        "enzymatic_in_top_k": payload.get("enzymatic_in_top_k"),
        "suggestions": out,
    }


def _proj_explore_branches(payload: Any) -> dict:
    if not isinstance(payload, dict):
        return {"raw": str(payload)[:300]}
    ranked = payload.get("ranked") or []
    out = []
    for i, r in enumerate(ranked[:6]):
        if not isinstance(r, dict):
            continue
        support = r.get("support") or []
        if not isinstance(support, list):
            support = []
        n_strict = sum(1 for s in support if isinstance(s, dict) and s.get("match") == "strict")
        n_partial = sum(1 for s in support if isinstance(s, dict) and s.get("match") == "partial")
        entry = {
            "rank": i + 1,
            # Branch name is what the LLM passes back when it wants to
            # promote / refer to a branch in a later turn — ordinal is
            # fragile across re-ranks.
            "name": r.get("name"),
            "score": r.get("score", 0),
            "feasibility": r.get("feasibility", 0),
            "cost": r.get("cost", 0),
            "greenness": r.get("greenness", 0),
            "reactants": r.get("reactants"),
            "support_count_strict": n_strict,
            "support_count_partial": n_partial,
        }
        # Surface per-branch failure reason so a score=0 entry isn't
        # silently opaque (e.g. invalid SMILES, no reactants).
        if r.get("error"):
            entry["error"] = str(r["error"])[:200]
        out.append(entry)
    return {"tool": "explore_branches", "ranked": out}


def _proj_score_route(payload: Any) -> dict:
    """Projector for the new 3-pillar verifier (`score_route`)."""
    if not isinstance(payload, dict):
        return {"raw": str(payload)[:300]}
    keep = (
        "total_score", "feasibility_score", "cost_score", "greenness_score",
        "bottleneck", "n_steps", "warnings", "note",
    )
    out = {"tool": "score_route", **{k: payload.get(k) for k in keep if k in payload}}
    # Surface failure diagnostics so a timed-out / errored score_route isn't
    # a near-empty stub the LLM can't act on (mirrors _proj_explore_branches
    # / _proj_recall_similar_routes).
    if payload.get("error"):
        out["error"] = str(payload["error"])[:300]
    if payload.get("hint"):
        out["hint"] = str(payload["hint"])[:300]
    return out


def _proj_recall_similar_routes(payload: Any) -> dict:
    """Project a recall result for the in-context message stream.

    Keeps the full per-step chain (product / reactants / reaction_smarts /
    rxn_class) so the LLM can read the precedent strategy directly without
    a follow-up ``load_result`` round-trip. Drops fields the LLM doesn't
    use for strategy transfer:
      - ``route_id`` (only consumed by the agent loop's provenance
        bookkeeping, which reads from the raw pre-stub payload — see
        ``core.agent`` ``_last_recall_payload`` capture).
      - ``source_db`` (``is_enzymatic`` already covers the
        chemo-vs-enzyme classification the LLM cares about).
      - Per-step extras the index builder may attach (``conditions``,
        ``yield``, ``doi``, ``notes``) — useful for post-mortem, not for
        strategic anchor reading. Still on disk via ``load_result``.

    Spec §7 mitigation #1 (originally stripped ``route`` entirely; relaxed
    after user feedback that strategic anchors need the actual step chain
    to be useful, and the per-recall payload is small enough — ~5-15KB
    for a k=3 result — that lazy-loading every example wastes more turns
    than the bytes save).
    """
    if not isinstance(payload, dict):
        return {"raw": str(payload)[:300]}
    examples = payload.get("examples") or []
    out_examples = []
    for ex in examples[:5]:
        if not isinstance(ex, dict):
            continue
        # Extract the strategic-anchor fields from the full route.
        route = ex.get("route") or {}
        full_steps = route.get("steps") or []
        slim_steps = []
        for s in full_steps:
            if not isinstance(s, dict):
                continue
            slim_steps.append({
                "product": s.get("product"),
                "reactants": s.get("reactants"),
                "reaction_smarts": s.get("reaction_smarts"),
                "rxn_class": s.get("rxn_class"),
            })
        out_examples.append({
            "similarity": ex.get("similarity"),
            "source_target": ex.get("source_target"),
            "n_steps": ex.get("n_steps"),
            "is_enzymatic": ex.get("is_enzymatic"),
            "rxn_classes": ex.get("rxn_classes"),
            "steps": slim_steps,
        })
    out: dict[str, Any] = {
        "tool": "recall_similar_routes",
        "confidence": payload.get("confidence"),
        "max_similarity": payload.get("max_similarity"),
        "examples": out_examples,
    }
    # Only surface query_canonical when canonicalization actually rewrote
    # the LLM's input — otherwise it's identical to the args the LLM just
    # passed and pure repetition.
    qc = payload.get("query_canonical")
    if qc and qc != payload.get("query"):
        out["query_canonical"] = qc
    if "error" in payload:
        out["error"] = payload["error"]
    return out


def _proj_analyze_molecule(payload: Any) -> dict:
    if not isinstance(payload, dict):
        return {"raw": str(payload)[:300]}
    # Field names must match what AnalyzeMoleculeTool.execute actually emits
    # (tools/builtins/analyze_molecule.py:97-149). The earlier keep-tuple
    # used `num_rings`/`stereo_centers`/`num_atoms` which never appear in
    # the payload — every stub silently degraded to `{smiles,
    # functional_groups}` until 2026-05-18.
    #
    # Drug-likeness descriptors (rotatable_bonds, hbd, hba, logp, tpsa) and
    # Bertz complexity are intentionally NOT included — no prompt section
    # or gate maps them to a retrosynthesis action. mw / sa_score / scaffold
    # are retained for `detail_level='full'` calls because SA score gates
    # the "is this even synthesizable?" prior, mw informs leaf-buyability
    # expectations, and scaffold drives template kind selection.
    keep = (
        # Brief mode (default)
        "smiles", "functional_groups", "ring_count", "aromatic_rings",
        "chiral_centers",
        # Full mode adds these — pass-through when present
        "mw", "sa_score", "scaffold",
    )
    out = {"tool": "analyze_molecule"}
    for k in keep:
        if k in payload:
            v = payload[k]
            if isinstance(v, list) and len(v) > 12:
                v = v[:12] + ["…"]
            out[k] = v
    return out


_PROJECTORS = {
    "neural_retro": _proj_neural_retro,
    "suggest_templates": _proj_suggest_templates,
    "explore_branches": _proj_explore_branches,
    "score_route": _proj_score_route,
    "analyze_molecule": _proj_analyze_molecule,
    "recall_similar_routes": _proj_recall_similar_routes,
}


# Tools whose output is already small enough — never project, never persist
# (no point writing a 100-byte payload to disk).
_BYPASS_TOOLS = frozenset({
    "validate_smiles",
    "check_buyability",
    "todo_read", "todo_write",
    "scratchpad_save", "scratchpad_list", "scratchpad_get",
    "save_best_route", "list_best_routes", "get_best_route",
    "backtrack_and_summarize",
    "read_doc", "load_result",
    "compare_molecules",
})


class ResultStore:
    """Persist heavy tool results to disk; return summary stubs for the chat.

    ``persist`` is the hot path called from the agent loop. ``load`` is the
    companion read API exposed via the ``load_result`` tool.

    The store also tracks a per-tool ring-buffer of ``keep_recent_full_per_tool``
    most-recent IDs; results inside that window are returned full, results
    older than that are summarised.
    """

    def __init__(
        self,
        session_dir: str,
        summary_max_chars: int = 800,
        keep_recent_full_per_tool: int = 4,
    ):
        self.session_dir = session_dir
        self.results_dir = os.path.join(session_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.summary_max_chars = int(summary_max_chars)
        self.keep_recent_full_per_tool = int(keep_recent_full_per_tool)
        # Ring buffer of recent (tool_name -> [tool_call_ids…]). The tail
        # `keep_recent_full_per_tool` IDs stay full; older IDs are demoted to
        # stubs the next time `persist` is called.
        self._recent: dict[str, list[str]] = {}
        # tool_call_id -> {path, tool_name, full_content}.
        self._index: dict[str, dict] = {}
        # Track which IDs have already been demoted (stub written) so we can
        # rewrite their content in the message stream lazily.
        self._demoted: set[str] = set()
        # Mutator lock. Concurrent tool execution (services/concurrent_tools)
        # can call persist() from multiple worker threads in the same
        # iteration; without this the ring buffer / index / demoted set are
        # subject to lost-update races (a key written by thread A can be
        # silently overwritten before thread B's append). RLock so a single
        # thread can re-enter (e.g. persist → stub_for path).
        self._lock = threading.RLock()

    # ---- write path -----------------------------------------------------

    def persist(self, tool_name: str, tool_call_id: str, content: str) -> str:
        """Persist full result, return either the original content (recent
        window) or a JSON-serialised projection stub.

        Returns: a string suitable to drop into ``message["content"]``.
        """
        if not tool_name or tool_name in _BYPASS_TOOLS:
            return content if isinstance(content, str) else json.dumps(content, default=str)

        content_str = content if isinstance(content, str) else json.dumps(content, default=str)

        # Try to parse JSON; non-JSON tool output (rare) → skip projection,
        # still persist so model can re-fetch verbatim later.
        try:
            payload = json.loads(content_str)
        except Exception:
            payload = None

        # Inject _result_id on first write so the LLM can copy it into
        # downstream audit fields (score_route_result_id, etc.). The same key
        # is later set by _project() on demoted stubs — values agree.
        if isinstance(payload, dict) and "_result_id" not in payload:
            payload["_result_id"] = tool_call_id
            content_str = json.dumps(payload, ensure_ascii=False, default=str)

        # `_detail` convention: a tool may attach a heavy, audit-only block
        # under the reserved `_detail` key (e.g. score_route's per-step
        # feasibility/cost/greenness breakdown, neural_retro's per-engine raw
        # top-K). It is written to DISK in full — recoverable post-run via
        # load_result / offline audit — but stripped from the string that
        # re-enters the message stream, so it never costs context tokens, not
        # even during the first-write recency window. (Projection stubs already
        # drop it via the per-tool allowlist; this also covers the first write.)
        disk_payload = payload
        if isinstance(payload, dict) and "_detail" in payload:
            slim_payload = {k: v for k, v in payload.items() if k != "_detail"}
            content_str = json.dumps(slim_payload, ensure_ascii=False, default=str)

        # Write full payload to disk. Use a tmpfile + os.replace for
        # atomicity: a torn write (process killed mid-json.dump) would
        # otherwise leave a half-flushed file that load() / iter_by_tool
        # crash on with json.JSONDecodeError. os.replace is atomic on
        # POSIX within the same filesystem (and we keep the tmpfile in
        # self.results_dir to guarantee that).
        path = os.path.join(self.results_dir, f"{tool_call_id}.json")
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                prefix=f".{tool_call_id}.", suffix=".tmp", dir=self.results_dir,
            )
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump({
                        "tool": tool_name,
                        "tool_call_id": tool_call_id,
                        "ts": _dt.datetime.utcnow().isoformat(timespec="seconds"),
                        # `disk_payload` carries the full `_detail` block when
                        # present; `content_str` (returned below) has it stripped.
                        "content": disk_payload if disk_payload is not None else content_str,
                    }, f, ensure_ascii=False, default=str)
                os.replace(tmp_path, path)
            except Exception:
                # Clean up the tmpfile on failure so we don't leak.
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.error("ResultStore: failed to persist %s: %s", tool_call_id, e)
            # Return inline content since we cannot persist
            return content_str

        with self._lock:
            self._index[tool_call_id] = {
                "path": path,
                "tool_name": tool_name,
                "full_content": content_str,
            }

            # Update recency window.
            ring = self._recent.setdefault(tool_name, [])
            ring.append(tool_call_id)
            # Anything that fell out of the window joins the demote set (the
            # agent loop reads `pop_demoted()` and rewrites old `tool` messages).
            # IMPORTANT: also drop the in-memory ``full_content`` for demoted IDs
            # so that a long-running session does not accumulate every heavy
            # tool result in RAM forever (case-0 OOM root cause: chembart's 50-
            # prediction blobs × 45 iters × ~3 calls/iter never released).
            # ``stub_for`` / ``load`` will re-read from disk on demand.
            while len(ring) > self.keep_recent_full_per_tool:
                old_id = ring.pop(0)
                self._demoted.add(old_id)
                old_info = self._index.get(old_id)
                if old_info is not None:
                    old_info.pop("full_content", None)

        # First-write: return full content (still inside recency window).
        return content_str

    def stub_for(self, tool_call_id: str) -> str | None:
        """Return the projection stub for a previously-persisted result.
        Used by the agent loop to rewrite older `tool` messages in-place.
        """
        with self._lock:
            info = self._index.get(tool_call_id)
            if info is None:
                return None
            full = info.get("full_content")
            path = info["path"]
            tool_name = info["tool_name"]
        if full is None:
            # Already demoted — re-read from disk to project. Done outside
            # the lock because the disk I/O can block; the path/tool_name
            # snapshot above is enough.
            try:
                with open(path) as f:
                    blob = json.load(f)
                payload = blob.get("content")
                full = payload if isinstance(payload, str) else json.dumps(payload, default=str)
            except Exception as e:
                logger.warning("stub_for: re-read failed for %s: %s", tool_call_id, e)
                return None
        return self._project(tool_name, tool_call_id, full)

    def pop_demoted(self) -> list[str]:
        with self._lock:
            ids = list(self._demoted)
            self._demoted.clear()
        return ids

    # ---- maintenance ---------------------------------------------------

    @staticmethod
    def cleanup_old_sessions(
        runs_dir: str,
        retention_days: int,
        prefix: str = "_session_",
    ) -> int:
        """Delete session subdirectories older than ``retention_days``.

        Without this the runs/ tree grows unboundedly — each agent.run()
        creates a fresh ``_session_<id>/`` with potentially hundreds of
        per-tool result files. On a long-running eval box this can hit
        millions of inodes.

        Match policy: any direct child of ``runs_dir`` whose basename
        starts with ``prefix`` and whose mtime is older than the cutoff.
        Returns the number of session dirs removed; logs each deletion.
        Failures are swallowed (best-effort) — startup must never abort
        because TTL cleanup hit a permission error on one stale session.
        """
        import shutil
        if not runs_dir or not os.path.isdir(runs_dir):
            return 0
        retention_days = float(retention_days)
        if retention_days <= 0:
            return 0
        cutoff = _dt.datetime.utcnow().timestamp() - retention_days * 86400
        removed = 0
        try:
            entries = os.listdir(runs_dir)
        except OSError as e:
            logger.warning("cleanup_old_sessions: listdir(%s) failed: %s", runs_dir, e)
            return 0
        for name in entries:
            if not name.startswith(prefix):
                continue
            path = os.path.join(runs_dir, name)
            try:
                if not os.path.isdir(path):
                    continue
                if os.path.getmtime(path) >= cutoff:
                    continue
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
                logger.info(
                    "cleanup_old_sessions: removed %s (older than %.1f days)",
                    path, retention_days,
                )
            except OSError as e:
                logger.warning(
                    "cleanup_old_sessions: skipped %s (%s)", path, e,
                )
        return removed

    # ---- read path ------------------------------------------------------

    def load(self, result_id: str, fields: list[str] | None = None) -> dict:
        info = self._index.get(result_id)
        if info is None:
            # Try disk fallback (e.g. resume from another process).
            path = os.path.join(self.results_dir, f"{result_id}.json")
            if not os.path.exists(path):
                return {"error": f"unknown result_id: {result_id}"}
        else:
            path = info["path"]
        try:
            with open(path) as f:
                blob = json.load(f)
        except Exception as e:
            return {"error": f"load failed: {e}"}
        payload = blob.get("content")
        if fields and isinstance(payload, dict):
            payload = {k: payload.get(k) for k in fields}
        return {"result_id": result_id, "tool": blob.get("tool"), "content": payload}

    def iter_by_tool(self, tool_name: str) -> Iterator[tuple[str, Any]]:
        """Yield ``(tool_call_id, parsed_payload)`` for every persisted
        result of the given tool. Reads from disk for demoted IDs,
        in-memory cache for recent ones. Read-only — recency window
        and demoted set are not mutated.

        Used by ExploreBranchesTool for cross-source reverse lookup.
        """
        # In-memory index yields first (parsed payload available cheaply); the
        # on-disk pass below only fills in IDs persisted in earlier processes
        # that aren't tracked in self._index.
        seen: set[str] = set()
        # In-memory first: faster path for the common case. Snapshot the
        # index under the lock — `persist` mutates `self._index` from worker
        # threads, so iterating it live races with concurrent writes
        # (RuntimeError: dictionary changed size during iteration).
        with self._lock:
            index_items = list(self._index.items())
        for tcid, info in index_items:
            if info.get("tool_name") != tool_name:
                continue
            seen.add(tcid)
            payload = self._read_payload(tcid)
            if payload is not None:
                yield tcid, payload
        # Disk fallback for IDs persisted in earlier processes.
        if not os.path.isdir(self.results_dir):
            return
        for fname in os.listdir(self.results_dir):
            if not fname.endswith(".json"):
                continue
            tcid = fname[:-5]
            if tcid in seen:
                continue
            try:
                with open(os.path.join(self.results_dir, fname)) as f:
                    blob = json.load(f)
            except Exception as e:
                logger.debug("iter_by_tool: skip %s (%s)", fname, e)
                continue
            if blob.get("tool") != tool_name:
                continue
            payload = blob.get("content")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    pass
            yield tcid, payload

    def _read_payload(self, tool_call_id: str) -> Any | None:
        """Internal: parse the persisted payload of an in-memory ID."""
        info = self._index.get(tool_call_id)
        if info is None:
            return None
        full = info.get("full_content")
        if full is None:
            try:
                with open(info["path"]) as f:
                    blob = json.load(f)
                payload = blob.get("content")
                if isinstance(payload, str):
                    try:
                        return json.loads(payload)
                    except Exception:
                        return payload
                return payload
            except Exception as e:
                logger.debug("_read_payload: %s read failed: %s", tool_call_id, e)
                return None
        try:
            return json.loads(full)
        except Exception:
            return full

    # ---- projection ----------------------------------------------------

    def _project(self, tool_name: str, tool_call_id: str, content_str: str) -> str:
        try:
            payload = json.loads(content_str)
        except Exception:
            payload = content_str
        proj = _PROJECTORS.get(tool_name)
        if proj is not None:
            try:
                stub = proj(payload)
            except Exception as e:
                logger.warning("Projector for %s failed: %s", tool_name, e)
                stub = {"tool": tool_name, "_proj_error": str(e)}
        else:
            # Default: truncate.
            txt = content_str if isinstance(content_str, str) else json.dumps(payload, default=str)
            stub = {
                "tool": tool_name,
                "preview": txt[: self.summary_max_chars],
                "truncated": len(txt) > self.summary_max_chars,
            }
        stub["_result_id"] = tool_call_id
        stub["_load_hint"] = f"load_result(result_id='{tool_call_id}') for full payload"
        out = json.dumps(stub, ensure_ascii=False, default=str)
        if len(out) > self.summary_max_chars * 2:
            # Hard cap — drop predictions/suggestions list further. Raised
            # 3→8 on 2026-05-18: at compaction.max_tokens=180k the room
            # exists, and clipping to 3 was repeatedly hiding viable
            # disconnections from the LLM on branchy targets.
            for shrink_key in ("predictions", "suggestions", "ranked"):
                if shrink_key in stub and isinstance(stub[shrink_key], list):
                    stub[shrink_key] = stub[shrink_key][:8]
            out = json.dumps(stub, ensure_ascii=False, default=str)
        return out
