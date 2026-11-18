"""Agent runner: runs ``agent.run(smiles)`` in a background thread and
bridges the synchronous ``trace_callback`` to an ``asyncio.Queue`` so an
SSE endpoint on the main event loop can drain events as they arrive.

Each ``RunSession`` is independent — its own ``LLMClient`` (so per-session
model/preset swap works), its own ``RetrosynthesisAgent``, its own
session directory under ``runs/_session_{sid}``. No global state is
mutated.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import traceback
from copy import deepcopy
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Event injected once the run thread finishes — drains the SSE stream.
SENTINEL_DONE = "_done"
SENTINEL_FAIL = "_failed"


def _safe_jsonable(obj: Any) -> Any:
    """Best-effort make ``obj`` JSON-serialisable. Falls back to ``str``."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        if isinstance(obj, dict):
            return {k: _safe_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_safe_jsonable(v) for v in obj]
        return str(obj)


class RunSession:
    """One in-flight retrosynthesis run.

    Lifecycle:
        rs = RunSession(...)
        rs.bind_loop(loop)        # main event loop ref for thread-safe pushes
        rs.start(...)             # spawns the background thread
        async for evt in rs.iter_events(): ...
        await rs.wait_result()    # awaits final agent.run() dict
    """

    def __init__(self, session_id: str, base_config: dict):
        self.session_id = session_id
        self._base_config = deepcopy(base_config)
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.status: str = "pending"     # pending → running → done / error / cancelled
        self.error: Optional[str] = None
        self.effective_config: dict[str, Any] = {}
        self.result: Optional[dict[str, Any]] = None

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self._thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        # Hold every event we emit so a client that connects mid-run can
        # replay history. Capped to avoid runaway memory if the client
        # never connects.
        self._event_log: list[dict] = []
        self._event_log_cap = 5000
        # Live agent ref so /routes can query the scratchpad mid-run.
        self.agent = None

    # ── setup ───────────────────────────────────────────────────────────

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def start(
        self,
        smiles: str,
        family: str,
        preset: str,
        overrides: dict | None = None,
    ) -> None:
        from webapp.server import presets as preset_mod

        if self._loop is None:
            raise RuntimeError("call bind_loop() before start()")
        if self._thread is not None:
            raise RuntimeError("session already started")

        config = deepcopy(self._base_config)
        preset_mod.apply(config, family, preset)
        if overrides:
            for k, v in overrides.items():
                if v is None:
                    continue
                if k == "max_iterations":
                    config["max_iterations"] = int(v)
                    config.setdefault("agent", {})["max_iterations"] = int(v)
                elif k == "max_depth":
                    config["max_depth"] = int(v)
                    config.setdefault("agent", {})["max_depth"] = int(v)
                elif k == "wallclock_s":
                    config.setdefault("agent", {})["wallclock_s"] = float(v)
                elif k == "stop_after_distinct_routes":
                    config.setdefault("agent", {})["stop_after_distinct_routes"] = int(v)

        self.effective_config = {
            "llm_model": config.get("llm_model"),
            "llm_reasoning_effort": config.get("llm_reasoning_effort"),
            "max_iterations": config.get("max_iterations"),
            "max_depth": config.get("max_depth"),
            "stop_after_distinct_routes": config.get("agent", {}).get("stop_after_distinct_routes"),
            "family": family,
            "preset": preset,
        }

        self._thread = threading.Thread(
            target=self._run_worker,
            args=(smiles, config),
            name=f"agent-run-{self.session_id}",
            daemon=True,
        )
        self.status = "running"
        self._thread.start()

    # ── cancellation ────────────────────────────────────────────────────

    def cancel(self) -> bool:
        if self.status not in ("pending", "running"):
            return False
        self._cancel_event.set()
        return True

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    # ── event bridge ────────────────────────────────────────────────────

    def _push_event(self, event: str, payload: dict) -> None:
        evt = {"event": event, "payload": _safe_jsonable(payload), "ts": time.time()}
        if len(self._event_log) < self._event_log_cap:
            self._event_log.append(evt)
        if self._loop is None or self._loop.is_closed():
            return
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, evt)
        except RuntimeError:
            pass

    def _trace_callback(self, event: str, **kwargs) -> None:
        # Mid-run cancellation: raise from the trace callback so it
        # propagates out of agent.run(). The agent loop has broad
        # except-Exception sites but the bail-out path still records the
        # bail reason in the final result.
        if self._cancel_event.is_set():
            raise RuntimeError("cancelled by user")
        self._push_event(event, kwargs)

    # ── worker ──────────────────────────────────────────────────────────

    def _run_worker(self, smiles: str, config: dict) -> None:
        # Force the agent's own result_store/transcript naming to align
        # with our session_id so /results/{tcid} can read directly off
        # runs/_session_{sid}/results/{tcid}.json.
        prior_env = os.environ.get("AGENT_SESSION_ID")
        os.environ["AGENT_SESSION_ID"] = self.session_id
        try:
            try:
                from core import build_agent
            except Exception as e:
                self._fail(f"failed to import core agent: {e}")
                return

            try:
                agent = build_agent(config)
                agent._user_trace = self._trace_callback
                agent.trace = self._trace_callback
            except Exception as e:
                logger.exception("build_agent failed")
                self._fail(f"build_agent failed: {e}")
                return
            self.agent = agent

            try:
                result = agent.run(smiles)
            except Exception as e:
                logger.exception("agent.run failed")
                self.result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
                self.status = "cancelled" if self._cancel_event.is_set() else "error"
                self.error = str(e)
                self.finished_at = time.time()
                self._push_event(SENTINEL_FAIL, {"error": str(e)})
                return

            self.result = _safe_jsonable(result)
            self.finished_at = time.time()
            self.status = "cancelled" if self._cancel_event.is_set() else "done"
            self._push_event(SENTINEL_DONE, {
                "success": bool(result.get("success")),
                "iterations": result.get("iterations"),
                "time": result.get("time"),
                "n_routes": len(result.get("all_routes") or []),
            })
        finally:
            if prior_env is None:
                os.environ.pop("AGENT_SESSION_ID", None)
            else:
                os.environ["AGENT_SESSION_ID"] = prior_env

    def _fail(self, msg: str) -> None:
        self.error = msg
        self.status = "error"
        self.finished_at = time.time()
        self._push_event(SENTINEL_FAIL, {"error": msg})

    # ── consumers ───────────────────────────────────────────────────────

    async def iter_events(self):
        """Async generator yielding events as they arrive.

        Replays the in-memory log first (so a client reconnecting mid-run
        sees what it missed), then streams live until the worker emits a
        ``_done`` / ``_failed`` sentinel.
        """
        replayed = 0
        for evt in list(self._event_log):
            yield evt
            replayed += 1

        while True:
            evt = await self._queue.get()
            # Skip events already replayed.
            if replayed > 0 and evt in self._event_log[:replayed]:
                continue
            yield evt
            if evt["event"] in (SENTINEL_DONE, SENTINEL_FAIL):
                break

    async def wait_result(self, timeout: float | None = None) -> dict:
        """Await the final agent.run() dict (or raise TimeoutError)."""
        deadline = time.time() + timeout if timeout else None
        while self.result is None and self.status in ("pending", "running"):
            if deadline and time.time() > deadline:
                raise asyncio.TimeoutError()
            await asyncio.sleep(0.1)
        return self.result or {"success": False, "error": self.error or "unknown"}
