"""
Concurrent tool execution.
Executes multiple independent tool calls in parallel using ThreadPoolExecutor.

Also exposes a ``run_parallel_branches`` helper used by
``ExploreBranchesTool`` to fan out the first-level retrosynthesis
disconnection across K alternative reactant sets in a single LLM turn.
This is beam-search-in-one-call: the agent proposes K branches, the
helper scores each in parallel with cheap heuristics (or a caller-
supplied scorer), and only the ranked top-K survive to the next turn —
so the agent can do breadth-first expansion without spending K whole
iterations on it.
"""

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeout, as_completed
from typing import Any, Callable, Iterable

from tools.base import ToolRegistry

logger = logging.getLogger(__name__)

# Default per-tool wallclock budget. Production reality check: a single hung
# HTTP engine (ChemBART has been observed to wedge on bootstrap for ~6 min)
# would otherwise freeze the entire iteration with no way for the LLM to
# route around. Anything past this returns a synthetic `{"error": "tool
# timeout after Ns"}` to the LLM so it can pick a different tool. Override
# per-call via `tool_timeout_s=` or globally via agent.tool_timeout_s in
# core.config / local override.
DEFAULT_TOOL_TIMEOUT_S: float = 300.0


def _hb(msg: str) -> None:
    """Heartbeat: stamped + flushed line so a 30+ minute tool call never
    looks like a hang in the batch stdout log."""
    line = f"[hb {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        sys.stdout.flush()
    except Exception:
        pass


def _timeout_payload(fn_name: str, timeout_s: float) -> str:
    """Synthetic tool result returned to the LLM when a tool exceeds its
    per-call wallclock budget. Phrased so the LLM can plausibly route
    around it (e.g. fall back to suggest_templates when neural_retro times
    out)."""
    return json.dumps({
        "error": f"tool timeout after {timeout_s:.0f}s",
        "tool": fn_name,
        "hint": (
            "The tool exceeded its wallclock budget and was cancelled. "
            "Try a different tool or simplify the input."
        ),
    })


def execute_tools_concurrent(
    registry: ToolRegistry,
    tool_calls: list[dict],
    max_workers: int = 4,
    tool_timeout_s: float | None = None,
) -> list[dict]:
    """
    Execute multiple tool calls concurrently.
    Returns list of {"tool_call_id": ..., "content": ..., "elapsed_ms": int}
    in original order.

    Each tool is bounded by ``tool_timeout_s`` (default
    ``DEFAULT_TOOL_TIMEOUT_S``). On timeout the future is abandoned (the
    underlying Python thread cannot be force-killed, but
    ``executor.shutdown(wait=False, cancel_futures=True)`` releases queued
    work and the result slot is filled with a synthetic timeout payload so
    the LLM sees a real ``tool_result`` and can pick a different tool.
    """
    timeout_s = float(
        tool_timeout_s if tool_timeout_s is not None else DEFAULT_TOOL_TIMEOUT_S
    )
    if len(tool_calls) <= 1:
        # No benefit from parallelism — still catch exceptions AND apply a
        # timeout so one bad tool doesn't freeze the iteration.
        results = []
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            _hb(f"tool ▶ {fn_name}")
            try:
                fn_args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError) as parse_err:
                logger.error("Tool %s: JSON parse error: %s", fn_name, parse_err)
                result = json.dumps({"error": f"Invalid JSON arguments: {parse_err}", "tool": fn_name})
                elapsed_ms = 0
                _hb(f"tool ◀ {fn_name} parse_error 0ms")
                results.append({
                    "tool_call_id": tc["id"],
                    "role": "tool",
                    "content": result,
                    "elapsed_ms": elapsed_ms,
                    "tool_name": fn_name,
                })
                continue
            t0 = time.monotonic()
            # Wrap single-tool execution in a thread pool too so the
            # ``future.result(timeout=)`` path is reused and timeouts are
            # enforced consistently with the parallel branch.
            #
            # NB: do NOT use ``with ThreadPoolExecutor() as solo:`` here. On a
            # timeout we want to abandon the (still-running) worker, but a
            # ``with`` block's ``__exit__`` calls ``shutdown(wait=True)`` which
            # re-joins the hung thread — silently defeating the timeout. We
            # manage shutdown manually instead: ``wait=False`` on timeout,
            # ``wait=True`` on the normal/error paths.
            solo = ThreadPoolExecutor(max_workers=1)
            fut = solo.submit(registry.execute, fn_name, fn_args)
            try:
                result = fut.result(timeout=timeout_s)
            except _FutureTimeout:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                logger.error(
                    "Tool %s timed out after %.0fs", fn_name, timeout_s
                )
                _hb(f"tool ◀ {fn_name} TIMEOUT {elapsed_ms}ms")
                # Best-effort: don't wait for the (likely hung) call.
                solo.shutdown(wait=False, cancel_futures=True)
                results.append({
                    "tool_call_id": tc["id"],
                    "role": "tool",
                    "content": _timeout_payload(fn_name, timeout_s),
                    "elapsed_ms": elapsed_ms,
                    "tool_name": fn_name,
                })
                continue
            except Exception as e:
                logger.error("Tool %s failed: %s", fn_name, e)
                result = json.dumps({"error": str(e), "tool": fn_name})
                solo.shutdown(wait=False)
            else:
                solo.shutdown(wait=False)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _hb(f"tool ◀ {fn_name} {elapsed_ms}ms")
            results.append({
                "tool_call_id": tc["id"],
                "role": "tool",
                "content": result,
                "elapsed_ms": elapsed_ms,
                "tool_name": fn_name,
            })
        return results

    results = [None] * len(tool_calls)

    def _run(idx: int, tc: dict) -> tuple[int, str, int]:
        fn_name = tc["function"]["name"]
        _hb(f"tool ▶ {fn_name} (parallel)")
        t0 = time.monotonic()
        try:
            fn_args = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, KeyError) as parse_err:
            logger.error("Tool %s: JSON parse error: %s", fn_name, parse_err)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _hb(f"tool ◀ {fn_name} parse_error {elapsed_ms}ms")
            return idx, json.dumps({"error": f"Invalid JSON arguments: {parse_err}", "tool": fn_name}), elapsed_ms
        try:
            out = registry.execute(fn_name, fn_args)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _hb(f"tool ◀ {fn_name} {elapsed_ms}ms")
            return idx, out, elapsed_ms
        except Exception as e:
            logger.error("Tool %s failed: %s", fn_name, e)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _hb(f"tool ◀ {fn_name} ERROR {elapsed_ms}ms: {e}")
            return idx, json.dumps({"error": str(e), "tool": fn_name}), elapsed_ms

    deadline = time.monotonic() + timeout_s
    executor = ThreadPoolExecutor(max_workers=min(max_workers, len(tool_calls)))
    try:
        futures = {executor.submit(_run, i, tc): i for i, tc in enumerate(tool_calls)}
        # Use as_completed with the global per-call deadline. If any future
        # is still pending when we exit, fill its slot with a timeout payload
        # rather than blocking the iteration forever.
        try:
            for future in as_completed(futures, timeout=timeout_s):
                try:
                    idx, result, elapsed_ms = future.result()
                    results[idx] = {
                        "tool_call_id": tool_calls[idx]["id"],
                        "role": "tool",
                        "content": result,
                        "elapsed_ms": elapsed_ms,
                        "tool_name": tool_calls[idx]["function"]["name"],
                    }
                except Exception as e:
                    # Prevent one future failure from blocking others
                    idx = futures[future]
                    logger.error(
                        "Future result failed for tool %s: %s",
                        tool_calls[idx]["function"]["name"], e,
                    )
                    results[idx] = {
                        "tool_call_id": tool_calls[idx]["id"],
                        "role": "tool",
                        "content": json.dumps({"error": f"Future execution failed: {e}"}),
                        "elapsed_ms": 0,
                        "tool_name": tool_calls[idx]["function"]["name"],
                    }
        except _FutureTimeout:
            # Any still-pending future blew the deadline. Mark each
            # uncompleted slot with a synthetic timeout payload. The
            # underlying threads cannot be force-killed (Python limitation),
            # but cancel_futures=True at shutdown drops queued work and
            # we no longer wait on the in-flight ones.
            for fut, idx in futures.items():
                if results[idx] is not None:
                    continue
                fn_name = tool_calls[idx]["function"]["name"]
                logger.error(
                    "Tool %s did not complete within %.0fs (concurrent batch)",
                    fn_name, timeout_s,
                )
                _hb(f"tool ◀ {fn_name} TIMEOUT (concurrent batch)")
                results[idx] = {
                    "tool_call_id": tool_calls[idx]["id"],
                    "role": "tool",
                    "content": _timeout_payload(fn_name, timeout_s),
                    "elapsed_ms": int(timeout_s * 1000),
                    "tool_name": fn_name,
                }
    finally:
        # wait=False so we don't block on hung tools; cancel_futures=True
        # drops anything that hadn't started yet.
        executor.shutdown(wait=False, cancel_futures=True)

    return results  # type: ignore


def run_parallel_branches(
    branches: list[Any],
    scorer: Callable[[Any], dict],
    max_workers: int = 8,
    timeout_s: float | None = None,
) -> list[dict]:
    """Score a list of candidate branches in parallel.

    ``scorer(branch)`` is called once per branch and must return a dict.
    The dict is passed through to the caller along with the original
    branch under ``"branch"``, plus ``"_index"`` preserving the input
    order. Exceptions in ``scorer`` are swallowed and reported as
    ``{"error": ...}`` so one bad branch cannot sink the whole fan-out.

    ``timeout_s`` (default ``DEFAULT_TOOL_TIMEOUT_S``) bounds the
    aggregate wallclock; any branch that hasn't completed by then is
    surfaced as an ``error`` slot rather than blocking the whole call.

    Order of the returned list matches the input order — the caller is
    responsible for sorting by whatever field they ranked on (usually
    ``score`` or a composite).
    """
    if not branches:
        return []

    deadline_s = float(
        timeout_s if timeout_s is not None else DEFAULT_TOOL_TIMEOUT_S
    )
    results: list[dict | None] = [None] * len(branches)

    def _run(idx: int, branch: Any) -> tuple[int, dict]:
        try:
            out = scorer(branch) or {}
            if not isinstance(out, dict):
                out = {"result": out}
            out.setdefault("branch", branch)
            out["_index"] = idx
            return idx, out
        except Exception as e:
            logger.exception("Branch scorer failed on index %d", idx)
            return idx, {"_index": idx, "branch": branch, "error": str(e)}

    workers = max(1, min(max_workers, len(branches)))
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {executor.submit(_run, i, b): i for i, b in enumerate(branches)}
        try:
            for future in as_completed(futures, timeout=deadline_s):
                try:
                    idx, out = future.result()
                    results[idx] = out
                except Exception as e:
                    idx = futures[future]
                    logger.exception("Future result failed for branch %d: %s", idx, e)
                    results[idx] = {"_index": idx, "branch": branches[idx], "error": f"Future failed: {e}"}
        except _FutureTimeout:
            for fut, idx in futures.items():
                if results[idx] is not None:
                    continue
                logger.error(
                    "Branch %d (%r) did not complete within %.0fs",
                    idx, branches[idx], deadline_s,
                )
                results[idx] = {
                    "_index": idx,
                    "branch": branches[idx],
                    "error": f"branch timeout after {deadline_s:.0f}s",
                }
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return [r for r in results if r is not None]
