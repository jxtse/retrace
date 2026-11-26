#!/usr/bin/env python
"""
Run the retrosynthesis agent on the retro-190 hard-test set
(`dataset/retro-190_dataset/cases.csv`, derived from routes_possible_test_hard.pkl).

Per-target outputs:
  runs/{idx:03d}.json
  transcripts/{idx:03d}.txt
  logs/{idx:03d}.log
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path

# Per-case RSS watchdog. retro-190 has a few cases whose SEARCH STATE (not the
# target molecule — the biggest target is only 67 atoms) balloons memory:
# explore_branches fan-out + multi-engine neural_retro candidate sets + the
# scratchpad accumulate until a single worker's RSS hits ~21GB, which OOM-kills
# the box (host-f has 30GB and vllm lives in VRAM, so all RAM is the worker's).
# The watchdog polls RSS; when a case crosses MEM_SKIP_GB it writes a `skipped`
# record (so --skip-existing won't re-run it) and hard-exits the worker — the
# ProcessPool (max_tasks_per_child=1) respawns a clean child for the next case.
# Tunable via env MEM_SKIP_GB (default 24GB: above normal big cases ~ slightly
# below the 30GB OOM wall, leaving headroom to write the skip json + exit).
MEM_SKIP_GB = float(os.environ.get("MEM_SKIP_GB", "24"))


def _rss_gb() -> float:
    """Current process resident-set size in GiB (Linux /proc, no deps)."""
    try:
        with open("/proc/self/statm") as fh:
            pages = int(fh.read().split()[1])  # resident pages
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3)
    except Exception:
        return 0.0


def _start_mem_watchdog(out_json: Path, record: dict, limit_gb: float):
    """Background thread: hard-exit this worker if RSS exceeds limit_gb.

    Writes a `skipped` result first so --skip-existing skips the case and the
    benchmark stats can tell a memory-skip apart from a real failure. Uses
    os._exit so no atexit/GC runs (we're already near the memory ceiling)."""
    stop = threading.Event()

    def _watch():
        while not stop.wait(5.0):
            rss = _rss_gb()
            if rss >= limit_gb:
                rec = dict(record)
                rec["success"] = False
                rec["skipped"] = "mem_limit"
                rec["skip_rss_gb"] = round(rss, 1)
                rec["error"] = f"mem watchdog: RSS {rss:.1f}GB >= {limit_gb}GB"
                rec["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                try:
                    with open(out_json, "w", encoding="utf-8") as f:
                        json.dump(rec, f, ensure_ascii=False, indent=2, default=str)
                except Exception:
                    pass
                sys.stderr.write(
                    f"[mem-watchdog] idx={record.get('idx')} RSS {rss:.1f}GB "
                    f">= {limit_gb}GB — skipping case, exiting worker\n")
                sys.stderr.flush()
                # Hard-exit THIS isolated child only. The driver (serial or
                # parallel) runs each case in its own spawned Process, reaps
                # this exit, and schedules the next job — see main()'s loop.
                os._exit(99)

    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    return stop

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BENCH_DIR = Path(__file__).resolve().parent
CASES_CSV = BENCH_DIR / "cases.csv"
RUNS_DIR = BENCH_DIR / "runs"
TR_DIR = BENCH_DIR / "transcripts"
LOG_DIR = BENCH_DIR / "logs"


def load_targets() -> list[dict]:
    import csv as _csv

    rows: list[dict] = []
    with open(CASES_CSV, encoding="utf-8", newline="") as f:
        reader = _csv.DictReader(f)
        for r in reader:
            smi = r["smiles"].strip()
            if not smi:
                continue
            n_steps_raw = (r.get("n_steps") or "").strip()
            try:
                n_steps = int(n_steps_raw) if n_steps_raw else None
            except ValueError:
                n_steps = None
            rows.append({
                "idx": int(r["idx"]),
                "smiles": smi,
                "ref_step": n_steps,
                "label": f"r190_{int(r['idx']):03d}",
            })
    return rows


def safe_name(s: str) -> str:
    s = s.replace(" ", "_").replace("/", "_")
    out = "".join(c if (c.isalnum() or c in "-_") else "_" for c in s)
    return out[:50] or "unknown"


def _run_one_pinned(job: dict, gpu_id: int | None):
    """Child entrypoint for parallel mode: pin this process to one GPU (via
    CUDA_VISIBLE_DEVICES) before running the case. Runs in its own spawned
    process so the case's memory watchdog (os._exit) only kills this child."""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    run_one(job)


def _run_jobs_parallel(jobs: list[dict], workers: int,
                       gpu_ids: list[int] | None,
                       child_target=_run_one_pinned) -> int:
    """Run `jobs` with at most `workers` ISOLATED child processes in flight.

    We deliberately do NOT use ProcessPoolExecutor: the per-case memory
    watchdog ends a worker with os._exit(99) (an abrupt, non-clean exit),
    which ProcessPoolExecutor treats as a dead worker and reacts to by marking
    the whole executor BrokenProcessPool — failing every pending/running future
    and stopping the run. max_tasks_per_child=1 only respawns after a CLEAN
    task return, not after os._exit, so it does not save us; empirically a
    single mem-skip zeroed out all remaining cases.

    Instead we spawn one Process per case (same isolation as the serial
    branch) and bound concurrency by hand. An os._exit then kills only that
    one child; it has already written its own case json (success or
    `skipped: mem_limit`), so the driver just reaps it and schedules the next
    job. Returns the number of jobs reaped.
    """
    import multiprocessing as _mp

    ctx = _mp.get_context("spawn")
    pending = list(jobs)
    running: dict = {}  # Process -> (job, slot)
    free_slots = list(range(max(1, workers)))
    done = 0
    total = len(jobs)

    def _launch(j: dict, slot: int):
        env_gpu = gpu_ids[slot % len(gpu_ids)] if gpu_ids else None
        p = ctx.Process(target=child_target, args=(j, env_gpu))
        p.start()
        running[p] = (j, slot)
        print(f"[bench] start idx={j['idx']:03d} {j['label']}"
              + (f" gpu={env_gpu}" if env_gpu is not None else ""), flush=True)

    while pending or running:
        while pending and free_slots:
            _launch(pending.pop(0), free_slots.pop())
        # Reap any finished child (poll; children self-write their json).
        time.sleep(0.2)
        for p in list(running):
            if p.is_alive():
                continue
            j, slot = running.pop(p)
            free_slots.append(slot)
            done += 1
            tag = f"{j['idx']:03d}_{safe_name(j['label'])}"
            note = "ok" if p.exitcode == 0 else f"child_exit={p.exitcode}"
            print(f"[bench] {done}/{total} done {tag} ({note})", flush=True)

    return done


def run_one(job: dict) -> dict:
    import logging as _lg

    idx = job["idx"]
    label = job["label"] or f"target{idx}"
    smiles = job["smiles"]

    runs_dir = Path(job.get("runs_dir") or RUNS_DIR)
    tr_dir = Path(job.get("tr_dir") or TR_DIR)
    log_dir = Path(job.get("log_dir") or LOG_DIR)
    runs_dir.mkdir(parents=True, exist_ok=True)
    tr_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{idx:03d}_{safe_name(label)}"
    out_json = runs_dir / f"{tag}.json"
    out_tr = tr_dir / f"{tag}.txt"
    out_log = log_dir / f"{tag}.log"

    root = _lg.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = _lg.FileHandler(out_log, mode="w", encoding="utf-8")
    fh.setFormatter(_lg.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    root.addHandler(fh)
    root.setLevel(_lg.INFO)

    from core import build_agent
    from core.config import load_benchmark_config

    config = load_benchmark_config()

    t0 = time.time()
    record = {
        "idx": idx,
        "label": label,
        "smiles": smiles,
        "ref_step": job.get("ref_step"),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    mem_stop = _start_mem_watchdog(out_json, record, MEM_SKIP_GB)
    try:
        agent = build_agent(config)
        result = agent.run(smiles)
        record["success"] = bool(result.get("success"))
        record["below_target"] = bool(result.get("below_target", False))
        record["iterations"] = result.get("iterations")
        record["wall_seconds"] = round(result.get("time", time.time() - t0), 1)
        record["route_text"] = result.get("route_text", "")
        record["best_route"] = result.get("best_route")
        record["all_routes"] = result.get("all_routes")
        record["postprocess"] = result.get("postprocess")
        record["session_dir"] = result.get("session_dir")
        record["transcript_json"] = result.get("transcript_json")

        msgs = result.get("messages", [])
        with open(out_tr, "w", encoding="utf-8") as f:
            f.write(f"# retro-190 target {idx}\n")
            f.write(f"# SMILES: {smiles}\n")
            f.write(f"# Ref step count: {job.get('ref_step')}\n\n")
            for m in msgs:
                role = m.get("role", "?")
                f.write(f"\n========== {role.upper()} ==========\n")
                if "content" in m and m.get("content"):
                    f.write(str(m["content"]))
                    f.write("\n")
                if "tool_calls" in m:
                    for tc in m["tool_calls"]:
                        fn = tc.get("function", tc) if isinstance(tc, dict) else tc.function
                        if isinstance(fn, dict):
                            tname = fn.get("name", "?")
                            targs = fn.get("arguments", "")
                        else:
                            tname = getattr(fn, "name", "?")
                            targs = getattr(fn, "arguments", "")
                        f.write(f"[tool_call] {tname}({targs})\n")
                if m.get("role") == "tool":
                    f.write(f"[tool_result for {m.get('tool_call_id','?')}]\n")
    except Exception as e:
        record["success"] = False
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()
        record["wall_seconds"] = round(time.time() - t0, 1)
        with open(out_tr, "w", encoding="utf-8") as f:
            f.write(record["traceback"])
    finally:
        mem_stop.set()
        record["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, default=str)

    return {
        "tag": tag,
        "success": record.get("success"),
        "below_target": record.get("below_target"),
        "wall": record.get("wall_seconds"),
        "iters": record.get("iterations"),
        "error": record.get("error"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", default="", help="comma list of idx to include")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--output-dir",
        default="",
        help="parent dir for runs/transcripts/logs (default: bench dir itself)",
    )
    ap.add_argument("--gpu-ids", default="",
                    help="comma list of GPU indices to round-robin across workers (sets CUDA_VISIBLE_DEVICES per worker)")
    args = ap.parse_args()
    from core.config import load_benchmark_config

    try:
        load_benchmark_config()
    except ValueError as exc:
        ap.error(str(exc))

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()] if args.gpu_ids else None

    global RUNS_DIR, TR_DIR, LOG_DIR
    if args.output_dir.strip():
        base = Path(args.output_dir).expanduser().resolve()
        RUNS_DIR = base / "runs"
        TR_DIR = base / "transcripts"
        LOG_DIR = base / "logs"

    wanted = set()
    if args.only.strip():
        wanted = {int(x) for x in args.only.split(",") if x.strip()}

    targets = load_targets()
    jobs: list[dict] = []
    for row in targets:
        if wanted and row["idx"] not in wanted:
            continue
        tag = f"{row['idx']:03d}_{safe_name(row['label'])}"
        if args.skip_existing and (RUNS_DIR / f"{tag}.json").exists():
            continue
        row = dict(row)
        row["runs_dir"] = str(RUNS_DIR)
        row["tr_dir"] = str(TR_DIR)
        row["log_dir"] = str(LOG_DIR)
        jobs.append(row)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    TR_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[bench] {len(jobs)} jobs (of {len(targets)} targets), workers={args.workers}", flush=True)

    t0 = time.time()
    done = 0
    if args.workers <= 1:
        # Serial mode, but run each case in an ISOLATED child process so the
        # per-case memory watchdog (_start_mem_watchdog → os._exit) kills only
        # that child, not this driver. The child writes the case json (success
        # or `skipped: mem_limit`) itself; the parent just waits and moves on.
        # Without this, run_one would execute in THIS process and os._exit
        # would terminate the whole run. (Plain Process, not ProcessPool, to
        # avoid BrokenProcessPool poisoning when a child self-exits.)
        import multiprocessing as _mp
        ctx = _mp.get_context("spawn")
        for j in jobs:
            print(f"[bench] start idx={j['idx']:03d} {j['label']}", flush=True)
            p = ctx.Process(target=run_one, args=(j,))
            p.start()
            p.join()
            done += 1
            tag = f"{j['idx']:03d}_{safe_name(j['label'])}"
            note = "ok" if p.exitcode == 0 else f"child_exit={p.exitcode}"
            print(f"[bench] {done}/{len(jobs)} done {tag} ({note})", flush=True)
    else:
        done = _run_jobs_parallel(jobs, args.workers, gpu_ids)

    print(f"[bench] elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
