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
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
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


def _pool_initializer(gpu_ids: list[int] | None):
    """Pin each pool worker to one GPU via CUDA_VISIBLE_DEVICES (round-robin over PIDs)."""
    if not gpu_ids:
        return
    gpu = gpu_ids[os.getpid() % len(gpu_ids)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


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
    from core.config import load_config

    overrides = {}
    if os.environ.get("AGENT_MODEL"):
        overrides.setdefault("llm", {})["model"] = os.environ["AGENT_MODEL"]
    config = load_config(
        os.environ.get("AGENT_CONFIG"),
        thinking=os.environ.get("AGENT_THINKING", "fast"),
        cli_overrides=overrides or None,
    )

    t0 = time.time()
    record = {
        "idx": idx,
        "label": label,
        "smiles": smiles,
        "ref_step": job.get("ref_step"),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

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
        for j in jobs:
            print(f"[bench] start idx={j['idx']:03d} {j['label']}", flush=True)
            r = run_one(j)
            done += 1
            print(f"[bench] {done}/{len(jobs)} done {r}", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            max_tasks_per_child=1,
            initializer=_pool_initializer,
            initargs=(gpu_ids,),
        ) as ex:
            futs = {ex.submit(run_one, j): j for j in jobs}
            for fut in as_completed(futs):
                j = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"tag": f"{j['idx']:03d}_{j.get('label','?')}", "error": str(e)}
                done += 1
                print(f"[bench] {done}/{len(jobs)} {r}", flush=True)

    print(f"[bench] elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
