#!/usr/bin/env python
"""
Run the retrosynthesis agent on the 2025 FDA small-molecule benchmark.

For every row in `2025FDA小分子汇总.csv` we run TWO jobs:
  - variant=stereo    -> uses the original `SMILES` column (with stereo)
  - variant=canonical -> uses `SMILES_canonical` (no stereo, same input
                          ChemBart MCTS used for the published comparison)

Per-job outputs:
  runs/{idx:02d}_{Name}_{variant}.json    structured result
  transcripts/{idx:02d}_{Name}_{variant}.txt   full message stream
  transcripts/{idx:02d}_{Name}_{variant}.json  agent-native structured transcript
  transcripts/{idx:02d}_{Name}_{variant}.jsonl agent-native event stream
  logs/{idx:02d}_{Name}_{variant}.log       stderr/info log

Usage:
  .venv/bin/python dataset/benchmark/small_mol/run_benchmark.py \
      --workers 4 [--variants stereo,canonical] [--only 0,1,2] [--skip-existing]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

BENCH_DIR = Path(__file__).resolve().parent
CSV_PATH = BENCH_DIR / "2025FDA小分子汇总.csv"
RUNS_DIR = BENCH_DIR / "runs"
TR_DIR = BENCH_DIR / "transcripts"
LOG_DIR = BENCH_DIR / "logs"


def load_targets() -> list[dict]:
    # The CSV is GBK-encoded.
    with open(CSV_PATH, encoding="gbk", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            name = (r.get("Name") or "").strip()
            smi = (r.get("SMILES") or "").strip()
            smi_c = (r.get("SMILES_canonical") or "").strip()
            idx_s = (r.get("idx") or "").strip()
            if not name or not smi:
                continue
            try:
                idx = int(idx_s)
            except Exception:
                continue
            rows.append({
                "idx": idx,
                "medicine": (r.get("Medicine") or "").strip(),
                "name": name,
                "smiles_stereo": smi,
                "smiles_canonical": smi_c or smi,
                "ref_step": (r.get("Step") or "").strip(),
                "chembart_step": (r.get("Step_MCTS") or "").strip(),
                "chembart_full_success": (r.get("FULL_success") or "").strip(),
                "chembart_notes": (r.get("逐步复现") or "").strip(),
            })
        rows.sort(key=lambda x: x["idx"])
        return rows


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:60]


def copy_structured_transcripts(result: dict, tag: str, tr_dir: Path,
                                record: dict) -> None:
    """Copy agent-native transcript artefacts into this benchmark output."""
    record["session_dir"] = result.get("session_dir")
    if result.get("transcript_stats") is not None:
        record["transcript_stats"] = result.get("transcript_stats")

    for key, suffix in (
        ("transcript_json", ".json"),
        ("transcript_jsonl", ".jsonl"),
    ):
        src_raw = result.get(key)
        if not src_raw:
            continue
        src = Path(src_raw)
        record[f"session_{key}"] = str(src)
        if not src.exists():
            record[f"{key}_copy_error"] = "source missing"
            continue
        dst = tr_dir / f"{tag}{suffix}"
        try:
            shutil.copy2(src, dst)
            record[key] = str(dst)
        except Exception as exc:
            record[key] = str(src)
            record[f"{key}_copy_error"] = str(exc)


def _pool_initializer(gpu_ids: list[int] | None):
    """Pin each pool worker to one GPU via CUDA_VISIBLE_DEVICES (round-robin over PIDs)."""
    if not gpu_ids:
        return
    gpu = gpu_ids[os.getpid() % len(gpu_ids)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def run_one(job: dict) -> dict:
    """Worker: build agent, run on one SMILES, persist outputs."""
    import logging as _lg

    idx = job["idx"]
    name = job["name"]
    variant = job["variant"]
    smiles = job["smiles"]

    runs_dir = Path(job.get("runs_dir") or RUNS_DIR)
    tr_dir = Path(job.get("tr_dir") or TR_DIR)
    log_dir = Path(job.get("log_dir") or LOG_DIR)
    runs_dir.mkdir(parents=True, exist_ok=True)
    tr_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{idx:02d}_{safe_name(name)}_{variant}"
    out_json = runs_dir / f"{tag}.json"
    out_tr = tr_dir / f"{tag}.txt"
    out_log = log_dir / f"{tag}.log"

    # configure per-process logging
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
        "medicine": job.get("medicine"),
        "name": name,
        "variant": variant,
        "smiles": smiles,
        "ref_step": job.get("ref_step"),
        "chembart_step": job.get("chembart_step"),
        "chembart_full_success": job.get("chembart_full_success"),
        "chembart_notes": job.get("chembart_notes"),
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
        copy_structured_transcripts(result, tag, tr_dir, record)

        # Persist transcript
        msgs = result.get("messages", [])
        with open(out_tr, "w", encoding="utf-8") as f:
            f.write(f"# Target {idx} {name} ({variant})\n# SMILES: {smiles}\n\n")
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
            json.dump(record, f, ensure_ascii=False, indent=2)

    return {
        "tag": tag, "success": record.get("success"),
        "below_target": record.get("below_target"),
        "wall": record.get("wall_seconds"),
        "iters": record.get("iterations"),
        "error": record.get("error"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--variants", default="stereo,canonical",
                    help="comma list: stereo,canonical")
    ap.add_argument("--only", default="",
                    help="comma list of idx to include")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--gpu-ids", default="",
                    help="comma list of GPU indices to round-robin across workers (sets CUDA_VISIBLE_DEVICES per worker)")
    ap.add_argument(
        "--output-dir",
        default="",
        help="parent dir for runs/transcripts/logs (default: bench dir itself)",
    )
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    wanted = set()
    if args.only.strip():
        wanted = {int(x) for x in args.only.split(",") if x.strip()}
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()] if args.gpu_ids else None

    global RUNS_DIR, TR_DIR, LOG_DIR
    if args.output_dir.strip():
        base = Path(args.output_dir).expanduser().resolve()
        RUNS_DIR = base / "runs"
        TR_DIR = base / "transcripts"
        LOG_DIR = base / "logs"

    targets = load_targets()
    jobs: list[dict] = []
    for row in targets:
        if wanted and row["idx"] not in wanted:
            continue
        for v in variants:
            smi = row["smiles_stereo"] if v == "stereo" else row["smiles_canonical"]
            if not smi:
                continue
            tag = f"{row['idx']:02d}_{safe_name(row['name'])}_{v}"
            if args.skip_existing and (RUNS_DIR / f"{tag}.json").exists():
                continue
            jobs.append({
                "idx": row["idx"],
                "medicine": row["medicine"],
                "name": row["name"],
                "variant": v,
                "smiles": smi,
                "ref_step": row["ref_step"],
                "chembart_step": row["chembart_step"],
                "chembart_full_success": row["chembart_full_success"],
                "chembart_notes": row["chembart_notes"],
                "runs_dir": str(RUNS_DIR),
                "tr_dir": str(TR_DIR),
                "log_dir": str(LOG_DIR),
            })

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    TR_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[bench] {len(jobs)} jobs ({len(targets)} targets × {len(variants)} variants), workers={args.workers}", flush=True)

    t0 = time.time()
    done = 0
    if args.workers <= 1:
        for j in jobs:
            print(f"[bench] start {j['idx']:02d} {j['name']} {j['variant']}", flush=True)
            r = run_one(j)
            done += 1
            print(f"[bench] {done}/{len(jobs)} done {r}", flush=True)
    else:
        # max_tasks_per_child=1 — recycle the worker after every target so
        # message-stream / sqlite cache / template-library memory does not
        # accumulate. On a 32 GB host two long-running workers easily push
        # past 30 GB anon-rss after a single 5000s target and OOM-kill the
        # pool, taking every pending future down with them. Recycling adds
        # ~5 s of import overhead per target — negligible vs 800-5000s runs.
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
                    r = {"tag": f"{j['idx']}_{j['variant']}", "error": str(e)}
                done += 1
                print(f"[bench] {done}/{len(jobs)} {r}", flush=True)

    print(f"[bench] elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
