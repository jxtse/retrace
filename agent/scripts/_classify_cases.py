#!/usr/bin/env python3
"""Supervision classifier for the opus-4.8 comparison run.

Scans a results directory (retro-190 batch layout OR JMC/FDA runs-json layout)
and labels every case so the supervisor can rerun only the ones that need it:

  SOLVED        success AND a real scored route (best total_score > 0)
  FAKE_SUCCESS  success banner / success=true BUT best route score is 0.0 / no
                real route — the native_bailout mislabel (see memory
                bailout-success-flag-mislabel). This is "假成功" — RERUN.
  NO_ROUTE      clean run, finished, but no route found (legit negative — keep)
  FLAP_KILLED   proxy 500-storm / prefetch TimeoutError signature — RERUN
  IMPORT_CRASH  ModuleNotFoundError / ImportError at startup — RERUN (env bug)
  TIMEOUT       killed_by_timeout / wall exceeded — inspect, usually RERUN
  CRASH         non-zero exit / fatal traceback, not flap/import — inspect
  RUNNING       no terminal state yet (live)
  MISSING       expected output absent

Only FAKE_SUCCESS / FLAP_KILLED / IMPORT_CRASH / CRASH / TIMEOUT /
RUNNING_OR_CRASH go on the rerun list by default; NO_ROUTE is a legitimate
result (the model genuinely found nothing) and SOLVED is done. Usage:

    python scripts/_classify_cases.py <results_dir> [--rerun-list]

Layout is auto-detected: a dir containing <NN>__<label>/stdout.log children is
batch; a dir containing runs/*.json (or *.json with a "success" field) is JMC/FDA.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from collections import Counter

FLAP_RE = re.compile(r"fetch failed|HTTP 500|attempt \d+/\d+|Prefetch \w+ failed: TimeoutError")
IMPORT_RE = re.compile(r"ModuleNotFoundError|ImportError|cannot import name")
SUCCESS_RE = re.compile(r"Success \|")
FAIL_RE = re.compile(r"❌ Failed|No route found|no viable route")


def _read(p: Path, limit: int = 200_000) -> str:
    try:
        return p.read_text(errors="replace")[:limit]
    except Exception:
        return ""


def classify_batch_case(case_dir: Path) -> dict:
    label = case_dir.name
    log = case_dir / "stdout.log"
    tr = case_dir / "transcript.json"
    if not log.exists():
        return {"case": label, "status": "MISSING", "why": "no stdout.log"}
    text = _read(log)
    flap_hits = len(FLAP_RE.findall(text))
    has_tr = tr.exists() and tr.stat().st_size > 0
    best_score = None
    if has_tr:
        try:
            td = json.loads(_read(tr, 5_000_000))
            routes = (td.get("scratchpad") or {}).get("routes") or td.get("routes") or []
            scores = []
            for r in routes:
                if isinstance(r, dict):
                    s = r.get("total_score")
                    if not isinstance(s, (int, float)):
                        s = (r.get("sub_scores") or {}).get("total_score")
                    if isinstance(s, (int, float)):
                        scores.append(s)
            best_score = max(scores) if scores else None
        except Exception:
            pass
    # priority order: import bug > flap > success-vs-fake > timeout > fail > running
    if IMPORT_RE.search(text):
        return {"case": label, "status": "IMPORT_CRASH", "why": "import error at startup", "flap": flap_hits}
    banner = bool(SUCCESS_RE.search(text))
    has_real_route = isinstance(best_score, (int, float)) and best_score > 0
    failed_cleanly = bool(FAIL_RE.search(text))
    if banner or has_tr:
        # success/transcript present — but trust the SCORE, not the flag
        # (native_bailout sets success=True on a 0.0 stub; that's 假成功).
        if has_real_route:
            return {"case": label, "status": "SOLVED", "score": best_score, "flap": flap_hits}
        # banner/transcript but no real scored route
        if banner:
            return {"case": label, "status": "FAKE_SUCCESS",
                    "why": f"success banner but best_score={best_score}", "flap": flap_hits}
        if failed_cleanly:
            return {"case": label, "status": "NO_ROUTE", "score": best_score, "flap": flap_hits}
        return {"case": label, "status": "RUNNING_OR_CRASH",
                "why": f"transcript present but no terminal marker; best_score={best_score}",
                "score": best_score, "flap": flap_hits}
    if flap_hits >= 15:
        return {"case": label, "status": "FLAP_KILLED", "why": f"{flap_hits} flap hits, no route", "flap": flap_hits}
    if "killed_by_timeout" in text and "true" in text.lower():
        return {"case": label, "status": "TIMEOUT", "flap": flap_hits}
    if FAIL_RE.search(text):
        return {"case": label, "status": "NO_ROUTE", "flap": flap_hits}
    # no terminal marker, no transcript
    if flap_hits >= 5:
        return {"case": label, "status": "FLAP_KILLED", "why": f"{flap_hits} flap hits", "flap": flap_hits}
    return {"case": label, "status": "RUNNING_OR_CRASH", "why": "no terminal marker", "flap": flap_hits}


def classify_json_case(jf: Path) -> dict:
    label = jf.stem
    try:
        rec = json.loads(_read(jf, 5_000_000))
    except Exception as e:
        return {"case": label, "status": "MISSING", "why": f"unreadable json: {e}"}
    err = (rec.get("error") or "") + (rec.get("traceback") or "")
    flap_hits = len(FLAP_RE.findall(err))
    if IMPORT_RE.search(err):
        return {"case": label, "status": "IMPORT_CRASH", "why": err[:120]}
    if FLAP_RE.search(err):
        return {"case": label, "status": "FLAP_KILLED", "why": err[:120], "flap": flap_hits}
    if rec.get("success"):
        # trust the score, not the flag: native_bailout sets success=True on a
        # 0.0-score stub (假成功). Pull best total_score from best_route/all_routes.
        best = rec.get("best_route") or {}
        score = None
        if isinstance(best, dict):
            score = best.get("total_score")
            if not isinstance(score, (int, float)):
                score = (best.get("sub_scores") or {}).get("total_score")
        if not isinstance(score, (int, float)):
            # fall back to max over all_routes
            allr = rec.get("all_routes") or []
            cand = []
            for r in allr if isinstance(allr, list) else []:
                if isinstance(r, dict):
                    s = r.get("total_score") or (r.get("sub_scores") or {}).get("total_score")
                    if isinstance(s, (int, float)):
                        cand.append(s)
            score = max(cand) if cand else None
        if isinstance(score, (int, float)) and score > 0:
            return {"case": label, "status": "SOLVED", "score": score}
        return {"case": label, "status": "FAKE_SUCCESS",
                "why": f"success=true but best total_score={score}"}
    if err:
        return {"case": label, "status": "CRASH", "why": err[:120]}
    # success false, no error -> legit no-route
    return {"case": label, "status": "NO_ROUTE"}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1]).expanduser().resolve()
    want_rerun = "--rerun-list" in sys.argv[2:]
    if not root.exists():
        print(f"ERR: {root} does not exist")
        return 2

    results = []
    def _is_current_path(p: Path) -> bool:
        rel = p.relative_to(root)
        return not any(part.startswith(("_bad", "_stale", "_partial")) for part in rel.parts)

    # batch layout: <root>/run/<NN>__<label>/stdout.log  OR <root>/<NN>__<label>/
    batch_dirs = sorted([
        d for d in root.rglob("*__*")
        if (d / "stdout.log").exists() and _is_current_path(d)
    ])
    if batch_dirs:
        for d in batch_dirs:
            results.append(classify_batch_case(d))
    else:
        # json layout: <root>/runs/*.json or <root>/*.json with success field
        json_files = sorted(root.rglob("*.json"))
        json_files = [
            j for j in json_files
            if j.name not in ("summary.json", "manifest.json")
            and "_launch" not in str(j)
            and _is_current_path(j)
        ]
        for j in json_files:
            txt = _read(j, 2000)
            if '"success"' in txt or '"smiles"' in txt:
                results.append(classify_json_case(j))

    counts = Counter(r["status"] for r in results)
    rerun = [r for r in results if r["status"] in
             ("FAKE_SUCCESS", "FLAP_KILLED", "IMPORT_CRASH", "CRASH",
              "TIMEOUT", "RUNNING_OR_CRASH")]

    if want_rerun:
        for r in rerun:
            print(r["case"])
        return 0

    print(f"=== {root} ===")
    print(f"total cases: {len(results)}")
    for status, n in counts.most_common():
        print(f"  {status:18s} {n}")
    if rerun:
        print(f"\n--- RERUN candidates ({len(rerun)}) ---")
        for r in rerun:
            print(f"  {r['case']:30s} {r['status']:14s} {r.get('why','')}")
    # show a few SOLVED scores for sanity
    solved = [r for r in results if r["status"] == "SOLVED"]
    if solved:
        scored = [r for r in solved if isinstance(r.get("score"), (int, float))]
        if scored:
            ss = sorted(r["score"] for r in scored)
            print(f"\nSOLVED scores: n={len(scored)} min={ss[0]:.3f} "
                  f"med={ss[len(ss)//2]:.3f} max={ss[-1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
