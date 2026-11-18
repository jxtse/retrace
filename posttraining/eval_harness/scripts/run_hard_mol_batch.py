#!/usr/bin/env python3
"""Run the agent across the hard_mol benchmark (11 Chematica targets).

Each target runs in a fresh subprocess (one-shot `python main.py run "<SMILES>"`)
so that:
  - sqlite caches in runs/_session_{session}/ are per-target (no cross-target locking)
  - one OOM / segfault doesn't take down the whole batch
  - per-target wall-clock timeout actually works (subprocess.kill on timeout)

Pre-flight checks before any target runs:
  - LLM proxy at config.llm.base_url is reachable
  - ReactionT5 HTTP server (config.verifier.forward.http_url) responds with loaded=true
  - ChemBart endpoint pings (best-effort; not a hard fail)

Outputs:
  - dataset/benchmark/hard_mol/results/<run_id>/
      summary.json           — one row per target with status / score / wall / iters / route
      <case_idx>__<label>/
          stdout.log         — full subprocess stdout+stderr
          session_dir.txt    — pointer to runs/_session_*/ for transcript.json
          transcript.json    — copy of the agent transcript for offline analysis
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import threading

# SIGUSR1 asks the batch scheduler to drain: finish in-flight cases but do not
# launch any new ones. This enables lossless worker-count changes.
_DRAIN_REQUESTED = threading.Event()


def _request_drain(signum, frame):
    _DRAIN_REQUESTED.set()
    print("[batch] SIGUSR1 received: draining in-flight cases, no new launches", flush=True)


signal.signal(signal.SIGUSR1, _request_drain)
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
SRC_XLSX = REPO / "dataset/benchmark/hard_mol/sources/chematica_exp_mol.xlsx"
SRC_CSV = REPO / "dataset/benchmark/hard_mol/cases.csv"
RESULTS_ROOT = REPO / "dataset/benchmark/hard_mol/results"


# ── Loading the dataset ──────────────────────────────────────────────────

def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def load_cases(csv_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load benchmark targets from a cases.csv.

    Schema-tolerant: hard_mol uses columns (idx, name, smiles, note, ref_step);
    retro-190 uses (idx, smiles, n_steps). Falls back to the legacy 11-row
    chematica xlsx only for the default hard_mol path when no csv exists."""
    if csv_path:
        src = Path(csv_path)
        if not src.is_absolute():
            src = REPO / src
        if not src.exists():
            raise FileNotFoundError(f"--cases-csv not found: {src}")
    else:
        src = SRC_CSV
    if src.exists():
        import csv as _csv
        out: List[Dict[str, Any]] = []
        with open(src, encoding="utf-8", newline="") as f:
            for row in _csv.DictReader(f):
                smi = (row.get("smiles") or "").strip()
                if not smi:
                    continue
                step_raw = (row.get("ref_step") or row.get("n_steps") or "").strip()
                try:
                    ref_steps = int(step_raw) if step_raw else None
                except ValueError:
                    ref_steps = None
                out.append({
                    "idx": int(row["idx"]),
                    "smiles": smi,
                    "ref_steps": ref_steps,
                    "chematica_note": (row.get("note") or "").strip(),
                    "label": (row.get("name") or f"case-{row['idx']}").strip(),
                })
        return out
    import openpyxl
    wb = openpyxl.load_workbook(SRC_XLSX, data_only=True)
    ws = wb["Sheet1"]
    out = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            continue  # header
        smi = row[0]
        if not smi:
            continue
        out.append({
            "idx": len(out),
            "smiles": smi.strip(),
            "ref_steps": row[1],
            "chematica_note": (row[2] or "").strip(),
            "label": (row[3] or f"case-{len(out)}").strip(),
        })
    return out


def slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s.strip())
    return s.strip("-").lower()[:60] or "case"


# ── Pre-flight ───────────────────────────────────────────────────────────

def http_get(url: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def preflight(config: Dict[str, Any]) -> Dict[str, Any]:
    report = {}

    # LLM proxy
    base = (config.get("llm") or {}).get("base_url", "")
    base = base.rstrip("/")
    if base:
        # OpenAI-compatible: a 404 on / is fine (proxy alive); only timeout / refused = down
        try:
            urllib.request.urlopen(base + "/models", timeout=5)
            report["llm"] = {"url": base, "ok": True, "method": "GET /models"}
        except urllib.error.HTTPError as e:
            # 404/401/etc still means a server is up
            report["llm"] = {"url": base, "ok": True, "method": f"GET /models -> HTTP {e.code}"}
        except Exception as e:
            report["llm"] = {"url": base, "ok": False, "error": f"{type(e).__name__}: {e}"}
    else:
        report["llm"] = {"ok": False, "error": "no base_url in config"}

    # ReactionT5 HTTP server (optional; if not set, in-process fallback is used)
    forward = (config.get("verifier") or {}).get("forward") or {}
    t5_url = (forward.get("http_url") or "").rstrip("/")
    if t5_url:
        info = http_get(t5_url + "/")
        if info and info.get("loaded"):
            report["reactiont5"] = {"url": t5_url, "ok": True, "device": info.get("device"),
                                    "model": info.get("model")}
        else:
            report["reactiont5"] = {"url": t5_url, "ok": False, "error": "no loaded=true response"}
    else:
        report["reactiont5"] = {"ok": True, "note": "in-process backend (no http_url)"}

    # ChemBart (best-effort)
    chembart = (config.get("neural_retro") or {}).get("chembart_url", "")
    if chembart:
        info = http_get(chembart.rstrip("/") + "/", timeout=10)
        report["chembart"] = {"url": chembart, "ok": bool(info), "note": "soft check"}

    return report


# ── Per-target run ───────────────────────────────────────────────────────

def find_session_dir(stdout_text: str, started_after: float) -> Optional[Path]:
    """Locate the runs/_session_*/ that this run created.

    Strategy: parse 'session_dir' or '_session_TIMESTAMP' from stdout, falling
    back to the newest runs/_session_*/ directory whose mtime is after
    `started_after`.
    """
    # Try to grab from stdout (the agent logs it as part of transcript writes).
    # Session dirs are named `_session_auto-YYYYMMDD-HHMMSS-PID-<hash>` or the
    # bare `_session_default` fallback when no per-run id was assigned. Match
    # the auto- form first; we intentionally do NOT accept `_session_default`
    # because it never holds a transcript.json (just shared cache dbs).
    # AUTHORITATIVE: the transcript-write banner the recorder prints at the
    # very end of the run ("transcript: wrote N events ... to <sess>/transcript").
    # Use the LAST such line — with postprocess continuations the recorder can
    # write more than once, and earlier session mentions in stdout (cache paths,
    # recall references) must not win. The old first-match re.search + the
    # mtime fallback mis-assigned 9/40 sessions in an earlier batch run
    # under --workers 2 (concurrent worker\'s newer session dir won the mtime
    # race), silently swapping per-case transcripts across cases.
    writes = re.findall(
        r"transcript: wrote \d+ events[^\n]*? to (/[^\s]+/runs/_session_[^/\s]+)/transcript",
        stdout_text,
    )
    for cand in reversed(writes):
        p = Path(cand)
        if p.exists():
            return p
    m = re.search(r"(/[^\s]+/runs/_session_auto-\d{8}-\d{6}-\d+-[0-9a-f]+)", stdout_text)
    if m:
        p = Path(m.group(1))
        if p.exists():
            return p
    # LAST-RESORT fallback: newest matching dir created after `started_after`.
    # Unsafe under concurrency (see above) — tag-only, callers must verify
    # target_smiles before trusting a session found this way.
    runs = REPO / "runs"
    if not runs.exists():
        return None
    candidates = []
    for d in runs.iterdir():
        if d.is_dir() and d.name.startswith("_session_"):
            try:
                if d.stat().st_mtime >= started_after - 5:
                    candidates.append((d.stat().st_mtime, d))
            except OSError:
                continue
    candidates.sort(reverse=True)
    return candidates[0][1] if candidates else None


def parse_route_text(text: str) -> Dict[str, Any]:
    """Extract structured info from the assistant's COMPLETE ROUTE block."""
    out: Dict[str, Any] = {"steps": [], "starting_materials": [], "raw_block": ""}
    # The marker can appear multiple times in stdout (heartbeat echoes of the
    # final message, tool-error lines, startup text). Scan ALL occurrences and
    # keep the LAST block that actually parses Step lines; a first-match
    # re.search silently returned a step-less echo block and the caller then
    # dropped the route entirely (observed: scorefix test40 cases 80/158/66/105).
    blocks = [m.group(0) for m in re.finditer(
        r"COMPLETE ROUTE FOUND:.*?(?=\n\n|\Z)", text, re.DOTALL)]
    if not blocks:
        return out
    step_re = re.compile(r"^\s*Step\s+\d+\s*:", re.M)
    with_steps = [b for b in blocks if step_re.search(b)]
    block = with_steps[-1] if with_steps else blocks[-1]
    out["raw_block"] = block
    for line in block.splitlines():
        sm = re.match(r"\s*Step\s+(\d+)\s*:\s*(.+?)\s*>>\s*(.+?)\s*(?:\(reaction:\s*(.+?)\))?\s*$", line)
        if sm:
            out["steps"].append({
                "step": int(sm.group(1)),
                "product": sm.group(2).strip(),
                "precursors": [p.strip() for p in sm.group(3).split(".")],
                "rxn": (sm.group(4) or "").strip(),
            })
        sm2 = re.match(r"\s*Starting materials\s*:\s*(.+)$", line)
        if sm2:
            out["starting_materials"] = [s.strip() for s in sm2.group(1).split(",")]
    return out


def _read_rss_kb(pid: int) -> Optional[int]:
    """Current RSS (VmRSS) in kB from /proc/<pid>/status. None if exited."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except Exception:
        return None
    return None


def _read_peak_rss_kb(pid: int) -> Optional[int]:
    """Best-effort: VmHWM (peak RSS) in kB from /proc/<pid>/status.
    Returns None if the process has already exited."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except Exception:
        return None
    return None


def run_one(case: Dict[str, Any], out_dir: Path, timeout_s: int,
            extra_env: Dict[str, str], config_path: str | None = None,
            rss_cap_gb: int = 12, gpu_id: Optional[int] = None,
            model: Optional[str] = None,
            thinking: str = "fast") -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Idempotent skip: if this case already produced a transcript.json, a
    # prior (or concurrent) batch process finished it. Don't re-run — this
    # lets a second batch process safely share the same slice / results-root
    # to add throughput without colliding on or overwriting done cases.
    existing_tr = out_dir / "transcript.json"
    if existing_tr.exists() and existing_tr.stat().st_size > 0:
        # A stale/contaminated copy (session mis-assignment, see
        # find_session_dir) must not be trusted as "done" for this case.
        try:
            _probe = json.loads(existing_tr.read_text())
            _t = _probe.get("target_smiles")
            if _t and _t != case["smiles"]:
                print(f"[batch] idx={case['idx']} existing transcript is for a "
                      f"DIFFERENT target; quarantining and re-running")
                existing_tr.rename(out_dir / "transcript.MISMATCHED.json")
                existing_tr = out_dir / "transcript.json"  # now absent
        except Exception:
            pass
    if existing_tr.exists() and existing_tr.stat().st_size > 0:
        print(f"[batch] idx={case['idx']} already done "
              f"({_display_path(existing_tr)}), skipping")
        rec = {
            "idx": case["idx"],
            "label": case["label"],
            "smiles": case["smiles"],
            "skipped_already_done": True,
            "out_dir": _display_path(out_dir),
        }
        log_path = out_dir / "stdout.log"
        try:
            stdout_text = log_path.read_text(errors="replace") if log_path.exists() else ""
            rec["success_marker"] = "Success |" in stdout_text
            rec["fail_marker"] = "❌ Failed" in stdout_text
        except Exception:
            pass
        try:
            td = json.loads(existing_tr.read_text())
            result = td.get("result") or {}
            rec["transcript_summary"] = {
                "elapsed_seconds": td.get("elapsed_seconds"),
                "iterations": result.get("iterations"),
                "success": result.get("success"),
                "below_target": result.get("below_target"),
                "n_events": len(td.get("events", [])),
            }
            if result.get("success") and not rec.get("success_marker"):
                rec["success_marker"] = True
            routes = (
                (td.get("scratchpad") or {}).get("routes")
                or td.get("routes")
                or td.get("ranked_routes")
                or []
            )

            def _route_score(r):
                if not isinstance(r, dict):
                    return None
                s = r.get("total_score")
                if not isinstance(s, (int, float)):
                    s = (r.get("sub_scores") or {}).get("total_score")
                return s if isinstance(s, (int, float)) else None

            if routes:
                scores = [s for s in (_route_score(r) for r in routes) if s is not None]
                rec["best_route_score"] = max(scores) if scores else None
                rec["n_ranked_routes"] = len(routes)
        except Exception as e:
            rec["transcript_parse_error"] = str(e)
        return rec
    log_path = out_dir / "stdout.log"
    py = str(REPO / ".venv/bin/python")
    cmd = [py, "-u", "main.py", "run", "--thinking", thinking]
    if config_path:
        cmd.extend(["--config", config_path])
    if model:
        cmd.extend(["--model", model])
    cmd.append(case["smiles"])
    # F6: prlimit --as is unusable here — PyTorch+CUDA reserve >32 GB of
    # address space at import time and immediately segfault under any AS
    # cap. Instead we poll /proc/<pid>/status and SIGKILL the process group
    # if RSS exceeds rss_cap_gb. Polled every 10 s while the agent runs.
    rss_cap_kb = rss_cap_gb * 1024 * 1024 if rss_cap_gb and rss_cap_gb > 0 else 0
    env = os.environ.copy()
    env.update(extra_env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    started = time.time()
    started_iso = datetime.now().isoformat(timespec="seconds")
    rec: Dict[str, Any] = {
        "idx": case["idx"],
        "label": case["label"],
        "smiles": case["smiles"],
        "ref_steps": case["ref_steps"],
        "chematica_note": case["chematica_note"],
        "started_at": started_iso,
        "timeout_s": timeout_s,
        "out_dir": _display_path(out_dir),
        "rss_cap_gb": rss_cap_gb,
    }

    peak_rss_kb = 0
    rss_capped = False
    try:
        with open(log_path, "wb") as f:
            f.write(f"# CMD: {' '.join(cmd)}\n# CWD: {REPO}\n# STARTED: {started_iso}\n# TIMEOUT: {timeout_s}s\n# RSS_CAP: {rss_cap_gb}G (poll-based)\n# CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', 'unset')}\n\n".encode())
            f.flush()
            proc = subprocess.Popen(
                cmd, cwd=str(REPO), env=env, stdout=f, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            child_pid = proc.pid
            try:
                rc: Optional[int] = None
                t_deadline = started + timeout_s
                while True:
                    try:
                        rc = proc.wait(timeout=10)
                        break
                    except subprocess.TimeoutExpired:
                        # Sample peak RSS for telemetry, current RSS for cap.
                        peak_kb = _read_peak_rss_kb(child_pid)
                        if peak_kb and peak_kb > peak_rss_kb:
                            peak_rss_kb = peak_kb
                        cur_kb = _read_rss_kb(child_pid)
                        if rss_cap_kb and cur_kb and cur_kb > rss_cap_kb:
                            try:
                                os.killpg(child_pid, signal.SIGKILL)
                            except Exception:
                                pass
                            rss_capped = True
                            rc = proc.wait(timeout=15)
                            break
                        if time.time() >= t_deadline:
                            raise
                killed = False
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=10)
                except Exception:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except Exception:
                        pass
                rc = -1
                killed = True
        elapsed = time.time() - started
        rec["wall_seconds"] = round(elapsed, 1)
        rec["return_code"] = rc
        rec["killed_by_timeout"] = killed
        rec["peak_rss_mb"] = round(peak_rss_kb / 1024, 1) if peak_rss_kb else None
        rec["killed_by_rss_cap"] = rss_capped
        # rc == -9 = SIGKILL: kernel OOM, manual rss-cap kill, or external kill
        rec["oom_killed"] = (rc == -9 and not rss_capped)
    except Exception as e:
        rec["wall_seconds"] = round(time.time() - started, 1)
        rec["fatal"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return rec

    # Parse the stdout for completion + iteration count
    try:
        stdout_text = log_path.read_text(errors="replace")
    except Exception:
        stdout_text = ""

    iter_match = re.search(r"Iterations?\s*[:=]\s*(\d+)", stdout_text)
    rec["iterations"] = int(iter_match.group(1)) if iter_match else None
    # `Success |` is main.py's authoritative completion banner — it prints
    # only after `agent.run()` returned with success=True. Earlier versions
    # also matched the bare substring `COMPLETE ROUTE FOUND`, but that
    # phrase now appears in the agent's own startup deprecation warnings
    # and in the system prompt loaded at boot, so a crashed/OOM-killed
    # process would still be tagged SOLVED. Stick to the post-run banner.
    rec["success_marker"] = "Success |" in stdout_text
    rec["fail_marker"] = "❌ Failed" in stdout_text

    route = parse_route_text(stdout_text)
    if route["steps"]:
        rec["route"] = route

    # Locate the session dir + copy the transcript
    sess = find_session_dir(stdout_text, started)
    if sess is not None:
        # Cross-check the located session actually belongs to THIS target
        # before recording/copying anything (guards the mtime fallback).
        try:
            _td_probe = json.loads((sess / "transcript.json").read_text())
            _sess_target = _td_probe.get("target_smiles")
            if _sess_target and _sess_target != case["smiles"]:
                rec["session_dir_mismatch_rejected"] = str(sess)
                sess = None
        except Exception:
            pass
    if sess is not None:
        rec["session_dir"] = str(sess)
        (out_dir / "session_dir.txt").write_text(str(sess) + "\n")
        tr = sess / "transcript.json"
        if tr.exists():
            try:
                shutil.copy2(tr, out_dir / "transcript.json")
                rec["transcript_copied"] = True
            except Exception as e:
                rec["transcript_copy_error"] = str(e)
            # Pull a few headline fields out of the transcript
            try:
                td = json.loads(tr.read_text())
                rec["transcript_summary"] = {
                    "elapsed_seconds": td.get("elapsed_seconds"),
                    "iterations": (td.get("result") or {}).get("iterations"),
                    "success": (td.get("result") or {}).get("success"),
                    "below_target": (td.get("result") or {}).get("below_target"),
                    "n_events": len(td.get("events", [])),
                }
                # Authoritative route source: the agent's own final
                # result.route_text. stdout parsing above is only a fallback
                # (log interleaving under --workers>1 can corrupt/echo blocks).
                rt = (td.get("result") or {}).get("route_text") or ""
                if rt:
                    route_t = parse_route_text(rt)
                    if route_t["steps"]:
                        rec["route"] = route_t
                # Best route info from transcript. The agent stores ranked
                # routes under `scratchpad.routes`; each route carries the
                # score both at the top level and (authoritatively) under
                # `sub_scores.total_score`. Older transcripts used top-level
                # `routes` / `ranked_routes`, so fall back to those.
                routes = (
                    (td.get("scratchpad") or {}).get("routes")
                    or td.get("routes")
                    or td.get("ranked_routes")
                    or []
                )

                def _route_score(r):
                    if not isinstance(r, dict):
                        return None
                    s = r.get("total_score")
                    if not isinstance(s, (int, float)):
                        s = (r.get("sub_scores") or {}).get("total_score")
                    return s if isinstance(s, (int, float)) else None

                if routes:
                    scores = [s for s in (_route_score(r) for r in routes) if s is not None]
                    rec["best_route_score"] = max(scores) if scores else None
                    rec["n_ranked_routes"] = len(routes)
            except Exception as e:
                rec["transcript_parse_error"] = str(e)

    return rec


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--thinking", choices=("fast", "standard", "thorough"), default="fast")
    p.add_argument("--config", default=None, help="Optional YAML override layered on the built-in preset.")
    p.add_argument("--cases-csv", default="",
                   help="Path to a cases.csv to run instead of the hard_mol default. "
                        "Schema-tolerant (hard_mol or retro-190 columns).")
    p.add_argument("--results-root", default="",
                   help="Parent dir for results/<run-id>/. Default = hard_mol results/.")
    p.add_argument("--indices", default="",
                   help="Comma-separated case indices (0-based) to run; "
                        "empty = all 11. e.g. --indices 3,4")
    p.add_argument("--timeout", type=int, default=2400,
                   help="Per-target wall-clock seconds (default 40 min)")
    p.add_argument("--run-id", default="",
                   help="Subdir name under results/. Default = timestamp.")
    p.add_argument("--label", default="full",
                   help="Free-form label included in summary.json (e.g. 'smoke', 'full').")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip pre-flight LLM/ReactionT5/ChemBart probes.")
    p.add_argument("--continue-on-failure", action="store_true", default=True,
                   help="Continue batch even if a case crashes (default true).")
    p.add_argument("--rss-cap-gb", type=int, default=0,
                   help="Per-target RSS cap (GB); 0 disables. Default 0 — rely on "
                        "the kernel OOM killer instead of a self-imposed false-positive kill.")
    p.add_argument("--workers", type=int, default=2,
                   help="Parallel worker count. Each worker runs one target "
                        "subprocess at a time; safe to raise now that the "
                        "one-step HTTP service supports concurrency. Default 2. "
                        "After the 2026-05-19 check_buyability sqlite migration "
                        "(was an in-RAM set of 10.3 M SMILES per worker), "
                        "steady-state per-worker RSS is ~500 MB-1 GB plus the "
                        "per-target working set (transient model outputs, "
                        "verifier caches). On the 30 GB box, `--workers 2` is "
                        "the comfortable default; `--workers 3` works headroom-"
                        "permitting but watch dmesg if you push higher.")
    p.add_argument("--gpu-ids", default="",
                   help="Comma-separated GPU ids to round-robin across "
                        "workers via CUDA_VISIBLE_DEVICES. Empty leaves "
                        "CUDA_VISIBLE_DEVICES untouched (single GPU / "
                        "external scheduler).")
    p.add_argument("--stagger-s", type=float, default=0.0,
                   help="Delay each case's launch by (dispatch_order * stagger_s) "
                        "seconds. Spreads out the cold-start burst so N workers "
                        "don't all hit the LLM proxy's concurrency ceiling at "
                        "t=0 (observed as HTTP 503 retry storms in the first "
                        "~10 min of a workers=4 batch, 2026-07-18 W4 run). "
                        "120-180 is a good starting point. Only the initial "
                        "launch is staggered; once a worker finishes a case it "
                        "picks up the next one immediately. 0 disables.")
    p.add_argument("--model",
                   help="Override llm.model for every subprocess. "
                        "e.g. --model claude-opus-4.7-xhigh or --model gpt-5.5.")
    args = p.parse_args()

    try:
        cases = load_cases(Path(args.cases_csv) if args.cases_csv else None)
    except FileNotFoundError as e:
        print(f"[batch] {e}", file=sys.stderr)
        return 2
    if args.indices:
        idxs = {int(x) for x in args.indices.split(",") if x.strip()}
        cases = [c for c in cases if c["idx"] in idxs]
    if not cases:
        print("[batch] no cases selected", file=sys.stderr)
        return 2

    # Resolve config for pre-flight + summary metadata.
    from core.config import load_config
    config = load_config(args.config, thinking=args.thinking)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    results_root = Path(args.results_root).expanduser().resolve() if args.results_root else RESULTS_ROOT
    out_root = results_root / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_id": run_id,
        "label": args.label,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "thinking": args.thinking,
        "config_override": args.config,
        "timeout_s": args.timeout,
        "model_override": args.model,
        "cases": [{"idx": c["idx"], "smiles": c["smiles"], "label": c["label"]} for c in cases],
        "preflight": None,
        "results": [],
    }

    if not args.skip_preflight:
        print("[batch] preflight...")
        pre = preflight(config)
        summary["preflight"] = pre
        for k, v in pre.items():
            mark = "✓" if v.get("ok") else "✗"
            print(f"  {mark} {k:12s} {v}")
        if not pre["llm"]["ok"]:
            print("[batch] FATAL: LLM endpoint not reachable", file=sys.stderr)
            return 3
        # ReactionT5 not reachable is fine — there's a fallback — but warn
        if not pre["reactiont5"]["ok"]:
            print("[batch] WARN: ReactionT5 HTTP not reachable; "
                  "agent will fall back to in-process inference (slow!)")

    # The first summary.json write is DEFERRED until after startup reconciliation
    # below (the preseed pass), so summary.json is never momentarily an empty
    # results:[] file — a reader (or a crash) during the startup window would
    # otherwise see/leave an under-count. See the preseed block.
    summary_path = out_root / "summary.json"

    extra_env = {}  # Could pin REACTIONT5_API_URL etc here

    workers = max(1, int(args.workers))
    print(f"[batch] running {len(cases)} cases, timeout={args.timeout}s, "
          f"workers={workers}, output={_display_path(out_root)}")
    t_batch = time.time()

    # Prepare per-case metadata once so both the sequential and parallel
    # paths share the same out_dir / banner layout.
    plan = []
    for i, c in enumerate(cases, 1):
        target_dir_name = f"{c['idx']:02d}__{slugify(c['label'])}"
        out_dir = out_root / target_dir_name
        plan.append((i, c, out_dir))

    # Startup reconciliation: fold ALREADY-COMPLETED transcripts on disk into
    # the summary BEFORE the first write. Without this, summary['results'] was
    # built only from this process's drained futures, so two failure modes lost
    # work silently:
    #   • a relaunch reusing the same run-id (deterministic per-node run-ids do)
    #     overwrote summary.json with results:[] and only re-added a completed
    #     case once its idempotent-skip future drained — a reader mid-run saw
    #     far fewer solved cases than exist on disk (the host-e under-count).
    #   • a mid-run death left every completed-but-not-yet-drained case absent
    #     from summary.json with no reconciliation.
    # run_one's skip branch already rebuilds a full rec (success_marker /
    # best_route_score / n_ranked_routes) from an existing transcript, so we
    # reuse it: for any case whose transcript.json already exists, call run_one
    # (it returns immediately via the skip path, never launching an agent) and
    # seed the result. Summary is then a superset of completed-on-disk work from
    # the first write onward.
    preseeded: Dict[int, Dict[str, Any]] = {}
    for _, c, out_dir in plan:
        tr = out_dir / "transcript.json"
        if tr.exists() and tr.stat().st_size > 0:
            try:
                rec = run_one(c, out_dir, args.timeout, extra_env, args.config,
                              args.rss_cap_gb, None,
                              model=args.model, thinking=args.thinking)
                preseeded[c["idx"]] = rec
            except Exception as e:
                print(f"[batch] preseed idx={c['idx']} failed: {e}", file=sys.stderr)
    if preseeded:
        print(f"[batch] startup reconciliation: folded {len(preseeded)} "
              f"already-completed transcript(s) into summary")
    # FIRST summary.json write — only now, AFTER reconciliation, so the file is
    # never momentarily an empty results:[] (a same-run-id relaunch or a crash
    # during the startup window can't under-count on-disk work). When nothing is
    # pre-completed this writes an empty results list, which is correct.
    summary["results"] = [
        preseeded[c["idx"]] for _, c, _ in plan if c["idx"] in preseeded
    ]
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    def _print_start(i, c, out_dir):
        print(f"\n[batch] [{i}/{len(cases)}] idx={c['idx']} label={c['label']!r}")
        print(f"        smiles={c['smiles']}")
        print(f"        chematica_note={c['chematica_note']!r} ref_steps={c['ref_steps']}")
        print(f"        out={_display_path(out_dir)}")

    def _print_done(i, c, rec):
        outcome = (
            "SOLVED" if rec.get("success_marker") and not rec.get("fail_marker") else
            "TIMEOUT" if rec.get("killed_by_timeout") else
            "FAILED"
        )
        print(f"[batch] [{i}/{len(cases)}] idx={c['idx']} -> {outcome}  "
              f"wall={rec.get('wall_seconds')}s  "
              f"iters={rec.get('iterations')}  rc={rec.get('return_code')}  "
              f"best_score={rec.get('best_route_score')}  "
              f"peak_rss={rec.get('peak_rss_mb')}MB  oom={rec.get('oom_killed')}")

    if workers == 1:
        gpu_list = [int(x) for x in args.gpu_ids.split(",") if x.strip()] or [None]
        # Seed with already-completed transcripts (H1/M3 reconciliation); build
        # summary in idx order from a dict so a preseeded case isn't double-listed.
        results_by_idx: Dict[int, Dict[str, Any]] = dict(preseeded)
        for n, (i, c, out_dir) in enumerate(plan):
            if _DRAIN_REQUESTED.is_set():
                print("[batch] drain complete: no in-flight case remains", flush=True)
                break
            _print_start(i, c, out_dir)
            gpu = gpu_list[n % len(gpu_list)]
            rec = run_one(c, out_dir, args.timeout, extra_env, args.config,
                          args.rss_cap_gb, gpu,
                          model=args.model, thinking=args.thinking)
            _print_done(i, c, rec)
            results_by_idx[c["idx"]] = rec
            summary["results"] = [
                results_by_idx[c2["idx"]]
                for _, c2, _ in plan
                if c2["idx"] in results_by_idx
            ]
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        # Parallel path: each worker spawns its own subprocess via run_one.
        # Subprocess isolation (per-target sqlite caches in
        # runs/_session_*/) means workers don't collide on disk; the shared
        # one-step HTTP service now serves concurrent requests so wall
        # time scales with --workers up to the HTTP server's own concurrency.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        write_lock = threading.Lock()
        # Print all starts up front so the operator knows the dispatch order.
        for i, c, out_dir in plan:
            _print_start(i, c, out_dir)
        # Seed with already-completed transcripts so the live summary is a
        # superset of on-disk work from the first refresh (H1/M3 reconciliation).
        results_by_idx: Dict[int, Dict[str, Any]] = dict(preseeded)
        gpu_list = [int(x) for x in args.gpu_ids.split(",") if x.strip()] or [None]

        def _run_one_staggered(delay_s, *a, **kw):
            # Stagger only the initial wave (first `workers` launches): the
            # sleep runs inside the worker thread, so by the time a thread
            # frees up for a queued case the burst is already spread out.
            if delay_s > 0:
                print(f"[batch] stagger: delaying launch by {delay_s:.0f}s")
                time.sleep(delay_s)
            if _DRAIN_REQUESTED.is_set():
                return {"_drain_skipped": True}
            return run_one(*a, **kw)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_to_meta = {
                ex.submit(
                    _run_one_staggered,
                    (n * args.stagger_s) if n < workers else 0.0,
                    c, out_dir, args.timeout, extra_env,
                    args.config, args.rss_cap_gb, gpu_list[n % len(gpu_list)],
                    model=args.model, thinking=args.thinking,
                ): (i, c)
                for n, (i, c, out_dir) in enumerate(plan)
            }
            for fut in as_completed(future_to_meta):
                i, c = future_to_meta[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {
                        "idx": c["idx"],
                        "label": c["label"],
                        "smiles": c["smiles"],
                        "fatal": f"worker raised: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                    }
                if rec.get("_drain_skipped"):
                    continue
                _print_done(i, c, rec)
                with write_lock:
                    results_by_idx[c["idx"]] = rec
                    # Reflect partial results in idx order so the live
                    # summary.json is readable while the batch is in flight.
                    summary["results"] = [
                        results_by_idx[c2["idx"]]
                        for _, c2, _ in plan
                        if c2["idx"] in results_by_idx
                    ]
                    summary_path.write_text(
                        json.dumps(summary, indent=2, ensure_ascii=False)
                    )

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary["batch_wall_seconds"] = round(time.time() - t_batch, 1)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    n_solved = sum(1 for r in summary["results"] if r.get("success_marker") and not r.get("fail_marker"))
    n_timeout = sum(1 for r in summary["results"] if r.get("killed_by_timeout"))
    n_fail = len(summary["results"]) - n_solved - n_timeout
    print(f"\n[batch] DONE  solved={n_solved}  failed={n_fail}  timeout={n_timeout}  "
          f"wall={summary['batch_wall_seconds']:.0f}s")
    print(f"[batch] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
