#!/usr/bin/env python
"""Run the agent on the experimental-validation molecules (实验验证分子.xlsx).

One process per case (serial), idempotent: skips a case whose out_dir already
holds a non-empty result.json. Each case writes:
  <out>/case_<idx>/result.json      structured summary (target, success, routes, scores)
  <out>/case_<idx>/transcript.json  full agent transcript (copied from session dir)
  <out>/case_<idx>/run.log          stdout/stderr of the run

Model + thinking are CLI-overridable; defaults: claude-opus-4-8 / standard.
"""
from __future__ import annotations
import argparse, csv, json, os, shutil, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_cases(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def newest_session_for_target(runs_dir: Path, target_canon: str, after_ts: float) -> Path | None:
    """Find the session dir whose transcript.json target matches target_canon,
    created after after_ts. Guards against M4 transcript mis-attribution."""
    from rdkit import Chem
    best = None
    best_mtime = after_ts
    for d in runs_dir.glob("_session_*"):
        tj = d / "transcript.json"
        if not tj.exists():
            continue
        if tj.stat().st_mtime < after_ts:
            continue
        try:
            data = json.loads(tj.read_text())
        except Exception:
            continue
        tgt = data.get("target_smiles") or data.get("target") or ""
        try:
            c = Chem.MolToSmiles(Chem.MolFromSmiles(tgt)) if tgt else ""
        except Exception:
            c = ""
        if c and c == target_canon and tj.stat().st_mtime >= best_mtime:
            best = d
            best_mtime = tj.stat().st_mtime
    return best


def run_one(case: dict, out_root: Path, model: str, thinking: str) -> dict:
    from rdkit import Chem
    from core.config import load_config
    from core import build_agent

    idx = case["idx"]
    smiles = case["smiles"]
    out_dir = out_root / f"case_{idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"

    if result_path.exists() and result_path.stat().st_size > 0:
        try:
            prev = json.loads(result_path.read_text())
            if prev.get("done"):
                print(f"[case {idx}] SKIP (already done)", flush=True)
                return prev
        except Exception:
            pass

    target_canon = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    cli_overrides = {"llm": {"model": model}}
    config = load_config(thinking=thinking, cli_overrides=cli_overrides)
    config["thinking"] = thinking

    runs_dir = ROOT / "runs"
    t0 = time.time()
    print(f"[case {idx}] START model={model} thinking={thinking} atoms={case.get('natoms')} MW={case.get('mw')}", flush=True)
    print(f"[case {idx}] target={target_canon}", flush=True)

    summary = {
        "idx": idx, "smiles": smiles, "target_canon": target_canon,
        "mw": case.get("mw"), "natoms": case.get("natoms"), "nrings": case.get("nrings"),
        "model": model, "thinking": thinking, "done": False,
    }
    try:
        agent = build_agent(config)
        result = agent.run(smiles)
        summary["success"] = bool(result.get("success"))
        summary["iterations"] = result.get("iterations")
        summary["time_s"] = round(result.get("time", time.time() - t0), 1)
        # ranked routes live under result["ranked"]["routes"] (and all_routes)
        ranked = result.get("ranked") or {}
        routes = (ranked.get("routes") if isinstance(ranked, dict) else None) \
            or result.get("all_routes") or []
        summary["n_routes"] = len(routes)
        best = result.get("best_route") or (routes[0] if routes else None)
        summary["best_route"] = best
        # rank-1 score for quick tally (best_route.total_score > 0 = real, not bailout)
        if isinstance(best, dict):
            summary["best_total_score"] = best.get("total_score") or (
                best.get("sub_scores", {}).get("total_score") if isinstance(best.get("sub_scores"), dict) else None
            )
        # keep all ranked routes (with scores + steps) for visualization
        summary["routes"] = routes
        # route_text tail (human readable)
        rt = result.get("route_text", "") or ""
        idx_marker = rt.find("COMPLETE ROUTE FOUND")
        summary["route_text"] = rt[idx_marker:] if idx_marker >= 0 else rt[-4000:]
        # locate + copy the matching session transcript
        sess = newest_session_for_target(runs_dir, target_canon, t0 - 1)
        if sess is not None:
            for fn in ("transcript.json", "transcript.jsonl"):
                src = sess / fn
                if src.exists():
                    shutil.copy2(src, out_dir / fn)
            summary["session_dir"] = str(sess)
        summary["done"] = True
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["traceback"] = traceback.format_exc()
        summary["done"] = False
        print(f"[case {idx}] ERROR {summary['error']}", flush=True)

    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    dt = time.time() - t0
    print(f"[case {idx}] DONE success={summary.get('success')} "
          f"routes={summary.get('n_routes')} time={dt:.0f}s -> {result_path}", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ROOT / "final_results/exp_validation_opus48/cases.csv"))
    ap.add_argument("--out", default=str(ROOT / "final_results/exp_validation_opus48"))
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--thinking", default="standard", choices=["fast", "standard", "thorough"])
    ap.add_argument("--only", default="", help="comma list of idx to run (default all)")
    args = ap.parse_args()

    cases = load_cases(Path(args.cases))
    if args.only:
        want = set(s.strip() for s in args.only.split(",") if s.strip())
        cases = [c for c in cases if c["idx"] in want]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"=== exp-validation run: {len(cases)} cases, model={args.model}, thinking={args.thinking} ===", flush=True)
    results = []
    for c in cases:
        results.append(run_one(c, out_root, args.model, args.thinking))

    # aggregate
    agg = out_root / "summary.json"
    agg.write_text(json.dumps({
        "model": args.model, "thinking": args.thinking,
        "n_cases": len(results),
        "n_success": sum(1 for r in results if r.get("success")),
        "cases": [{k: r.get(k) for k in ("idx", "success", "n_routes", "time_s", "error")} for r in results],
    }, ensure_ascii=False, indent=2))
    print(f"=== ALL DONE -> {agg} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
