#!/usr/bin/env python
"""Resume each ExpValidation2 case agent with chemist feedback and continue.

Human-in-the-loop continuation: for each case, rebuild ``resume_from`` exactly
the way ``core/repl.py:_load_resume_state`` does —
  - ``messages``    from the session's ``conversation/latest_messages.json``
                    (path stored in transcript.json -> resume_state)
  - ``all_routes``  the prior scratchpad ``to_dict()`` payloads
                    (transcript.json -> scratchpad.routes; has full ``steps``)
  - ``todos``       transcript.json -> todos
  - ``iterations``  prior iteration count (continuation offset)
then call ``agent.run(target, resume_from=, feedback=<chemist feedback>,
extra_rounds=)`` so the agent ingests the feedback ON TOP of the conversation
it actually had, and keeps iterating.

Feedback per case is read from a JSON file (--feedback-json) keyed by case idx:
  { "1": "<chemist feedback text>", "2": "...", "3": "..." }

Idempotent: skips a case whose ``case_<idx>/result_resumed.json`` exists and is
done, unless --force.

Writes per case:
  <out>/case_<idx>/result_resumed.json   structured summary (before/after scores)
  <out>/case_<idx>/transcript_resumed.json   full continuation transcript
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_resume_from(case_dir: Path) -> dict | None:
    """Mirror core/repl.py:_load_resume_state from on-disk artifacts."""
    tj = case_dir / "transcript.json"
    if not tj.exists():
        return None
    transcript = json.loads(tj.read_text())
    # messages from latest_messages.json (path inside resume_state)
    messages: list[dict] = []
    lm_path = (transcript.get("resume_state") or {}).get("latest_messages_path")
    if lm_path and os.path.exists(lm_path):
        messages = (json.loads(Path(lm_path).read_text()) or {}).get("messages") or []
    if not messages:
        return None
    scratch = transcript.get("scratchpad") or {}
    all_routes = scratch.get("routes") or []          # to_dict()-shaped (has steps)
    todos = transcript.get("todos") or []
    iterations = int((transcript.get("result") or {}).get("iterations")
                     or transcript.get("iterations") or 0)
    return {
        "messages": messages,
        "all_routes": all_routes,
        "todos": todos,
        "iterations": iterations,
        "_session_dir": (transcript.get("resume_state") or {}).get("session_dir"),
    }


def best_score(routes: list) -> float | None:
    sc = [r.get("total_score") for r in routes if isinstance(r, dict) and r.get("total_score") is not None]
    return max(sc) if sc else None


def run_one(idx: str, case_dir: Path, target: str, feedback: str,
            model: str, thinking: str, extra_rounds: int, force: bool) -> dict:
    from rdkit import Chem
    from core.config import load_config
    from core import build_agent

    out_path = case_dir / "result_resumed.json"
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        try:
            prev = json.loads(out_path.read_text())
            if prev.get("done"):
                print(f"[case {idx}] SKIP (already resumed)", flush=True)
                return prev
        except Exception:
            pass

    target_canon = Chem.MolToSmiles(Chem.MolFromSmiles(target))
    resume_from = load_resume_from(case_dir)
    if resume_from is None:
        return {"idx": idx, "done": False, "error": "no resumable prior state"}

    prior_routes = resume_from["all_routes"]
    prior_best = best_score(prior_routes)
    print(f"[case {idx}] RESUME model={model} thinking={thinking} "
          f"prior_routes={len(prior_routes)} prior_best={prior_best} "
          f"prior_iters={resume_from['iterations']} +{extra_rounds} rounds", flush=True)

    cli_overrides = {"llm": {"model": model}}
    config = load_config(thinking=thinking, cli_overrides=cli_overrides)

    summary = {
        "idx": idx, "target": target, "target_canon": target_canon,
        "model": model, "thinking": thinking,
        "prior_n_routes": len(prior_routes), "prior_best_score": prior_best,
        "prior_iterations": resume_from["iterations"],
        "feedback": feedback, "done": False,
    }
    t0 = time.time()
    try:
        agent = build_agent(config)
        rf = {k: v for k, v in resume_from.items() if not k.startswith("_")}
        result = agent.run(target, resume_from=rf, feedback=feedback,
                           extra_rounds=extra_rounds)
        ranked = result.get("ranked") or {}
        routes = (ranked.get("routes") if isinstance(ranked, dict) else None) \
            or result.get("all_routes") or []
        summary["success"] = bool(result.get("success"))
        summary["iterations_total"] = result.get("iterations")
        summary["new_n_routes"] = len(routes)
        summary["new_best_score"] = best_score(routes)
        best = result.get("best_route") or (routes[0] if routes else None)
        summary["new_best_route"] = best
        summary["routes"] = routes
        summary["delta_best"] = (
            (summary["new_best_score"] or 0) - (prior_best or 0)
            if summary["new_best_score"] is not None else None
        )
        rt = result.get("route_text", "") or ""
        m = rt.find("COMPLETE ROUTE FOUND")
        summary["route_text"] = rt[m:] if m >= 0 else rt[-4000:]
        # copy continuation transcript
        for fn, dst in (("transcript_json", "transcript_resumed.json"),):
            src = result.get(fn)
            if src and os.path.exists(src):
                shutil.copy2(src, case_dir / dst)
        summary["session_dir"] = result.get("session_dir")
        summary["time_s"] = round(time.time() - t0, 1)
        summary["done"] = True
        print(f"[case {idx}] DONE new_best={summary['new_best_score']} "
              f"(Δ{summary['delta_best']:+.4f}) routes={summary['new_n_routes']} "
              f"iters={summary['iterations_total']} time={summary['time_s']:.0f}s", flush=True)
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["traceback"] = traceback.format_exc()
        summary["time_s"] = round(time.time() - t0, 1)
        print(f"[case {idx}] ERROR {summary['error']}", flush=True)

    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "final_results/ExpValidation2_GPT55-standard_RETRACE"))
    ap.add_argument("--feedback-json", required=True,
                    help="JSON file: {idx: feedback_text}")
    ap.add_argument("--cases", default="", help="comma list of idx (default all in feedback-json)")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--thinking", default="standard", choices=["fast", "standard", "thorough"])
    ap.add_argument("--extra-rounds", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out)
    cases_csv = out_root / "cases.csv"
    import csv as _csv
    targets = {r["idx"]: (r.get("canon") or r["smiles"])
               for r in _csv.DictReader(open(cases_csv))}
    feedback = json.loads(Path(args.feedback_json).read_text())
    idxs = [s.strip() for s in args.cases.split(",") if s.strip()] or list(feedback.keys())

    print(f"=== resume-with-feedback: {len(idxs)} cases, model={args.model}, "
          f"thinking={args.thinking}, +{args.extra_rounds} rounds ===", flush=True)
    results = []
    for idx in idxs:
        case_dir = out_root / f"case_{idx}"
        fb = feedback.get(idx, "")
        if not fb:
            print(f"[case {idx}] SKIP (no feedback)", flush=True)
            continue
        results.append(run_one(idx, case_dir, targets[idx], fb,
                               args.model, args.thinking, args.extra_rounds, args.force))

    agg = out_root / "summary_resumed.json"
    agg.write_text(json.dumps({
        "model": args.model, "thinking": args.thinking, "extra_rounds": args.extra_rounds,
        "cases": [{k: r.get(k) for k in
                   ("idx", "success", "prior_best_score", "new_best_score",
                    "delta_best", "new_n_routes", "iterations_total", "time_s", "error")}
                  for r in results],
    }, ensure_ascii=False, indent=2))
    print(f"=== ALL RESUMED -> {agg} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
