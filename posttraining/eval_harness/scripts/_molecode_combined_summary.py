#!/usr/bin/env python
"""Combined 3-arm retro-190 summary: baseline vs prior-passive vs new-stack.

Usage:
  _molecode_combined_summary.py <stack_arm_dir> <baseline_arm_dir> <prior_passive_arm_dir>

Each *_arm_dir has runs/{*}.json and logs/{*}.log. Prints a per-case table
with scores, deltas (stack-vs-baseline, stack-vs-passive), and stop reason
(floor/plateau) for the stack arm, plus aggregate means.
"""
from __future__ import annotations
import sys, glob, json, os, statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _molecode_stopreason import arm_table  # noqa: E402


def scores(arm_dir: str) -> dict[int, float | None]:
    out: dict[int, float | None] = {}
    for p in sorted(glob.glob(os.path.join(arm_dir, "runs", "*.json"))):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        idx = r.get("idx")
        if not isinstance(idx, int):
            continue
        br = r.get("best_route") or {}
        v = br.get("total_score")
        out[idx] = float(v) if isinstance(v, (int, float)) else None
    return out


def iters(arm_dir: str) -> dict[int, int | None]:
    out: dict[int, int | None] = {}
    for p in sorted(glob.glob(os.path.join(arm_dir, "runs", "*.json"))):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        idx = r.get("idx")
        if isinstance(idx, int):
            it = r.get("iterations")
            out[idx] = int(it) if isinstance(it, int) else None
    return out


def main() -> int:
    stack_dir, base_dir, passive_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    s_sc, b_sc, p_sc = scores(stack_dir), scores(base_dir), scores(passive_dir)
    s_it = iters(stack_dir)
    s_stop = arm_table(stack_dir)

    idxs = sorted(set(s_sc) | set(b_sc) | set(p_sc))
    hdr = (f"{'idx':>3} {'base':>7} {'passive':>8} {'STACK':>7} "
           f"{'Δbase':>8} {'Δpass':>8} {'st_it':>5} {'stop':>8} {'distinct':>8}")
    print(hdr)
    print("-" * len(hdr))
    d_base, d_pass = [], []
    for i in idxs:
        b, p, s = b_sc.get(i), p_sc.get(i), s_sc.get(i)
        db = round(s - b, 4) if (s is not None and b is not None) else None
        dp = round(s - p, 4) if (s is not None and p is not None) else None
        if db is not None:
            d_base.append(db)
        if dp is not None:
            d_pass.append(dp)
        stp = s_stop.get(i, {})
        print(f"{i:>3} {str(b):>7} {str(p):>8} {str(s):>7} "
              f"{str(db):>8} {str(dp):>8} {str(s_it.get(i)):>5} "
              f"{str(stp.get('stop')):>8} {str(stp.get('distinct')):>8}")
    print("-" * len(hdr))
    def agg(xs):
        return (f"mean={round(st.mean(xs),4)} median={round(st.median(xs),4)} "
                f"min={round(min(xs),4)} max={round(max(xs),4)} n={len(xs)}") if xs else "n=0"
    print(f"Δ stack-vs-baseline : {agg(d_base)}")
    print(f"Δ stack-vs-passive  : {agg(d_pass)}")
    # stop-reason tally
    tally: dict[str, int] = {}
    for i in s_sc:
        st_ = (s_stop.get(i) or {}).get("stop", "??")
        tally[st_] = tally.get(st_, 0) + 1
    print(f"stack stop reasons  : {tally}")
    base_stop = arm_table(base_dir)
    btally: dict[str, int] = {}
    for i in b_sc:
        st_ = (base_stop.get(i) or {}).get("stop", "??")
        btally[st_] = btally.get(st_, 0) + 1
    print(f"baseline stop reasons: {btally}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
