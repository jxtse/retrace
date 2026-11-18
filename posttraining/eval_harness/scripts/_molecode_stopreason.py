#!/usr/bin/env python
"""Extract per-case stop reason (floor vs plateau) from retro-190 run logs.

Reads <arm>/logs/{idx:03d}*.log, finds the last
  "Soft threshold reached: distinct_routes=N (floor=F, plateau_turns=P/L, ...)"
line, and classifies:
  floor   -> distinct >= floor   (search saturated the distinct-route cap)
  plateau -> plateau_turns >= L  (no new distinct route for L turns)
  maxiter -> neither soft line present (hit max_iterations / wallclock)
"""
from __future__ import annotations
import re, sys, glob, json, os

LINE = re.compile(
    r"Soft threshold reached: distinct_routes=(\d+) \(floor=(\d+), plateau_turns=(\d+)/(\d+)"
)

def classify(log_path: str):
    last = None
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = LINE.search(line)
                if m:
                    last = m
    except OSError:
        return None
    if not last:
        return {"stop": "maxiter_or_wall", "distinct": None,
                "floor": None, "plateau": None}
    distinct, floor, pturns, plimit = (int(last.group(i)) for i in range(1, 5))
    if distinct >= floor:
        stop = "floor"
    elif pturns >= plimit:
        stop = "plateau"
    else:
        stop = "other"
    return {"stop": stop, "distinct": distinct, "floor": floor,
            "plateau": f"{pturns}/{plimit}"}

def arm_table(arm_dir: str):
    out = {}
    logs_dir = os.path.join(arm_dir, "logs")
    for p in sorted(glob.glob(os.path.join(logs_dir, "*.log"))):
        base = os.path.basename(p)
        m = re.match(r"(\d+)", base)
        if not m:
            continue
        idx = int(m.group(1))
        out[idx] = classify(p)
    return out

if __name__ == "__main__":
    arm = sys.argv[1]
    tbl = arm_table(arm)
    print(json.dumps(tbl, indent=2))
