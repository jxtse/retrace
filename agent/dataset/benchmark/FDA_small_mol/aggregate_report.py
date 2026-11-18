#!/usr/bin/env python
"""
Aggregate per-job JSONs from runs/ into a comparison CSV + markdown report
against ChemBart MCTS results in 2025FDA小分子汇总.csv.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

BENCH = Path(__file__).resolve().parent
RUNS = BENCH / "runs"
CSV_OUT = BENCH / "agent_vs_chembart.csv"
MD_OUT = BENCH / "REPORT.md"


def load_chembart() -> dict[int, dict]:
    rows = {}
    with open(BENCH / "2025FDA小分子汇总.csv", encoding="gbk", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                idx = int((row.get("idx") or "").strip())
            except Exception:
                continue
            rows[idx] = row
    return rows


def parse_route_steps(route_text: str) -> int | None:
    if not route_text:
        return None
    m = re.search(r"COMPLETE\s+ROUTE\s+FOUND", route_text, re.I)
    body = route_text[m.end():] if m else route_text
    # count "Step N:" occurrences
    steps = re.findall(r"^\s*Step\s+\d+\s*[:.]", body, re.M | re.I)
    if steps:
        return len(steps)
    arrows = body.count(">>") + body.count("→")
    return arrows or None


def parse_score(route_text: str) -> float | None:
    if not route_text:
        return None
    # look for "agent_total_score" or "total_score" numbers
    for pat in [r"agent_total_score[\"\s:=]+([0-9.]+)",
                r"total[_ ]score[\"\s:=]+([0-9.]+)",
                r"route\s*score[\"\s:=]+([0-9.]+)"]:
        m = re.search(pat, route_text, re.I)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def main():
    chembart = load_chembart()
    records: dict[tuple[int, str], dict] = {}
    for jf in sorted(RUNS.glob("*.json")):
        d = json.loads(jf.read_text(encoding="utf-8"))
        key = (d["idx"], d["variant"])
        records[key] = d

    # Build CSV
    fieldnames = [
        "idx", "medicine", "name", "ref_step",
        "chembart_step", "chembart_full_success", "chembart_notes",
        "agent_stereo_success", "agent_stereo_below_target",
        "agent_stereo_step", "agent_stereo_iters", "agent_stereo_wall_s",
        "agent_canonical_success", "agent_canonical_below_target",
        "agent_canonical_step", "agent_canonical_iters", "agent_canonical_wall_s",
        "agent_stereo_score", "agent_canonical_score",
    ]
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for idx in sorted(chembart.keys()):
            cb = chembart[idx]
            row = {
                "idx": idx,
                "medicine": (cb.get("Medicine") or "").strip(),
                "name": (cb.get("Name") or "").strip(),
                "ref_step": (cb.get("Step") or "").strip(),
                "chembart_step": (cb.get("Step_MCTS") or "").strip(),
                "chembart_full_success": (cb.get("FULL_success") or "").strip(),
                "chembart_notes": (cb.get("逐步复现") or "").strip(),
            }
            for variant in ("stereo", "canonical"):
                d = records.get((idx, variant))
                if not d:
                    continue
                row[f"agent_{variant}_success"] = d.get("success")
                row[f"agent_{variant}_below_target"] = d.get("below_target")
                row[f"agent_{variant}_step"] = parse_route_steps(d.get("route_text", ""))
                row[f"agent_{variant}_iters"] = d.get("iterations")
                row[f"agent_{variant}_wall_s"] = d.get("wall_seconds")
                row[f"agent_{variant}_score"] = parse_score(d.get("route_text", ""))
            w.writerow(row)

    # Build markdown report
    n_total = len(chembart)
    stereo_rec = [d for (i, v), d in records.items() if v == "stereo"]
    canon_rec = [d for (i, v), d in records.items() if v == "canonical"]

    def stat(rs):
        if not rs:
            return (0, 0, 0.0, 0.0)
        succ = sum(1 for r in rs if r.get("success"))
        gold = sum(1 for r in rs if r.get("success") and not r.get("below_target"))
        wall = sum(r.get("wall_seconds") or 0 for r in rs) / len(rs)
        iters = sum(r.get("iterations") or 0 for r in rs) / len(rs)
        return succ, gold, wall, iters

    s_succ, s_gold, s_wall, s_iters = stat(stereo_rec)
    c_succ, c_gold, c_wall, c_iters = stat(canon_rec)

    cb_failed = [int(cb["idx"]) for cb in chembart.values()
                 if (cb.get("FULL_success") or "").strip().lower() == "failed"]

    md = []
    md.append("# 2025 FDA Small-Molecule Benchmark — Agent vs ChemBart MCTS\n")
    md.append(f"- Targets: **{n_total}** (idx 0–{n_total-1})")
    md.append(f"- Agent variants: **stereo** (original SMILES) and **canonical** (no stereo, ChemBart input)")
    md.append(f"- ChemBart MCTS column: `FULL_success` flagged 'failed' on idx={cb_failed}")
    md.append("")
    md.append("## Headline numbers\n")
    md.append("| Variant | Runs | success (`success=true`) | gold (`success ∧ ¬below_target`) | mean iters | mean wall (s) |")
    md.append("|---|---:|---:|---:|---:|---:|")
    md.append(f"| stereo    | {len(stereo_rec)} | {s_succ}/{len(stereo_rec)} | {s_gold}/{len(stereo_rec)} | {s_iters:.1f} | {s_wall:.1f} |")
    md.append(f"| canonical | {len(canon_rec)} | {c_succ}/{len(canon_rec)} | {c_gold}/{len(canon_rec)} | {c_iters:.1f} | {c_wall:.1f} |")
    md.append("")
    md.append("Definitions:")
    md.append("- *success*: agent emitted `COMPLETE ROUTE FOUND` (may include `below_target` warning, i.e. score < 0.7).")
    md.append("- *gold*: success with `below_target=false` under the current verifier gate.")
    md.append("- ChemBart entries with `FULL_success=failed` failed end-to-end at the published MCTS reproduction step.")
    md.append("")
    md.append("## Per-target detail\n")
    md.append("| idx | Name | Ref step | ChemBart MCTS | Agent (canonical) | Agent (stereo) |")
    md.append("|---:|---|---:|---|---|---|")
    for idx in sorted(chembart.keys()):
        cb = chembart[idx]
        nm = (cb.get("Name") or "").strip()
        ref_step = (cb.get("Step") or "").strip()
        cb_step = (cb.get("Step_MCTS") or "").strip()
        cb_full = (cb.get("FULL_success") or "").strip() or "—"
        cb_note = (cb.get("逐步复现") or "").strip()
        cb_cell = f"{cb_step}步, FULL={cb_full}; {cb_note}" if cb_note else f"{cb_step}步, FULL={cb_full}"

        def cell(d):
            if not d:
                return "—"
            if d.get("error"):
                return f"ERR: {d['error'][:60]}"
            ok = "✓" if d.get("success") else "✗"
            bt = " (below_target)" if d.get("below_target") else ""
            steps = parse_route_steps(d.get("route_text", ""))
            return f"{ok}{bt}, step={steps}, iters={d.get('iterations')}, {d.get('wall_seconds')}s"

        md.append(f"| {idx} | {nm} | {ref_step} | {cb_cell} | {cell(records.get((idx,'canonical')))} | {cell(records.get((idx,'stereo')))} |")

    md.append("")
    md.append("## Files\n")
    md.append("- `agent_vs_chembart.csv` — machine-readable comparison (this script's output).")
    md.append("- `runs/{idx:02d}_{Name}_{variant}.json` — per-run structured result (route_text, iterations, wall, etc.).")
    md.append("- `transcripts/{idx:02d}_{Name}_{variant}.txt` — full message stream incl. tool calls and results.")
    md.append("- `logs/{idx:02d}_{Name}_{variant}.log` — agent INFO/DEBUG log per run.")

    MD_OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {CSV_OUT}")
    print(f"wrote {MD_OUT}")


if __name__ == "__main__":
    main()
