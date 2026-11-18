#!/usr/bin/env python
"""Trace the provenance + per-step scoring of a run's saved routes.

After the `_detail` traceability upgrade (2026-06-20), every persisted
`score_route` result carries a per-step / per-pillar breakdown, and every
`neural_retro` ensemble result carries each engine's raw top-K — both on disk
only (stripped from the LLM message stream). This script joins them back to the
routes a run saved, answering the two questions the slim transcript cannot:

  * which engine produced which precursor set (and that engine's own score,
    before the ensemble max-collapsed it), and
  * how each step of a saved route finally scored — rater confidence, whether
    an LLM judge ran (rater-only vs rater+judge), plus the cost / greenness
    per-step detail.

Usage
-----
    .venv/bin/python scripts/trace_route.py <session_dir> [--route LABEL]
                                            [--engines] [--json]

    <session_dir>   a runs/_session_* directory (or any dir with
                    transcript.json + results/).
    --route LABEL   only trace the route whose label matches (substring, ci).
    --engines       also dump per-engine neural_retro predictions for the run.
    --json          emit the assembled trace as JSON instead of text.

The join path: scratchpad route  ->  the save_best_route event that carries its
`score_route_result_id`  ->  results/<id>.json  ->  `_detail`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _result_detail(session_dir: str, result_id: Optional[str]) -> Optional[dict]:
    """Load the on-disk `_detail` block for a persisted result id (or None)."""
    if not result_id:
        return None
    path = os.path.join(session_dir, "results", f"{result_id}.json")
    if not os.path.exists(path):
        return None
    try:
        blob = _load_json(path)
    except Exception:
        return None
    content = blob.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except Exception:
            return None
    if not isinstance(content, dict):
        return None
    return content.get("_detail")


def _events(transcript: dict) -> List[dict]:
    return transcript.get("events") or transcript.get("timeline") or []


def _save_events(transcript: dict) -> Dict[str, str]:
    """Map saved-route label -> score_route_result_id from save_best_route calls."""
    out: Dict[str, str] = {}
    for e in _events(transcript):
        et = e.get("event") or e.get("type")
        name = e.get("name") or e.get("tool")
        if et != "tool_call" or name != "save_best_route":
            continue
        args = e.get("args") or e.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                continue
        label = args.get("label")
        rid = args.get("score_route_result_id")
        if label and rid:
            out[label] = rid
    return out


def _fmt_feasibility(detail: dict) -> List[str]:
    lines: List[str] = []
    feas = (detail or {}).get("feasibility") or {}
    per_step = feas.get("per_step") or []
    lines.append(f"  feasibility = {feas.get('score')}  ({len(per_step)} steps)")
    for ps in per_step:
        i = ps.get("step")
        rater = ps.get("rater") or {}
        backend = "rater+judge" if "step_judge" in ps else "rater-only"
        final = ps.get("final_score")
        extra = []
        if "carbon_fabrication_penalty" in ps:
            extra.append(f"carbonΔ-penalty={ps['carbon_fabrication_penalty']}")
        if "ring_carbon_warning" in ps:
            extra.append("ring-carbon-warn")
        sj = ps.get("step_judge") or {}
        if sj:
            extra.append(
                f"judge {sj.get('lifted_from')}→{sj.get('lifted_to')}"
                f" ({sj.get('verdict')})"
            )
        lines.append(
            f"    step {i}: final={final}  rater_conf={rater.get('confidence')}"
            f"  [{backend}]" + (("  " + "; ".join(extra)) if extra else "")
        )
    return lines


def _fmt_cost(detail: dict) -> List[str]:
    lines: List[str] = []
    cost = (detail or {}).get("cost") or {}
    lines.append(
        f"  cost = {cost.get('score')}  "
        f"(step_eff={cost.get('step_efficiency')}, "
        f"leaf_econ={cost.get('leaf_economics')}, "
        f"sc_decrease={cost.get('sc_decrease_score')})"
    )
    for lf in cost.get("leaves") or []:
        lines.append(
            f"    leaf {lf.get('smiles')}: score={lf.get('leaf_score')}"
            f"  ${lf.get('price_per_g')}/g  vendor={lf.get('vendor')}"
        )
    return lines


def _fmt_greenness(detail: dict) -> List[str]:
    lines: List[str] = []
    green = (detail or {}).get("greenness") or {}
    per_step = green.get("per_step") or []
    lines.append(f"  greenness = {green.get('score')}  ({len(per_step)} steps)")
    for ps in per_step:
        solv = ps.get("solvent") or {}
        haz = ps.get("hazardous_reagents") or []
        lines.append(
            f"    step {ps.get('step')}: AE={ps.get('atom_economy')}"
            f"  solvent={solv.get('name')}({solv.get('class')})"
            f"  T={ps.get('temperature_C')}C"
            + (f"  hazards={[h.get('name') or h.get('smiles') for h in haz]}" if haz else "")
        )
    return lines


def trace_routes(session_dir: str, route_filter: Optional[str]) -> List[dict]:
    transcript = _load_json(os.path.join(session_dir, "transcript.json"))
    sp = transcript.get("scratchpad") or {}
    routes = sp.get("routes") or []
    label_to_rid = _save_events(transcript)

    traced: List[dict] = []
    for r in routes:
        label = r.get("label")
        if route_filter and route_filter.lower() not in (label or "").lower():
            continue
        rid = label_to_rid.get(label)
        detail = _result_detail(session_dir, rid)
        traced.append({
            "label": label,
            "total_score": r.get("total_score"),
            "sub_scores": r.get("sub_scores"),
            "n_steps": r.get("num_steps") or len(r.get("steps") or []),
            "score_route_result_id": rid,
            "detail": detail,
            "steps": r.get("steps"),
        })
    return traced


def per_engine_predictions(session_dir: str) -> List[dict]:
    """Every neural_retro call's per-engine raw top-K (from `_detail`)."""
    out: List[dict] = []
    for path in sorted(glob.glob(os.path.join(session_dir, "results", "*.json"))):
        try:
            blob = _load_json(path)
        except Exception:
            continue
        if blob.get("tool") != "neural_retro":
            continue
        content = blob.get("content")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except Exception:
                continue
        if not isinstance(content, dict):
            continue
        detail = content.get("_detail") or {}
        pep = detail.get("per_engine_predictions")
        if not pep:
            continue
        out.append({
            "result_id": blob.get("tool_call_id"),
            "target": content.get("target"),
            "per_engine_counts": detail.get("per_engine_counts"),
            "engine_errors": detail.get("engine_errors"),
            "per_engine_predictions": pep,
        })
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir")
    ap.add_argument("--route", default=None, help="filter routes by label substring")
    ap.add_argument("--engines", action="store_true",
                    help="also dump per-engine neural_retro predictions")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args(argv)

    sd = args.session_dir
    if not os.path.exists(os.path.join(sd, "transcript.json")):
        print(f"error: no transcript.json under {sd}", file=sys.stderr)
        return 2

    traced = trace_routes(sd, args.route)
    engines = per_engine_predictions(sd) if args.engines else None

    if args.json:
        print(json.dumps({"routes": traced, "engines": engines},
                         ensure_ascii=False, indent=2, default=str))
        return 0

    print(f"=== {sd} ===")
    print(f"saved routes: {len(traced)}"
          + (f"  (filter: {args.route!r})" if args.route else ""))
    for t in traced:
        print()
        print(f"# {t['label']}  total={t['total_score']}  steps={t['n_steps']}")
        print(f"  result_id: {t['score_route_result_id']}")
        d = t["detail"]
        if d is None:
            print("  (no _detail on disk — pre-upgrade run, or result expired)")
            continue
        for line in _fmt_feasibility(d):
            print(line)
        for line in _fmt_cost(d):
            print(line)
        for line in _fmt_greenness(d):
            print(line)

    if engines is not None:
        print()
        print(f"=== per-engine neural_retro predictions ({len(engines)} calls) ===")
        for c in engines:
            print()
            print(f"# target {c['target']}  (result {c['result_id']})")
            print(f"  counts: {c['per_engine_counts']}")
            if c["engine_errors"]:
                print(f"  errors: {c['engine_errors']}")
            for eng, preds in (c["per_engine_predictions"] or {}).items():
                print(f"  [{eng}] {len(preds)} predictions:")
                for p in preds[:8]:
                    tmpl = f"  template={p['template']}" if p.get("template") else ""
                    print(f"      {p.get('reactants')}  score={p.get('score')}{tmpl}")
                if len(preds) > 8:
                    print(f"      … +{len(preds) - 8} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
