#!/usr/bin/env python3
"""Unified post-hoc rescoring for final_results with the current verifier.

This script re-runs stored routes only; it does not re-run search. Results are
written append-only as JSONL so a long rescore can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@dataclass(frozen=True)
class RouteRecord:
    suite: str
    case_id: str
    route_id: str
    source_file: str
    target_smiles: str
    steps: tuple[dict[str, Any], ...]
    old_total: float | None
    label: str | None
    source_format: str

    @property
    def route_key(self) -> str:
        return f"{self.suite}::{self.case_id}::{self.route_id}::{self.source_file}"


def parse_step_str(value: str) -> dict[str, Any] | None:
    rxn = value.split(" (reaction", 1)[0].strip()
    if ">>" not in rxn:
        return None
    lhs, rhs = rxn.split(">>", 1)
    reactants = [x.strip() for x in lhs.split(".") if x.strip()]
    product = rhs.strip()
    if not reactants or not product:
        return None
    return {"reactants": reactants, "product": product}


def normalize_step(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        return parse_step_str(value)
    if isinstance(value, dict):
        reactants = value.get("reactants")
        product = value.get("product")
        if isinstance(reactants, str):
            reactants = [x.strip() for x in reactants.split(".") if x.strip()]
        if isinstance(reactants, list) and isinstance(product, str):
            reactants = [str(x).strip() for x in reactants if str(x).strip()]
            product = product.strip()
            if reactants and product:
                return {"reactants": reactants, "product": product}
    return None


def normalize_steps(values: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(values, list):
        return ()
    out = [s for s in (normalize_step(v) for v in values) if s]
    return tuple(out)


def parse_route_text(route_text: str) -> tuple[str | None, tuple[dict[str, Any], ...]]:
    label = None
    m = re.search(r"label=([^,)\s]+)", route_text)
    if m:
        label = m.group(1)
    steps: list[dict[str, Any]] = []
    for line in route_text.splitlines():
        line = line.strip()
        m = re.match(r"Step\s+\d+:\s*(.+)", line)
        if not m:
            continue
        step = parse_step_str(m.group(1))
        if step:
            steps.append(step)
    return label, tuple(steps)


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def suite_from_path(path: Path) -> str:
    try:
        rel = path.relative_to(REPO / "final_results")
        return rel.parts[0]
    except ValueError:
        return path.parts[-3] if len(path.parts) >= 3 else "unknown"


def case_from_path(path: Path, data: dict[str, Any]) -> str:
    for key in ("case_id", "label", "name"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    idx = data.get("idx")
    if idx is not None:
        return f"idx-{idx}"
    m = re.search(r"case[-_](\d+)", str(path))
    if m:
        return f"case-{m.group(1)}"
    return path.stem


def target_from_data(data: dict[str, Any]) -> str | None:
    for key in ("target_smiles", "smiles", "target_canon"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    result = data.get("result")
    if isinstance(result, dict):
        for key in ("target_smiles", "smiles"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def route_records_from_run_json(path: Path) -> Iterable[RouteRecord]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return
    if not isinstance(data, dict):
        return
    target = target_from_data(data)
    if not target:
        return
    suite = suite_from_path(path)
    case_id = case_from_path(path, data)

    all_routes = data.get("all_routes")
    if isinstance(all_routes, list) and all_routes:
        for i, route in enumerate(all_routes):
            if not isinstance(route, dict):
                continue
            steps = normalize_steps(route.get("steps"))
            if not steps:
                continue
            yield RouteRecord(
                suite=suite,
                case_id=case_id,
                route_id=str(route.get("rank") or route.get("label") or i),
                source_file=str(path.relative_to(REPO)),
                target_smiles=target,
                steps=steps,
                old_total=as_float(route.get("total_score")),
                label=route.get("label") if isinstance(route.get("label"), str) else None,
                source_format="all_routes_json",
            )
        return

    steps = normalize_steps(data.get("steps"))
    if steps:
        yield RouteRecord(
            suite=suite,
            case_id=case_id,
            route_id="route_text_steps",
            source_file=str(path.relative_to(REPO)),
            target_smiles=target,
            steps=steps,
            old_total=None,
            label=None,
            source_format="top_level_steps_json",
        )
        return

    route_text = data.get("route_text")
    if isinstance(route_text, str) and route_text.strip():
        label, parsed_steps = parse_route_text(route_text)
        if parsed_steps:
            old = None
            m = re.search(r"total_score=([0-9.]+)", route_text)
            if m:
                old = as_float(m.group(1))
            yield RouteRecord(
                suite=suite,
                case_id=case_id,
                route_id=label or "route_text",
                source_file=str(path.relative_to(REPO)),
                target_smiles=target,
                steps=parsed_steps,
                old_total=old,
                label=label,
                source_format="route_text_json",
            )


def route_records_from_transcript(path: Path) -> Iterable[RouteRecord]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return
    if not isinstance(data, dict):
        return
    target = target_from_data(data)
    if not target:
        return
    suite = suite_from_path(path)
    case_id = case_from_path(path, data)
    m = re.search(r"case[-_](\d+)", str(path))
    if m:
        case_id = f"case-{m.group(1)}"
    scratchpad = data.get("scratchpad") or {}
    routes = scratchpad.get("routes") if isinstance(scratchpad, dict) else None
    if not isinstance(routes, list):
        return
    for i, route in enumerate(routes):
        if not isinstance(route, dict):
            continue
        steps = normalize_steps(route.get("steps"))
        if not steps:
            continue
        yield RouteRecord(
            suite=suite,
            case_id=case_id,
            route_id=str(route.get("rank") or route.get("label") or i),
            source_file=str(path.relative_to(REPO)),
            target_smiles=target,
            steps=steps,
            old_total=as_float(route.get("total_score")),
            label=route.get("label") if isinstance(route.get("label"), str) else None,
            source_format="scratchpad_transcript",
        )


def iter_route_records(roots: list[Path]) -> Iterable[RouteRecord]:
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        patterns = [
            "**/runs/*.json",
            "**/transcript.json",
            "**/transcripts/*.json",
        ]
        for pattern in patterns:
            for raw in glob.iglob(str(root / pattern), recursive=True):
                path = Path(raw)
                if any(part in {"verifier_scores", "_reports"} for part in path.parts):
                    continue
                extractor = (
                    route_records_from_transcript
                    if path.name == "transcript.json" or path.parent.name == "transcripts"
                    else route_records_from_run_json
                )
                for rec in extractor(path) or ():
                    if rec.route_key in seen:
                        continue
                    seen.add(rec.route_key)
                    yield rec


def completed_keys(jsonl_path: Path) -> set[str]:
    keys: set[str] = set()
    if not jsonl_path.exists():
        return keys
    with jsonl_path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except Exception:
                continue
            key = row.get("route_key")
            if isinstance(key, str) and not row.get("error"):
                keys.add(key)
    return keys


def summarize(jsonl_path: Path, csv_path: Path, summary_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    if jsonl_path.exists():
        with jsonl_path.open() as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    fields = [
        "suite",
        "case_id",
        "route_id",
        "label",
        "source_format",
        "source_file",
        "old_total",
        "new_total",
        "delta",
        "feasibility_score",
        "raw_mean",
        "worst_step",
        "pre_judge_score",
        "cost_score",
        "greenness_score",
        "n_steps",
        "error",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    by_suite: dict[str, list[float]] = {}
    errors = 0
    for row in rows:
        if row.get("error"):
            errors += 1
            continue
        score = as_float(row.get("new_total"))
        if score is None:
            continue
        by_suite.setdefault(str(row.get("suite")), []).append(score)
    summary = {
        "n_rows": len(rows),
        "n_errors": errors,
        "jsonl": str(jsonl_path.relative_to(REPO)),
        "csv": str(csv_path.relative_to(REPO)),
        "suites": {
            suite: {
                "n": len(vals),
                "mean": round(statistics.mean(vals), 6),
                "median": round(statistics.median(vals), 6),
                "min": round(min(vals), 6),
                "max": round(max(vals), 6),
            }
            for suite, vals in sorted(by_suite.items())
            if vals
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", default=["final_results"])
    ap.add_argument("--output-dir", default="final_results/_rescored_dispersion")
    ap.add_argument("--config", default=None)
    ap.add_argument("--thinking", default="fast")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--route-timeout-s", type=float, default=180.0)
    ap.add_argument(
        "--disable-judge",
        action="store_true",
        help="Isolation/debug mode: keep reaction_rater + dispersion penalty, but disable LLM judge.",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "rescored_routes.jsonl"
    csv_path = out_dir / "rescored_routes.csv"
    summary_path = out_dir / "summary.json"

    if args.summarize_only:
        summarize(jsonl_path, csv_path, summary_path)
        return 0

    roots = [Path(r) if Path(r).is_absolute() else REPO / r for r in args.roots]
    records = list(iter_route_records(roots))
    done = completed_keys(jsonl_path)
    pending = [r for r in records if r.route_key not in done]
    if args.limit:
        pending = pending[: args.limit]

    print(f"discovered_routes={len(records)} completed={len(done)} pending={len(pending)}")
    if args.dry_run:
        for rec in pending[:20]:
            print(f"DRY {rec.suite} {rec.case_id} {rec.route_id} steps={len(rec.steps)} old={rec.old_total}")
        summarize(jsonl_path, csv_path, summary_path)
        return 0

    os.environ.pop("AGENT_SESSION_ID", None)
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
        os.environ.pop(name, None)

    from core import build_agent
    from core.config import load_config

    cli_overrides = None
    if args.disable_judge:
        cli_overrides = {"verifier": {"feasibility": {"judge": {"enabled": False}}}}
    config = load_config(args.config, thinking=args.thinking, cli_overrides=cli_overrides)
    agent = build_agent(config)
    tool = agent.tools.get("score_route")
    if tool is None:
        raise RuntimeError("score_route tool not registered")
    if abs(float(getattr(tool, "feasibility_dispersion_lambda", 0.0)) - 0.3) > 1e-9:
        raise RuntimeError(f"unexpected dispersion lambda: {tool.feasibility_dispersion_lambda}")
    if not tool.judge_cfg.enabled and not args.disable_judge:
        raise RuntimeError("feasibility judge is disabled; unified rescore requires pinned judge")
    judge = tool._judge._get_llm() if tool._judge is not None else None
    if judge is not None:
        judge.max_retries = min(getattr(judge, "max_retries", 2), 2)
        judge.read_timeout = min(getattr(judge, "read_timeout", 120.0) or 120.0, 90.0)

    git_rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "verifier_git": git_rev,
        "feasibility_dispersion_lambda": tool.feasibility_dispersion_lambda,
        "judge_enabled": tool.judge_cfg.enabled,
        "judge_model": getattr(judge, "model", None),
        "judge_base_url": getattr(judge, "base_url", None),
        "judge_disabled_for_debug": bool(args.disable_judge),
        "roots": [str(r.relative_to(REPO) if r.is_relative_to(REPO) else r) for r in roots],
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, ensure_ascii=False))

    pool = ThreadPoolExecutor(max_workers=1)
    with jsonl_path.open("a") as out:
        for n, rec in enumerate(pending, 1):
            t0 = time.time()
            row: dict[str, Any] = {
                "route_key": rec.route_key,
                "suite": rec.suite,
                "case_id": rec.case_id,
                "route_id": rec.route_id,
                "label": rec.label,
                "source_file": rec.source_file,
                "source_format": rec.source_format,
                "target_smiles": rec.target_smiles,
                "n_steps": len(rec.steps),
                "old_total": rec.old_total,
                "verifier_git": git_rev,
                "feasibility_dispersion_lambda": tool.feasibility_dispersion_lambda,
            }
            try:
                fut = pool.submit(
                    tool.execute,
                    smiles=rec.target_smiles,
                    route_steps=json.dumps(list(rec.steps)),
                )
                raw = fut.result(timeout=args.route_timeout_s)
                res = json.loads(raw) if isinstance(raw, str) else raw
                row.update(
                    {
                        "new_total": res.get("total_score"),
                        "delta": (
                            round(float(res["total_score"]) - rec.old_total, 6)
                            if rec.old_total is not None and isinstance(res.get("total_score"), (int, float))
                            else None
                        ),
                        "feasibility_score": res.get("feasibility_score"),
                        "cost_score": res.get("cost_score"),
                        "greenness_score": res.get("greenness_score"),
                    }
                )
                detail = res.get("_detail") if isinstance(res.get("_detail"), dict) else {}
                feas = detail.get("feasibility", {}) if isinstance(detail.get("feasibility"), dict) else {}
                if isinstance(feas, dict):
                    per_step = feas.get("per_step") if isinstance(feas.get("per_step"), list) else []
                    step_scores = [
                        float(ps["final_score"])
                        for ps in per_step
                        if isinstance(ps, dict) and isinstance(ps.get("final_score"), (int, float))
                    ]
                    raw_mean = round(sum(step_scores) / len(step_scores), 4) if step_scores else None
                    worst_step = round(min(step_scores), 4) if step_scores else None
                    pre_judge = (
                        round(raw_mean - tool.feasibility_dispersion_lambda * (raw_mean - worst_step), 4)
                        if raw_mean is not None and worst_step is not None
                        else None
                    )
                    row.update(
                        {
                            "raw_mean": feas.get("raw_mean", raw_mean),
                            "worst_step": feas.get("worst_step", worst_step),
                            "pre_judge_score": feas.get("pre_judge_score", pre_judge),
                            "feasibility_notes": " | ".join(feas.get("notes") or []),
                        }
                    )
            except FutureTimeout:
                row["error"] = f"timeout_after_{args.route_timeout_s}s"
                pool.shutdown(wait=False, cancel_futures=True)
                pool = ThreadPoolExecutor(max_workers=1)
            except Exception as exc:
                row["error"] = f"{exc.__class__.__name__}: {exc}"
            row["elapsed_s"] = round(time.time() - t0, 3)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            status = "ERR" if row.get("error") else f"{row.get('new_total')}"
            print(f"[{n}/{len(pending)}] {rec.suite} {rec.case_id} {rec.route_id} -> {status} ({row['elapsed_s']}s)")

    pool.shutdown(wait=False, cancel_futures=True)
    summarize(jsonl_path, csv_path, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
