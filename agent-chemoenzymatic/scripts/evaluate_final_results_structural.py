#!/usr/bin/env python3
"""Structural success metrics for final_results using unified verifier scores.

The script deliberately does not trust the agent's own ``success`` flags.  A
route is considered structurally successful only if RDKit can parse every
molecule, RDKit can parse every retrosynthetic reaction SMILES, and the route
DAG connects back to the target.  Ranking uses the post-hoc verifier scores in
``final_results/_rescored_dispersion``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem


REPO = Path(__file__).resolve().parents[1]
RESCORE_SCRIPT = REPO / "scripts" / "rescore_final_results_dispersion.py"
DEFAULT_RESCored = REPO / "final_results" / "_rescored_dispersion" / "rescored_routes.jsonl"
DEFAULT_OUT = REPO / "final_results" / "_structural_eval_dispersion"

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class RetroStep:
    parents: tuple[str, ...]
    precursors: tuple[str, ...]


def load_rescore_module():
    spec = importlib.util.spec_from_file_location("rescore_final_results_dispersion", RESCORE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RESCORE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def canon(smiles: str, *, isomeric: bool = False) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric)


def split_mols(blob: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(blob).split(".") if x.strip())


def retro_steps_from_record_steps(steps: tuple[dict[str, Any], ...]) -> list[RetroStep]:
    out: list[RetroStep] = []
    for st in steps:
        # rescore_final_results_dispersion.parse_step_str stores the left side
        # of the emitted retrosynthetic step in "reactants", and the right side
        # in "product".  Preserve that emitted parent>>precursors direction.
        parents_raw = st.get("reactants") or []
        if isinstance(parents_raw, str):
            parents = split_mols(parents_raw)
        else:
            parents = tuple(str(x).strip() for x in parents_raw if str(x).strip())
        precursors = split_mols(str(st.get("product", "")).split(" (reaction", 1)[0])
        out.append(RetroStep(parents=parents, precursors=precursors))
    return out


def reaction_parses(step: RetroStep) -> bool:
    if not step.parents or not step.precursors:
        return False
    rxn_smiles = ".".join(step.parents) + ">>" + ".".join(step.precursors)
    try:
        rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)
        return (
            rxn is not None
            and rxn.GetNumReactantTemplates() > 0
            and rxn.GetNumProductTemplates() > 0
        )
    except Exception:
        return False


def validate_route(target_smiles: str, steps: list[RetroStep]) -> dict[str, Any]:
    problems: list[str] = []
    target_relaxed = canon(target_smiles, isomeric=False)
    target_strict = canon(target_smiles, isomeric=True)
    if target_relaxed is None:
        problems.append("target_smiles_unparseable")

    mol_tokens: list[str] = []
    for step in steps:
        mol_tokens.extend(step.parents)
        mol_tokens.extend(step.precursors)

    bad_mols = sorted({s for s in mol_tokens if canon(s, isomeric=False) is None})
    if bad_mols:
        problems.append("molecule_smiles_unparseable")

    bad_reactions = [i + 1 for i, step in enumerate(steps) if not reaction_parses(step)]
    if bad_reactions:
        problems.append("reaction_smiles_unparseable")

    parent_relaxed: set[str] = set()
    parent_strict: set[str] = set()
    precursor_relaxed: set[str] = set()
    multi_parent_steps = 0
    empty_side_steps = 0
    for step in steps:
        if len(step.parents) != 1:
            multi_parent_steps += 1
        if not step.parents or not step.precursors:
            empty_side_steps += 1
        for smi in step.parents:
            c0 = canon(smi, isomeric=False)
            c1 = canon(smi, isomeric=True)
            if c0:
                parent_relaxed.add(c0)
            if c1:
                parent_strict.add(c1)
        for smi in step.precursors:
            c0 = canon(smi, isomeric=False)
            if c0:
                precursor_relaxed.add(c0)

    target_match_relaxed = bool(target_relaxed and target_relaxed in parent_relaxed)
    target_match_strict = bool(target_strict and target_strict in parent_strict)
    if not target_match_relaxed:
        problems.append("target_not_produced")
    if multi_parent_steps:
        problems.append("multi_parent_step")
    if empty_side_steps:
        problems.append("empty_reaction_side")

    dangling_products = sorted(parent_relaxed - precursor_relaxed - ({target_relaxed} if target_relaxed else set()))
    if dangling_products:
        problems.append("disconnected_intermediate")

    structural_success = (
        bool(steps)
        and target_relaxed is not None
        and not bad_mols
        and not bad_reactions
        and target_match_relaxed
        and not dangling_products
        and empty_side_steps == 0
        and multi_parent_steps == 0
    )
    return {
        "structural_success": structural_success,
        "rdkit_mol_valid": not bad_mols and target_relaxed is not None,
        "reaction_parse_valid": not bad_reactions,
        "target_match_relaxed": target_match_relaxed,
        "target_match_strict": target_match_strict,
        "connected_to_target": target_match_relaxed and not dangling_products,
        "n_steps": len(steps),
        "n_bad_molecules": len(bad_mols),
        "bad_reaction_steps": ";".join(map(str, bad_reactions)),
        "problems": ";".join(problems),
    }


def case_tag_from_source(source_file: str) -> str:
    path = Path(source_file)
    text = str(path)
    m = re.search(r"case[-_](\d+)", text)
    if m:
        return f"case-{int(m.group(1))}"
    if path.name == "transcript.json":
        return path.parent.name
    if path.suffix in {".json", ".log", ".txt"}:
        return path.stem
    return text


def collect_expected_cases(final_results: Path, suites: set[str]) -> dict[str, set[str]]:
    expected: dict[str, set[str]] = {s: set() for s in suites}
    for suite in suites:
        root = final_results / suite
        if not root.exists():
            continue
        runs = sorted((root / "runs").glob("*.json"))
        if runs:
            expected[suite].update(case_tag_from_source(str(p.relative_to(REPO))) for p in runs)
        transcripts = sorted(root.glob("**/transcript.json"))
        for p in transcripts:
            rel = p.relative_to(REPO)
            expected[suite].add(case_tag_from_source(str(rel)))
        # Plain Codex baseline stores one result per json under results/.
        for p in sorted((root / "results").glob("*.json")):
            expected[suite].add(case_tag_from_source(str(p.relative_to(REPO))))
    return expected


def max_iteration_from_log(path: Path) -> int | None:
    if not path.exists():
        return None
    max_iter: int | None = None
    try:
        for line in path.open(errors="ignore"):
            m = re.search(r"=== Iteration\s+(\d+)\s+===", line)
            if m:
                value = int(m.group(1))
                max_iter = value if max_iter is None else max(max_iter, value)
    except Exception:
        return None
    return max_iter


def save_iterations_from_log(path: Path) -> tuple[dict[str, int], int | None]:
    by_label: dict[str, int] = {}
    current_iter: int | None = None
    max_iter: int | None = None
    if not path.exists():
        return by_label, None
    try:
        for line in path.open(errors="ignore"):
            m_iter = re.search(r"=== Iteration\s+(\d+)\s+===", line)
            if m_iter:
                current_iter = int(m_iter.group(1))
                max_iter = current_iter if max_iter is None else max(max_iter, current_iter)
            if "save_best_route" not in line or current_iter is None:
                continue
            if '"status": "invalid_step_rejected"' in line or '"status": "duplicate"' in line:
                continue
            m_label = re.search(r'"label"\s*:\s*"([^"]+)"', line)
            if m_label:
                label = m_label.group(1)
                by_label[label] = min(by_label.get(label, current_iter), current_iter)
    except Exception:
        return by_label, max_iter
    return by_label, max_iter


def load_scores(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    with jsonl_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("route_key")
            if isinstance(key, str):
                scores[key] = row
    return scores


def summarize_suite(rows: list[dict[str, Any]], cases: list[dict[str, Any]], budgets: list[int]) -> dict[str, Any]:
    n = len(cases)
    solved = [c for c in cases if c["solved"]]
    valid_routes = [r for r in rows if r["structural_success"]]
    scored_routes = [r for r in rows if r.get("new_total") is not None]
    rank1_scores = [c["rank1_score"] for c in cases if c.get("rank1_score") is not None]
    best_valid_scores = [c["best_valid_score"] for c in cases if c.get("best_valid_score") is not None]

    def pct(count: int) -> float:
        return round(100.0 * count / n, 2) if n else 0.0

    pass_at = {}
    pass_at_count = {}
    for k in (1, 3, 5, 10):
        count = sum(1 for c in cases if c.get("first_valid_rank") is not None and c["first_valid_rank"] <= k)
        pass_at[str(k)] = pct(count)
        pass_at_count[str(k)] = count

    success_at_budget = {}
    success_at_budget_count = {}
    for b in budgets:
        count = sum(
            1
            for c in cases
            if c.get("first_valid_budget") is not None and c["first_valid_budget"] <= b
        )
        success_at_budget[str(b)] = pct(count)
        success_at_budget_count[str(b)] = count

    def mean(values: list[float]) -> float | None:
        return round(statistics.mean(values), 6) if values else None

    def median(values: list[float]) -> float | None:
        return round(statistics.median(values), 6) if values else None

    return {
        "n_cases": n,
        "n_cases_with_scored_routes": sum(1 for c in cases if c["n_scored_routes"] > 0),
        "n_scored_routes": len(scored_routes),
        "n_structurally_valid_routes": len(valid_routes),
        "solved": len(solved),
        "solved_rate_percent": pct(len(solved)),
        "pass_at": pass_at,
        "pass_at_count": pass_at_count,
        "success_at_budget": success_at_budget,
        "success_at_budget_count": success_at_budget_count,
        "rank1_score_mean": mean(rank1_scores),
        "rank1_score_median": median(rank1_scores),
        "best_valid_score_mean": mean(best_valid_scores),
        "best_valid_score_median": median(best_valid_scores),
        "mean_valid_route_steps": mean([float(r["n_steps"]) for r in valid_routes]),
        "reaction_parse_route_rate_percent": round(
            100.0 * sum(1 for r in rows if r["reaction_parse_valid"]) / len(rows), 2
        )
        if rows
        else 0.0,
        "rdkit_mol_route_rate_percent": round(
            100.0 * sum(1 for r in rows if r["rdkit_mol_valid"]) / len(rows), 2
        )
        if rows
        else 0.0,
        "score_coverage_note": "metrics rank only routes present in _rescored_dispersion",
    }


def write_report(path: Path, summary: dict[str, Any], budgets: list[int]) -> None:
    lines = [
        "# Structural Evaluation With 6/20 Dispersion Verifier",
        "",
        "Success is computed from RDKit molecule validity, RDKit reaction-SMILES parsing, and route connectivity to the target. Top-k ranking uses `new_total` from `_rescored_dispersion`; success@budget uses first valid save iteration * 4 when the save log is available, otherwise final run iteration * 4.",
        "",
        "| suite | n | solved | solved % | pass@1 | pass@3 | pass@5 | best valid score | rank1 score | scored routes | valid routes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for suite, s in sorted(summary["suites"].items()):
        lines.append(
            f"| {suite} | {s['n_cases']} | {s['solved']} | {s['solved_rate_percent']:.2f} | "
            f"{s['pass_at']['1']:.2f} | {s['pass_at']['3']:.2f} | {s['pass_at']['5']:.2f} | "
            f"{s['best_valid_score_mean'] if s['best_valid_score_mean'] is not None else ''} | "
            f"{s['rank1_score_mean'] if s['rank1_score_mean'] is not None else ''} | "
            f"{s['n_scored_routes']} | {s['n_structurally_valid_routes']} |"
        )
    lines += ["", "## Success@Budget", ""]
    header = "| suite | " + " | ".join(f"@{b}" for b in budgets) + " |"
    lines.append(header)
    lines.append("|---|" + "|".join("---:" for _ in budgets) + "|")
    for suite, s in sorted(summary["suites"].items()):
        lines.append(
            f"| {suite} | "
            + " | ".join(f"{s['success_at_budget'][str(b)]:.2f}" for b in budgets)
            + " |"
        )
    lines += [
        "",
        "## Caveats",
        "",
        "- FDA_GPT55_RETRACE has only one route per case in the existing 6/20 rescore artifact; its historical saved candidates are recoverable from transcripts but cannot be ranked by final verifier score without rerunning the verifier.",
        "- No Chematica agent-result directory was found under `final_results`; reference-route scoring for the Chematica zip is a separate input-only evaluation.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescored-jsonl", default=str(DEFAULT_RESCored))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--suites",
        nargs="*",
        default=[
            "FDA_GPT55_RETRACE",
            "FDA_Qwen36_RETRACE",
            "JMC_GPT55_RETRACE",
            "JMC_Qwen36_RETRACE",
            "Retro190_GPT55-fast_RETRACE",
            "Retro190_GPT55-standard_RETRACE",
            "Retro190_Opus48_RETRACE",
            "Retro190_Qwen36_RETRACE",
            "Retro190_GPT55_Codex",
        ],
    )
    ap.add_argument("--budgets", nargs="*", type=int, default=[50, 100, 200, 300, 400])
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = load_scores(Path(args.rescored_jsonl))
    rescore_mod = load_rescore_module()
    suite_filter = set(args.suites)
    roots = [REPO / "final_results" / suite for suite in sorted(suite_filter)]
    records = list(rescore_mod.iter_route_records(roots))
    expected = collect_expected_cases(REPO / "final_results", suite_filter)

    route_rows: list[dict[str, Any]] = []
    for rec in records:
        if rec.suite not in suite_filter:
            continue
        score_row = scores.get(rec.route_key)
        if not score_row or score_row.get("error"):
            continue
        new_total = as_float(score_row.get("new_total"))
        if new_total is None:
            continue
        steps = retro_steps_from_record_steps(rec.steps)
        val = validate_route(rec.target_smiles, steps)
        tag = case_tag_from_source(rec.source_file)
        expected.setdefault(rec.suite, set()).add(tag)
        route_rows.append(
            {
                "suite": rec.suite,
                "case_tag": tag,
                "case_id": rec.case_id,
                "route_id": rec.route_id,
                "label": rec.label or rec.route_id,
                "source_file": rec.source_file,
                "source_format": rec.source_format,
                "new_total": new_total,
                "feasibility_score": as_float(score_row.get("feasibility_score")),
                "raw_mean": as_float(score_row.get("raw_mean")),
                "worst_step": as_float(score_row.get("worst_step")),
                "cost_score": as_float(score_row.get("cost_score")),
                "greenness_score": as_float(score_row.get("greenness_score")),
                **val,
            }
        )

    # Iteration budget is route-level when save_best_route can be recovered from logs.
    case_iterations: dict[tuple[str, str], int | None] = {}
    save_iterations: dict[tuple[str, str, str], int] = {}
    for suite, tags in expected.items():
        root = REPO / "final_results" / suite
        for p in (root / "runs").glob("*.json"):
            try:
                data = json.loads(p.read_text())
            except Exception:
                data = {}
            tag = case_tag_from_source(str(p.relative_to(REPO)))
            case_iterations[(suite, tag)] = int(data["iterations"]) if isinstance(data.get("iterations"), int) else None
        for p in (root / "logs").glob("*.log"):
            tag = case_tag_from_source(str(p.relative_to(REPO)))
            label_iters, max_iter = save_iterations_from_log(p)
            if max_iter is not None:
                case_iterations.setdefault((suite, tag), max_iter)
            for label, it in label_iters.items():
                save_iterations[(suite, tag, label)] = it
        for p in root.glob("**/transcript.json"):
            tag = case_tag_from_source(str(p.relative_to(REPO)))
            try:
                data = json.loads(p.read_text())
            except Exception:
                data = {}
            it = data.get("iterations")
            if isinstance(it, int):
                case_iterations[(suite, tag)] = it
            else:
                label_iters, log_iter = save_iterations_from_log(p.parent / "stdout.log")
                if log_iter is not None:
                    case_iterations[(suite, tag)] = log_iter
                for label, save_it in label_iters.items():
                    save_iterations[(suite, tag, label)] = save_it

    by_case: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in route_rows:
        by_case.setdefault((row["suite"], row["case_tag"]), []).append(row)

    case_rows: list[dict[str, Any]] = []
    for suite in sorted(expected):
        for tag in sorted(expected[suite]):
            routes = sorted(
                by_case.get((suite, tag), []),
                key=lambda r: (-(r["new_total"] if r["new_total"] is not None else -math.inf), r["label"]),
            )
            first_valid_rank = None
            best_valid_score = None
            best_valid_label = None
            best_valid_steps = None
            for i, route in enumerate(routes, start=1):
                route["rank_by_final_score"] = i
                save_it = save_iterations.get((suite, tag, route["label"]))
                if save_it is not None:
                    route["save_iteration"] = save_it
                    route["save_budget"] = save_it * 4
                if route["structural_success"] and first_valid_rank is None:
                    first_valid_rank = i
                    best_valid_score = route["new_total"]
                    best_valid_label = route["label"]
                    best_valid_steps = route["n_steps"]
            iterations = case_iterations.get((suite, tag))
            budget = iterations * 4 if isinstance(iterations, int) else None
            valid_budgets = [
                r.get("save_budget")
                for r in routes
                if r["structural_success"] and isinstance(r.get("save_budget"), int)
            ]
            first_valid_budget = min(valid_budgets) if valid_budgets else (budget if first_valid_rank is not None else None)
            case_rows.append(
                {
                    "suite": suite,
                    "case_tag": tag,
                    "n_scored_routes": len(routes),
                    "solved": first_valid_rank is not None,
                    "first_valid_rank": first_valid_rank,
                    "first_valid_budget": first_valid_budget,
                    "iterations": iterations,
                    "budget": budget,
                    "rank1_score": routes[0]["new_total"] if routes else None,
                    "rank1_label": routes[0]["label"] if routes else None,
                    "rank1_structural_success": routes[0]["structural_success"] if routes else None,
                    "best_valid_score": best_valid_score,
                    "best_valid_label": best_valid_label,
                    "best_valid_steps": best_valid_steps,
                }
            )

    summary = {
        "method": "RDKit structural success; final verifier score ranking",
        "rescored_jsonl": str(Path(args.rescored_jsonl).relative_to(REPO)),
        "budget_rule": "success@budget = first valid save iteration * 4 one-step-model calls when recoverable; otherwise final run iteration * 4 fallback",
        "budgets": args.budgets,
        "suites": {},
    }
    for suite in sorted(expected):
        summary["suites"][suite] = summarize_suite(
            [r for r in route_rows if r["suite"] == suite],
            [c for c in case_rows if c["suite"] == suite],
            args.budgets,
        )

    route_fields = [
        "suite",
        "case_tag",
        "case_id",
        "rank_by_final_score",
        "label",
        "new_total",
        "save_iteration",
        "save_budget",
        "structural_success",
        "rdkit_mol_valid",
        "reaction_parse_valid",
        "target_match_relaxed",
        "target_match_strict",
        "connected_to_target",
        "n_steps",
        "problems",
        "source_file",
    ]
    with (out_dir / "per_route.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=route_fields)
        writer.writeheader()
        for row in sorted(route_rows, key=lambda r: (r["suite"], r["case_tag"], r.get("rank_by_final_score") or 9999)):
            writer.writerow({k: row.get(k) for k in route_fields})

    case_fields = [
        "suite",
        "case_tag",
        "n_scored_routes",
        "solved",
        "first_valid_rank",
        "first_valid_budget",
        "iterations",
        "budget",
        "rank1_score",
        "rank1_label",
        "rank1_structural_success",
        "best_valid_score",
        "best_valid_label",
        "best_valid_steps",
    ]
    with (out_dir / "per_case.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=case_fields)
        writer.writeheader()
        for row in sorted(case_rows, key=lambda r: (r["suite"], r["case_tag"])):
            writer.writerow({k: row.get(k) for k in case_fields})

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_report(out_dir / "REPORT.md", summary, args.budgets)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
