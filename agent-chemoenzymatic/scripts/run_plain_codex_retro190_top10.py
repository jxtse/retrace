#!/usr/bin/env python
"""Run the plain Codex retro-190 baseline ONE case at a time, emitting up to TOP-10
distinct routes per target (model-enumerated) instead of a single rank-1 route.

This is the top-10 sibling of scripts/run_plain_codex_retro190.py (recovered from git
ac52ac7). Everything about the *isolation* is identical — each nested Codex run is placed
in a temporary directory containing only the per-case input, output schema, and a stock
buyability helper; it does NOT receive MoleCode, AtomRetro, project files, skills, or prior
result directories. The ONLY differences are:
  1. ROUTE_SCHEMA is a top-level object with a `routes` array (1..10), each route ranked.
  2. The prompt asks for up to 10 DISTINCT complete routes, best-first, no filler.
  3. postprocess re-runs buyability PER route; target success = rank-1 route all-buyable.
  4. summarize reports per-route stats; success_true/non_all_buyable stay rank-1-keyed
     so they remain directly comparable to the rank-1 run's fields.
  5. A pre-launch assertion that the case workspace contains ONLY the 3 allowed files.

Methodology note: this top-10 is MODEL-ENUMERATED (one Codex response lists up to 10 routes
it prefers, ranked) — NOT a search-ranked top-K and NOT 10 independent re-runs. The downstream
summary footnotes this; solved@10 here estimates route-coverage and is not strictly comparable
to the search-ranked top-10 of the other systems.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_CSV = REPO_ROOT / "dataset/benchmark/retro-190_dataset/cases.csv"
DEFAULT_OUT_ROOT = REPO_ROOT / "final_results/Retro190_GPT55_Codex_top10"
DEFAULT_CASE_ROOT = Path("/tmp/retro190_plain_codex_top10_cases")
DEFAULT_CODEX_HOME = Path("/tmp/retro190_plain_codex_top10_home")
STOCK_CACHE = REPO_ROOT / ".cache/stock_canonical"
PYTHON = REPO_ROOT / ".venv/bin/python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

MAX_ROUTES = 10
ALLOWED_WORKSPACE_FILES = {"input.json", "route_schema.json", "check_buyability.py"}


# --- TOP-10 schema: top-level object with a ranked `routes` array (1..MAX_ROUTES). ----
# Each route keeps the EXACT inner key shape of the rank-1 baseline, plus an integer rank,
# so the per-route buyability postprocessor reuses the rank-1 logic unchanged.
_ROUTE_ITEM = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rank", "route_text", "steps", "starting_materials", "buyability_checks", "notes"],
    "properties": {
        "rank": {"type": "integer"},
        "route_text": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["step", "product", "precursors", "reaction"],
                "properties": {
                    "step": {"type": "integer"},
                    "product": {"type": "string"},
                    "precursors": {"type": "array", "items": {"type": "string"}},
                    "reaction": {"type": "string"},
                },
            },
        },
        "starting_materials": {"type": "array", "items": {"type": "string"}},
        "buyability_checks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["smiles", "buyable", "source"],
                "properties": {
                    "smiles": {"type": "string"},
                    "buyable": {"type": "boolean"},
                    "source": {"type": "string"},
                },
            },
        },
        "notes": {"type": "string"},
    },
}

ROUTE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["idx", "target_smiles", "success", "routes"],
    "properties": {
        "idx": {"type": "integer"},
        "target_smiles": {"type": "string"},
        "success": {"type": "boolean"},
        "routes": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_ROUTES,
            "items": _ROUTE_ITEM,
        },
    },
}


CHECK_BUYABILITY = f'''#!/usr/bin/env python
"""Standalone stock buyability helper for the plain Codex baseline."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from rdkit import Chem


STOCK_CACHE = Path({str(STOCK_CACHE)!r})


def validate(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None or mol.GetNumAtoms() == 0:
        return {{"input_smiles": smiles, "valid": False, "buyable": False}}
    return {{
        "input_smiles": smiles,
        "valid": True,
        "canonical_smiles": Chem.MolToSmiles(mol),
    }}


def buyability(smiles_list: list[str]) -> dict:
    stock_paths = {{
        "BioNav-stock": sorted(STOCK_CACHE.glob("BioNav-stock_*.sqlite")),
        "Zinc_Fix-stock": sorted(STOCK_CACHE.glob("Zinc_Fix-stock_*.sqlite")),
    }}
    results = []
    for smiles in smiles_list:
        item = validate(smiles)
        if not item.get("valid"):
            results.append(item)
            continue
        canonical = item["canonical_smiles"]
        found_in = []
        for stock_name, paths in stock_paths.items():
            for path in paths:
                with sqlite3.connect(f"file:{{path}}?mode=ro", uri=True) as con:
                    hit = con.execute(
                        "SELECT 1 FROM smi WHERE s = ? LIMIT 1", (canonical,)
                    ).fetchone()
                if hit is not None:
                    found_in.append(stock_name)
                    break
        item.update({{
            "smiles": canonical,
            "buyable": bool(found_in),
            "found_in_stocks": found_in,
        }})
        results.append(item)
    return {{
        "n": len(smiles_list),
        "n_buyable": sum(1 for r in results if r.get("buyable")),
        "results": results,
    }}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    buy = sub.add_parser("buyability")
    buy.add_argument("smiles", nargs="+")
    args = parser.parse_args()
    print(json.dumps(buyability(args.smiles), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def safe_tag(idx: int) -> str:
    return f"{idx:03d}__case-{idx}"


def load_cases() -> list[dict]:
    rows = []
    with CASES_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "idx": int(row["idx"]),
                "target_smiles": row["smiles"].strip(),
                "n_steps": (row.get("n_steps") or "").strip(),
            })
    return rows


def parse_only(spec: str) -> set[int]:
    out: set[int] = set()
    if not spec.strip():
        return out
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            out.update(range(int(start), int(end) + 1))
        else:
            out.add(int(chunk))
    return out


def ensure_codex_home(codex_home: Path) -> None:
    codex_home.mkdir(parents=True, exist_ok=True)
    auth_dst = codex_home / "auth.json"
    if not auth_dst.exists():
        auth_src = Path.home() / ".codex/auth.json"
        if not auth_src.exists():
            raise FileNotFoundError(f"Missing Codex auth file: {auth_src}")
        shutil.copy2(auth_src, auth_dst)
        auth_dst.chmod(0o600)
    skills = codex_home / "skills"
    if skills.exists():
        shutil.rmtree(skills)


def write_case_workspace(case_dir: Path, row: dict, *, overwrite: bool) -> tuple[Path, Path]:
    if case_dir.exists() and overwrite:
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    input_path = case_dir / "input.json"
    schema_path = case_dir / "route_schema.json"
    helper_path = case_dir / "check_buyability.py"
    input_path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    schema_path.write_text(json.dumps(ROUTE_SCHEMA, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    helper_path.write_text(CHECK_BUYABILITY, encoding="utf-8")
    helper_path.chmod(0o755)
    return input_path, schema_path


def assert_clean_workspace(case_dir: Path) -> None:
    """Anti-leakage guard: the per-case cwd handed to the nested Codex must contain ONLY
    the three allowed files. Any extra entry (a stray answer file, a prior result, etc.)
    aborts the case rather than risk exposing it to the model."""
    present = {p.name for p in case_dir.iterdir()}
    extra = present - ALLOWED_WORKSPACE_FILES
    missing = ALLOWED_WORKSPACE_FILES - present
    if extra or missing:
        raise RuntimeError(
            f"workspace contract violated in {case_dir}: extra={sorted(extra)} missing={sorted(missing)}"
        )


def make_prompt(row: dict) -> str:
    return f"""Solve one retrosynthesis benchmark case as the plain Codex baseline, returning up to {MAX_ROUTES} candidate routes.

Input:
- idx: {row['idx']}
- target_smiles: {row['target_smiles']}
- reference_step_count_for_context_only: {row['n_steps'] or 'unknown'}

Workspace contract:
- Use only the files in the current working directory: input.json,
  route_schema.json, and check_buyability.py.
- Do not inspect parent directories, repositories, prior result directories,
  hidden Codex files, AtomRetro/MoleCode files, or any other benchmark outputs.
- Do not use or install any Codex skills. This is the no-skill plain Codex
  baseline.
- Do not call this repository's RetrosynthesisAgent, neural_retro,
  suggest_templates, score_route, run_benchmark, or web search.

Required workflow:
1. Propose UP TO {MAX_ROUTES} DISTINCT, complete retrosynthetic routes for the target
   SMILES, ordered best-first by your own chemical preference (rank 1 = your single
   best route, rank 2 = your next best, and so on). Each route must be a GENUINELY
   different disconnection strategy — do NOT pad the list with reorderings, trivial
   reagent swaps, or near-duplicates of another route. Returning fewer than {MAX_ROUTES}
   routes is completely acceptable; never invent filler routes just to reach {MAX_ROUTES}.
2. For EACH route independently, the terminal starting_materials must all be valid SMILES
   strings and must all be buyable in the stock database used by our Agent. Reagent names
   alone are not acceptable starting_materials.
3. You MUST actively check each route's terminal starting materials before finalizing by
   running:
   {PYTHON} check_buyability.py buyability "SMILES1" "SMILES2" ...
4. If any terminal material in a route is not buyable, revise that route and check again,
   or drop that route. Keep every route you return all-buyable. If you cannot find any
   all-buyable route at all, return a single route with your best attempt and set
   success=false, explaining in that route's notes.
5. Make your final response exactly one JSON object matching route_schema.json, with the
   `routes` array ranked best-first (rank 1, 2, 3, ...).

For each route's route_text, use this style:
COMPLETE ROUTE FOUND: (plain Codex baseline, rank N)
Step 1: PRODUCT>>PRECURSOR1.PRECURSOR2 (reaction: ...)
...
Starting materials: ...
"""


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        input=input_text,
        text=True,
        capture_output=True,
    )


def normalize_buyability(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        out.append({
            "smiles": row.get("smiles") or row.get("canonical_smiles") or row.get("input_smiles", ""),
            "buyable": bool(row.get("buyable")),
            "source": ",".join(row.get("found_in_stocks") or []) or "not_found",
        })
    return out


def _recheck_route_buyability(case_dir: Path, route: dict) -> dict:
    """Re-run the harness buyability check on one route's declared starting_materials,
    against the main Agent stock sqlite. Returns the route dict with buyability_checks
    and all_starting_materials_buyable filled in. Mirrors the rank-1 postprocess logic."""
    starts = [str(s).strip() for s in route.get("starting_materials") or [] if str(s).strip()]
    route["starting_materials"] = starts
    if starts:
        check = run_cmd(
            [str(PYTHON), "check_buyability.py", "buyability", *starts],
            cwd=case_dir,
        )
        if check.returncode == 0:
            buy = json.loads(check.stdout)
            route["buyability_checks"] = normalize_buyability(buy.get("results", []))
        else:
            route["buyability_checks"] = []
            route["notes"] = (
                str(route.get("notes", "")) + f"\nHarness buyability check failed: {check.stderr}"
            ).strip()
    else:
        route["buyability_checks"] = []
        route["notes"] = (
            str(route.get("notes", "")) + "\nHarness enforcement: no terminal starting_materials were provided."
        ).strip()
    route["all_starting_materials_buyable"] = bool(route["buyability_checks"]) and all(
        item.get("buyable") for item in route["buyability_checks"]
    )
    return route


def postprocess_result(case_dir: Path, last_path: Path, result_path: Path, row: dict) -> dict:
    if not last_path.exists():
        payload = {
            "idx": row["idx"],
            "target_smiles": row["target_smiles"],
            "success": False,
            "routes": [],
            "notes": "No structured Codex result was produced.",
        }
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return payload

    payload = json.loads(last_path.read_text(encoding="utf-8"))
    payload["idx"] = row["idx"]
    payload["target_smiles"] = row["target_smiles"]

    routes = payload.get("routes") or []
    # Preserve the model's best-first order; renumber rank 1..N defensively.
    processed = []
    for i, route in enumerate(routes, 1):
        if not isinstance(route, dict):
            continue
        route["rank"] = i
        processed.append(_recheck_route_buyability(case_dir, route))
    payload["routes"] = processed

    # Target success = the rank-1 route is fully buyable (matches the rank-1 baseline
    # semantics, so success_true / non_all_buyable_idx stay directly comparable).
    rank1 = processed[0] if processed else None
    payload["success"] = bool(rank1 and rank1.get("all_starting_materials_buyable"))
    payload["n_routes"] = len(processed)
    payload["n_routes_all_buyable"] = sum(1 for r in processed if r.get("all_starting_materials_buyable"))
    payload["notes"] = (
        str(payload.get("notes", ""))
        + "\nHarness buyability check used the main Agent stock sqlite cache (per route)."
    ).strip()

    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def run_one(
    row: dict,
    *,
    out_root: Path,
    case_root: Path,
    codex_home: Path,
    overwrite: bool,
    model: str | None,
) -> dict:
    tag = safe_tag(row["idx"])
    dirs = {
        "runs": out_root / "runs",
        "transcripts": out_root / "transcripts",
        "last_messages": out_root / "last_messages",
        "logs": out_root / "logs",
        "case_inputs": out_root / "case_inputs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    result_path = dirs["runs"] / f"{tag}.json"
    meta_path = dirs["runs"] / f"{tag}.meta.json"
    transcript_path = dirs["transcripts"] / f"{tag}.jsonl"
    last_path = dirs["last_messages"] / f"{tag}.json"
    log_path = dirs["logs"] / f"{tag}.log"
    input_copy_path = dirs["case_inputs"] / f"{tag}.input.json"

    if result_path.exists() and transcript_path.exists() and meta_path.exists() and not overwrite:
        return {"idx": row["idx"], "tag": tag, "skipped": True}

    case_dir = case_root / tag
    input_path, schema_path = write_case_workspace(case_dir, row, overwrite=overwrite)
    shutil.copy2(input_path, input_copy_path)

    # Anti-leakage guard: refuse to launch if the workspace has anything beyond the 3
    # allowed files.
    assert_clean_workspace(case_dir)

    env = os.environ.copy()
    env["CODEX_HOME"] = str(codex_home)
    prompt = make_prompt(row)
    cmd = [
        "codex",
        "exec",
        "--json",
        "--ignore-user-config",
        "--ignore-rules",
        "--disable",
        "plugins",
        "--disable",
        "apps",
        "--skip-git-repo-check",
        "--output-schema",
        str(schema_path),
        "-o",
        str(last_path),
        "-C",
        str(case_dir),
        "--add-dir",
        str(STOCK_CACHE),
        "--sandbox",
        "workspace-write",
    ]
    if model:
        cmd.extend(["--model", model])
    cmd.append("-")

    t0 = time.time()
    proc = run_cmd(cmd, cwd=case_dir, env=env, input_text=prompt)
    elapsed = round(time.time() - t0, 1)
    transcript_path.write_text(proc.stdout, encoding="utf-8")
    log_path.write_text(proc.stderr, encoding="utf-8")

    meta = {
        "idx": row["idx"],
        "tag": tag,
        "target_smiles": row["target_smiles"],
        "case_dir": str(case_dir),
        "codex_home": str(codex_home),
        "codex_returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "last_message_path": str(last_path),
        "transcript_path": str(transcript_path),
        "log_path": str(log_path),
        "no_skills": True,
        "ignore_user_config": True,
        "ignore_rules": True,
        "max_routes": MAX_ROUTES,
        "exposed_inputs": sorted(ALLOWED_WORKSPACE_FILES),
        "exposed_external_read": str(STOCK_CACHE),
    }

    try:
        postprocess_result(case_dir, last_path, result_path, row)
    except Exception as exc:
        payload = {
            "idx": row["idx"],
            "target_smiles": row["target_smiles"],
            "success": False,
            "routes": [],
            "notes": f"Postprocess failed: {type(exc).__name__}: {exc}",
        }
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        meta["postprocess_error"] = f"{type(exc).__name__}: {exc}"

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**meta, "result_exists": result_path.exists()}


def summarize(out_root: Path, expected_rows: list[dict]) -> dict:
    runs_dir = out_root / "runs"
    tr_dir = out_root / "transcripts"
    expected_tags = {safe_tag(row["idx"]) for row in expected_rows}
    result_paths = sorted(p for p in runs_dir.glob("*.json") if not p.name.endswith(".meta.json"))
    meta_paths = sorted(runs_dir.glob("*.meta.json"))
    transcript_paths = sorted(tr_dir.glob("*.jsonl"))
    result_tags = {p.stem for p in result_paths}
    meta_tags = {p.name.removesuffix(".meta.json") for p in meta_paths}
    transcript_tags = {p.stem for p in transcript_paths}

    payloads = []
    for path in result_paths:
        with path.open(encoding="utf-8") as f:
            payloads.append(json.load(f))
    metas = []
    for path in meta_paths:
        with path.open(encoding="utf-8") as f:
            metas.append(json.load(f))

    elapsed = [m.get("elapsed_seconds") for m in metas if isinstance(m.get("elapsed_seconds"), (int, float))]

    # Per-route stats over all targets.
    routes_per_target = [len(p.get("routes") or []) for p in payloads]
    total_routes = sum(routes_per_target)
    all_route_checks = [
        check
        for p in payloads
        for r in (p.get("routes") or [])
        for check in (r.get("buyability_checks") or [])
    ]
    # rank-1-keyed buyability (comparable to the rank-1 run's 189/1 and [162]).
    def _rank1(p):
        rs = p.get("routes") or []
        return rs[0] if rs else None

    non_all_buyable = [
        p.get("idx")
        for p in payloads
        if not (_rank1(p) and _rank1(p).get("all_starting_materials_buyable"))
    ]

    summary = {
        "dataset": "retro-190",
        "method": "plain_codex_no_skills_top10",
        "depth": "top-10 (model-enumerated, single response)",
        "max_routes": MAX_ROUTES,
        "n_cases_expected": len(expected_rows),
        "n_results": len(result_paths),
        "n_meta": len(meta_paths),
        "n_transcripts": len(transcript_paths),
        "success_true": sum(1 for p in payloads if p.get("success") is True),
        "success_false": sum(1 for p in payloads if p.get("success") is False),
        "all_starting_materials_buyable": len(payloads) - len(non_all_buyable),
        "non_all_buyable_idx": sorted(x for x in non_all_buyable if x is not None),
        "total_routes": total_routes,
        "routes_per_target_mean": round(statistics.mean(routes_per_target), 2) if routes_per_target else 0.0,
        "routes_per_target_median": round(statistics.median(routes_per_target), 1) if routes_per_target else 0.0,
        "routes_per_target_min": min(routes_per_target) if routes_per_target else 0,
        "routes_per_target_max": max(routes_per_target) if routes_per_target else 0,
        "targets_with_ge1_buyable_route": sum(
            1 for p in payloads if any(r.get("all_starting_materials_buyable") for r in (p.get("routes") or []))
        ),
        "codex_returncode_zero": sum(1 for m in metas if m.get("codex_returncode") == 0),
        "nonzero_codex_returncodes": [
            {"idx": m.get("idx"), "returncode": m.get("codex_returncode")}
            for m in metas
            if m.get("codex_returncode") != 0
        ],
        "missing_result_tags": sorted(expected_tags - result_tags),
        "extra_result_tags": sorted(result_tags - expected_tags),
        "missing_meta_tags": sorted(expected_tags - meta_tags),
        "missing_transcript_tags": sorted(expected_tags - transcript_tags),
        "total_route_buyability_checks": len(all_route_checks),
        "buyable_starting_materials": sum(1 for c in all_route_checks if c.get("buyable")),
        "buyable_starting_material_fraction": (
            round(sum(1 for c in all_route_checks if c.get("buyable")) / len(all_route_checks), 4)
            if all_route_checks
            else 0.0
        ),
        "elapsed_seconds_total": round(sum(elapsed), 1) if elapsed else 0.0,
        "elapsed_seconds_mean": round(statistics.mean(elapsed), 1) if elapsed else 0.0,
        "elapsed_seconds_median": round(statistics.median(elapsed), 1) if elapsed else 0.0,
        "elapsed_seconds_max": round(max(elapsed), 1) if elapsed else 0.0,
        "out_root": str(out_root),
        "runner": "scripts/run_plain_codex_retro190_top10.py",
        "no_skills": True,
        "isolation": (
            "Nested Codex used a temporary CODEX_HOME containing auth.json only, "
            "with --ignore-user-config, --ignore-rules, --skip-git-repo-check, "
            "and per-case cwd containing ONLY input.json, route_schema.json, and "
            "check_buyability.py (asserted before launch). No MoleCode/AtomRetro skill "
            "or prior results were added to the nested workspace."
        ),
        "topk_caveat": (
            "MODEL-ENUMERATED top-10: a single Codex response listed up to 10 routes ranked "
            "by the model's own preference — not a search-ranked top-K and not 10 independent "
            "re-runs. Answer-key leakage screened post-hoc by _codex_top10_leakage_check.py."
        ),
        "buyability": (
            "Nested Codex was required to call check_buyability.py per route. The harness "
            "rechecked each route's terminal starting_materials against the main Agent stock "
            f"sqlite cache at {STOCK_CACHE}; target success = rank-1 route all-buyable."
        ),
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--codex-home", type=Path, default=DEFAULT_CODEX_HOME)
    parser.add_argument("--only", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model")
    args = parser.parse_args()

    args.out_root = args.out_root.expanduser()
    if not args.out_root.is_absolute():
        args.out_root = (Path.cwd() / args.out_root).resolve()
    else:
        args.out_root = args.out_root.resolve()
    args.case_root = args.case_root.expanduser().resolve()
    args.codex_home = args.codex_home.expanduser().resolve()

    ensure_codex_home(args.codex_home)
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = load_cases()
    wanted = parse_only(args.only)
    if wanted:
        rows = [row for row in rows if row["idx"] in wanted]
    elif args.start or args.limit is not None:
        stop = None if args.limit is None else args.start + args.limit
        rows = rows[args.start:stop]

    manifest = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_root": str(args.out_root),
        "case_root": str(args.case_root),
        "codex_home": str(args.codex_home),
        "no_skills": True,
        "max_routes": MAX_ROUTES,
        "n_cases": len(rows),
        "cases": [],
    }
    for i, row in enumerate(rows, 1):
        print(f"[plain-codex-top10] {i}/{len(rows)} idx={row['idx']:03d}", flush=True)
        result = run_one(
            row,
            out_root=args.out_root,
            case_root=args.case_root,
            codex_home=args.codex_home,
            overwrite=args.overwrite,
            model=args.model,
        )
        manifest["cases"].append(result)
        (args.out_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[plain-codex-top10] done idx={row['idx']:03d} {result}", flush=True)
    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (args.out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = summarize(args.out_root, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
