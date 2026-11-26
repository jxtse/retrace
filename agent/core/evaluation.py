"""Benchmark evaluation and cluster distribution for the retrosynthesis agent.

The public CLI uses the maintained benchmark drivers for hard_mol, FDA, JMC,
and retro-190, then shards target indices across cluster hosts when requested.
The older serial retro-190 comparator is kept here as an internal fallback for
existing shard scripts that still need ground-truth matching output.
"""

import csv
import json
import logging
import os
import pickle
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

from core.agent_factory import build_agent
from core.config import load_config
from core.llm_client import _load_dotenv_once
from core.route_text import parse_route_from_text

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pillar_score(scored: dict, name: str):
    """Pull one pillar score from a score_route payload.

    score_route emits FLAT pillar keys (``feasibility_score`` / ``cost_score`` /
    ``greenness_score``); older payloads nested them as ``{name: {"score": ..}}``.
    Read flat first, fall back to nested. Returns None if absent/non-numeric.
    """
    flat = scored.get(f"{name}_score")
    if isinstance(flat, (int, float)):
        return flat
    nested = scored.get(name)
    if isinstance(nested, dict) and isinstance(nested.get("score"), (int, float)):
        return nested["score"]
    return None


def _avg_pillar_score(scored_list, name: str):
    """Mean pillar score over a list of score_route payloads, or None."""
    vals = [v for v in (_pillar_score(s, name) for s in scored_list)
            if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else None

# Empirical defaults (May 2026 thorough preset, gpt-5.5/xhigh/200 iter).
PER_WORKER_RAM_GB = 9
PER_WORKER_GPU_MB = 4000
BASE_RAM_GB = 7

DEFAULT_CLUSTER_HOSTS = ()
DEFAULT_HARD_CAPS = {
    "host-c": 5,
    "host-e": 5,
    "host-f": 1,
    "host-g": 1,
}


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    cli_name: str
    driver: str
    cases_csv: str
    aliases: tuple[str, ...] = ()
    default_variants: str | None = None
    idx_field: str = "idx"
    csv_encoding: str = "utf-8"


BENCHMARK_SUITES: dict[str, BenchmarkSuite] = {}


def _register_suite(suite: BenchmarkSuite) -> None:
    names = {suite.name, suite.cli_name, *suite.aliases}
    for name in names:
        BENCHMARK_SUITES[name] = suite


_register_suite(BenchmarkSuite(
    name="hard_mol",
    cli_name="hard-mol",
    driver="dataset/benchmark/hard_mol/run_benchmark.py",
    cases_csv="dataset/benchmark/hard_mol/cases.csv",
    aliases=("hard-mol", "hardmol"),
))
_register_suite(BenchmarkSuite(
    name="fda",
    cli_name="fda",
    driver="dataset/benchmark/FDA_small_mol/run_benchmark.py",
    cases_csv="dataset/benchmark/FDA_small_mol/2025FDA小分子汇总.csv",
    aliases=("FDA", "fda-small-mol", "FDA_small_mol"),
    default_variants="stereo,canonical",
    csv_encoding="gbk",
))
_register_suite(BenchmarkSuite(
    name="jmc",
    cli_name="jmc",
    driver="dataset/benchmark/JMC/run_benchmark.py",
    cases_csv="dataset/benchmark/JMC/cases.csv",
    aliases=("JMC",),
))
_register_suite(BenchmarkSuite(
    name="retro190",
    cli_name="retro190",
    driver="dataset/retro-190_dataset/run_benchmark.py",
    cases_csv="dataset/retro-190_dataset/cases.csv",
    aliases=("retro-190", "r190"),
))


def get_benchmark_suite(name: str) -> BenchmarkSuite:
    try:
        return BENCHMARK_SUITES[name]
    except KeyError as exc:
        choices = ", ".join(sorted({s.cli_name for s in BENCHMARK_SUITES.values()}))
        raise ValueError(f"unknown benchmark suite {name!r}; expected one of: {choices}") from exc


def benchmark_suite_names() -> tuple[str, ...]:
    return tuple(sorted({s.cli_name for s in BENCHMARK_SUITES.values()}))


def parse_indices(spec: str | None) -> list[int]:
    if not spec:
        return []
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    seen: set[int] = set()
    uniq: list[int] = []
    for idx in out:
        if idx not in seen:
            seen.add(idx)
            uniq.append(idx)
    return uniq


def indices_from_start_and_count(start_idx: int, max_samples: int | None) -> list[int]:
    if max_samples is None:
        return []
    return list(range(int(start_idx), int(start_idx) + int(max_samples)))


def default_indices_for_suite(suite_name: str) -> list[int]:
    suite = get_benchmark_suite(suite_name)
    path = REPO_ROOT / suite.cases_csv
    if not path.exists():
        raise FileNotFoundError(f"benchmark cases file not found: {path}")
    out: list[int] = []
    with open(path, encoding=suite.csv_encoding, errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get(suite.idx_field) or "").strip()
            if raw:
                out.append(int(raw))
    return out


def parse_hosts(hosts: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(hosts, str):
        return [h.strip() for h in hosts.split(",") if h.strip()]
    return [str(h).strip() for h in hosts if str(h).strip()]


def parse_worker_caps(spec: str | None) -> dict[str, int]:
    caps = dict(DEFAULT_HARD_CAPS)
    if not spec:
        return caps
    for kv in spec.split(","):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        caps[k.strip()] = int(v.strip())
    return caps


def ssh(host: str, cmd: str, timeout: int = 15) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}", host, cmd],
        capture_output=True,
        text=True,
        timeout=timeout + 5,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ssh {host}: rc={result.returncode} stderr={result.stderr.strip()[:200]}"
        )
    return result.stdout


def probe_host(host: str) -> dict:
    """Return free RAM (MB) and per-GPU free memory (MB) on a remote host."""
    free_ram = ssh(host, "free -m | awk '/^Mem:/ {print $7}'").strip()
    gpu_csv = ssh(
        host,
        "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits",
    ).strip()
    gpus = []
    for line in gpu_csv.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            gpus.append({"idx": int(parts[0]), "free_mb": int(parts[1])})
    return {
        "host": host,
        "free_ram_mb": int(free_ram) if free_ram.isdigit() else 0,
        "gpus": gpus,
    }


def compute_workers(
    probe: dict,
    reserve_ram_gb: int,
    reserve_gpu_mb: int,
    hard_cap: int | None,
) -> int:
    free_ram_mb = max(
        0,
        int(probe["free_ram_mb"]) - reserve_ram_gb * 1024 - BASE_RAM_GB * 1024,
    )
    ram_budget = free_ram_mb // (PER_WORKER_RAM_GB * 1024)

    gpu0 = next((g for g in probe["gpus"] if g["idx"] == 0), None)
    gpu_budget = 0
    if gpu0:
        gpu_free = max(0, int(gpu0["free_mb"]) - reserve_gpu_mb)
        gpu_budget = gpu_free // PER_WORKER_GPU_MB

    n = min(ram_budget, gpu_budget)
    if hard_cap is not None:
        n = min(n, hard_cap)
    return max(0, n)


def shard_indices(
    indices: list[int],
    host_budgets: dict[str, int],
) -> tuple[dict[str, list[int]], list[int]]:
    out: dict[str, list[int]] = {h: [] for h in host_budgets}
    total_budget = sum(host_budgets.values())
    if total_budget == 0:
        return out, list(indices)

    host_seq = list(host_budgets.keys())
    cursor = 0
    for pos, idx in enumerate(indices):
        placed = False
        for _ in range(len(host_seq)):
            host = host_seq[cursor % len(host_seq)]
            cursor += 1
            if len(out[host]) < host_budgets[host]:
                out[host].append(idx)
                placed = True
                break
        if not placed:
            return out, list(indices[pos:])
    return out, []


def default_output_dir(suite_name: str) -> Path:
    suite = get_benchmark_suite(suite_name)
    tag = time.strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / "runs" / "benchmarks" / suite.name / tag


def build_distributed_launch_cmd(
    *,
    host: str,
    suite_name: str,
    indices: list[int],
    output_dir: Path,
    thinking: str,
    config_path: Path | None = None,
    model: str | None = None,
    variants: str | None = None,
    skip_existing: bool = True,
    repo_root: Path = REPO_ROOT,
) -> str:
    suite = get_benchmark_suite(suite_name)
    only = ",".join(str(i) for i in indices)
    workers = max(1, len(indices))
    args = [
        ".venv/bin/python",
        "main.py",
        "evaluate",
        suite.cli_name,
        "--local",
        "--indices",
        only,
        "--workers",
        str(workers),
        "--gpu-ids",
        "0",
        "--output-dir",
        str(output_dir),
        "--thinking",
        thinking,
    ]
    if skip_existing:
        args.append("--skip-existing")
    else:
        args.append("--no-skip-existing")
    if config_path is not None:
        args.extend(["--config", str(config_path)])
    if model:
        args.extend(["--model", model])
    if suite.default_variants is not None and variants:
        args.extend(["--variants", variants])

    run_cmd = " ".join(shlex.quote(x) for x in args)
    safe_host = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in host)
    log_path = f"/tmp/distribute_{safe_host}_$(date +%s).log"
    inner = (
        f"cd {shlex.quote(str(repo_root))} && "
        "if [ -f .env ]; then source .env; fi && "
        "export LLM_API_KEY && "
        f"nohup {run_cmd} > {log_path} 2>&1 < /dev/null &"
    )
    # The remote login shell parses this argument before bash receives the script.
    remote_command = shlex.join(["bash", "-lc", inner])
    return shlex.join([
        "ssh", "-n", "-f", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "--", host, remote_command,
    ])


def run_local_benchmark(
    *,
    suite_name: str,
    indices: list[int] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    thinking: str = "fast",
    config_path: str | os.PathLike[str] | None = None,
    model: str | None = None,
    workers: int = 2,
    gpu_ids: str = "",
    skip_existing: bool = True,
    variants: str | None = None,
) -> int:
    """Run one benchmark suite locally via its maintained suite driver."""
    suite = get_benchmark_suite(suite_name)
    cmd = [
        sys.executable,
        str(REPO_ROOT / suite.driver),
        "--workers",
        str(workers),
    ]
    if indices is not None:
        cmd.extend(["--only", ",".join(str(i) for i in indices)])
    if output_dir:
        cmd.extend(["--output-dir", str(output_dir)])
    if gpu_ids:
        cmd.extend(["--gpu-ids", gpu_ids])
    if skip_existing:
        cmd.append("--skip-existing")
    if suite.default_variants is not None and variants:
        cmd.extend(["--variants", variants])

    env = os.environ.copy()
    env["AGENT_THINKING"] = thinking
    if config_path:
        env["AGENT_CONFIG"] = str(config_path)
    if model:
        env["AGENT_MODEL"] = model
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode


def distribute_benchmark(
    *,
    suite_name: str,
    indices: list[int] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    thinking: str = "fast",
    config_path: str | os.PathLike[str] | None = None,
    model: str | None = None,
    hosts: str | list[str] | tuple[str, ...] = DEFAULT_CLUSTER_HOSTS,
    max_workers: str | None = "",
    reserve_ram_gb: int = 10,
    reserve_gpu_mb: int = 4000,
    dry_run: bool = False,
    full_cap: bool = False,
    variants: str | None = None,
    skip_existing: bool = True,
    launch_timeout_s: int = 120,
) -> dict:
    """Shard and optionally launch a benchmark suite across the gu* cluster."""
    suite = get_benchmark_suite(suite_name)
    selected_indices = default_indices_for_suite(suite.name) if indices is None else indices
    host_list = parse_hosts(hosts)
    if not host_list:
        raise ValueError("provide --hosts for distributed evaluation or use --local")
    hard_caps = parse_worker_caps(max_workers)
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else default_output_dir(suite.name)
    _load_dotenv_once()
    selected = config_path or os.getenv("AGENT_CONFIG")
    config = (REPO_ROOT / Path(selected).expanduser()).resolve() if selected else None
    effective_model = model or os.getenv("AGENT_MODEL")
    if config is not None:
        loaded_config = load_config(
            config, thinking=thinking,
            cli_overrides={"llm": {"model": effective_model}} if effective_model else None,
        )
        effective_model = loaded_config["llm_model"]
        if not (effective_model or "").strip():
            raise ValueError("llm.model requires a served model ID")

    print(f"[evaluate] suite={suite.cli_name} indices={len(selected_indices)}", file=sys.stderr)
    print(f"[evaluate] output_dir={out_dir}", file=sys.stderr)

    probes: dict[str, dict | None] = {}
    budgets: dict[str, int] = {}
    if full_cap:
        for host in host_list:
            budgets[host] = hard_caps.get(host, 0)
            probes[host] = None
            print(f"[evaluate] {host}: --full-cap -> {budgets[host]} workers", file=sys.stderr)
    else:
        print(f"[evaluate] probing {len(host_list)} hosts...", file=sys.stderr)
        for host in host_list:
            try:
                probes[host] = probe_host(host)
            except Exception as exc:
                print(f"[evaluate] {host}: probe failed - {exc}", file=sys.stderr)
                probes[host] = None
        for host in host_list:
            probe = probes.get(host)
            if probe is None:
                budgets[host] = 0
                continue
            budgets[host] = compute_workers(
                probe,
                reserve_ram_gb=reserve_ram_gb,
                reserve_gpu_mb=reserve_gpu_mb,
                hard_cap=hard_caps.get(host),
            )
            gpu0_free = next((g["free_mb"] for g in probe["gpus"] if g["idx"] == 0), 0)
            cap_note = f" (cap {hard_caps[host]})" if host in hard_caps else ""
            print(
                f"[evaluate] {host}: free_ram={probe['free_ram_mb']}MB "
                f"gpu0_free={gpu0_free}MB -> {budgets[host]} workers{cap_note}",
                file=sys.stderr,
            )

    total_workers = sum(budgets.values())
    if total_workers == 0:
        raise RuntimeError("no host has evaluation capacity")

    plan, unassigned = shard_indices(selected_indices, budgets)
    commands = []
    assignments = {}
    for host, idxs in plan.items():
        if not idxs:
            continue
        cmd = build_distributed_launch_cmd(
            host=host,
            suite_name=suite.name,
            indices=idxs,
            output_dir=out_dir,
            thinking=thinking,
            config_path=config,
            model=effective_model,
            variants=variants or suite.default_variants,
            skip_existing=skip_existing,
        )
        assignments[host] = {
            "indices": idxs,
            "workers": len(idxs),
            "thinking": thinking,
            "config_override": str(config) if config else "",
            "model": effective_model or "",
            "variants": variants or suite.default_variants or "",
        }
        commands.append({"host": host, "cmd": cmd})

    summary = {
        "suite": suite.cli_name,
        "output_dir": str(out_dir),
        "thinking": thinking,
        "config_override": str(config) if config else "",
        "model": effective_model or "",
        "total_indices": len(selected_indices),
        "total_workers": total_workers,
        "unassigned": unassigned,
        "budgets": budgets,
        "assignments": assignments,
        "commands": commands,
    }

    print("[evaluate] plan:", file=sys.stderr)
    for entry in commands:
        host = entry["host"]
        idxs = assignments[host]["indices"]
        print(f"  {host}: {len(idxs)} workers on indices {idxs}", file=sys.stderr)
        print(f"    $ {entry['cmd']}", file=sys.stderr)
    if unassigned:
        print(f"[evaluate] WARNING: unassigned indices: {unassigned}", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    plan_path = out_dir / "distribute_plan.json"
    plan_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[evaluate] wrote plan -> {plan_path}", file=sys.stderr)

    if dry_run:
        return summary

    for entry in commands:
        host = entry["host"]
        print(f"[evaluate] launching {host}", file=sys.stderr)
        result = subprocess.run(
            shlex.split(entry["cmd"]),
            capture_output=True,
            text=True,
            timeout=launch_timeout_s,
        )
        print(
            f"[evaluate] {host}: rc={result.returncode} "
            f"out={result.stdout.strip()[:200]} err={result.stderr.strip()[:200]}",
            file=sys.stderr,
        )
    return summary


def canonicalize(smiles: str) -> str:
    """Canonicalize a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol)


def load_retro190(pkl_path: str) -> list[list[str]]:
    """Load retro-190 dataset. Returns list of routes, each is a list of rxn SMILES."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def extract_target_from_route(route: list[str]) -> str:
    """Extract target molecule from a route (product of the first reaction)."""
    first_rxn = route[0]
    product = first_rxn.split(">>")[0]
    return product


def extract_route_reactions(route: list[str]) -> list[dict]:
    """Parse ground truth route into structured format."""
    steps = []
    for rxn in route:
        parts = rxn.split(">>")
        if len(parts) == 2:
            steps.append({
                "product": canonicalize(parts[0]),
                "reactants": ".".join(sorted([canonicalize(r) for r in parts[1].split(".")])),
            })
    return steps


def compare_routes(predicted_steps: list[dict], gt_steps: list[dict]) -> dict:
    """
    Compare predicted route with ground truth.
    Returns match statistics.
    """
    if not predicted_steps:
        return {"match": False, "matched_steps": 0, "total_gt_steps": len(gt_steps), "reason": "no prediction"}

    # Build sets of canonical reactions for comparison
    gt_rxns = set()
    for step in gt_steps:
        rxn_key = f"{step['product']}>>{step['reactants']}"
        gt_rxns.add(rxn_key)

    pred_rxns = set()
    for step in predicted_steps:
        prod = canonicalize(step.get("product", ""))
        reacts = ".".join(sorted([canonicalize(r.strip()) for r in step.get("reactants", "").split(".")]))
        rxn_key = f"{prod}>>{reacts}"
        pred_rxns.add(rxn_key)

    matched = gt_rxns & pred_rxns

    # Also check partial matches (just the product decomposition target matches)
    gt_products = {step['product'] for step in gt_steps}
    pred_products = {canonicalize(step.get("product", "")) for step in predicted_steps}
    product_matches = gt_products & pred_products

    return {
        "match": len(matched) == len(gt_rxns),
        "matched_steps": len(matched),
        "total_gt_steps": len(gt_rxns),
        "total_pred_steps": len(pred_rxns),
        "product_overlap": len(product_matches),
        "total_gt_products": len(gt_products),
    }


def run_evaluation(
    dataset_path: str,
    config: dict | None = None,
    max_samples: int | None = None,
    output_path: str = "eval_results.json",
    start_idx: int = 0,
    per_target_timeout: int | None = None,
):
    """Run evaluation on retro-190 dataset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger = logging.getLogger("evaluate")

    # Load dataset
    routes = load_retro190(dataset_path)
    logger.info("Loaded %d routes from %s", len(routes), dataset_path)

    if max_samples:
        routes = routes[start_idx:start_idx + max_samples]
    else:
        routes = routes[start_idx:]

    # Build agent
    agent = build_agent(config)

    # Grab the score_route tool from the registry so the evaluator can compute
    # `total_score` on predicted routes (the gate the agent uses internally).
    scorer = None
    try:
        scorer = agent.tools.get("score_route")
        if scorer is not None:
            logger.info("Route verifier loaded: will record total_score on predictions.")
    except Exception as e:
        logger.warning("score_route tool unavailable: %s", e)

    # Resume support: if output_path already exists, skip already-done indices.
    already_done = set()
    if os.path.isfile(output_path):
        try:
            with open(output_path) as f_prev:
                prev = json.load(f_prev)
            for entry in prev.get("results", []):
                if entry.get("index") is not None:
                    already_done.add(entry["index"])
            logger.info("Resume mode: %d previously completed routes will be skipped.", len(already_done))
        except Exception as e:
            logger.warning("Could not load previous results (%s); running fresh.", e)

    results = []
    total_success = 0
    total_matched = 0

    def _checkpoint(results_so_far, total_so_far, success_so_far, matched_so_far):
        # Merge with any previous results on disk to keep full history.
        out = {"summary": {
            "total": total_so_far,
            "success": success_so_far,
            "exact_match": matched_so_far,
        }, "results": results_so_far}
        if already_done:
            try:
                with open(output_path) as f_prev:
                    prev = json.load(f_prev)
                prev_map = {r.get("index"): r for r in prev.get("results", [])
                             if r.get("index") is not None}
                # Merge: new entries overwrite same index
                cur_map = {r.get("index"): r for r in results_so_far
                            if r.get("index") is not None}
                merged = {**prev_map, **cur_map}
                out["results"] = sorted(merged.values(), key=lambda r: r.get("index", 0))
                out["summary"]["total"] = len(out["results"])
                out["summary"]["success"] = sum(1 for r in out["results"] if r.get("success"))
                out["summary"]["exact_match"] = sum(
                    1 for r in out["results"]
                    if r.get("comparison", {}).get("match"))
            except Exception:
                pass
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)

    for i, gt_route in enumerate(routes):
        idx = start_idx + i
        if idx in already_done:
            logger.info("Skip route %d (already done).", idx)
            continue
        target = extract_target_from_route(gt_route)
        gt_steps = extract_route_reactions(gt_route)

        logger.info("\n%s", "=" * 60)
        logger.info("Route %d/%d | Target: %s", idx + 1, len(routes) + start_idx, target[:80])
        logger.info("Ground truth: %d steps", len(gt_steps))

        try:
            if per_target_timeout:
                import signal as _sig
                def _timeout_handler(signum, frame):
                    raise TimeoutError("per-target timeout")
                _prev = _sig.signal(_sig.SIGALRM, _timeout_handler)
                _sig.alarm(int(per_target_timeout))
                try:
                    result = agent.run(target)
                except TimeoutError:
                    logger.warning("Target %d timed out after %ds", idx, per_target_timeout)
                    result = {"success": False, "iterations": 0, "time": per_target_timeout, "route_text": "", "timed_out": True}
                except Exception as _e:
                    logger.exception("Target %d crashed: %s", idx, _e)
                    result = {"success": False, "iterations": 0, "time": 0.0, "route_text": "", "error": str(_e)}
                finally:
                    _sig.alarm(0)
                    _sig.signal(_sig.SIGALRM, _prev)
            else:
                try:
                    result = agent.run(target)
                except Exception as _e:
                    logger.exception("Target %d crashed: %s", idx, _e)
                    result = {"success": False, "iterations": 0, "time": 0.0, "route_text": "", "error": str(_e)}
            raw_success = result["success"]
            pred_steps = parse_route_from_text(result.get("route_text", "")) if raw_success else []
            # Real success requires both agent-claimed success AND parseable non-empty route.
            success = bool(raw_success and pred_steps)

            comparison = compare_routes(pred_steps, gt_steps)

            # Score predicted route with the production verifier (total_score).
            scored = None
            if success and pred_steps and scorer is not None:
                try:
                    steps_for_scorer = [
                        {
                            "product": canonicalize(step.get("product", "")),
                            "reactants": [canonicalize(r.strip())
                                          for r in step.get("reactants", "").split(".")
                                          if r.strip()],
                        }
                        for step in pred_steps
                    ]
                    raw = scorer.execute(
                        smiles=target,
                        route_steps=json.dumps(steps_for_scorer),
                    )
                    scored = json.loads(raw)
                except Exception as e:
                    logger.warning("score_route failed on route %d: %s", idx, e)
                    scored = {"error": str(e)}

            entry = {
                "index": idx,
                "target": target,
                "gt_steps": len(gt_steps),
                "success": success,
                "raw_success": raw_success,
                "pred_steps": len(pred_steps),
                "iterations": result["iterations"],
                "time": round(result["time"], 1),
                "comparison": comparison,
                "scored": scored,  # verifier total_score + sub-scores
            }
            results.append(entry)

            if success:
                total_success += 1
            if comparison.get("match"):
                total_matched += 1

            ts = (scored or {}).get("total_score") if isinstance(scored, dict) else None
            logger.info(
                "Result: success=%s, pred_steps=%d, matched=%d/%d, total_score=%s, time=%.1fs",
                success, len(pred_steps), comparison["matched_steps"],
                comparison["total_gt_steps"],
                f"{ts:.3f}" if isinstance(ts, (int, float)) else "-",
                result["time"],
            )
            # Checkpoint every route so a crash doesn't lose progress.
            _checkpoint(results, len(results), total_success, total_matched)

        except Exception as e:
            logger.exception("Error on route %d", idx)
            results.append({
                "index": idx,
                "target": target,
                "gt_steps": len(gt_steps),
                "success": False,
                "error": str(e),
            })

    # Summary
    total = len(results)
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("Total routes: %d", total)
    logger.info("Agent found route: %d (%.1f%%)", total_success, 100 * total_success / max(total, 1))
    logger.info("Exact match with GT: %d (%.1f%%)", total_matched, 100 * total_matched / max(total, 1))

    # Aggregate score stats (over successful runs that got scored)
    scored_list = [r["scored"] for r in results
                   if r.get("scored") and isinstance(r.get("scored"), dict)
                   and "total_score" in r["scored"]]
    def _avg(k):
        vals = [s.get(k) for s in scored_list if isinstance(s.get(k), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None
    avg_total = _avg("total_score")
    avg_feas = _avg_pillar_score(scored_list, "feasibility")
    avg_cost = _avg_pillar_score(scored_list, "cost")
    avg_green = _avg_pillar_score(scored_list, "greenness")
    logger.info("Avg total_score: %s  (n=%d)", avg_total, len(scored_list))
    logger.info("Avg feasibility / cost / greenness: %s / %s / %s",
                avg_feas, avg_cost, avg_green)

    # Merge with previous results on disk (resume mode)
    final_results = results
    if already_done:
        try:
            with open(output_path) as f_prev:
                prev = json.load(f_prev)
            prev_map = {r.get("index"): r for r in prev.get("results", [])
                         if r.get("index") is not None}
            cur_map = {r.get("index"): r for r in results if r.get("index") is not None}
            merged = {**prev_map, **cur_map}
            final_results = sorted(merged.values(), key=lambda r: r.get("index", 0))
        except Exception:
            pass
    total_final = len(final_results)
    success_final = sum(1 for r in final_results if r.get("success"))
    matched_final = sum(1 for r in final_results if r.get("comparison", {}).get("match"))

    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "total": total_final,
                "success": success_final,
                "exact_match": matched_final,
                "success_rate": round(success_final / max(total_final, 1), 4),
                "match_rate": round(matched_final / max(total_final, 1), 4),
                "avg_total_score": avg_total,
                "avg_feasibility_score": avg_feas,
                "avg_cost_score": avg_cost,
                "avg_greenness_score": avg_green,
                "scored_n": len(scored_list),
            },
            "results": final_results,
        }, f, indent=2)
    logger.info("Results saved to %s", output_path)
