#!/usr/bin/env python
"""Unified entry point for the retrosynthesis agent."""

from __future__ import annotations

import argparse
import json
import logging
import textwrap

from core.deployment import validate_deployment
from core.llm_client import _load_dotenv_once
from core import build_agent
from core.config import available_thinking_levels, load_config


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    WHITE = "\033[37m"
    BG_BLUE = "\033[44m"
    BG_GREEN = "\033[42m"

    @classmethod
    def disable(cls):
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith("_"):
                setattr(cls, attr, "")


def _wrap(text: str, width: int = 100, indent: str = "  ") -> str:
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        if len(line) <= width:
            wrapped.append(indent + line)
        else:
            wrapped.extend(
                textwrap.wrap(
                    line,
                    width=width,
                    initial_indent=indent,
                    subsequent_indent=indent,
                )
            )
    return "\n".join(wrapped)


def _truncate_json(text: str, max_len: int = 500) -> str:
    try:
        obj = json.loads(text)
        pretty = json.dumps(obj, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        pretty = text
    if len(pretty) > max_len:
        return pretty[:max_len] + f"\n  ... ({len(pretty) - max_len} chars truncated)"
    return pretty


def trace_callback(event: str, **kwargs):
    if event == "iteration":
        print(f"\n{C.BG_BLUE}{C.WHITE}{C.BOLD} Step {kwargs['n']} {C.RESET}")
        print(f"{C.DIM}{'-' * 78}{C.RESET}")
    elif event == "thinking":
        print(f"{C.CYAN}{C.BOLD}Thinking:{C.RESET}")
        print(_wrap(kwargs["content"], width=100))
    elif event == "tool_call":
        args = kwargs["args"]
        if isinstance(args, str):
            try:
                args = json.dumps(json.loads(args), ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                pass
        else:
            args = json.dumps(args, ensure_ascii=False)
        print(f"{C.YELLOW}{C.BOLD}Tool call:{C.RESET} {kwargs['name']}({C.DIM}{args}{C.RESET})")
    elif event == "tool_result":
        content = kwargs["content"]
        color = C.RED if '"error"' in content else C.GREEN
        print(f"{color}Result:{C.RESET}")
        print(_wrap(_truncate_json(content, max_len=600), indent=f"  {C.DIM}"))
        print(C.RESET, end="")
    elif event == "complete":
        print(f"\n{C.BG_GREEN}{C.WHITE}{C.BOLD} ROUTE FOUND {C.RESET}")
        content = kwargs["content"]
        idx = content.find("COMPLETE ROUTE FOUND")
        print(f"{C.GREEN}{content[idx:] if idx >= 0 else content[-1000:]}{C.RESET}")
    elif event == "error":
        print(f"{C.RED}{C.BOLD}Error: {kwargs['message']}{C.RESET}")


def _config_from_args(args) -> dict:
    _load_dotenv_once()
    overrides: dict = {}
    if getattr(args, "model", None):
        overrides.setdefault("llm", {})["model"] = args.model
    try:
        config = load_config(
            getattr(args, "config", None),
            thinking=getattr(args, "thinking", "fast"),
            cli_overrides=overrides or None,
        )
        validate_deployment(config)
    except (ValueError, OSError) as exc:
        import sys
        print(exc, file=sys.stderr)
        raise SystemExit(2) from None
    return config


def _benchmark_indices_from_args(args) -> list[int] | None:
    from core.evaluation import default_indices_for_suite, parse_indices

    if getattr(args, "indices", None):
        if getattr(args, "max_samples", None) is not None or getattr(args, "start_idx", 0) != 0:
            raise SystemExit("--indices cannot be combined with --start-idx/--max-samples")
        return parse_indices(args.indices)

    start_idx = getattr(args, "start_idx", 0)
    max_samples = getattr(args, "max_samples", None)
    if max_samples is None and start_idx == 0:
        return None
    if start_idx < 0:
        raise SystemExit("--start-idx must be >= 0")
    if max_samples is not None and max_samples < 0:
        raise SystemExit("--max-samples must be >= 0")

    all_indices = default_indices_for_suite(args.suite)
    stop_idx = None if max_samples is None else start_idx + max_samples
    selected = all_indices[start_idx:stop_idx]
    if not selected:
        # An empty slice (e.g. --max-samples 0, or --start-idx past the end)
        # must NOT silently fall through to the driver, which treats an empty
        # --only filter as "run every case". Fail loudly instead.
        raise SystemExit(
            f"--start-idx={start_idx} --max-samples={max_samples} selects no "
            f"cases from suite '{args.suite}' ({len(all_indices)} available)"
        )
    return selected


def _cmd_check_config(args) -> int:
    _config_from_args(args)
    print("Deployment configuration valid (no services contacted).")
    return 0


def _cmd_run(args) -> int:
    config = _config_from_args(args)
    if args.trace:
        print(f"{C.BOLD}Loading agent ({config['thinking']})...{C.RESET}", end="", flush=True)
    agent = build_agent(config)
    if args.trace:
        agent._user_trace = trace_callback
        agent.trace = trace_callback
        print(f" {C.GREEN}done{C.RESET}")
        print(f"{C.BOLD}Target:{C.RESET} {C.MAGENTA}{args.smiles}{C.RESET}")
        print(f"{C.DIM}Tools: {', '.join(agent.tools.tool_names)}{C.RESET}")
        print(f"{C.DIM}Max iterations: {agent.max_iterations} | Max depth: {agent.max_depth}{C.RESET}")

    result = agent.run(args.smiles)
    print(f"\n{'=' * 78}")
    if result["success"]:
        print(
            f"{C.GREEN}{C.BOLD}Success{C.RESET} | "
            f"Iterations: {result['iterations']} | Time: {result['time']:.1f}s"
        )
    else:
        print(
            f"{C.RED}{C.BOLD}Failed{C.RESET} | "
            f"Iterations: {result['iterations']} | Time: {result['time']:.1f}s"
        )
    print(f"{'=' * 78}")

    if result["success"] and not args.trace:
        text = result["route_text"]
        idx = text.find("COMPLETE ROUTE FOUND")
        print(text[idx:] if idx >= 0 else text[-1000:])
    elif not result["success"]:
        print("Route not found within iteration limit.")
        for msg in reversed(result.get("messages", [])):
            if msg.get("role") == "assistant" and msg.get("content"):
                print(f"\n{C.DIM}Last assistant message:{C.RESET}")
                print(msg["content"][-500:])
                break
    return 0 if result["success"] else 1


def _cmd_evaluate_benchmark(args) -> int:
    from core.evaluation import distribute_benchmark, run_evaluation, run_local_benchmark

    if getattr(args, "serial_retro190", False):
        config = _config_from_args(args)
        run_evaluation(
            dataset_path=args.dataset,
            config=config,
            max_samples=args.max_samples,
            output_path=args.output or "eval_results.json",
            start_idx=args.start_idx,
            per_target_timeout=args.per_target_timeout,
        )
        return 0

    if getattr(args, "output", None):
        raise SystemExit("--output is only valid with --serial-retro190; use --output-dir")

    indices = _benchmark_indices_from_args(args)
    if args.local:
        return run_local_benchmark(
            suite_name=args.suite,
            indices=indices,
            output_dir=args.output_dir,
            thinking=args.thinking,
            config_path=args.config,
            model=args.model,
            workers=args.workers,
            gpu_ids=args.gpu_ids,
            skip_existing=args.skip_existing,
            variants=args.variants,
        )

    distribute_benchmark(
        suite_name=args.suite,
        indices=indices,
        output_dir=args.output_dir,
        thinking=args.thinking,
        config_path=args.config,
        model=args.model,
        hosts=args.hosts,
        max_workers=args.max_workers,
        reserve_ram_gb=args.reserve_ram_gb,
        reserve_gpu_mb=args.reserve_gpu_mb,
        dry_run=args.dry_run,
        full_cap=args.full_cap,
        variants=args.variants,
        skip_existing=args.skip_existing,
        launch_timeout_s=args.launch_timeout_s,
    )
    return 0


def _cmd_repl(args) -> int:
    from core.repl import RetroSynthesisREPL

    config = _config_from_args(args)
    repl = RetroSynthesisREPL(config)
    repl.run()
    return 0


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--thinking",
        choices=available_thinking_levels(),
        default="fast",
        help="Built-in reasoning/budget preset.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML override file layered on top of the built-in preset.",
    )
    parser.add_argument("--model", help="Override llm.model for this run.")


def _add_evaluate_args(parser: argparse.ArgumentParser, *, serial_retro190: bool = False) -> None:
    parser.add_argument(
        "--indices",
        default="",
        help="Case indices to run, e.g. 0-14 or 0,1,2,5-7. Defaults to all cases.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Run this many cases from the suite case list.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start offset into the suite case list when used with --max-samples.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Parent directory for benchmark runs/transcripts/logs.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run this suite on the current host via the maintained benchmark driver.",
    )
    parser.add_argument("--workers", type=int, default=2, help="Local worker count.")
    parser.add_argument("--gpu-ids", default="", help="Local GPU ids for worker round-robin.")
    parser.add_argument("--variants", default=None, help="FDA only: comma list such as stereo,canonical.")

    skip_group = parser.add_mutually_exclusive_group()
    skip_group.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    skip_group.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")

    parser.add_argument("--hosts", default="", help="Cluster hosts for distributed runs.")
    parser.add_argument("--max-workers", default="", help="Host caps, e.g. host-c=5,host-e=5.")
    parser.add_argument("--reserve-ram-gb", type=int, default=10)
    parser.add_argument("--reserve-gpu-mb", type=int, default=4000)
    parser.add_argument("--full-cap", action="store_true", help="Use configured host caps without probing.")
    parser.add_argument("--dry-run", action="store_true", help="Write and print the distribution plan only.")
    parser.add_argument("--launch-timeout-s", type=int, default=120)
    _add_config_args(parser)

    if serial_retro190:
        parser.add_argument("--serial-retro190", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument(
            "--dataset",
            default="dataset/retro-190_dataset/routes_possible_test_hard.pkl",
            help=argparse.SUPPRESS,
        )
        parser.add_argument("--output", default=None, help=argparse.SUPPRESS)
        parser.add_argument("--per-target-timeout", type=int, default=600, help=argparse.SUPPRESS)

    parser.add_argument("-v", "--verbose", action="store_true")
    parser.set_defaults(func=_cmd_evaluate_benchmark)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chemoenzymatic Retrosynthesis Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check-config", help="Validate deployment settings offline.")
    _add_config_args(check)
    check.set_defaults(func=_cmd_check_config)

    run_p = sub.add_parser("run", help="Run the agent on one target SMILES.")
    run_p.add_argument("smiles", help="Target molecule SMILES.")
    run_p.add_argument("--trace", action="store_true", help="Print step-by-step trace.")
    run_p.add_argument("-v", "--verbose", action="store_true")
    _add_config_args(run_p)
    run_p.set_defaults(func=_cmd_run)

    eval_p = sub.add_parser("evaluate", help="Run benchmark evaluation.")
    eval_sub = eval_p.add_subparsers(dest="suite", required=True)
    hard_p = eval_sub.add_parser(
        "hard-mol",
        aliases=["hard_mol", "hardmol"],
        help="Distribute or locally run the hard-molecule benchmark.",
    )
    _add_evaluate_args(hard_p)

    fda_p = eval_sub.add_parser(
        "fda",
        aliases=["FDA", "fda-small-mol", "FDA_small_mol"],
        help="Distribute or locally run the FDA small-molecule benchmark.",
    )
    _add_evaluate_args(fda_p)

    jmc_p = eval_sub.add_parser(
        "jmc",
        aliases=["JMC"],
        help="Distribute or locally run the JMC benchmark.",
    )
    _add_evaluate_args(jmc_p)

    retro_p = eval_sub.add_parser(
        "retro190",
        aliases=["retro-190", "r190"],
        help="Distribute or locally run the retro-190 hard set.",
    )
    _add_evaluate_args(retro_p, serial_retro190=True)

    repl_p = sub.add_parser("repl", help="Start the interactive REPL.")
    repl_p.add_argument("-v", "--verbose", action="store_true")
    _add_config_args(repl_p)
    repl_p.set_defaults(func=_cmd_repl)

    return parser


def main() -> int:
    import sys

    if not sys.stdout.isatty():
        C.disable()
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "trace", False):
        level = logging.WARNING
    else:
        level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
