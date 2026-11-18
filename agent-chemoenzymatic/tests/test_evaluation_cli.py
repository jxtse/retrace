from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import main as cli
from core.evaluation import (
    benchmark_suite_names,
    build_distributed_launch_cmd,
    parse_indices,
    shard_indices,
)


def test_benchmark_suite_registry_exposes_all_public_suites():
    assert benchmark_suite_names() == ("fda", "hard-mol", "jmc", "retro190")


def test_parse_indices_accepts_ranges_and_deduplicates():
    assert parse_indices("0-2,2,5,7-8") == [0, 1, 2, 5, 7, 8]


def test_shard_indices_assigns_at_most_one_host_per_case():
    plan, unassigned = shard_indices([0, 1, 2, 3, 4], {"host-c": 2, "host-e": 1})

    assert plan == {"host-c": [0, 2], "host-e": [1]}
    assert unassigned == [3, 4]


def test_distributed_launch_command_uses_main_local_entrypoint():
    cmd = build_distributed_launch_cmd(
        host="host-c",
        suite_name="hard-mol",
        indices=[0, 2],
        output_dir=Path("/tmp/eval-hard"),
        thinking="thorough",
        config_path=Path("/tmp/overlay.yaml"),
        model="gpt-test",
        skip_existing=False,
    )

    assert "main.py evaluate hard-mol --local --indices 0,2" in cmd
    assert "--workers 2 --gpu-ids 0" in cmd
    assert "--output-dir /tmp/eval-hard --thinking thorough" in cmd
    assert "--no-skip-existing --config /tmp/overlay.yaml --model gpt-test" in cmd
    assert "run_benchmark.py" not in cmd


def test_evaluate_cli_distributes_by_default(tmp_path):
    parser = cli.build_parser()
    args = parser.parse_args([
        "evaluate",
        "hard-mol",
        "--indices",
        "0-1",
        "--output-dir",
        str(tmp_path),
        "--thinking",
        "thorough",
        "--model",
        "gpt-test",
        "--dry-run",
        "--full-cap",
    ])

    with patch("core.evaluation.distribute_benchmark") as distribute:
        assert args.func(args) == 0

    kwargs = distribute.call_args.kwargs
    assert kwargs["suite_name"] == "hard-mol"
    assert kwargs["indices"] == [0, 1]
    assert kwargs["output_dir"] == str(tmp_path)
    assert kwargs["thinking"] == "thorough"
    assert kwargs["model"] == "gpt-test"
    assert kwargs["dry_run"] is True
    assert kwargs["full_cap"] is True


def test_evaluate_cli_local_delegates_to_suite_driver():
    parser = cli.build_parser()
    args = parser.parse_args([
        "evaluate",
        "fda",
        "--local",
        "--indices",
        "0",
        "--variants",
        "canonical",
        "--workers",
        "1",
        "--gpu-ids",
        "0",
    ])

    with patch("core.evaluation.run_local_benchmark", return_value=7) as run_local:
        assert args.func(args) == 7

    kwargs = run_local.call_args.kwargs
    assert kwargs["suite_name"] == "fda"
    assert kwargs["indices"] == [0]
    assert kwargs["variants"] == "canonical"
    assert kwargs["workers"] == 1
    assert kwargs["gpu_ids"] == "0"


def test_evaluate_cli_can_slice_suite_case_list():
    parser = cli.build_parser()
    args = parser.parse_args([
        "evaluate",
        "jmc",
        "--start-idx",
        "1",
        "--max-samples",
        "2",
        "--dry-run",
    ])

    with (
        patch("core.evaluation.default_indices_for_suite", return_value=[10, 11, 12, 13]),
        patch("core.evaluation.distribute_benchmark") as distribute,
    ):
        assert args.func(args) == 0

    assert distribute.call_args.kwargs["indices"] == [11, 12]


def test_hidden_retro190_serial_mode_remains_parseable_for_legacy_shards():
    parser = cli.build_parser()
    args = parser.parse_args([
        "evaluate",
        "retro190",
        "--serial-retro190",
        "--max-samples",
        "1",
        "--output",
        "shard.json",
    ])

    assert args.serial_retro190 is True
    assert args.output == "shard.json"
