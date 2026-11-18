"""Regression: a per-case memory skip (os._exit in a worker) must NOT stop the
parallel benchmark run.

The retro-190 runner's memory watchdog ends an over-budget case with
os._exit(99). The original parallel path used ProcessPoolExecutor, which treats
that abrupt exit as a dead worker and marks the whole pool BrokenProcessPool —
failing every remaining case (empirically: one mem-skip zeroed the rest of the
run). The fix runs each case in its own spawned Process with hand-bounded
concurrency, so an os._exit kills only that child and the driver schedules the
rest.

This test drives the REAL `_run_jobs_parallel` scheduler with an injected,
module-level (picklable) child target that writes a marker file per case and
os._exit(99)s on one chosen idx — proving the others still complete despite the
abrupt exit, which is exactly what ProcessPoolExecutor failed to do.
"""
import importlib.util
import os
from pathlib import Path


def _load_runner():
    path = (Path(__file__).resolve().parents[1]
            / "dataset" / "retro-190_dataset" / "run_benchmark.py")
    spec = importlib.util.spec_from_file_location("retro190_runner_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Module-level so a spawn child can re-import it by qualified name (a local
# closure would fail to pickle). Signature matches the real `_run_one_pinned`.
def _case_double(job: dict, gpu_id):
    if job.get("_explode"):
        # Mimic the mem watchdog: hard-exit before writing any success marker.
        os._exit(99)
    marker_dir = Path(job["_marker_dir"])
    (marker_dir / f"{job['idx']:03d}.done").write_text("ok", encoding="utf-8")


def test_parallel_survives_worker_os_exit(tmp_path):
    module = _load_runner()

    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()

    jobs = []
    for i in range(6):
        jobs.append({
            "idx": i,
            "label": f"case{i}",
            "_marker_dir": str(marker_dir),
            "_explode": (i == 2),  # one case mem-skips mid-run
        })

    done = module._run_jobs_parallel(
        jobs, workers=2, gpu_ids=None, child_target=_case_double
    )

    # All 6 jobs were reaped (none stranded by a broken pool).
    assert done == 6
    # Every non-exploding case wrote its marker; the exploding one did not —
    # and crucially, cases scheduled AFTER the explosion (003-005) still ran.
    produced = sorted(p.stem for p in marker_dir.glob("*.done"))
    assert produced == ["000", "001", "003", "004", "005"]
    assert not (marker_dir / "002.done").exists()
