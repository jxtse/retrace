import importlib.util
import json
from pathlib import Path


def _load_retro190_benchmark_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "dataset"
        / "retro-190_dataset"
        / "run_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("retro190_run_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_one_persists_postprocess_artifacts(monkeypatch, tmp_path):
    bench = _load_retro190_benchmark_module()

    class FakeAgent:
        def run(self, smiles):
            return {
                "success": True,
                "iterations": 1,
                "time": 1.2,
                "route_text": "COMPLETE ROUTE FOUND:\nStep 1: C>>C\nStarting materials: C",
                "best_route": {"label": "fake"},
                "all_routes": [{"label": "fake"}],
                "postprocess": {
                    "status": "accepted",
                    "action": "accept",
                    "review_path": "/tmp/review.json",
                },
                "session_dir": "/tmp/session",
                "transcript_json": "/tmp/session/transcript.json",
                "messages": [{"role": "assistant", "content": "COMPLETE ROUTE FOUND"}],
            }

    monkeypatch.setattr("core.build_agent", lambda config: FakeAgent())
    monkeypatch.setattr("core.config.load_benchmark_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(bench, "MEM_SKIP_GB", 999999)

    result = bench.run_one(
        {
            "idx": 2,
            "label": "r190_002",
            "smiles": "C",
            "ref_step": 1,
            "runs_dir": str(tmp_path / "runs"),
            "tr_dir": str(tmp_path / "transcripts"),
            "log_dir": str(tmp_path / "logs"),
        }
    )

    assert result["success"] is True
    payload = json.loads((tmp_path / "runs" / "002_r190_002.json").read_text())
    assert payload["postprocess"]["status"] == "accepted"
    assert payload["postprocess"]["action"] == "accept"
    assert payload["session_dir"] == "/tmp/session"
    assert payload["transcript_json"] == "/tmp/session/transcript.json"
