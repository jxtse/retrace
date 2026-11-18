from __future__ import annotations

import json

from services.transcript import TranscriptRecorder


class FakeRoute:
    label = "route-a"
    total_score = 0.81
    sub_scores = {"feasibility": 0.8}
    steps = [{"rxn": "A>>B"}]
    starting_materials = ["A"]
    notes = "short note"


class FakeScratchpad:
    def list(self):
        return [FakeRoute()]

    def lessons(self):
        return []


class FakeTodo:
    def read(self):
        return []


def test_transcript_training_trace_records_tool_decisions(tmp_path):
    recorder = TranscriptRecorder(
        session_dir=str(tmp_path),
        target_smiles="CCO",
        scratchpad_store=FakeScratchpad(),
        todo_store=FakeTodo(),
    )
    recorder.update_metadata(
        config={"thinking": "fast"},
        model_settings={"main": {"model": "fake-main"}, "reviewer": {"model": "fake-reviewer"}},
        prompt_artifacts={"core": {"path": "prompts/core.md"}},
        postprocess={"status": "accepted"},
    )
    recorder.write_session_manifest(tool_names=["score_route"])
    recorder.write_latest_messages([{"role": "user", "content": "hi"}], stage="test", iteration=0)
    recorder.record_llm_request(
        phase="main",
        messages=[{"role": "user", "content": "hi"}],
        model_settings={"model": "fake-main"},
    )

    recorder("iteration", n=1)
    recorder(
        "tool_call",
        tool_call_id="call-1",
        name="score_route",
        args='{"route_label":"route-a"}',
    )
    recorder(
        "tool_result",
        tool_call_id="call-1",
        tool_name="score_route",
        elapsed_ms=12,
        content='{"total_score":0.64,"sub_scores":{"feasibility":0.6}}',
    )
    recorder(
        "native_bailout",
        iteration=1,
        n_saved=0,
        error="native LLM call failed twice: 502",
    )
    payload = recorder.finalize(
        "CCO",
        {"success": False, "below_target": True, "iterations": 1},
    )

    records = payload["training_trace"]["records"]
    assert [r["action"] for r in records] == [
        "iteration_start",
        "call_score_route",
        "score_route_result",
        "native_bailout",
    ]
    assert records[1]["state"]["scratchpad"]["top_routes"][0]["label"] == "route-a"
    assert records[2]["payload"]["result_summary"]["total_score"] == 0.64
    assert records[3]["kind"] == "stop_or_gate"
    assert records[3]["payload"]["error"] == "native LLM call failed twice: 502"
    assert payload["config"]["thinking"] == "fast"
    assert payload["model_settings"]["reviewer"]["model"] == "fake-reviewer"
    assert payload["prompt_artifacts"]["core"]["path"] == "prompts/core.md"
    assert payload["postprocess"]["status"] == "accepted"

    manifest_path = tmp_path / "session_manifest.json"
    latest_path = tmp_path / "conversation" / "latest_messages.json"
    requests_path = tmp_path / "conversation" / "llm_requests.jsonl"
    assert manifest_path.exists()
    assert latest_path.exists()
    assert requests_path.exists()
    assert json.loads(latest_path.read_text())["messages"][0]["content"] == "hi"
    assert json.loads(requests_path.read_text().splitlines()[0])["phase"] == "main"
