import json
import pytest

from core import llm_client
from tests.test_lean_profile import RecordingLLM, TARGET, make_agent, tool_script


def test_session_binding_and_logical_identity(make_agent, tmp_path):
    agent = make_agent({"max_iterations": 1, "target_prefetch": "off", "iteration_mode": "plain"})
    client = RecordingLLM(tool_script(2))
    client.wire_log_path = None
    client._wire_seq = 99
    agent.llm = client
    seen = []
    original = client.chat_with_tools

    def chat(messages, tools=None):
        seen.append((client.wire_log_path, client._wire_seq, client.wire_phase, client.wire_request_index))
        client._wire_seq += 1
        return original(messages, tools)

    client.chat_with_tools = chat
    for index in range(2):
        directory = tmp_path / f"session-{index}"
        directory.mkdir()
        agent.result_store.session_dir = str(directory)
        agent.run(TARGET)
        expected = str(directory / "conversation" / "wire_log.jsonl")
        assert seen[-1] == (expected, 0, "main", 1)
        manifest = json.loads((directory / "session_manifest.json").read_text())
        assert manifest["paths"]["wire_log"] == expected
        requests = [json.loads(line) for line in (directory / "conversation" / "llm_requests.jsonl").read_text().splitlines()]
        assert requests[0]["request_index"] == seen[-1][3]


def test_agent_does_not_retry_capture_integrity_failure(make_agent):
    agent = make_agent({"max_iterations": 2, "target_prefetch": "off"})
    client = RecordingLLM([])
    client.wire_log_path = None
    calls = []

    def chat(messages, tools=None):
        calls.append(messages)
        raise llm_client.CaptureIntegrityError("capture failed")

    client.chat_with_tools = chat
    agent.llm = client
    with pytest.raises(llm_client.CaptureIntegrityError):
        agent.run(TARGET)
    assert len(calls) == 1
