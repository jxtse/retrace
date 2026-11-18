"""Offline contracts for the opt-in training profile."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from core import build_agent
from core.config import load_config
from tests.test_agent_e2e_loop import _FakeLLMClient, _make_config, _tc
from tools.base import Tool


TARGET = "CC(=O)OC1CCC(C)CC1"
ALLOWED = [
    "validate_smiles", "analyze_molecule", "check_buyability", "neural_retro",
    "suggest_templates", "explore_branches", "score_route", "save_best_route",
    "list_best_routes",
]


class FakeTool(Tool):
    description = "Offline tool"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, name):
        self.name = name
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps({"tool": self.name, "payload": "data " * 600})


class RecordingLLM(_FakeLLMClient):
    def __init__(self, script):
        super().__init__(script)
        self.requests = []

    def chat_with_tools(self, messages, tools=None):
        self.requests.append(deepcopy({"messages": messages, "tools": tools}))
        response = super().chat_with_tools(messages, tools)
        if isinstance(response, Exception):
            raise response
        return response


def tool_script(count, name="analyze_molecule"):
    return [
        (None, [_tc(f"call_{index}", name, {"smiles": "C" * (index + 1)})], "tool_calls")
        for index in range(count)
    ]


@pytest.fixture
def make_agent(monkeypatch, tmp_path):
    import core.agent_factory as factory
    import tools.builtins as builtins

    original_register = builtins.register_all_builtins

    def register(registry, **kwargs):
        original_register(registry, **kwargs)
        for name in ["suggest_templates", "neural_retro", "score_route", "check_buyability", "analyze_molecule"]:
            registry.register(FakeTool(name))

    monkeypatch.setattr(builtins, "register_all_builtins", register)
    monkeypatch.setattr(factory, "make_llm_client", lambda **kwargs: RecordingLLM([]))
    monkeypatch.delenv("RETRO_AGENT_DISABLE_TARGET_PREFETCH", raising=False)

    def make(options=None, *, compaction=False, keep_recent=4):
        config = _make_config(str(tmp_path))
        config["agent"].update(options or {})
        config["result_store"]["keep_recent_full_per_tool"] = keep_recent
        config["compaction"] = {
            "enabled": compaction, "max_tokens": 1,
            "keep_recent_tail": 2, "keep_head_messages": 2,
        }
        return build_agent(config)

    return make


def test_allowlist_filters_registry_and_llm_schemas(make_agent, caplog):
    with caplog.at_level("INFO"):
        agent = make_agent({"tools_allowlist": ALLOWED})
    assert set(agent.tools.tool_names) == set(ALLOWED)
    assert {tool["function"]["name"] for tool in agent.tools.get_openai_tools()} == set(ALLOWED)
    assert agent.tools.get("score_route") is not None
    assert agent.tools.get("save_best_route") is not None
    assert "Registered tools:" in caplog.text
    assert agent.tools.get("explore_branches")._result_store is agent.result_store
    agent.max_iterations = 1
    agent.llm = RecordingLLM(tool_script(1, "neural_retro"))
    agent.run(TARGET)
    assert {tool["function"]["name"] for tool in agent.llm.requests[0]["tools"]} == set(ALLOWED)


def test_allowlist_unknown_fails_at_build(make_agent):
    with pytest.raises(ValueError, match="typo_tool"):
        make_agent({"tools_allowlist": ["score_route", "typo_tool"]})


def test_empty_allowlist_hides_every_tool(make_agent):
    assert make_agent({"tools_allowlist": []}).tools.get_openai_tools() == []


@pytest.mark.parametrize("mode,roles", [
    (None, ["assistant", "tool", "tool", "user"]),
    ("synthetic_turn", ["assistant", "tool", "tool", "user"]),
    ("user_message", ["user"]),
    ("off", []),
])
def test_prefetch_first_request(make_agent, mode, roles):
    options = {"max_iterations": 1}
    if mode is not None:
        options["target_prefetch"] = mode
    agent = make_agent(options)
    agent.llm = RecordingLLM(tool_script(1, "neural_retro"))
    agent.run(TARGET)
    prefetch = agent.llm.requests[0]["messages"][2:]
    assert [message["role"] for message in prefetch] == roles
    if mode == "user_message":
        lines = prefetch[0]["content"].splitlines()
        assert len(lines) == 2
        for line, name in zip(lines, ["suggest_templates", "neural_retro"]):
            label = f"[harness prefetch] {name}: "
            assert line.startswith(label)
            assert json.loads(line[len(label):])["tool"] == name
    if mode != "off":
        assert {"suggest_templates", "neural_retro"} <= agent._target_tools_called_set
        assert agent.tools.get("suggest_templates").calls == [
            {"product_smiles": TARGET, "top_k": 15},
        ]
    else:
        assert agent.tools.get("suggest_templates").calls == []


@pytest.mark.parametrize("mode,enabled", [(None, False), ("synthetic_turn", True), ("user_message", True), ("off", False)])
def test_prefetch_config_overrides_environment(make_agent, monkeypatch, mode, enabled):
    agent = make_agent({"max_iterations": 1, **({"target_prefetch": mode} if mode else {})})
    monkeypatch.setenv("RETRO_AGENT_DISABLE_TARGET_PREFETCH", "1")
    agent.llm = RecordingLLM(tool_script(1, "neural_retro"))
    agent.run(TARGET)
    assert bool(agent.llm.requests[0]["messages"][2:]) is enabled


def test_defaults_keep_tools_and_prefetch(make_agent):
    implicit = make_agent()
    explicit = make_agent({
        "tools_allowlist": None, "target_prefetch": "synthetic_turn",
        "mini_status_every": 5, "iteration_mode": "billable",
    })
    assert implicit.tools.get_openai_tools() == explicit.tools.get_openai_tools()
    assert "todo_write" in implicit.tools.tool_names
    assert "backtrack_and_summarize" in implicit.tools.tool_names
    messages = []
    implicit._prefetch_disconnections(messages, TARGET)
    assert [message["role"] for message in messages] == ["assistant", "tool", "tool", "user"]
    assert messages[0]["content"] == (
        "Prefetching the curated template library and the multi-engine "
        "neural ensemble for the target so we can compare cross-source "
        "candidates before committing to any disconnection."
    )
    for call, message in zip(messages[0]["tool_calls"], messages[1:3]):
        assert message["tool_call_id"] == call["id"]
        assert json.loads(message["content"])["tool"] == call["function"]["name"]


def test_user_prefetch_preserves_separate_recall_envelope(make_agent):
    agent = make_agent({"target_prefetch": "user_message"})
    agent.precedent_prefetch_enabled = True
    agent.tools.register(FakeTool("recall_similar_routes"))
    messages = []
    agent._prefetch_disconnections(messages, TARGET)
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "user"]
    assert [call["function"]["name"] for call in messages[1]["tool_calls"]] == ["recall_similar_routes"]
    assert messages[2]["name"] == "recall_similar_routes"
    assert json.loads(messages[2]["content"])["suppressed"] is True


def test_default_billable_mode_keeps_refunds(make_agent):
    agent = make_agent({"max_iterations": 1, "target_prefetch": "off"})
    agent.llm = RecordingLLM(tool_script(3) + tool_script(1, "neural_retro"))
    result = agent.run(TARGET)
    assert result["iterations"] == 1
    assert agent.llm.call_count == 4
    assert result["bail_reason"] == "max_iterations"


@pytest.mark.parametrize("every,expected", [(0, []), (5, [5, 10]), (3, [3, 6, 9])])
def test_mini_status_schedule(make_agent, monkeypatch, every, expected):
    agent = make_agent({"max_iterations": 10, "mini_status_every": every, "target_prefetch": "off"})
    monkeypatch.setattr(agent, "_build_mini_status", lambda iteration: f"STATUS:{iteration}")
    agent.llm = RecordingLLM(tool_script(10, "neural_retro"))
    result = agent.run(TARGET)
    assert [message["content"] for message in result["messages"] if message.get("content", "").startswith("STATUS:")] == [f"STATUS:{iteration}" for iteration in expected]


def test_disabled_compaction_preserves_history_and_recorded_requests(make_agent, monkeypatch):
    import core.agent as agent_module

    agent = make_agent({"max_iterations": 7, "target_prefetch": "off", "mini_status_every": 0}, keep_recent=1)
    def forbidden(*args, **kwargs):
        pytest.fail("Compaction ran despite enabled=false")
    monkeypatch.setattr(agent_module, "maybe_compact", forbidden)
    monkeypatch.setattr(agent_module, "compact_to_checkpoint", forbidden)
    agent.llm = RecordingLLM(tool_script(7, "neural_retro"))
    result = agent.run(TARGET)
    history = result["messages"]
    for request in agent.llm.requests:
        assert request["messages"] == history[:len(request["messages"])]
    request_file = Path(result["session_dir"]) / "conversation/llm_requests.jsonl"
    recorded = [json.loads(line) for line in request_file.read_text().splitlines()]
    assert len(recorded) == len(agent.llm.requests)
    for record, request in zip(recorded, agent.llm.requests):
        assert record["messages"] == request["messages"]
        assert record["tools"] == request["tools"]
    assert "data " * 600 in history[3]["content"]


@pytest.mark.parametrize("kind", ["free", "complete", "empty"])
def test_plain_mode_hard_call_budget(make_agent, kind):
    agent = make_agent({"max_iterations": 4, "iteration_mode": "plain", "target_prefetch": "off"})
    scripts = {
        "free": tool_script(100),
        "complete": [("COMPLETE ROUTE FOUND: premature", None, "stop")] * 100,
        "empty": [(None, None, "stop")] * 100,
    }
    agent.llm = RecordingLLM(scripts[kind])
    events = []
    agent._user_trace = lambda event, **kwargs: events.append((event, kwargs))
    result = agent.run(TARGET)
    assert agent.llm.call_count == result["iterations"] == 4
    assert [data["n"] for event, data in events if event == "iteration"] == [1, 2, 3, 4]
    assert not any(event == "iteration_refunded" for event, _ in events)
    assert result["bail_reason"] == "max_iterations"


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_plain_mode_counts_inline_retry(make_agent, monkeypatch, budget):
    import core.agent as agent_module

    monkeypatch.setattr(agent_module.time, "sleep", lambda seconds: None)
    agent = make_agent({"max_iterations": budget, "iteration_mode": "plain", "target_prefetch": "off"})
    agent.llm = RecordingLLM([RuntimeError("transient")] + tool_script(100))
    result = agent.run(TARGET)
    assert agent.llm.call_count == result["iterations"] == budget
    assert result["bail_reason"] == "max_iterations"


@pytest.mark.parametrize("kind", ["empty", "text", "complete", "repeated", "saved"])
def test_lean_disables_other_midrun_user_nudges(make_agent, kind):
    agent = make_agent({
        "max_iterations": 12, "iteration_mode": "plain", "target_prefetch": "user_message",
        "mini_status_every": 0, "user_nudges_enabled": False,
        "auto_finish_after_saved": 1, "stop_after_distinct_routes": 1,
    })
    scripts = {
        "empty": [(None, None, "stop")] * 12,
        "text": [("Still planning", None, "stop")] * 12,
        "complete": [("COMPLETE ROUTE FOUND: premature", None, "stop")] * 12,
        "repeated": tool_script(1) * 12,
        "saved": [(None, [_tc("save", "save_best_route", {
            "target_smiles": TARGET, "steps": [f"{TARGET}>>CCO.CC=O"],
            "total_score": 0.0, "label": "candidate",
        })], "tool_calls")] + tool_script(11),
    }
    agent.llm = RecordingLLM(scripts[kind])
    result = agent.run(TARGET)
    user_msgs = [message for message in result["messages"] if message["role"] == "user"]
    initial_user = agent.llm.requests[0]["messages"][1:]
    if kind in ("empty", "text", "complete"):
        # A text-only / empty assistant turn (including an ignored COMPLETE
        # declaration) MUST still be followed by a user turn so the transcript keeps strict role alternation (two adjacent
        # assistant messages would be an invalid chat). Only advisory nudges
        # are suppressed; the continuation turn is structural and stays.
        extra = user_msgs[len(initial_user):]
        assert user_msgs[:len(initial_user)] == initial_user
        assert extra and all(
            "continue" in m["content"].lower() for m in extra
        ), extra
        # Two adjacent assistant messages would be an invalid chat; adjacent
        # user messages (kickoff + prefetch, or an empty assistant turn that
        # is not appended) are tolerated by the chat template.
        roles = [m["role"] for m in result["messages"] if m["role"] in ("user", "assistant")]
        assert not any(a == b == "assistant" for a, b in zip(roles, roles[1:])), roles
    else:
        assert user_msgs == initial_user


def test_hidden_rank_tool_still_finalises_saved_routes(make_agent):
    agent = make_agent({
        "tools_allowlist": ALLOWED, "max_iterations": 1, "iteration_mode": "plain",
        "target_prefetch": "off",
    })
    agent.llm = RecordingLLM([(None, [_tc("save", "save_best_route", {
        "target_smiles": TARGET, "steps": [f"{TARGET}>>CCO.CC=O"],
        "total_score": 0.0, "label": "candidate",
    })], "tool_calls")])
    result = agent.run(TARGET)
    assert agent.tools.get("rank_routes") is None
    assert result["ranked"] is not None
    assert result["best_route"]["label"] == "candidate"


def test_lean_yaml_loads_and_preserves_student_settings():
    config = load_config("configs/student_27b_vllm_lean.yaml")
    original = yaml.safe_load(Path("configs/student_27b_vllm.yaml").read_text())
    lean = yaml.safe_load(Path("configs/student_27b_vllm_lean.yaml").read_text())
    for block in ["postprocess", "neural_retro", "verifier"]:
        assert lean[block] == original[block]
    # llm block: sampling knobs identical. Two keys are expected to differ:
    # context_window (the lean profile runs against the 128k server) and
    # fixed_output_budget (the lean profile pins max_tokens so every attempt sends
    # the identical sampling request; the base rollout config leaves it off).
    differing = {"context_window", "fixed_output_budget"}
    assert {k: v for k, v in lean["llm"].items() if k not in differing} == \
        {k: v for k, v in original["llm"].items() if k not in differing}
    assert lean["llm"]["fixed_output_budget"] is True
    assert original["llm"].get("fixed_output_budget", False) is False
    assert lean["llm"]["context_window"] in (65536, 131072)
    assert config["agent"]["tools_allowlist"] == ALLOWED
    assert config["agent"]["target_prefetch"] == "user_message"
    assert config["agent"]["iteration_mode"] == "plain"
    assert config["agent"]["mini_status_every"] == 0
    assert config["agent"]["user_nudges_enabled"] is False
    assert config["max_iterations"] == 40
    assert config["compaction"]["enabled"] is False
    assert config["precedent_recall"]["enabled"] is False
    assert config["prompts"]["core"] == "prompts/core_lean.md"
    prompt = Path(config["prompts"]["core"]).read_text()
    for hidden in ["todo_write", "read_doc", "backtrack_and_summarize", "rank_routes", "load_result", "apply_template", "enzymatic_retro"]:
        assert hidden not in prompt
    assert prompt.count("product_SMILES>>reactant1_SMILES.reactant2_SMILES (reaction: name)") == 1
    prompt.format(target_smiles=TARGET, max_depth=12, max_iterations=40)


@pytest.mark.parametrize("options", [
    {"tools_allowlist": "score_route"}, {"tools_allowlist": [1]},
    {"target_prefetch": "bad"}, {"iteration_mode": "bad"},
    {"mini_status_every": -1}, {"mini_status_every": True},
])
def test_invalid_options_rejected(options):
    with pytest.raises(ValueError):
        load_config(cli_overrides={"agent": options})


def test_new_keys_do_not_disable_existing_typo_validation():
    with pytest.raises(ValueError, match="max_iterations_"):
        load_config(cli_overrides={"agent": {"iteration_mode": "plain", "max_iterations_": 10}})
