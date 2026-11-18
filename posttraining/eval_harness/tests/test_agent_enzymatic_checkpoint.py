"""Behavioral coverage for the coordinator's explicit enzyme checkpoint."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from core.agent import RetrosynthesisAgent
from core.prompt_loader import PromptLoader
from tools.base import Tool, ToolRegistry
from tools.scratchpad import SaveBestRouteTool, ScratchpadStore


def _tool_call(call_id: str, product_smiles: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "enzymatic_retro",
            "arguments": json.dumps({"product_smiles": product_smiles}),
        },
    }


def _save_call(
    call_id: str,
    *,
    label: str,
    target: str,
    complex_intermediate: str,
    step_provenance=None,
) -> dict:
    arguments = {
        "label": label,
        "steps": [f"{target}>>{complex_intermediate}.CN"],
        "total_score": 0.7,
    }
    if step_provenance is not None:
        arguments["step_provenance"] = step_provenance
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "save_best_route",
            "arguments": json.dumps(arguments),
        },
    }


class _EnzymaticRetroStub(Tool):
    name = "enzymatic_retro"
    description = "stub"
    parameters = {
        "type": "object",
        "properties": {"product_smiles": {"type": "string"}},
        "required": ["product_smiles"],
    }

    def execute(self, product_smiles: str, **kwargs) -> str:
        return json.dumps({
            "product": product_smiles,
            "num_suggestions": 1,
            "suggestions": [{
                "template_kind": "enzymatic",
                "template_source": "retrobiocat",
                "reactants": ["CC=O"],
                "reactants_smiles": "CC=O",
            }],
        })


class _RecordingLLM:
    model = "fake"
    max_tokens = 1024
    temperature = 0.0

    def __init__(self, products: list[str]):
        self.products = products
        self.calls: list[list[dict]] = []

    def chat_with_tools(self, messages, tools=None):
        self.calls.append(deepcopy(messages))
        idx = len(self.calls) - 1
        return None, [_tool_call(f"enzyme-{idx}", self.products[idx])], "tool_calls"


class _ScriptedLLM:
    model = "fake"
    max_tokens = 1024
    temperature = 0.0

    def __init__(self, turns: list[list[dict]]):
        self.turns = turns
        self.calls: list[list[dict]] = []

    def chat_with_tools(self, messages, tools=None):
        self.calls.append(deepcopy(messages))
        return None, self.turns[len(self.calls) - 1], "tool_calls"


def _route_checkpoint_agent(llm, target: str):
    store = ScratchpadStore(max_routes=5)
    registry = ToolRegistry()
    registry.register(_EnzymaticRetroStub())
    registry.register(SaveBestRouteTool(store))
    prompts = PromptLoader(
        str(Path(__file__).resolve().parents[1] / "prompts" / "messages")
    )
    agent = RetrosynthesisAgent(
        llm_client=llm,
        tool_registry=registry,
        max_iterations=len(llm.turns),
        compaction_enabled=False,
        scratchpad_store=store,
        system_prompt=(
            "Target {target_smiles}; depth {max_depth}; budget {max_iterations}. "
            "{precedent_recall_block}{reaction_ordering_block}"
        ),
        iteration_billable_tools=["enzymatic_retro", "save_best_route"],
        prompts=prompts,
        auto_finish_after_saved=None,
        reaction_ordering_block="",
        postprocess_config={"enabled": False},
        wallclock_s=10,
    )
    return agent, store


def _route_checkpoints(messages):
    return [
        message for message in messages
        if message.get("role") == "user"
        and "[ROUTE-AWARE ENZYME CHECKPOINT]"
        in (message.get("content") or "")
    ]


def test_successful_enzyme_checkpoint_is_once_per_canonical_product(
    monkeypatch,
):
    """Equivalent SMILES share one checkpoint; another product gets its own.

    This runs the real native tool loop so the assertion covers placement in
    the conversation after successful tool results, not just a text helper.
    """
    monkeypatch.setenv("RETRO_AGENT_DISABLE_TARGET_PREFETCH", "1")
    registry = ToolRegistry()
    registry.register(_EnzymaticRetroStub())
    llm = _RecordingLLM(["OCC", "CCO", "CC=O"])
    prompts = PromptLoader(
        str(Path(__file__).resolve().parents[1] / "prompts" / "messages")
    )
    agent = RetrosynthesisAgent(
        llm_client=llm,
        tool_registry=registry,
        max_iterations=3,
        compaction_enabled=False,
        system_prompt=(
            "Target {target_smiles}; depth {max_depth}; budget {max_iterations}. "
            "{precedent_recall_block}{reaction_ordering_block}"
        ),
        iteration_billable_tools=["enzymatic_retro"],
        prompts=prompts,
        auto_finish_after_saved=None,
        reaction_ordering_block="",
        postprocess_config={"enabled": False},
        wallclock_s=10,
    )

    result = agent.run("CCO")

    def checkpoints(messages):
        return [
            message for message in messages
            if message.get("role") == "user"
            and "[ENZYME BRANCH CHECKPOINT]" in (message.get("content") or "")
        ]

    # The next model turn sees the checkpoint from the first successful call.
    assert len(checkpoints(llm.calls[1])) == 1
    # OCC and CCO canonicalize to the same product, so no repeat is injected.
    assert len(checkpoints(llm.calls[2])) == 1
    # The distinct intermediate receives one additional checkpoint.
    final_checkpoints = checkpoints(result["messages"])
    assert len(final_checkpoints) == 2
    assert any("target" in item["content"].lower() for item in final_checkpoints)
    assert all("rank 1" in item["content"].lower() for item in final_checkpoints)


def test_saved_route_checkpoint_names_only_plausible_complex_rhs_intermediate(
    monkeypatch,
):
    monkeypatch.setenv("RETRO_AGENT_DISABLE_TARGET_PREFETCH", "1")
    target = "CCN[C@@H](C)[C@H](O)c1ccccc1"
    intermediate = "CC(=O)[C@H](O)c1ccccc1"
    llm = _ScriptedLLM([[
        _save_call(
            "save-1",
            label="mixed-route",
            target=target,
            complex_intermediate=intermediate,
        )
    ]])
    agent, store = _route_checkpoint_agent(llm, target)

    result = agent.run(target)

    assert store.get("mixed-route") is not None
    checkpoints = _route_checkpoints(result["messages"])
    assert len(checkpoints) == 1
    text = checkpoints[0]["content"]
    assert intermediate in text
    assert "CN" not in text
    assert "rather than assuming rank 1 is valid" in text.lower()


def test_saved_route_checkpoint_suppressed_for_already_queried_product(
    monkeypatch,
):
    monkeypatch.setenv("RETRO_AGENT_DISABLE_TARGET_PREFETCH", "1")
    target = "CCN[C@@H](C)[C@H](O)c1ccccc1"
    intermediate = "CC(=O)[C@H](O)c1ccccc1"
    llm = _ScriptedLLM([
        [_tool_call("enzyme-1", intermediate)],
        [_save_call(
            "save-1",
            label="queried-route",
            target=target,
            complex_intermediate=intermediate,
        )],
    ])
    agent, _ = _route_checkpoint_agent(llm, target)

    result = agent.run(target)

    assert _route_checkpoints(result["messages"]) == []


def test_saved_route_checkpoint_suppressed_by_exact_enzymatic_step_provenance(
    monkeypatch,
):
    monkeypatch.setenv("RETRO_AGENT_DISABLE_TARGET_PREFETCH", "1")
    target = "CCN[C@@H](C)[C@H](O)c1ccccc1"
    intermediate = "CC(=O)[C@H](O)c1ccccc1"
    llm = _ScriptedLLM([[
        _save_call(
            "save-1",
            label="enzyme-proven-route",
            target=target,
            complex_intermediate=intermediate,
            step_provenance=[{
                "template_kind": "enzymatic",
                "template_source": "curated_enzyme_source",
                "rank": 2,
            }],
        )
    ]])
    agent, store = _route_checkpoint_agent(llm, target)

    result = agent.run(target)

    assert store.get("enzyme-proven-route").step_provenance[0][
        "template_kind"
    ] == "enzymatic"
    assert _route_checkpoints(result["messages"]) == []


def test_saved_route_checkpoint_is_once_per_structural_fingerprint(
    monkeypatch,
):
    monkeypatch.setenv("RETRO_AGENT_DISABLE_TARGET_PREFETCH", "1")
    target = "CCN[C@@H](C)[C@H](O)c1ccccc1"
    intermediate = "CC(=O)[C@H](O)c1ccccc1"
    save = _save_call(
        "save-1",
        label="rescored-route",
        target=target,
        complex_intermediate=intermediate,
    )
    updated_save = deepcopy(save)
    updated_save["id"] = "save-2"
    updated_args = json.loads(updated_save["function"]["arguments"])
    updated_args["total_score"] = 0.8
    updated_save["function"]["arguments"] = json.dumps(updated_args)
    llm = _ScriptedLLM([[save], [updated_save]])
    agent, _ = _route_checkpoint_agent(llm, target)

    result = agent.run(target)

    assert len(_route_checkpoints(result["messages"])) == 1


def test_saved_route_checkpoint_has_global_three_nudge_cap(monkeypatch):
    monkeypatch.setenv("RETRO_AGENT_DISABLE_TARGET_PREFETCH", "1")
    target = "CCN[C@@H](C)[C@H](O)c1ccccc1"
    intermediates = [
        "CC(=O)c1ccccc1",
        "O=C(O)c1ccccc1",
        "CCOC(=O)c1ccccc1",
        "CC(=O)Oc1ccccc1",
    ]
    llm = _ScriptedLLM([
        [_save_call(
            f"save-{index}",
            label=f"route-{index}",
            target=target,
            complex_intermediate=intermediate,
        )]
        for index, intermediate in enumerate(intermediates, start=1)
    ])
    agent, _ = _route_checkpoint_agent(llm, target)

    result = agent.run(target)

    checkpoints = _route_checkpoints(result["messages"])
    assert len(checkpoints) == 3
    assert intermediates[3] not in "\n".join(
        checkpoint["content"] for checkpoint in checkpoints
    )
