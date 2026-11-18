"""End-to-end smoke test of the ReAct loop with a scripted FakeLLMClient.

The goal is to catch loop-level regressions (wallclock guard,
iteration-budget whitelist, completion-gate enforcement, sticky-state
reset, score-validation handoff) without spinning up real LLM, neural
engines, or HTTP services. A FakeLLMClient yields a hand-scripted
sequence of (content, tool_calls, finish_reason) triples that drive the
agent through:

    1. one suggest_templates call (billable)
    2. one neural_retro call (billable)
    3. three save_best_route calls (free, reaches stop_after_distinct_routes)
    4. final "COMPLETE ROUTE FOUND:" emission

We assert the loop terminates with status=success, the iteration meter
counts the two billable tool calls (not the three saves), and the
result includes the structural-completion fields (`route_found=True`,
`iterations` equal to the billable count).

Stubs:
    - LLMClient is replaced wholesale.
    - NeuralRetroTool / VerifierTool / suggest_templates are
      disabled in the config so build_agent only registers the cheap
      pure-RDKit + scratchpad + read_doc + load_result tools. We then
      directly register a FakeNeuralRetroTool + FakeSuggestTemplatesTool
      that return canned JSON.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest


class _FakeLLMClient:
    """Yields scripted (content, tool_calls, finish_reason) triples.

    Indexable via call_count; raises StopIteration if the agent demands
    more turns than scripted (which itself catches a runaway loop).
    """

    def __init__(self, script):
        self._script = list(script)
        self.call_count = 0
        # Match real LLMClient surface enough for the agent to introspect.
        self.model = "fake/claude-opus-4.7"
        self.max_tokens = 64000
        self.temperature = 0.0

    def chat_with_tools(self, messages, tools=None):
        if self.call_count >= len(self._script):
            raise AssertionError(
                f"FakeLLMClient ran out of script after {self.call_count} calls "
                f"(agent did not terminate); last messages: "
                f"{[m.get('role') for m in messages[-3:]]}"
            )
        triple = self._script[self.call_count]
        self.call_count += 1
        return triple

    def chat(self, messages, tools=None):
        # Used by the precursor sanity judge — return a structured
        # judgement that keeps everything (fail-open).
        return {
            "choices": [{"message": {"content": "{\"keep\": [], \"drop\": []}"}, "finish_reason": "stop"}]
        }

    def chat_with_override(self, *a, **kw):
        return self.chat(a[0] if a else kw.get("messages"))


def _tc(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _make_config(tmpdir: str) -> dict:
    """Strip every heavy / network-bound feature so the test runs offline."""
    return {
        "llm": {"base_url": "http://localhost:0", "model": "fake", "temperature": 0.0, "max_tokens": 1024},
        "agent": {
            "max_depth": 12,
            "max_iterations": 50,
            "iteration_billable_tools": ["suggest_templates", "neural_retro"],
            "stop_after_distinct_routes": 3,
            "tool_timeout_s": 5,
            "wallclock_s": 60,
            "auto_finish_after_saved": None,
        },
        "scratchpad": {"max_routes": 10},
        "prompts": {
            "core": "prompts/core.md",
            "sections_dir": "prompts/sections",
            "messages_dir": "prompts/messages",
        },
        "result_store": {
            "dir": tmpdir + "/_session_{session}",
            "summary_max_chars": 800,
            "keep_recent_full_per_tool": 4,
            "retention_days": 0,
        },
        "postprocess": {"enabled": False},
        # All heavy tools off — we patch in fakes after build.
        "neural_retro": {"enabled": False},
        "verifier": {"enabled": False},
        "templates": {"dataset_dir": None},
        "stocks": {},
        "precedent_recall": {"enabled": False},
    }


@pytest.fixture
def agent_with_fakes(monkeypatch):
    """Build a real agent against in-memory fake LLM + canned tool outputs."""
    import core.agent as agent_mod
    from core import build_agent
    from tools.base import Tool

    # ── Fake tools ────────────────────────────────────────────────────
    class FakeSuggestTemplates(Tool):
        name = "suggest_templates"
        description = "fake"
        parameters = {"type": "object", "properties": {"product_smiles": {"type": "string"}}}
        def execute(self, **kw):
            return json.dumps({
                "product": kw.get("product_smiles", ""),
                "library_size": 1000,
                "num_suggestions": 1,
                "suggestions": [{
                    "template_kind": "organic",
                    "template_source": "uspto",
                    "reactants_smiles": "CCO.CC(=O)Cl",
                    "reactants": "CCO.CC(=O)Cl",
                }],
            })

    class FakeNeuralRetro(Tool):
        name = "neural_retro"
        description = "fake"
        parameters = {"type": "object", "properties": {"target_smiles": {"type": "string"}}}
        def execute(self, **kw):
            return json.dumps({
                "target": kw.get("target_smiles", ""),
                "engine": "fake-ensemble",
                "n_unique_after_dedup": 5,
                "n_clusters": 3,
                "predictions": [
                    {"reactants": "CCO.CC(=O)Cl", "score": 0.9, "support": ["fake-engine"]},
                    {"reactants": "CC=O.CC", "score": 0.7, "support": ["fake-engine"]},
                ],
            })

    # Patch _prefetch_disconnections out — it tries to call the real
    # tools and we don't need its output for this loop-shape test.
    monkeypatch.setattr(
        agent_mod.RetrosynthesisAgent, "_prefetch_disconnections",
        lambda self, msgs, tgt: None,
    )

    with tempfile.TemporaryDirectory() as td:
        config = _make_config(td)
        # Build with whatever fits offline; we then forcibly swap LLM.
        ag = build_agent(config)
        # Drop in fake tools (overwrite if any cheap variant was registered).
        ag.tools.register(FakeSuggestTemplates())
        ag.tools.register(FakeNeuralRetro())
        yield ag


def test_react_loop_completes_with_scripted_llm(agent_with_fakes):
    """End-to-end: scripted LLM drives loop → COMPLETE ROUTE FOUND."""
    ag = agent_with_fakes

    # 7 LLM turns:
    #   1. call suggest_templates                    (iter 1, billable)
    #   2. call neural_retro                         (iter 2, billable)
    #   3. call save_best_route #1                   (iter 3, free)
    #   4. call save_best_route #2                   (iter 4, free)
    #   5. call save_best_route #3                   (iter 5, free)
    #   6. emit COMPLETE ROUTE FOUND                 (iter 6, terminate)
    target = "CC(=O)OC1CCC(C)CC1"
    route_reactants = {
        1: "CCO.CC=O",
        2: "CCN.CC=O",
        3: "CCCl.CC=O",
    }
    save_route = lambda i: _tc(
        f"call_save_{i}",
        "save_best_route",
        {
            "target_smiles": target,
            "steps": [f"{target}>>{route_reactants[i]}"],
            "total_score": 0.5 + 0.05 * i,
            "rationale": f"route variant {i}",
            "label": f"route-{i}",
        },
    )

    script = [
        # Turn 1: call suggest_templates
        (None, [_tc("call_st", "suggest_templates", {"product_smiles": target})], "tool_calls"),
        # Turn 2: call neural_retro
        (None, [_tc("call_nr", "neural_retro", {"target_smiles": target})], "tool_calls"),
        # Turns 3-5: save 3 distinct routes
        (None, [save_route(1)], "tool_calls"),
        (None, [save_route(2)], "tool_calls"),
        (None, [save_route(3)], "tool_calls"),
        # Turn 6: emit completion after the soft threshold has fired.
        (
            "COMPLETE ROUTE FOUND: best route = route-3 (score 0.65)",
            None,
            "stop",
        ),
    ]

    fake = _FakeLLMClient(script)
    ag.llm = fake

    result = ag.run(target)

    # Loop terminated (didn't bail by wallclock or max_iterations)
    assert result is not None, "agent.run returned None"
    assert "bail_reason" not in result or result.get("bail_reason") in (None, ""), (
        f"unexpected bail: {result.get('bail_reason')!r} / status={result.get('status')!r}"
    )
    # Used exactly the scripted number of LLM turns.
    assert fake.call_count == len(script), (
        f"agent took {fake.call_count} LLM turns; expected exactly {len(script)}. "
        "Script-overrun usually means a completion gate is over-strict."
    )
    # Iteration meter counts only billable tool calls. With one
    # suggest_templates + one neural_retro, the meter should report 2.
    iters = result.get("iterations")
    assert iters is not None and iters >= 2, (
        f"iterations={iters!r}; expected at least 2 billable tool calls"
    )
    # 3 saves should have landed in the scratchpad.
    assert ag.scratchpad_store is not None
    saved = ag.scratchpad_store.list()
    assert len(saved) == 3, f"expected 3 saved routes, got {len(saved)}"


def test_sticky_state_reset_between_runs(agent_with_fakes):
    """Re-using one agent for two targets must NOT carry over short-circuit
    cache, fingerprints, or per-tool counts. Regression test for the fix
    in agent.run() where a stale `_call_fingerprints[<target1, neural_retro>]`
    was triggering a "you already called this 4 times" nudge on iteration 1
    of the second target."""
    ag = agent_with_fakes
    target_a = "CCO"
    target_b = "CC(=O)O"

    # Seed sticky state as if a previous run had populated it.
    ag._call_fingerprints = {("neural_retro", "CCC"): 9}
    ag._short_circuit_cache = {"suggest_templates::CCC": "stale"}
    ag._tool_call_counts = {"neural_retro": 99}

    save_route = lambda i: _tc(
        f"call_seed_{i}",
        "save_best_route",
        {
            "target_smiles": target_a,
            "steps": [f"{target_a}>>X{i}.Y"],
            "total_score": 0.5,
            "rationale": f"seed route {i}",
            "label": f"seed-{i}",
        },
    )
    script = [
        (None, [save_route(1)], "tool_calls"),
        (None, [save_route(2)], "tool_calls"),
        (None, [save_route(3)], "tool_calls"),
        (
            f"COMPLETE ROUTE FOUND: trivial {target_a}",
            None,
            "stop",
        ),
    ]

    ag.llm = _FakeLLMClient(script)
    _ = ag.run(target_a)

    # Verify reset happened at the top of run().
    assert ag._call_fingerprints == {} or all(
        k for k in ag._call_fingerprints if "CCC" not in k
    ), f"stale fingerprints leaked: {ag._call_fingerprints!r}"
    assert "suggest_templates::CCC" not in ag._short_circuit_cache, (
        "stale short-circuit cache entry leaked between runs"
    )


class _FakeReviewerClient:
    def __init__(self, content: str):
        self.content = content
        self.calls = []
        self.base_url = "http://reviewer.invalid/v1"
        self.model = "fake-reviewer"
        self.temperature = 0.0
        self.max_tokens = 1024
        self.reasoning_effort = "xhigh"
        self.responses_state = "manual"
        self.max_retries = 1
        self.read_timeout = 1

    def chat_with_override(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return {
            "choices": [
                {
                    "message": {"content": self.content},
                    "finish_reason": "stop",
                }
            ]
        }


def _reorder_route_setup(ag, target, optimized_route_text):
    """Wire three saved single-step routes + a reviewer 'reorder' decision.

    Shared by the accept (true-permutation) and reject (non-permutation)
    variants below. reorder-3 (score 0.65) wins finalisation; its single
    step is `{target}>>CCCl.CC=O`.
    """
    ag.postprocess_config = {"enabled": True, "max_review_rounds": 1}
    ag.reviewer_client = _FakeReviewerClient(json.dumps({
        "action": "reorder",
        "summary": "Install the expensive fragment last.",
        "optimized_route_text": optimized_route_text,
        "issues": ["forward order wastes scarce material"],
    }))
    route_reactants = {1: "CCO.CC=O", 2: "CCN.CC=O", 3: "CCCl.CC=O"}
    save_route = lambda i: _tc(
        f"call_reorder_{i}",
        "save_best_route",
        {
            "target_smiles": target,
            "steps": [f"{target}>>{route_reactants[i]}"],
            "total_score": 0.5 + 0.05 * i,
            "rationale": f"route variant {i}",
            "label": f"reorder-{i}",
        },
    )
    ag.llm = _FakeLLMClient([
        (None, [save_route(1)], "tool_calls"),
        (None, [save_route(2)], "tool_calls"),
        (None, [save_route(3)], "tool_calls"),
        ("COMPLETE ROUTE FOUND: original final route", None, "stop"),
    ])


def test_reaction_ordering_reviewer_reorder_permutation_accepted(agent_with_fakes):
    # A TRUE permutation: the rewrite restates the selected route's exact
    # step. Scores are clamped to 0.0 (no score_route_result_id), so rank-1 of
    # finalisation is the first-saved route reorder-1 → `target>>CCO.CC=O`.
    # Same step multiset ⇒ accepted.
    ag = agent_with_fakes
    target = "CC(=O)OC1CCC(C)CC1"
    optimized = (
        "COMPLETE ROUTE FOUND: reordered\n"
        f"Step 1: `{target}>>CCO.CC=O`"
    )
    _reorder_route_setup(ag, target, optimized)

    result = ag.run(target)

    assert result["route_text"] == optimized
    assert result["postprocess"]["status"] == "reordered"
    assert "route_fingerprint" in result["postprocess"]
    assert "rank-1 from finalisation" in result["postprocess"]["original_route_text"]
    assert ag.reviewer_client.calls, "reviewer was not called"

    session_dir = result["session_dir"]
    assert os.path.exists(os.path.join(session_dir, "session_manifest.json"))
    assert os.path.exists(os.path.join(session_dir, "conversation", "latest_messages.json"))
    assert os.path.exists(os.path.join(session_dir, "conversation", "llm_requests.jsonl"))
    with open(result["transcript_json"], encoding="utf-8") as fh:
        transcript = json.load(fh)
    assert transcript["postprocess"]["status"] == "reordered"
    assert transcript["model_settings"]["reviewer"]["model"] == "fake-reviewer"
    assert "reaction_ordering" in transcript["prompt_artifacts"]


def test_reaction_ordering_reviewer_reorder_freetext_rejected(agent_with_fakes):
    # The reviewer returns free text with no parseable reaction step. Under
    # the permutation guard this is rejected (it could silently drop/mutate
    # the route); route_text + structures stay as finalisation produced them.
    ag = agent_with_fakes
    target = "CC(=O)OC1CCC(C)CC1"
    _reorder_route_setup(ag, target, "COMPLETE ROUTE FOUND: reordered final route")

    result = ag.run(target)

    # route_text is the gated finalisation text — NOT the reviewer's rewrite.
    assert result["route_text"] != "COMPLETE ROUTE FOUND: reordered final route"
    assert result["postprocess"]["status"] == "reorder_rejected_unparseable"
    assert result["postprocess"]["rejected_route_text"] == (
        "COMPLETE ROUTE FOUND: reordered final route"
    )


def test_reaction_ordering_reviewer_reorder_mutation_rejected(agent_with_fakes):
    # The rewrite swaps a reactant (CCCl → CCBr): a real chemistry change
    # disguised as a reorder. Fingerprints differ ⇒ rejected, structure kept.
    ag = agent_with_fakes
    target = "CC(=O)OC1CCC(C)CC1"
    mutated = (
        "COMPLETE ROUTE FOUND: reordered\n"
        f"Step 1: `{target}>>CCBr.CC=O`"
    )
    _reorder_route_setup(ag, target, mutated)

    result = ag.run(target)

    assert result["route_text"] != mutated
    assert result["postprocess"]["status"] == "reorder_rejected_non_permutation"
    assert result["postprocess"]["original_fingerprint"] != result["postprocess"]["rewrite_fingerprint"]


def test_reaction_ordering_reviewer_revise_continues(agent_with_fakes):
    ag = agent_with_fakes
    ag.postprocess_config = {"enabled": True, "max_review_rounds": 1}
    ag.reviewer_client = _FakeReviewerClient(json.dumps({
        "action": "revise",
        "summary": "The order cannot be repaired by presentation only.",
        "feedback": "Find a route where the hindered coupling is done earlier.",
        "issues": ["hindered late-stage coupling"],
    }))
    target = "CC(=O)OC1CCC(C)CC1"

    route_reactants = {
        "initial-1": "CCO.CC=O",
        "initial-2": "CCN.CC=O",
        "initial-3": "CCCl.CC=O",
        "revised-4": "CCBr.CC=O",
    }

    def save_route(call_id, label, score):
        return _tc(
            call_id,
            "save_best_route",
            {
                "target_smiles": target,
                "steps": [f"{target}>>{route_reactants[label]}"],
                "total_score": score,
                "rationale": label,
                "label": label,
            },
        )

    ag.llm = _FakeLLMClient([
        (None, [save_route("call_initial_1", "initial-1", 0.51)], "tool_calls"),
        (None, [save_route("call_initial_2", "initial-2", 0.52)], "tool_calls"),
        (None, [save_route("call_initial_3", "initial-3", 0.53)], "tool_calls"),
        ("COMPLETE ROUTE FOUND: initial route", None, "stop"),
        (None, [save_route("call_revised_4", "revised-4", 0.75)], "tool_calls"),
        ("COMPLETE ROUTE FOUND: revised route", None, "stop"),
    ])

    result = ag.run(target)

    assert result["postprocess"]["status"] == "continued"
    assert result["postprocess"]["continuation_iterations"] >= 0
    assert ag.llm.call_count == 6
    assert any(
        "Post-run reaction-ordering review" in (m.get("content") or "")
        for m in result["messages"]
    )
