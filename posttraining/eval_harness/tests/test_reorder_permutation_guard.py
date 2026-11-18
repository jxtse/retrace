"""Reorder postprocess is a verified permutation, never a mutation (audit #6).

The reviewer's "reorder" action used to assign its free-text
optimized_route_text straight into result["route_text"], leaving best_route /
ranked / all_routes pointing at the OLD structure — so the user-facing route
could silently disagree with its attached score (typo, dropped step,
infeasible reordering shipped as success).

The guard proves the rewrite is a pure permutation via order-insensitive
route_fingerprint equality against the selected route's steps:
  * equal fingerprint  -> accept, route_text updated, status "reordered".
  * different / mutated -> reject, route_text + structures untouched,
    status "reorder_rejected_non_permutation".
  * unparseable rewrite -> reject, status "reorder_rejected_unparseable".

These call _run_reaction_ordering_postprocess directly on a bypass-built
agent with a fake reviewer, isolating the branch from the full LLM loop.
"""
import json

from core.agent import RetrosynthesisAgent


class _FakeReviewer:
    """Reviewer client whose chat_with_override returns a canned decision."""

    def __init__(self, decision: dict):
        self._raw = json.dumps(decision)
        self.model = "fake-reviewer"
        self.base_url = "http://localhost:0/v1"
        self.max_tokens = 32000
        self.reasoning_effort = "low"

    def chat_with_override(self, messages, **kwargs):
        # Mirror the OpenAI chat-completion shape execute() expects.
        return {"choices": [{"message": {"content": self._raw}}]}


def _agent(decision):
    ag = RetrosynthesisAgent.__new__(RetrosynthesisAgent)
    ag.postprocess_config = {"enabled": True, "max_review_rounds": 1,
                             "max_tokens": 32000, "temperature": 0.0,
                             "reasoning_effort": "low"}
    ag.reviewer_client = _FakeReviewer(decision)
    ag.max_iterations = 50
    # Helpers the postprocess path calls — keep them inert.
    ag._record_llm_request = lambda **kw: None
    ag._reviewer_message_text = lambda tgt, res: "review please"
    ag._write_review_artifact = lambda **kw: None
    ag._update_postprocess_metadata = lambda payload: None
    ag._active_recorder = lambda: None
    return ag


# The selected route's single verified step (what route_text was built from).
_TARGET = "CC(=O)OC1CCC(C)CC1"
_STEP = f"{_TARGET}>>CCO.CC=O"
_ORIG_TEXT = "COMPLETE ROUTE FOUND: original\nStep 1: `" + _STEP + "`"


def _result():
    return {
        "route_text": _ORIG_TEXT,
        "best_route": {"label": "R1", "steps": [_STEP]},
        "iterations": 5,
    }


def test_true_permutation_accepted():
    # Same step restated (reactant order flipped) — fingerprint is invariant.
    new_text = f"COMPLETE ROUTE FOUND: reordered\nStep 1: `{_TARGET}>>CC=O.CCO`"
    ag = _agent({"status": "ok", "action": "reorder",
                 "optimized_route_text": new_text,
                 "summary": "s", "issues": []})
    out = ag._run_reaction_ordering_postprocess(_TARGET, _result())
    assert out["postprocess"]["status"] == "reordered"
    assert out["route_text"] == new_text
    assert "route_fingerprint" in out["postprocess"]


def test_mutation_rejected_structure_untouched():
    # Reactant swapped CCO -> CCN: real chemistry change disguised as reorder.
    new_text = f"COMPLETE ROUTE FOUND: reordered\nStep 1: `{_TARGET}>>CCN.CC=O`"
    ag = _agent({"status": "ok", "action": "reorder",
                 "optimized_route_text": new_text,
                 "summary": "s", "issues": []})
    res = _result()
    out = ag._run_reaction_ordering_postprocess(_TARGET, res)
    assert out["postprocess"]["status"] == "reorder_rejected_non_permutation"
    # route_text and best_route left exactly as finalisation produced them.
    assert out["route_text"] == _ORIG_TEXT
    assert out["best_route"] == {"label": "R1", "steps": [_STEP]}
    assert out["postprocess"]["original_fingerprint"] != out["postprocess"]["rewrite_fingerprint"]


def test_dropped_step_rejected():
    # A two-step original; the rewrite drops a step. Multiset changed -> reject.
    step2 = "CCCC>>CCCCBr"
    res = {
        "route_text": _ORIG_TEXT,
        "best_route": {"label": "R1", "steps": [_STEP, step2]},
        "iterations": 5,
    }
    new_text = f"COMPLETE ROUTE FOUND: reordered\nStep 1: `{_STEP}`"
    ag = _agent({"status": "ok", "action": "reorder",
                 "optimized_route_text": new_text,
                 "summary": "s", "issues": []})
    out = ag._run_reaction_ordering_postprocess(_TARGET, res)
    assert out["postprocess"]["status"] == "reorder_rejected_non_permutation"
    assert out["route_text"] == _ORIG_TEXT


def test_unparseable_rewrite_rejected():
    # Free text with no parseable reaction step.
    ag = _agent({"status": "ok", "action": "reorder",
                 "optimized_route_text": "COMPLETE ROUTE FOUND: trust me",
                 "summary": "s", "issues": []})
    res = _result()
    out = ag._run_reaction_ordering_postprocess(_TARGET, res)
    assert out["postprocess"]["status"] == "reorder_rejected_unparseable"
    assert out["route_text"] == _ORIG_TEXT


def test_accept_action_unchanged():
    # The non-reorder 'accept' path must still pass straight through.
    ag = _agent({"status": "ok", "action": "accept", "summary": "s", "issues": []})
    out = ag._run_reaction_ordering_postprocess(_TARGET, _result())
    assert out["postprocess"]["status"] == "accepted"
    assert out["route_text"] == _ORIG_TEXT
