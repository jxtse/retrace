"""Model tool-call arguments are evidence and must not be mutated in place.

train_round1_batch01 (2026-09-17): the agent injected ``_provenance`` /
``_rag_source_ids`` into the *same* ``save_best_route`` tool_call object that
had already been appended to ``messages``. Later requests therefore carried
assistant history that differed from the captured model response and the
training admission verifier rejected both trajectories.

Contract: the message history (and every later request) carries the model's
verbatim arguments; provenance is attached only to the execution copy handed
to the tool, and the scratchpad still receives it.
"""
import json

from tests.test_lean_profile import make_agent  # noqa: F401  (fixture re-export)
from tests.test_lean_profile import RecordingLLM, _tc

TARGET = "CCOC(=O)c1nc(N2CC[C@H](N)[C@H](OC)C2)sc1C"


def _save_args():
    return {"target_smiles": TARGET, "steps": [f"{TARGET}>>CCO.CC=O"],
            "total_score": 0.0, "label": "candidate"}


def test_history_keeps_verbatim_model_arguments(make_agent):
    agent = make_agent({
        "max_iterations": 3, "iteration_mode": "plain", "target_prefetch": "off",
        "mini_status_every": 0, "user_nudges_enabled": False,
        "stop_after_distinct_routes": 10, "auto_finish_after_saved": 20,
    })
    save_call = _tc("save1", "save_best_route", _save_args())
    original_args = save_call["function"]["arguments"]
    agent.llm = RecordingLLM([
        (None, [_tc("n1", "neural_retro", {"smiles": TARGET})], "tool_calls"),
        (None, [save_call], "tool_calls"),
        (None, [_tc("n2", "neural_retro", {"smiles": "CCO"})], "tool_calls"),
    ])
    result = agent.run(TARGET)
    # The response object handed back by the model must be untouched.
    assert save_call["function"]["arguments"] == original_args
    # Every later request's assistant history must equal the verbatim response.
    seen = False
    for req in agent.llm.requests:
        for m in req["messages"]:
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                if tc["id"] == "save1":
                    seen = True
                    assert tc["function"]["arguments"] == original_args
                    assert "_provenance" not in json.loads(tc["function"]["arguments"])
    assert seen
    for m in result["messages"]:
        for tc in (m.get("tool_calls") or []) if m.get("role") == "assistant" else []:
            assert "_provenance" not in json.loads(tc["function"]["arguments"])
    # Provenance still reaches the scratchpad through the execution copy.
    entries = agent.scratchpad_store.all()
    assert entries and entries[0].provenance == "neural_consensus"
