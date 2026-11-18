"""LLM-judge precursor sanity filter.

Drops candidates that are obviously implausible (a single small leaving
group, e.g. methylamine alone, that cannot account for the product's
skeleton). Fail-open: any LLM error keeps everything.
"""
import json

import pytest

from tools._precursor_sanity import filter_precursors
from tests.conftest import FakeLLMClient


def _candidates():
    return [
        {"reactants": "CN", "engine": "rxngraphormer", "score": 0.4},      # methylamine alone
        {"reactants": "CC", "engine": "rxngraphormer", "score": 0.3},      # ethane alone
        {"reactants": "c1ccccc1Br.CN", "engine": "chembart", "score": 0.7},  # plausible
        {"reactants": "c1ccccc1Cl.CN", "engine": "graph_retrosyn", "score": 0.6},  # plausible
    ]


def test_filter_drops_implausible_singletons():
    judge_response = json.dumps([
        {"idx": 0, "plausible": False, "reason": "single methylamine; no skeleton"},
        {"idx": 1, "plausible": False, "reason": "ethane has no functional handle"},
        {"idx": 2, "plausible": True,  "reason": "aryl bromide + amine; sensible"},
        {"idx": 3, "plausible": True,  "reason": "aryl chloride + amine; sensible"},
    ])
    llm = FakeLLMClient(responses=[judge_response])

    kept, dropped, reasons = filter_precursors(
        target_smiles="c1ccc(NC)cc1",
        candidates=_candidates(),
        llm_client=llm,
    )
    assert [c["reactants"] for c in kept] == ["c1ccccc1Br.CN", "c1ccccc1Cl.CN"]
    assert len(dropped) == 2
    assert "methylamine" in reasons[0].lower()


def test_filter_fails_open_on_bad_json():
    llm = FakeLLMClient(responses=["not json at all"])
    kept, dropped, reasons = filter_precursors(
        target_smiles="c1ccc(NC)cc1",
        candidates=_candidates(),
        llm_client=llm,
    )
    assert len(kept) == len(_candidates())
    assert dropped == []


def test_filter_fails_open_on_llm_exception():
    class BoomClient:
        def chat(self, *_a, **_k):
            raise RuntimeError("connection refused")

    kept, dropped, _ = filter_precursors(
        target_smiles="c1ccc(NC)cc1",
        candidates=_candidates(),
        llm_client=BoomClient(),
    )
    assert len(kept) == len(_candidates())
    assert dropped == []


def test_filter_uses_cache_for_identical_input():
    judge_response = json.dumps([
        {"idx": 0, "plausible": False, "reason": "trivial fragment"},
        {"idx": 1, "plausible": False, "reason": "trivial fragment"},
        {"idx": 2, "plausible": True, "reason": "ok"},
        {"idx": 3, "plausible": True, "reason": "ok"},
    ])
    llm = FakeLLMClient(responses=[judge_response])
    cache: dict = {}
    target = "c1ccc(NC)cc1"
    cands = _candidates()
    filter_precursors(target, cands, llm, cache=cache)
    # Second call: cache hit — no further LLM calls.
    filter_precursors(target, cands, llm, cache=cache)
    assert len(llm.calls) == 1


def test_filter_empty_input_is_noop():
    llm = FakeLLMClient()  # no responses; should never be called
    kept, dropped, reasons = filter_precursors(
        target_smiles="c1ccc(NC)cc1",
        candidates=[],
        llm_client=llm,
    )
    assert kept == []
    assert dropped == []
    assert reasons == []
    assert llm.calls == []
