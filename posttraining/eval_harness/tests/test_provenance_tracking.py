"""Tests for provenance attribution on `save_best_route`.

Covers:
- `RouteEntry` carries `provenance` + `rag_source_ids` and roundtrips
  through `to_dict()`.
- `SaveBestRouteTool.execute` honors the internal `_provenance` /
  `_rag_source_ids` kwargs (not in the JSON schema).
- Default for legacy callers (no kwarg) is `"unknown"`.
- `_compute_provenance` priority ladder: rag_anchored > neural_consensus
  > template_suggested > zero_shot.
"""

from __future__ import annotations

import json

import pytest

from tools.scratchpad import RouteEntry, SaveBestRouteTool, ScratchpadStore


def _make_store():
    return ScratchpadStore(max_routes=5)


VALID_STEP = "CCO>>CC.O"


def test_route_entry_defaults():
    e = RouteEntry(label="r1", total_score=0.8, steps=["A>>B"])
    assert e.provenance == "unknown"
    assert e.rag_source_ids is None
    d = e.to_dict()
    assert d["provenance"] == "unknown"
    assert d["rag_source_ids"] is None


def test_route_entry_with_provenance_roundtrips():
    e = RouteEntry(
        label="r2", total_score=0.9, steps=["X>>Y"],
        provenance="rag_anchored",
        rag_source_ids=["uspto_000001", "uspto_000042"],
    )
    d = e.to_dict()
    assert d["provenance"] == "rag_anchored"
    assert d["rag_source_ids"] == ["uspto_000001", "uspto_000042"]


def test_save_best_route_accepts_internal_kwargs():
    store = _make_store()
    tool = SaveBestRouteTool(store)
    tool.execute(
        label="r3", steps=[VALID_STEP], total_score=0.75,
        _provenance="rag_anchored",
        _rag_source_ids=["uspto_000007"],
    )
    saved = store.all()
    assert any(e.provenance == "rag_anchored" for e in saved)
    e = next(e for e in saved if e.provenance == "rag_anchored")
    assert e.rag_source_ids == ["uspto_000007"]


def test_save_best_route_default_unknown_provenance():
    store = _make_store()
    tool = SaveBestRouteTool(store)
    tool.execute(label="r4", steps=[VALID_STEP], total_score=0.5)
    saved = store.all()
    assert any(e.provenance == "unknown" for e in saved)


def test_save_best_route_excludes_provenance_from_json_schema():
    store = _make_store()
    tool = SaveBestRouteTool(store)
    schema_props = tool.parameters["properties"]
    assert "_provenance" not in schema_props
    assert "_rag_source_ids" not in schema_props
    assert "provenance" not in schema_props


def test_dedup_preserves_known_provenance():
    """If the first save tagged 'rag_anchored' and a second identical save
    comes in as 'unknown', the existing provenance must not be demoted."""
    store = _make_store()
    tool = SaveBestRouteTool(store)
    tool.execute(label="r5", steps=[VALID_STEP], total_score=0.6,
                 _provenance="rag_anchored", _rag_source_ids=["uspto_x"])
    tool.execute(label="r5b", steps=[VALID_STEP], total_score=0.6)
    matching = [e for e in store.all() if e.steps == [VALID_STEP]]
    assert any(e.provenance == "rag_anchored" for e in matching)


# ---- _compute_provenance priority --------------------------------------

class _FakeAgent:
    """Minimal stub exposing exactly the attributes _compute_provenance
    reads — lets us exercise the priority ladder without instantiating
    the full agent (which requires an LLM client, registry, etc.)."""

    def __init__(self, recent_turns, last_recall_payload=None, target_tools=None):
        from core.agent import RetrosynthesisAgent
        self._recent_tool_turns = recent_turns
        self._last_recall_payload = last_recall_payload
        self._target_tools_called_set = set(target_tools or [])
        # Bind the real method onto this stub.
        self._compute_provenance = RetrosynthesisAgent._compute_provenance.__get__(self)


def test_compute_provenance_rag_wins():
    payload = {"examples": [
        {"route_id": "u_001"}, {"route_id": "u_002"},
    ]}
    a = _FakeAgent(
        recent_turns=[["recall_similar_routes", "neural_retro", "suggest_templates"]],
        last_recall_payload=payload,
    )
    prov, ids = a._compute_provenance()
    assert prov == "rag_anchored"
    assert ids == ["u_001", "u_002"]


def test_compute_provenance_rag_needs_payload():
    a = _FakeAgent(
        recent_turns=[["recall_similar_routes", "neural_retro"]],
        last_recall_payload=None,
    )
    prov, ids = a._compute_provenance()
    # No cached payload -> fall through to neural_consensus.
    assert prov == "neural_consensus"
    assert ids is None


def test_compute_provenance_neural_over_template():
    a = _FakeAgent(
        recent_turns=[["neural_retro", "suggest_templates"]],
    )
    prov, _ = a._compute_provenance()
    assert prov == "neural_consensus"


def test_compute_provenance_template():
    a = _FakeAgent(recent_turns=[["suggest_templates", "apply_template"]])
    prov, _ = a._compute_provenance()
    assert prov == "template_suggested"


def test_compute_provenance_zero_shot():
    a = _FakeAgent(recent_turns=[["analyze_molecule", "check_buyability"]])
    prov, _ = a._compute_provenance()
    assert prov == "zero_shot"


def test_compute_provenance_uses_prefetch_when_window_empty():
    a = _FakeAgent(
        recent_turns=[],
        target_tools={"suggest_templates", "neural_retro"},
    )
    prov, _ = a._compute_provenance()
    # Window empty -> prefetch falls back; neural beats template by priority.
    assert prov == "neural_consensus"
