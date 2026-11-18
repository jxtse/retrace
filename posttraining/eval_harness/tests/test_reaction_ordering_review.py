from __future__ import annotations

import json

from services.reaction_ordering_review import parse_reviewer_decision


def test_parse_accept_decision():
    decision = parse_reviewer_decision(json.dumps({
        "action": "accept",
        "summary": "Ordering is sound.",
    }))

    assert decision["status"] == "ok"
    assert decision["action"] == "accept"
    assert decision["summary"] == "Ordering is sound."


def test_parse_reorder_decision_from_fence():
    decision = parse_reviewer_decision(
        "```json\n"
        + json.dumps({
            "action": "reorder",
            "summary": "Install labile group late.",
            "optimized_route_text": "COMPLETE ROUTE FOUND: better order",
        })
        + "\n```"
    )

    assert decision["status"] == "ok"
    assert decision["action"] == "reorder"
    assert decision["optimized_route_text"].endswith("better order")


def test_parse_revise_decision_requires_feedback():
    missing = parse_reviewer_decision(json.dumps({
        "action": "revise",
        "summary": "Needs more work.",
    }))
    valid = parse_reviewer_decision(json.dumps({
        "action": "revise",
        "summary": "Needs more work.",
        "feedback": "Try an earlier hindered coupling.",
    }))

    assert missing["status"] == "failed"
    assert "feedback" in missing["error"]
    assert valid["status"] == "ok"
    assert valid["action"] == "revise"


def test_parse_invalid_json_fails_softly():
    decision = parse_reviewer_decision("not json")

    assert decision["status"] == "failed"
    assert "invalid reviewer JSON" in decision["error"]
    assert decision["raw_output"] == "not json"
