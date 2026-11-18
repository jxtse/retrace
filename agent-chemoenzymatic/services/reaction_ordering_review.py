"""Post-run reaction-ordering reviewer helpers."""

from __future__ import annotations

import json
from typing import Any


VALID_ACTIONS = {"accept", "reorder", "revise"}


def _strip_fence(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_text(text: str) -> str:
    candidate = _strip_fence(text)
    if candidate.startswith("{") and candidate.endswith("}"):
        return candidate
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        return candidate[start:end + 1]
    return candidate


def parse_reviewer_decision(text: str) -> dict[str, Any]:
    """Parse and validate the reviewer JSON contract.

    Returns a dict whose ``status`` is either ``ok`` or ``failed``. Failed
    parses are deliberately data-shaped so callers can persist the raw output
    without raising out of the main Agent result path.
    """
    raw = text or ""
    try:
        parsed = json.loads(_extract_json_text(raw))
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"invalid reviewer JSON: {exc}",
            "raw_output": raw,
        }
    if not isinstance(parsed, dict):
        return {
            "status": "failed",
            "error": "reviewer JSON must be an object",
            "raw_output": raw,
        }

    action = str(parsed.get("action", "")).strip().lower()
    if action not in VALID_ACTIONS:
        return {
            "status": "failed",
            "error": f"unknown reviewer action: {action!r}",
            "raw_output": raw,
            "parsed": parsed,
        }

    summary = str(parsed.get("summary") or "").strip()
    issues = parsed.get("issues") if isinstance(parsed.get("issues"), list) else []
    decision: dict[str, Any] = {
        "status": "ok",
        "action": action,
        "summary": summary,
        "issues": issues,
        "raw_output": raw,
        "parsed": parsed,
    }

    if action == "reorder":
        optimized = str(
            parsed.get("optimized_route_text") or parsed.get("route_text") or ""
        ).strip()
        if not optimized:
            return {
                "status": "failed",
                "error": "reorder action requires optimized_route_text",
                "raw_output": raw,
                "parsed": parsed,
            }
        decision["optimized_route_text"] = optimized
    elif action == "revise":
        feedback = str(parsed.get("feedback") or "").strip()
        if not feedback:
            return {
                "status": "failed",
                "error": "revise action requires feedback",
                "raw_output": raw,
                "parsed": parsed,
            }
        decision["feedback"] = feedback

    return decision
