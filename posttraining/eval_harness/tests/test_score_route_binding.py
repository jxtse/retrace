"""Regression test for the score-gaming gap (audit finding #1).

save_best_route's score cross-check must verify the cited
score_route_result_id actually scored the SAME route being saved — not
just that the score value matches. Otherwise an agent can score route A
(high score), capture its _result_id, then save a different route B citing
A's id + A's score, and the value-only check would pass.
"""
from __future__ import annotations
import json

from tools.scratchpad import ScratchpadStore, SaveBestRouteTool
from tools._route_fp import route_fingerprint


class FakeResultStore:
    """Minimal stand-in: maps result_id -> score_route blob."""
    def __init__(self):
        self._blobs = {}

    def add_score_route(self, result_id, steps_retro, total_score, with_fp=True):
        # steps_retro: list of "product>>reactants" strings (retro direction)
        payload = {"total_score": total_score, "n_steps": len(steps_retro)}
        if with_fp:
            payload["route_fp"] = route_fingerprint(steps_retro)
        self._blobs[result_id] = {"tool": "score_route", "content": payload}

    def load(self, result_id):
        if result_id not in self._blobs:
            return {"error": f"no such result {result_id}"}
        return self._blobs[result_id]


def _make_tool():
    store = ScratchpadStore(max_routes=10)
    rs = FakeResultStore()
    tool = SaveBestRouteTool(store, result_store=rs)
    return store, rs, tool


ROUTE_A = ["CC(=O)Oc1ccccc1C(=O)O>>Oc1ccccc1C(=O)O.CC(=O)Cl"]
ROUTE_B = ["CCOC(=O)c1ccccc1>>CCO.OC(=O)c1ccccc1"]


def test_matching_route_and_score_is_accepted():
    store, rs, tool = _make_tool()
    rs.add_score_route("res_A", ROUTE_A, 0.85)
    out = json.loads(tool.execute(
        label="A", steps=ROUTE_A, total_score=0.85, score_route_result_id="res_A"))
    assert out.get("ok") or out.get("status") in ("saved", "updated", None), out
    assert "error" not in out, out
    assert len(store.list()) == 1


def test_reused_result_id_for_different_route_is_rejected():
    """The core gaming scenario: score A, save B citing A's id + A's score."""
    store, rs, tool = _make_tool()
    rs.add_score_route("res_A", ROUTE_A, 0.85)
    # Save route B but cite route A's result_id and its 0.85 score.
    out = json.loads(tool.execute(
        label="B", steps=ROUTE_B, total_score=0.85, score_route_result_id="res_A"))
    assert "error" in out, f"expected route-mismatch rejection, got {out}"
    assert "route mismatch" in out["error"].lower(), out
    assert len(store.list()) == 0, "gamed route must not be saved"


def test_legacy_payload_without_fp_falls_back_to_nsteps():
    """Pre-fingerprint payloads still catch obvious step-count mismatches."""
    store, rs, tool = _make_tool()
    two_step = [
        "A>>B.C",
        "B>>D.E",
    ]
    rs.add_score_route("res_old", two_step, 0.80, with_fp=False)
    # Cite the 2-step result but save a 1-step route at the same score.
    out = json.loads(tool.execute(
        label="X", steps=ROUTE_A, total_score=0.80, score_route_result_id="res_old"))
    assert "error" in out, f"expected n_steps mismatch rejection, got {out}"
    assert "step" in out["error"].lower() and "mismatch" in out["error"].lower(), out
    assert len(store.list()) == 0
