"""Per-step enzymatic template provenance survives score -> save."""

from __future__ import annotations

import json

from core.agent import RetrosynthesisAgent
from tools.scratchpad import RouteEntry, SaveBestRouteTool, ScratchpadStore
from tools.verifier.score_route import RouteVerifierTool


class _TemplateResults:
    def __init__(self, payloads):
        self._payloads = payloads

    def iter_by_tool(self, tool_name):
        assert tool_name == "suggest_templates"
        return iter((f"suggest-{i}", payload) for i, payload in enumerate(self._payloads))


def _score_tool(monkeypatch, payloads):
    tool = RouteVerifierTool.__new__(RouteVerifierTool)
    tool.weights = {"feasibility": 1.0, "cost": 0.0, "greenness": 0.0}
    tool.pillar_parallel = False
    tool._result_store = _TemplateResults(payloads)
    monkeypatch.setattr(
        tool,
        "_score_feasibility",
        lambda steps, notes: {"score": 0.8, "summary": {"score": 0.8}},
    )
    monkeypatch.setattr(
        tool,
        "_score_cost",
        lambda smiles, steps, notes: {"score": 0.5, "summary": {"score": 0.5}},
    )
    monkeypatch.setattr(
        tool,
        "_score_greenness",
        lambda steps, notes: {"score": 0.5, "summary": {"score": 0.5}},
    )
    return tool


def test_score_route_surfaces_exact_template_match(monkeypatch):
    tool = _score_tool(monkeypatch, [{
        "product": "CCO",
        "suggestions": [
            {
                "reactants_smiles": "CC",
                "template_source": "bkms",
                "template_kind": "enzymatic",
            },
            {
                "reactants_smiles": "CC=O",
                "template_source": "reaxys_enzyme",
                "template_kind": "enzymatic",
            },
        ],
    }])

    out = json.loads(tool.execute(
        smiles="CCO",
        route_steps=[{"product": "CCO", "reactants": ["CC=O"]}],
    ))

    assert out["step_provenance"] == [{
        "template_source": "reaxys_enzyme",
        "template_kind": "enzymatic",
        "rank": 2,
    }]


def test_score_route_uses_null_when_reactants_do_not_match(monkeypatch):
    tool = _score_tool(monkeypatch, [{
        "product": "CCO",
        "suggestions": [{
            "reactants_smiles": "CC",
            "template_source": "bkms",
            "template_kind": "enzymatic",
        }],
    }])

    out = json.loads(tool.execute(
        smiles="CCO",
        route_steps=[{"product": "CCO", "reactants": ["CC=O"]}],
    ))

    assert out["step_provenance"] == [None]


def test_save_best_route_persists_step_provenance():
    store = ScratchpadStore(max_routes=3)
    tool = SaveBestRouteTool(store)
    provenance = [{
        "template_source": "reaxys_enzyme",
        "template_kind": "enzymatic",
        "rank": 2,
    }]

    out = json.loads(tool.execute(
        label="enzyme-route",
        steps=["CCO>>CC=O"],
        total_score=0.8,
        step_provenance=provenance,
    ))

    assert out["ok"] is True
    saved = store.best()
    assert saved is not None
    assert saved.step_provenance == provenance
    assert RouteEntry.from_dict(saved.to_dict()).step_provenance == provenance


class _CapturingSaveTool:
    def __init__(self):
        self.kwargs = None

    def execute(self, **kwargs):
        self.kwargs = kwargs
        return json.dumps({"ok": True})


def test_auto_capture_forwards_score_route_step_provenance():
    save_tool = _CapturingSaveTool()
    agent = RetrosynthesisAgent.__new__(RetrosynthesisAgent)
    agent.tools = {"save_best_route": save_tool}
    agent.trace = lambda *args, **kwargs: None
    provenance = [{
        "template_source": "reaxys_enzyme",
        "template_kind": "enzymatic",
        "rank": 1,
    }]

    ok = agent._auto_capture_scored_route(
        tool_call_id="call-score-12345678",
        arguments={
            "route_steps": [{"product": "CCO", "reactants": ["CC=O"]}],
        },
        result={
            "total_score": 0.8,
            "route_grounding": {"checked": True, "ok": True},
            "step_provenance": provenance,
        },
    )

    assert ok is True
    assert save_tool.kwargs["step_provenance"] == provenance
