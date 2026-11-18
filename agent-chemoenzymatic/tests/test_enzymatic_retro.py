"""Contract tests for the explicit enzymatic retrosynthesis builtin."""

from __future__ import annotations

import json

from core.config import load_config
from tools.base import ToolRegistry
from tools.builtins import register_all_builtins
from tools.builtins.enzymatic_retro import EnzymaticRetroTool
from tools.builtins.suggest_templates import SuggestTemplatesTool
from tools.verifier.score_route import RouteVerifierTool


def test_wrapper_forces_enzymatic_kind_even_if_caller_attempts_override(monkeypatch):
    captured = {}
    delegated = {
        "product": "CC(O)CN",
        "suggestions": [{
            "template_source": "bkms",
            "template_kind": "enzymatic",
            "reactants": ["CC(=O)CN"],
            "reactants_smiles": "CC(=O)CN",
        }],
    }

    def fake_execute(self, **kwargs):
        captured.update(kwargs)
        return json.dumps(delegated)

    monkeypatch.setattr(SuggestTemplatesTool, "execute", fake_execute)

    out = json.loads(EnzymaticRetroTool().execute(
        product_smiles="CC(O)CN",
        top_k=7,
        max_templates_to_try=321,
        kind="organic",
    ))

    assert captured == {
        "product_smiles": "CC(O)CN",
        "kind": "enzymatic",
        "top_k": 7,
        "max_templates_to_try": 321,
    }
    assert out == delegated
    suggestion = out["suggestions"][0]
    assert suggestion["template_source"] == "bkms"
    assert suggestion["template_kind"] == "enzymatic"
    assert suggestion["reactants"] == ["CC(=O)CN"]
    assert not any(key.startswith("enzyme") for key in suggestion)


def test_registry_registers_enzymatic_retro_when_enabled():
    registry = ToolRegistry()
    register_all_builtins(
        registry,
        config={"enzymatic_retro": {"enabled": True}},
    )
    assert "enzymatic_retro" in registry.tool_names


def test_registry_omits_enzymatic_retro_when_disabled():
    registry = ToolRegistry()
    register_all_builtins(
        registry,
        config={"enzymatic_retro": {"enabled": False}},
    )
    assert "enzymatic_retro" not in registry.tool_names


def test_enzymatic_retro_is_enabled_by_default():
    assert load_config()["enzymatic_retro"]["enabled"] is True


class _ToolResults:
    def __init__(self, payloads_by_tool):
        self._payloads_by_tool = payloads_by_tool

    def iter_by_tool(self, tool_name):
        payloads = self._payloads_by_tool.get(tool_name, [])
        return iter(
            (f"{tool_name}-{i}", payload)
            for i, payload in enumerate(payloads)
        )


def test_score_route_finds_exact_provenance_from_enzymatic_retro(monkeypatch):
    tool = RouteVerifierTool.__new__(RouteVerifierTool)
    tool.weights = {"feasibility": 1.0, "cost": 0.0, "greenness": 0.0}
    tool.pillar_parallel = False
    tool._result_store = _ToolResults({
        "enzymatic_retro": [{
            "product": "CC(O)CN",
            "suggestions": [
                {
                    "reactants_smiles": "CC(N)CN",
                    "template_source": "brenda",
                    "template_kind": "enzymatic",
                },
                {
                    "reactants": ["CC(=O)CN"],
                    "template_source": "bkms",
                    "template_kind": "enzymatic",
                },
            ],
        }],
    })
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

    out = json.loads(tool.execute(
        smiles="CC(O)CN",
        route_steps=[{
            "product": "CC(O)CN",
            "reactants": ["CC(=O)CN"],
        }],
    ))

    assert out["step_provenance"] == [{
        "template_source": "bkms",
        "template_kind": "enzymatic",
        "rank": 2,
    }]
