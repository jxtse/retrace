"""Bail-out return contract: success must mean the SAME thing on the crash
path as on the happy path.

The native loop bails via _fail_native_with_best_so_far on an unrecoverable
LLM error. The old return did `success=bool(best_dict)` with
`route_text=best_dict.get("route_text","")` — but RouteEntry.to_dict() has no
route_text key, so a "successful" bail-out shipped an EMPTY route and skipped
the buyability gate (the bailout-success-flag mislabel the project memory
warns about). These tests pin the fixed contract: success requires a saved
route AND a synthesised route_text AND all-leaves-buyable.
"""
import types

from core.agent import RetrosynthesisAgent
from tools.scratchpad import RouteEntry


class _FakeStore:
    def __init__(self, entry):
        self._entry = entry
        self._all = [entry] if entry is not None else []

    def best(self):
        return self._entry

    def list(self):
        return self._all

    def all(self):
        return self._all


class _FakeBuyTool:
    """check_buyability stub: buyable iff SMILES in `buyable_set`."""

    def __init__(self, buyable_set):
        self._buyable = buyable_set

    def execute(self, smiles_list=None, **kw):
        import json
        results = [{"input_smiles": s, "buyable": s in self._buyable}
                   for s in (smiles_list or [])]
        return json.dumps({"results": results})


class _FakeTools:
    def __init__(self, buy_tool):
        self._buy = buy_tool

    def get(self, name):
        return self._buy if name == "check_buyability" else None


def _make_agent(entry, buyable_set):
    a = RetrosynthesisAgent.__new__(RetrosynthesisAgent)
    a.scratchpad_store = _FakeStore(entry)
    a.tools = _FakeTools(_FakeBuyTool(buyable_set))
    a._buyability_checked = {}
    a.trace = lambda event, **kw: None
    return a


def _bail(agent):
    return agent._fail_native_with_best_so_far(
        target_smiles="CCO", iteration=7, t0=0.0,
        error="boom", messages=[{"role": "user", "content": "x"}],
    )


def test_bailout_success_requires_buyable_leaves():
    # step is retrosynthetic: product>>reactants. Product CCO, leaf CC=O.
    entry = RouteEntry(label="R1", total_score=0.8,
                       steps=["CCO>>CC=O"], starting_materials=["CC=O"])
    agent = _make_agent(entry, buyable_set={"CC=O"})
    out = _bail(agent)
    assert out["success"] is True
    assert out["bailed_out"] is True
    assert out["leaves_buyable"] is True
    # route_text is synthesised from the entry's steps, never empty on success.
    assert "COMPLETE ROUTE FOUND" in out["route_text"]
    assert "CCO>>CC=O" in out["route_text"]


def test_bailout_unbuyable_leaves_is_not_success():
    # step is retrosynthetic: product>>reactants. Product CCO, leaf CC=O.
    entry = RouteEntry(label="R1", total_score=0.8,
                       steps=["CCO>>CC=O"], starting_materials=["CC=O"])
    agent = _make_agent(entry, buyable_set=set())  # nothing buyable
    out = _bail(agent)
    assert out["success"] is False
    assert out["leaves_buyable"] is False
    # the route is still surfaced for inspection even though success=False
    assert out["best_route"] is not None
    assert "COMPLETE ROUTE FOUND" in out["route_text"]


def test_bailout_no_saved_route_is_not_success():
    agent = _make_agent(entry=None, buyable_set=set())
    out = _bail(agent)
    assert out["success"] is False
    assert out["route_text"] == ""
    assert out["best_route"] is None


# ── Structural-leaf buyability gate (audit #3) ────────────────────────────────


def test_gate_uses_structural_leaves_not_declared_field():
    # The route's true leaf (from steps: product CCO >> reactant CC(=O)Cl) is
    # CC(=O)Cl, which is NOT buyable. The LLM mis-declares a trivially-buyable
    # block (CCO) as the starting material. The gate must derive CC(=O)Cl from
    # the steps and FAIL, ignoring the declared field.
    from rdkit import Chem
    canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
    entry = RouteEntry(label="R1", total_score=0.9,
                       steps=["CCO>>CC(=O)Cl"],
                       starting_materials=["CCO"])           # mis-declared
    agent = _make_agent(entry, buyable_set={canon("CCO")})   # only CCO buyable
    assert agent._verify_route_leaves_buyable(entry) is False


def test_gate_passes_when_structural_leaves_buyable():
    from rdkit import Chem
    canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
    entry = RouteEntry(label="R1", total_score=0.9,
                       steps=["CCO>>CC(=O)Cl"],
                       starting_materials=["CCO"])  # declared field irrelevant
    agent = _make_agent(entry, buyable_set={canon("CC(=O)Cl")})
    assert agent._verify_route_leaves_buyable(entry) is True


def test_gate_falls_back_to_declared_when_steps_unparseable():
    from rdkit import Chem
    canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
    # Steps can't be parsed (placeholder) -> derivation returns None -> gate
    # falls back to the declared starting_materials.
    entry = RouteEntry(label="R1", total_score=0.9,
                       steps=["target>>intermediate.CCO"],
                       starting_materials=["CCO"])
    agent = _make_agent(entry, buyable_set={canon("CCO")})
    assert agent._verify_route_leaves_buyable(entry) is True


def test_terminal_policy_rejects_route_that_expands_a_known_buyable_molecule():
    """A buyable molecule must be terminal everywhere in the run."""
    from rdkit import Chem
    canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
    entry = RouteEntry(
        label="R2", total_score=0.9,
        steps=["CCO>>CC=O", "CC=O>>C"], starting_materials=["C"],
    )
    agent = _make_agent(entry, buyable_set={canon("C")})
    agent._buyability_checked[canon("CC=O")] = True
    assert agent._verify_route_leaves_buyable(entry) is False


def test_terminal_policy_allows_internal_molecule_known_nonbuyable():
    """Known non-buyable intermediates may be expanded normally."""
    from rdkit import Chem
    canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
    entry = RouteEntry(
        label="R2", total_score=0.9,
        steps=["CCO>>CC=O", "CC=O>>C"], starting_materials=["C"],
    )
    agent = _make_agent(entry, buyable_set={canon("C")})
    agent._buyability_checked[canon("CC=O")] = False
    assert agent._verify_route_leaves_buyable(entry) is True
