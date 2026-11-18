"""save_best_route hard-rejects an ungrounded route (audit #2, option A).

The grounding gate only fires when a target has been set via set_target()
(threaded in per-run from agent.run()). Without it the gate is dormant
(legacy callers). Steps are retrosynthetic: product>>reactants.
"""
import json

from rdkit import Chem

from tools.scratchpad import ScratchpadStore, SaveBestRouteTool

canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))


def _tool():
    # No result_store: score is clamped to 0.0 but the save still archives,
    # so the only thing that can REJECT here is the grounding gate.
    return SaveBestRouteTool(ScratchpadStore(max_routes=10))


def test_grounded_route_saves():
    t = _tool()
    t.set_target("CCCCO")
    out = json.loads(t.execute(
        label="R1",
        steps=["CCCCO>>CCCC.OCC", "CCCC>>CCCCBr"],
        total_score=0.8,
        starting_materials=["OCC", "CCCCBr"],
    ))
    assert out.get("status") != "ungrounded_route_rejected"


def test_root_not_target_rejected():
    t = _tool()
    t.set_target("CCCCO")  # route never produces CCCCO
    out = json.loads(t.execute(
        label="R1",
        steps=["CCCC>>CCCCBr"],
        total_score=0.8,
        starting_materials=["CCCCBr"],
    ))
    assert out["status"] == "ungrounded_route_rejected"
    assert out["root_matches_target"] is False


def test_dangling_intermediate_rejected():
    t = _tool()
    t.set_target("CCCCO")
    out = json.loads(t.execute(
        label="R1",
        steps=["CCCCO>>CCCC.OCC", "CCCl>>CCBr"],  # CCCl produced, never consumed
        total_score=0.8,
        starting_materials=["OCC", "CCBr", "CCCC"],
    ))
    assert out["status"] == "ungrounded_route_rejected"
    assert out["n_dangling"] == 1


def test_gate_dormant_without_target():
    t = _tool()  # set_target never called
    out = json.loads(t.execute(
        label="R1",
        steps=["CCCC>>CCCCBr"],  # would be ungrounded vs CCCCO
        total_score=0.8,
        starting_materials=["CCCCBr"],
    ))
    assert out.get("status") != "ungrounded_route_rejected"
