"""Buyability-tool advice surface tests.

Note: the iteration-billing rule (formerly tested here against the old
`exempt`-set blacklist) moved to `tests/test_iteration_billable.py`
when the meter switched to a `suggest_templates` + `neural_retro`
whitelist (Task 9). The check_buyability advice tests below remain
because they verify orthogonal tool-side behaviour.
"""
import pytest


import json

from tools.check_buyability import CheckBuyabilityTool


def test_check_buyability_schema_avoids_top_level_combinators():
    assert "anyOf" not in CheckBuyabilityTool.parameters
    assert "oneOf" not in CheckBuyabilityTool.parameters
    assert "allOf" not in CheckBuyabilityTool.parameters


def test_check_buyability_still_validates_missing_runtime_input(tmp_path):
    stock = tmp_path / "stock.csv"
    stock.write_text("smiles\nCCO\n")
    tool = CheckBuyabilityTool({"test_stock": str(stock)})

    out = json.loads(tool.execute())

    assert out["error"].startswith("Missing argument:")


def test_check_buyability_advice_appears_after_repeated_singletons(tmp_path):
    # Build a minimal stock so the constructor succeeds.
    stock = tmp_path / "stock.csv"
    stock.write_text("smiles\nCCO\n")
    tool = CheckBuyabilityTool({"test_stock": str(stock)})

    # Three sequential single-SMILES calls within the recent window.
    out1 = json.loads(tool.execute(smiles="CCO"))
    out2 = json.loads(tool.execute(smiles="CCN"))
    out3 = json.loads(tool.execute(smiles="CCC"))
    # First two: no advice yet (counter ramps).
    assert "_advice" not in out1
    # Third: advice should kick in (≥ 2 prior singletons in window).
    assert "_advice" in out3
    assert "smiles_list" in out3["_advice"]


def test_check_buyability_batch_call_clears_advice(tmp_path):
    stock = tmp_path / "stock.csv"
    stock.write_text("smiles\nCCO\n")
    tool = CheckBuyabilityTool({"test_stock": str(stock)})

    # Two singletons, then a batch — batch must NOT add the advice.
    json.loads(tool.execute(smiles="CCO"))
    json.loads(tool.execute(smiles="CCN"))
    batch_out = json.loads(tool.execute(smiles_list=["CCC", "CCCC"]))
    assert "_advice" not in batch_out


def test_check_buyability_merges_singleton_without_crashing_on_bad_batch_item(tmp_path):
    stock = tmp_path / "stock.csv"
    stock.write_text("smiles\nCCO\n")
    tool = CheckBuyabilityTool({"test_stock": str(stock)})

    out = json.loads(tool.execute(smiles="CCO", smiles_list=[["bad"]]))

    assert out["n"] == 2
    assert out["results"][0]["error"] == "not a string"
    assert out["results"][1]["input_smiles"] == "CCO"
