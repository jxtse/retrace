"""Regression for the per-pillar average reader in the eval summary.

score_route emits FLAT pillar keys (feasibility_score / cost_score /
greenness_score). The summary's _avg_pillar previously read a NESTED shape
({name: {"score": ..}}) that score_route never produces, so the per-pillar
benchmark averages were silently None on every run. These tests pin the flat
read (with a nested fallback for legacy payloads).
"""
import pytest

from core.evaluation import _avg_pillar_score, _pillar_score


def test_pillar_score_reads_flat_keys():
    scored = {"total_score": 0.7, "feasibility_score": 0.8,
              "cost_score": 0.5, "greenness_score": 0.6}
    assert _pillar_score(scored, "feasibility") == pytest.approx(0.8)
    assert _pillar_score(scored, "cost") == pytest.approx(0.5)
    assert _pillar_score(scored, "greenness") == pytest.approx(0.6)


def test_pillar_score_nested_fallback():
    scored = {"feasibility": {"score": 0.42}}
    assert _pillar_score(scored, "feasibility") == pytest.approx(0.42)


def test_pillar_score_absent_is_none():
    assert _pillar_score({"total_score": 0.7}, "feasibility") is None


def test_avg_pillar_non_none_on_flat_payloads():
    scored_list = [
        {"total_score": 0.7, "feasibility_score": 0.8, "cost_score": 0.5, "greenness_score": 0.6},
        {"total_score": 0.6, "feasibility_score": 0.6, "cost_score": 0.4, "greenness_score": 0.5},
    ]
    assert _avg_pillar_score(scored_list, "feasibility") == pytest.approx(0.7)
    assert _avg_pillar_score(scored_list, "cost") == pytest.approx(0.45)
    assert _avg_pillar_score(scored_list, "greenness") == pytest.approx(0.55)


def test_avg_pillar_empty_is_none():
    assert _avg_pillar_score([], "feasibility") is None
    assert _avg_pillar_score([{"total_score": 0.7}], "feasibility") is None
