"""Regression tests for built-in config defaults."""

from __future__ import annotations

from core.config import load_config


def test_precedent_recall_off_by_default():
    cfg = load_config()
    assert cfg["precedent_recall"]["enabled"] is False


def test_precedent_db_root_is_not_machine_specific():
    cfg = load_config()
    db_root = cfg["precedent_recall"].get("db_root", "")
    # A shipped default must stay relative: no machine- or user-specific root.
    assert not db_root.startswith("/")


def test_legacy_scoring_config_is_absent():
    cfg = load_config()
    assert "scoring" not in cfg
