"""Integration tests for the recall_similar_routes wiring into the agent.

Two surfaces:
- `register_all_builtins(precedent_db=...)` registers the tool iff a DB
  is provided. With `precedent_db=None`, the tool is absent and the
  rest of the registry is unaffected.
- The full `build_agent(config)` factory tolerates `precedent_recall:
  enabled: false` (production default) — no DB load, no extra tool.

We do NOT spin a live LLM here — the agent loop itself is exercised
elsewhere. These tests just guard the wiring contract.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("rdkit")

from services.precedent_db import (  # noqa: E402
    PrecedentDB,
    PrecedentDBConfig,
    maybe_load,
)
from tools.base import ToolRegistry  # noqa: E402
from tools.builtins import register_all_builtins  # noqa: E402

from tests.test_precedent_db import SAMPLE_SMILES, _write_fixture  # noqa: E402


@pytest.fixture
def db(tmp_path):
    _write_fixture(str(tmp_path), SAMPLE_SMILES)
    return PrecedentDB(db_root=str(tmp_path), config=PrecedentDBConfig(top_n_for_mmr=15))


def test_tool_absent_when_db_is_none():
    reg = ToolRegistry()
    register_all_builtins(reg, config={}, precedent_db=None)
    assert "recall_similar_routes" not in reg.tool_names


def test_tool_registered_when_db_present(db):
    reg = ToolRegistry()
    register_all_builtins(reg, config={}, precedent_db=db)
    assert "recall_similar_routes" in reg.tool_names
    # And it actually executes through the registry path.
    tool = reg.get("recall_similar_routes")
    out = json.loads(tool.execute(query_smiles="c1ccccc1", k=2, min_similarity=0.0))
    assert "examples" in out
    assert out["query_canonical"]


def test_maybe_load_returns_none_when_disabled():
    db = maybe_load({"enabled": False}, base_dir="/")
    assert db is None


def test_maybe_load_returns_none_when_db_root_missing():
    db = maybe_load(
        {"enabled": True, "db_root": "/nonexistent/precedent-db-xxx-not-here"},
        base_dir="/",
    )
    assert db is None


def test_maybe_load_loads_when_present(tmp_path):
    _write_fixture(str(tmp_path), SAMPLE_SMILES)
    db = maybe_load(
        {"enabled": True, "db_root": str(tmp_path)},
        base_dir="/",
    )
    assert db is not None
    assert db.n_rows == len(SAMPLE_SMILES)


def test_build_agent_skips_recall_tool_when_disabled():
    """The production default has precedent_recall.enabled=False, so the
    factory must not crash and the tool must not appear."""
    pytest.importorskip("yaml")
    import yaml

    # Build a minimal config that build_agent will accept. We can't run
    # the full thing without an LLM endpoint, so we go straight to
    # `register_all_builtins` with the same precedent_db plumbing
    # `build_agent` performs.
    config = {
        "precedent_recall": {"enabled": False},
    }
    db = maybe_load(config.get("precedent_recall"), base_dir=".")
    assert db is None
    reg = ToolRegistry()
    register_all_builtins(reg, config=config, precedent_db=db)
    assert "recall_similar_routes" not in reg.tool_names


def test_recall_tool_can_be_called_via_registry(db):
    """End-to-end on the tool surface — same call path the agent loop takes."""
    reg = ToolRegistry()
    register_all_builtins(reg, config={}, precedent_db=db)
    tool = reg.get("recall_similar_routes")
    out_str = tool.execute(query_smiles=SAMPLE_SMILES[0], k=3, min_similarity=0.0)
    out = json.loads(out_str)
    assert out["query_canonical"]
    assert isinstance(out["examples"], list)
    # Schema fields we promised.
    assert "confidence" in out
    assert "max_similarity" in out
    assert "advisory" in out


def test_system_prompt_block_empty_when_disabled(tmp_path, monkeypatch):
    """When precedent_recall is disabled the rendered system prompt MUST NOT
    mention recall_similar_routes or the prefetch claim. Otherwise the
    LLM gets told to use a tool that is not registered."""
    from core.agent import RetrosynthesisAgent
    from core.prompt_loader import PromptLoader
    # Render directly through the same machinery the agent uses.
    prompts_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "prompts"
    )
    loader = PromptLoader(prompts_dir)
    core_template = loader.render("core")
    rendered = core_template.format(
        target_smiles="CCO",
        max_depth=10,
        max_iterations=40,
        precedent_recall_block="",  # what __init__ produces when disabled
        reaction_ordering_block="",
    )
    assert "recall_similar_routes" not in rendered, (
        "system prompt mentions recall_similar_routes while feature disabled"
    )
    # The 'precedent database' phrasing comes ONLY from the recall block —
    # disconnection-rule's prefetch language is about template+neural, not RAG.
    assert "precedent database" not in rendered, (
        "system prompt still references precedent database while feature disabled"
    )


def test_system_prompt_block_present_when_enabled(tmp_path):
    """When enabled, the rendered system prompt must include the recall
    guidance — both the tool name and the 'prefetched' claim."""
    from core.prompt_loader import PromptLoader
    prompts_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "prompts"
    )
    # Section file lives under prompts/sections/ — read it the same way
    # agent.__init__ does (file read, not through _PromptLoader).
    section_path = os.path.join(
        prompts_dir, "sections", "core_precedent_recall.md"
    )
    with open(section_path) as f:
        block = f.read().rstrip("\n")
    assert "recall_similar_routes" in block
    assert "prefetched" in block.lower()
    # And rendered into core.md via the {precedent_recall_block} placeholder.
    loader = PromptLoader(prompts_dir)
    core_template = loader.render("core")
    rendered = core_template.format(
        target_smiles="CCO",
        max_depth=10,
        max_iterations=40,
        precedent_recall_block=block,
        reaction_ordering_block="",
    )
    assert "recall_similar_routes" in rendered
    assert "prefetched" in rendered.lower()


def test_reaction_ordering_block_renders_before_consensus_shortcut():
    from core.prompt_loader import PromptLoader
    prompts_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "prompts"
    )
    section_path = os.path.join(
        prompts_dir, "sections", "core_reaction_ordering.md"
    )
    with open(section_path) as f:
        ordering_block = f.read().rstrip("\n")

    loader = PromptLoader(prompts_dir)
    rendered = loader.render("core").format(
        target_smiles="CCO",
        max_depth=10,
        max_iterations=40,
        precedent_recall_block="PRECEDENT_BLOCK",
        reaction_ordering_block=ordering_block,
    )

    assert "PRECEDENT_BLOCK" in rendered
    assert "Reaction ordering strategy" in rendered
    assert rendered.index("PRECEDENT_BLOCK") < rendered.index(
        "Reaction ordering strategy"
    )
    assert rendered.index("Reaction ordering strategy") < rendered.index(
        "Consensus shortcut"
    )
