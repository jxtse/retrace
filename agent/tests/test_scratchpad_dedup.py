"""Backend deduplication for save_best_route.

Two saves with structurally identical routes (same canonical SMILES,
order-insensitive) must result in:
- one entry in the store
- the second save returning status:"duplicate" with duplicate_of pointing
  at the first label.
"""

import json

import pytest


def test_save_duplicate_route_is_rejected():
    from tools.scratchpad import ScratchpadStore, SaveBestRouteTool

    store = ScratchpadStore(max_routes=10)
    tool = SaveBestRouteTool(store)

    steps = ["CCO>>CC.O (reaction: dummy)"]
    r1 = json.loads(tool.execute(label="route-A", steps=steps, total_score=0.7))
    r2 = json.loads(tool.execute(label="route-B", steps=steps, total_score=0.7))

    assert r1.get("status") != "duplicate"
    assert r2.get("status") == "duplicate"
    assert r2.get("duplicate_of") == "route-A"
    assert len(store.list()) == 1


def test_save_reactant_order_does_not_matter():
    """Reactant order in 'A.B' should not be treated as a different route."""
    from tools.scratchpad import ScratchpadStore, SaveBestRouteTool

    store = ScratchpadStore(max_routes=10)
    tool = SaveBestRouteTool(store)

    s1 = ["c1ccccc1Br>>c1ccccc1B(O)O.Br[Cu]"]
    s2 = ["c1ccccc1Br>>Br[Cu].c1ccccc1B(O)O"]
    json.loads(tool.execute(label="suzuki-v1", steps=s1, total_score=0.6))
    r2 = json.loads(tool.execute(label="suzuki-v2", steps=s2, total_score=0.6))

    assert r2.get("status") == "duplicate", (
        f"Reordered reactants should dedup, got: {r2}"
    )
    assert len(store.list()) == 1


def test_distinct_routes_are_both_saved():
    from tools.scratchpad import ScratchpadStore, SaveBestRouteTool

    store = ScratchpadStore(max_routes=10)
    tool = SaveBestRouteTool(store)

    json.loads(tool.execute(
        label="route-A",
        steps=["CCO>>CC.O"],
        total_score=0.5,
    ))
    r2 = json.loads(tool.execute(
        label="route-B",
        steps=["CCC>>CC.C"],
        total_score=0.6,
    ))

    assert r2.get("status") != "duplicate"
    assert len(store.list()) == 2


def test_evicted_route_can_be_resaved_without_duplicate_status():
    """If an entry is evicted by capacity, its fingerprint must also be
    dropped — re-saving the same chemistry should NOT trigger duplicate."""
    from tools.scratchpad import ScratchpadStore, SaveBestRouteTool

    store = ScratchpadStore(max_routes=2)
    tool = SaveBestRouteTool(store)

    # Fill capacity with two high-scoring routes.
    json.loads(tool.execute(label="hi-A", steps=["CCO>>CC.O"], total_score=0.9))
    json.loads(tool.execute(label="hi-B", steps=["CCC>>CC.C"], total_score=0.8))

    # Insert a third route that displaces the lowest-scoring kept one.
    # (max_routes=2; new entry beats hi-B at 0.8.)
    r3 = json.loads(tool.execute(label="hi-C", steps=["CCN>>CC.N"], total_score=0.85))
    assert r3.get("status") != "duplicate"

    # hi-B's fingerprint should be gone with it; re-saving its chemistry
    # under a new label must succeed (and may itself be rejected by score,
    # but NOT marked duplicate).
    r4 = json.loads(tool.execute(label="reborn-B", steps=["CCC>>CC.C"], total_score=0.95))
    assert r4.get("status") != "duplicate", (
        f"Evicted route's fingerprint must be dropped, got: {r4}"
    )


def test_clear_resets_fingerprints():
    """After clear(), saving a previously-saved route must NOT trigger duplicate."""
    from tools.scratchpad import ScratchpadStore, SaveBestRouteTool

    store = ScratchpadStore(max_routes=10)
    tool = SaveBestRouteTool(store)

    json.loads(tool.execute(label="A", steps=["CCO>>CC.O"], total_score=0.7))
    store.clear()
    r = json.loads(tool.execute(label="A2", steps=["CCO>>CC.O"], total_score=0.7))
    assert r.get("status") != "duplicate"
    assert len(store.list()) == 1


def test_save_same_label_updates_in_place():
    """Save low-score draft, then re-save same label/steps with higher
    score → store keeps the higher score, status is 'updated'.

    Regression: the original dedup branch returned ``"duplicate"`` for any
    fingerprint hit, silently keeping the stale low score and breaking the
    legitimate "save draft → rescore → re-save" workflow.
    """
    from tools.scratchpad import ScratchpadStore, SaveBestRouteTool

    store = ScratchpadStore(max_routes=10)
    tool = SaveBestRouteTool(store)

    steps = ["CCO>>CC.O (reaction: dummy)"]
    r1 = json.loads(tool.execute(label="route-A", steps=steps, total_score=0.1))
    r2 = json.loads(tool.execute(label="route-A", steps=steps, total_score=0.9))

    assert r1.get("ok") is True
    assert r2.get("status") == "updated", f"expected 'updated', got {r2}"
    assert len(store.list()) == 1
    assert store.list()[0].total_score == pytest.approx(0.9)


def test_save_same_steps_different_label_still_duplicates():
    """Different-label fingerprint hit must still return 'duplicate' so
    the LLM can't game capacity by re-saving under a new name."""
    from tools.scratchpad import ScratchpadStore, SaveBestRouteTool

    store = ScratchpadStore(max_routes=10)
    tool = SaveBestRouteTool(store)

    steps = ["CCO>>CC.O (reaction: dummy)"]
    json.loads(tool.execute(label="route-A", steps=steps, total_score=0.5))
    r2 = json.loads(tool.execute(label="route-B", steps=steps, total_score=0.9))

    assert r2.get("status") == "duplicate"
    assert r2.get("duplicate_of") == "route-A"
    assert len(store.list()) == 1


def test_save_same_label_update_refreshes_sub_scores_and_notes():
    """The 'updated' branch must refresh sub_scores, notes, and
    starting_materials too — not just total_score."""
    from tools.scratchpad import ScratchpadStore, SaveBestRouteTool

    store = ScratchpadStore(max_routes=10)
    tool = SaveBestRouteTool(store)

    steps = ["CCO>>CC.O (reaction: dummy)"]
    json.loads(tool.execute(
        label="route-A",
        steps=steps,
        total_score=0.1,
        sub_scores={"feasibility": 0.1, "cost": 0.1, "greenness": 0.1},
        starting_materials=["CC.O"],
        notes="initial draft",
    ))
    r2 = json.loads(tool.execute(
        label="route-A",
        steps=steps,
        total_score=0.85,
        sub_scores={"feasibility": 0.9, "cost": 0.8, "greenness": 0.85},
        starting_materials=["CC", "O"],
        notes="re-scored after verification",
    ))
    assert r2.get("status") == "updated"

    entry = store.get("route-A")
    assert entry is not None
    assert entry.total_score == pytest.approx(0.85)
    assert entry.sub_scores.get("feasibility") == pytest.approx(0.9)
    assert entry.notes == "re-scored after verification"
    assert entry.starting_materials == ["CC", "O"]


def test_overwriting_label_with_different_steps_drops_old_fingerprint():
    """When a label is overwritten with a structurally different route,
    the OLD fingerprint must be removed from the dedup index. Otherwise a
    later save of the original chemistry under a new label gets falsely
    flagged ``duplicate_of=<old_label>`` even though that label no longer
    holds those steps — blocking a legitimate distinct route from
    entering the scratchpad and depressing the distinct-route count.

    Reproducer (per the bug report):
      1. save r1 = A>>B.C
      2. save r1 = X>>Y.Z   (label r1 now holds X>>Y.Z)
      3. save r2 = A>>B.C   (must NOT report duplicate_of=r1)
    """
    from tools.scratchpad import ScratchpadStore, SaveBestRouteTool

    store = ScratchpadStore(max_routes=10)
    tool = SaveBestRouteTool(store)

    # 1) initial save under label "r1"
    r1 = json.loads(tool.execute(
        label="r1", steps=["CCO>>CC.O"], total_score=0.5,
    ))
    assert r1.get("ok") is True

    # 2) overwrite "r1" with a DIFFERENT chemistry — fingerprint changes
    r2 = json.loads(tool.execute(
        label="r1", steps=["CCC>>CC.C"], total_score=0.6,
    ))
    assert r2.get("status") != "duplicate", (
        f"Same-label overwrite with new chemistry must succeed, got: {r2}"
    )

    # 3) save the ORIGINAL chemistry under a fresh label — must be a
    # genuine insert, not a stale-fingerprint duplicate
    r3 = json.loads(tool.execute(
        label="r2", steps=["CCO>>CC.O"], total_score=0.7,
    ))
    assert r3.get("status") != "duplicate", (
        f"Stale fingerprint flagged a fresh route as duplicate: {r3}"
    )

    labels = sorted(e.label for e in store.list())
    assert labels == ["r1", "r2"], (
        f"Both routes must be in the store, got labels: {labels}"
    )
    # And the chemistry under each label must match what we last saved.
    assert store.get("r1").steps == ["CCC>>CC.C"]
    assert store.get("r2").steps == ["CCO>>CC.O"]
