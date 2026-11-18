"""Store rehydration for the human-in-the-loop continuation seam (audit #4).

A continuation run must NOT clear the stores — it restores the prior
scratchpad + todos so the chemist's feedback lands on the routes they were
reviewing. These cover the round-trip helpers:
  * RouteEntry.from_dict round-trips to_dict (and drops unknown keys).
  * ScratchpadStore.restore repopulates _routes AND the structural
    _fingerprints index (dedup still fires after restore).
  * TodoStore.restore handles the camelCase activeForm key.
"""
from tools.scratchpad import RouteEntry, ScratchpadStore
from tools.todo import TodoStore, TodoItem


# ── RouteEntry.from_dict ─────────────────────────────────────────────────────


def test_route_entry_from_dict_round_trips():
    e = RouteEntry(
        label="R1",
        total_score=0.73,
        steps=["CCCCO>>CCCC.OCC", "CCCC>>CCCCBr"],
        sub_scores={"feasibility": 0.8, "cost": 0.5},
        starting_materials=["OCC", "CCCCBr"],
        notes="anchored on precedent X",
        provenance="rag_anchored",
        rag_source_ids=["rt-12"],
    )
    rebuilt = RouteEntry.from_dict(e.to_dict())
    assert rebuilt.label == e.label
    assert rebuilt.steps == e.steps
    assert rebuilt.sub_scores == e.sub_scores
    assert rebuilt.starting_materials == e.starting_materials
    assert rebuilt.notes == e.notes
    assert rebuilt.provenance == e.provenance
    assert rebuilt.rag_source_ids == e.rag_source_ids
    # total_score is rounded in to_dict but should be numerically equal here.
    assert abs(rebuilt.total_score - e.total_score) < 1e-9


def test_route_entry_from_dict_drops_unknown_keys():
    # A summary()-shaped blob carries num_steps / notes_excerpt — must not raise.
    payload = {
        "label": "R2",
        "total_score": 0.4,
        "steps": ["CCO>>CC=O"],
        "num_steps": 1,           # unknown to the constructor
        "notes_excerpt": "…",     # unknown to the constructor
    }
    rebuilt = RouteEntry.from_dict(payload)
    assert rebuilt.label == "R2"
    assert rebuilt.steps == ["CCO>>CC=O"]


# ── ScratchpadStore.restore ──────────────────────────────────────────────────


def _entry(label, steps, score=0.5):
    return RouteEntry(label=label, total_score=score, steps=steps)


def test_restore_repopulates_routes():
    store = ScratchpadStore(max_routes=10)
    a = _entry("A", ["CCCCO>>CCCC.OCC", "CCCC>>CCCCBr"], 0.6)
    b = _entry("B", ["CCO>>CC=O"], 0.4)
    payload = [a.to_dict(), b.to_dict()]

    n = store.restore(payload)
    assert n == 2
    labels = {r.label for r in store.list()}
    assert labels == {"A", "B"}
    # best() reflects the restored scores.
    assert store.best().label == "A"


def test_restore_clears_prior_state():
    store = ScratchpadStore(max_routes=10)
    store.save(_entry("OLD", ["XXX>>YYY".replace("X", "C").replace("Y", "O")]))
    store.restore([_entry("NEW", ["CCO>>CC=O"]).to_dict()])
    labels = {r.label for r in store.list()}
    assert labels == {"NEW"}
    assert "OLD" not in labels


def test_restore_rebuilds_fingerprints_so_dedup_fires():
    store = ScratchpadStore(max_routes=10)
    a = _entry("A", ["CCCCO>>CCCC.OCC"], 0.6)
    store.restore([a.to_dict()])

    # A structurally identical route under a NEW label must be detected as a
    # duplicate — proving _fingerprints was rebuilt by restore(), not left empty.
    dup = _entry("A-again", ["CCCCO>>OCC.CCCC"], 0.6)  # reactant order flipped
    _entry_out, status = store.save(dup)
    assert status == "duplicate"


def test_restore_ignores_non_dicts():
    store = ScratchpadStore(max_routes=10)
    n = store.restore([_entry("A", ["CCO>>CC=O"]).to_dict(), "garbage", None, 42])
    assert n == 1
    assert {r.label for r in store.list()} == {"A"}


# ── TodoStore.restore ────────────────────────────────────────────────────────


def test_todo_restore_handles_activeform_key():
    store = TodoStore()
    # to_dict emits camelCase activeForm; naive TodoItem(**d) would raise.
    payload = [TodoItem(
        id="t1", content="decompose core", status="in_progress",
        active_form="Decomposing core", smiles="CCO",
    ).to_dict()]
    assert "activeForm" in payload[0]

    summary = store.restore(payload)
    items = store.read()
    assert len(items) == 1
    assert items[0].id == "t1"
    assert items[0].active_form == "Decomposing core"
    assert items[0].status == "in_progress"
    assert summary["total"] == 1


def test_todo_restore_replaces_prior():
    store = TodoStore()
    store.restore([{"id": "old", "content": "old task"}])
    store.restore([{"id": "new", "content": "new task"}])
    ids = {t.id for t in store.read()}
    assert ids == {"new"}
