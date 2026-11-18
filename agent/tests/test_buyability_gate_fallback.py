"""Regression tests for the finalisation buyability gate.

The gate previously checked only the rank-1 (best-scoring) route: if its leaves
contained one non-buyable item — a complex intermediate, or a reagent like H2/O2
absent from the building-block catalogues — the whole run was reported FAILED even
when rank-2+ routes terminated cleanly in commercial blocks. The gate now walks
the ranked list and selects the first fully-buyable route.

These tests exercise `_select_first_buyable_route` directly with a stubbed
`_verify_route_leaves_buyable`, so they need no network, no stock CSVs, and no LLM.
"""
import types

from core.agent import RetrosynthesisAgent as Agent


class _FakeEntry:
    def __init__(self, label, buyable):
        self.label = label
        self.starting_materials = [f"SM_{label}"]
        self._buyable = buyable


class _FakeStore:
    def __init__(self, entries):
        self._entries = entries

    def all(self):
        return list(self._entries)


def _make_agent(entries, buyable_by_label):
    """Build a bare Agent without running __init__, wiring only what the gate reads."""
    agent = Agent.__new__(Agent)
    agent.scratchpad_store = _FakeStore(entries)
    agent._all_leaves_buyable_flag = True
    # Stub the per-route check so we don't touch check_buyability / stock files.
    agent._verify_route_leaves_buyable = lambda entry: buyable_by_label[entry.label]
    return agent


def _ranked(*labels):
    return {"routes": [{"label": lbl} for lbl in labels]}


def test_rank1_buyable_selects_rank1():
    labels = ["r1", "r2", "r3"]
    entries = [_FakeEntry(lbl, True) for lbl in labels]
    agent = _make_agent(entries, {"r1": True, "r2": True, "r3": True})
    assert agent._select_first_buyable_route(_ranked(*labels)) == 0
    assert agent._all_leaves_buyable_flag is True


def test_rank1_unbuyable_falls_back_to_rank2():
    """The exact shape of idx7/idx12: rank-1 has a non-buyable leaf, rank-2 is clean."""
    labels = ["r1", "r2", "r3"]
    entries = [_FakeEntry(lbl, None) for lbl in labels]
    agent = _make_agent(entries, {"r1": False, "r2": True, "r3": True})
    assert agent._select_first_buyable_route(_ranked(*labels)) == 1
    assert agent._all_leaves_buyable_flag is True


def test_skips_multiple_unbuyable_to_first_clean():
    labels = ["r1", "r2", "r3", "r4"]
    entries = [_FakeEntry(lbl, None) for lbl in labels]
    agent = _make_agent(
        entries, {"r1": False, "r2": False, "r3": True, "r4": True}
    )
    assert agent._select_first_buyable_route(_ranked(*labels)) == 2
    assert agent._all_leaves_buyable_flag is True


def test_all_unbuyable_returns_minus_one():
    labels = ["r1", "r2"]
    entries = [_FakeEntry(lbl, None) for lbl in labels]
    agent = _make_agent(entries, {"r1": False, "r2": False})
    assert agent._select_first_buyable_route(_ranked(*labels)) == -1
    assert agent._all_leaves_buyable_flag is False


def test_empty_payload_returns_zero_and_leaves_flag():
    agent = _make_agent([], {})
    assert agent._select_first_buyable_route({"routes": []}) == 0
    # No gate ran; flag stays at its initial value.
    assert agent._all_leaves_buyable_flag is True


def test_no_store_returns_zero():
    agent = Agent.__new__(Agent)
    agent.scratchpad_store = None
    agent._all_leaves_buyable_flag = True
    assert agent._select_first_buyable_route(_ranked("r1")) == 0


def test_missing_entry_for_ranked_label_is_skipped():
    """A ranked label with no matching scratchpad entry is skipped, not fatal."""
    labels = ["ghost", "r2"]
    entries = [_FakeEntry("r2", True)]  # 'ghost' has no entry
    agent = _make_agent(entries, {"r2": True})
    # rank-1 ('ghost') unresolved -> skipped; rank-2 ('r2') buyable -> index 1.
    assert agent._select_first_buyable_route(_ranked(*labels)) == 1
    assert agent._all_leaves_buyable_flag is True


def test_synthesise_route_text_honours_route_index():
    """The selected index must drive which route's text is emitted."""
    agent = Agent.__new__(Agent)
    agent.scratchpad_store = None  # force the header-only path
    ranked = {
        "routes": [
            {"label": "best", "total_score": 0.9},
            {"label": "buyable", "total_score": 0.7},
        ]
    }
    txt0 = agent._synthesise_route_text_from_ranked(ranked, route_index=0)
    txt1 = agent._synthesise_route_text_from_ranked(ranked, route_index=1)
    assert "rank-1" in txt0 and "label=best" in txt0
    assert "rank-2" in txt1 and "label=buyable" in txt1


# ── best_route must match the gate-selected route (P2#2 regression) ────────

class _RouteEntry:
    """Minimal stand-in for a scratchpad RouteEntry: a label and a to_dict()."""

    def __init__(self, label):
        self.label = label

    def to_dict(self):
        return {"label": self.label}


def test_best_route_follows_gate_selection_to_rank2():
    """The exact rank-1-unbuyable / rank-2-buyable shape this PR fixes: the
    gate selects index 1, so best_route must be rank-2 — NOT routes_sorted[0]
    (rank-1). score_best_route still carries the top-by-score rank-1."""
    agent = Agent.__new__(Agent)
    # routes_sorted is score-descending: rank-1 first.
    routes_sorted = [_RouteEntry("r1"), _RouteEntry("r2"), _RouteEntry("r3")]
    ranked = _ranked("r1", "r2", "r3")
    best, score_best = agent._resolve_best_route(routes_sorted, ranked, 1)
    assert best == {"label": "r2"}        # gate-selected rank-2
    assert score_best == {"label": "r1"}  # unconditional top-by-score


def test_best_route_is_rank1_when_gate_selects_rank1():
    agent = Agent.__new__(Agent)
    routes_sorted = [_RouteEntry("r1"), _RouteEntry("r2")]
    ranked = _ranked("r1", "r2")
    best, score_best = agent._resolve_best_route(routes_sorted, ranked, 0)
    assert best == {"label": "r1"}
    assert score_best == {"label": "r1"}


def test_best_route_falls_back_to_score_best_when_gate_fails():
    """selected_index == -1 (no route buyable): there is no selected route, so
    best_route falls back to the score-best entry rather than vanishing."""
    agent = Agent.__new__(Agent)
    routes_sorted = [_RouteEntry("r1"), _RouteEntry("r2")]
    ranked = _ranked("r1", "r2")
    best, score_best = agent._resolve_best_route(routes_sorted, ranked, -1)
    assert best == {"label": "r1"}
    assert score_best == {"label": "r1"}


def test_best_route_empty_store_returns_none():
    agent = Agent.__new__(Agent)
    best, score_best = agent._resolve_best_route([], _ranked("r1"), 0)
    assert best is None
    assert score_best is None


def test_best_route_unresolvable_label_falls_back_to_score_best():
    """A selected index whose ranked label has no matching entry must not crash
    or null out best_route — it falls back to the score-best entry."""
    agent = Agent.__new__(Agent)
    routes_sorted = [_RouteEntry("r1"), _RouteEntry("r2")]
    # ranked carries a 'ghost' label at the selected index with no entry.
    ranked = {"routes": [{"label": "r1"}, {"label": "ghost"}]}
    best, score_best = agent._resolve_best_route(routes_sorted, ranked, 1)
    assert best == {"label": "r1"}        # fell back to score-best
    assert score_best == {"label": "r1"}
