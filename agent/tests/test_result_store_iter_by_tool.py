"""Verifies ResultStore.iter_by_tool returns every persisted result for
the requested tool, regardless of recency-window state. Used by the
redesigned ExploreBranchesTool for cross-source reverse lookup."""

import json
import os
import tempfile

from services.result_store import ResultStore


def _store(tmp):
    return ResultStore(
        session_dir=tmp,
        summary_max_chars=600,
        keep_recent_full_per_tool=2,
    )


def test_iter_by_tool_returns_recent_results():
    with tempfile.TemporaryDirectory() as tmp:
        rs = _store(tmp)
        rs.persist("neural_retro", "n1",
                   json.dumps({"target": "CCO", "predictions": []}))
        rs.persist("suggest_templates", "s1",
                   json.dumps({"product": "CCO", "suggestions": []}))

        out = list(rs.iter_by_tool("neural_retro"))
        assert len(out) == 1
        tool_call_id, payload = out[0]
        assert tool_call_id == "n1"
        assert payload["target"] == "CCO"


def test_iter_by_tool_includes_demoted_results():
    """Demoted IDs must still be reachable — they live on disk."""
    with tempfile.TemporaryDirectory() as tmp:
        rs = _store(tmp)
        # keep_recent_full_per_tool=2, so the third call demotes the first.
        for i in range(3):
            rs.persist("neural_retro", f"n{i}",
                       json.dumps({"target": f"C{'C' * i}O", "predictions": []}))

        ids_seen = sorted(tcid for tcid, _ in rs.iter_by_tool("neural_retro"))
        assert ids_seen == ["n0", "n1", "n2"]


def test_iter_by_tool_unknown_tool_yields_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        rs = _store(tmp)
        rs.persist("neural_retro", "n1",
                   json.dumps({"target": "CCO", "predictions": []}))

        assert list(rs.iter_by_tool("apply_template")) == []


def test_iter_by_tool_does_not_mutate_recency():
    """Reverse lookup must be read-only — the recency window is unaffected."""
    with tempfile.TemporaryDirectory() as tmp:
        rs = _store(tmp)
        rs.persist("neural_retro", "n1",
                   json.dumps({"target": "CCO", "predictions": []}))
        before = list(rs._recent.get("neural_retro", []))
        list(rs.iter_by_tool("neural_retro"))  # consume iterator
        after = list(rs._recent.get("neural_retro", []))
        assert before == after


def test_iter_by_tool_skips_corrupt_json_on_disk():
    """A malformed JSON file in results/ must be silently skipped, not raised."""
    with tempfile.TemporaryDirectory() as tmp:
        rs = _store(tmp)
        rs.persist("neural_retro", "n_valid",
                   json.dumps({"target": "CCO", "predictions": []}))
        # Drop a malformed JSON file directly into results/.
        corrupt_path = os.path.join(rs.results_dir, "n_corrupt.json")
        with open(corrupt_path, "w") as f:
            f.write("{not valid")

        out = list(rs.iter_by_tool("neural_retro"))
        assert len(out) == 1
        assert out[0][0] == "n_valid"


def test_iter_by_tool_skips_non_json_files():
    """Non-.json files in results/ must be ignored by the disk pass."""
    with tempfile.TemporaryDirectory() as tmp:
        rs = _store(tmp)
        rs.persist("neural_retro", "n_valid",
                   json.dumps({"target": "CCO", "predictions": []}))
        # Drop a stray non-JSON file into results/.
        readme_path = os.path.join(rs.results_dir, "README.txt")
        with open(readme_path, "w") as f:
            f.write("not a result")

        out = list(rs.iter_by_tool("neural_retro"))
        assert len(out) == 1
        assert out[0][0] == "n_valid"


def test_iter_by_tool_dedupes_disk_and_memory():
    """An ID present in both _index (in-memory) and on disk must yield once."""
    with tempfile.TemporaryDirectory() as tmp:
        rs = _store(tmp)
        # persist() writes to disk AND records in _index, so the entry lives
        # in both places simultaneously.
        rs.persist("neural_retro", "n1",
                   json.dumps({"target": "CCO", "predictions": []}))
        # Sanity-check the dual presence.
        assert "n1" in rs._index
        assert os.path.exists(os.path.join(rs.results_dir, "n1.json"))

        out = list(rs.iter_by_tool("neural_retro"))
        assert len(out) == 1
        assert out[0][0] == "n1"
