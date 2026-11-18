"""Regression tests for batch attribution and lossless draining."""

from scripts import run_hard_mol_batch as batch
from scripts.run_hard_mol_batch import find_session_dir, parse_route_text


def test_parse_route_text_uses_last_complete_block_with_steps():
    text = """COMPLETE ROUTE FOUND: echoed marker without a route

noise

COMPLETE ROUTE FOUND:
Step 1: CCO >> CC.O (reaction: test)
Starting materials: CC, O
"""

    route = parse_route_text(text)

    assert route["steps"] == [{
        "step": 1,
        "product": "CCO",
        "precursors": ["CC", "O"],
        "rxn": "test",
    }]
    assert route["starting_materials"] == ["CC", "O"]


def test_find_session_dir_prefers_last_transcript_write_banner(tmp_path):
    runs = tmp_path / "runs"
    first = runs / "_session_auto-20260821-120000-1-aaaa"
    last = runs / "_session_auto-20260821-120001-2-bbbb"
    first.mkdir(parents=True)
    last.mkdir(parents=True)
    stdout = (
        f"transcript: wrote 10 events to {first}/transcript.json\n"
        f"transcript: wrote 12 events to {last}/transcript.json\n"
    )

    assert find_session_dir(stdout, started_after=0) == last


def test_sigusr1_handler_requests_drain():
    batch._DRAIN_REQUESTED.clear()
    try:
        batch._request_drain(None, None)
        assert batch._DRAIN_REQUESTED.is_set()
    finally:
        batch._DRAIN_REQUESTED.clear()
