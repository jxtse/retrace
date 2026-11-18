"""Unit tests for the shared retroplanner HTTP client.

Mocks `requests.Session.post` so no network is required. (May 2026:
the client moved off the global `requests.post` to a pooled
`requests.Session` for connection reuse — patches now target the
session method instead.) Covers success parsing, score normalisation
across all three regimes (log-space, already-[0,1], mixed-sign), retry
behaviour, and the failure classification matrix from the spec's Error
Handling table.
"""
from unittest.mock import MagicMock, patch

import pytest
import requests

from tools._retroplanner_http import RetroplannerHTTPClient, RetroplannerHTTPError


def _resp(
    status: int,
    payload: dict | None = None,
    raises: Exception | None = None,
    headers: dict | None = None,
):
    """Build a fake `requests.Response`-like object."""
    m = MagicMock()
    m.status_code = status
    m.headers = dict(headers or {})
    if raises is not None:
        m.raise_for_status.side_effect = raises
    else:
        m.raise_for_status.return_value = None
    m.json.return_value = payload or {}
    return m


URL = "http://example.invalid/single_step"


# ---------- success parsing ----------


def test_success_returns_normalised_predictions():
    body = {
        "status": "success",
        "one_step_results": {
            "reactants": ["A.B", "C.D", "E"],
            "scores": [0.50, 0.14, 0.12],
            "template": ["t1", "t2", "t3"],
        },
    }
    client = RetroplannerHTTPClient(url=URL, timeout_s=5, retries=0)
    with patch("requests.Session.post", return_value=_resp(200, body)) as p:
        out = client.predict("CCO", model_name="GraphFP (USPTO-Full)", topk=3)
    assert p.call_count == 1
    sent = p.call_args
    assert sent.kwargs["json"] == {
        "smiles": "CCO",
        "savedOptions": {"topk": 3, "oneStepModel": ["DMPNNFP (USPTO-Full)"]},
    }
    assert [d["reactants"] for d in out] == ["A.B", "C.D", "E"]
    assert out[0]["score"] == pytest.approx(0.50)
    assert out[0].get("template") == "t1"


def test_multiple_urls_round_robin_first_attempts():
    body = {
        "status": "success",
        "one_step_results": {
            "reactants": ["A"],
            "scores": [0.5],
            "template": [""],
        },
    }
    client = RetroplannerHTTPClient(
        url=["http://host-a.invalid/single_step", "http://host-d.invalid/single_step"],
        timeout_s=5,
        retries=0,
    )
    with patch("requests.Session.post", return_value=_resp(200, body)) as p:
        client.predict("CCO", model_name="RxNGormer (USPTO-Full)", topk=1)
        client.predict("CCN", model_name="RxNGormer (USPTO-Full)", topk=1)
    assert p.call_args_list[0].args[0] == "http://host-a.invalid/single_step"
    assert p.call_args_list[1].args[0] == "http://host-d.invalid/single_step"


def test_weighted_round_robin_favours_primary():
    """Weights [8, 1] send ~8/9 of first hops to the primary host.

    Over one full SWRR cycle (sum of weights = 9) the primary is picked
    exactly 8 times and the fallback once, interleaved rather than
    bursty (the single fallback hit lands mid-cycle, not at either end).
    """
    body = {
        "status": "success",
        "one_step_results": {"reactants": ["A"], "scores": [0.5], "template": [""]},
    }
    primary = "http://host-d.invalid/single_step"
    fallback = "http://host-a.invalid/single_step"
    client = RetroplannerHTTPClient(
        url=[primary, fallback], weights=[8, 1], timeout_s=5, retries=0,
    )
    with patch("requests.Session.post", return_value=_resp(200, body)) as p:
        for _ in range(9):
            client.predict("CCO", model_name="RxNGormer (USPTO-Full)", topk=1)
    hosts = [c.args[0] for c in p.call_args_list]
    assert hosts.count(primary) == 8
    assert hosts.count(fallback) == 1
    # Interleaved, not bursty: fallback is not the very first hop.
    assert hosts[0] == primary


def test_equal_weights_reproduce_plain_round_robin():
    """Default (no weights) keeps the historical 0, 1, 0, 1 sequence."""
    body = {
        "status": "success",
        "one_step_results": {"reactants": ["A"], "scores": [0.5], "template": [""]},
    }
    a = "http://a.invalid/single_step"
    b = "http://b.invalid/single_step"
    client = RetroplannerHTTPClient(url=[a, b], timeout_s=5, retries=0)
    with patch("requests.Session.post", return_value=_resp(200, body)) as p:
        for _ in range(4):
            client.predict("CCO", model_name="RxNGormer (USPTO-Full)", topk=1)
    assert [c.args[0] for c in p.call_args_list] == [a, b, a, b]
    body = {
        "status": "success",
        "one_step_results": {
            "reactants": ["A"],
            "scores": [0.5],
            "template": [""],
        },
    }
    client = RetroplannerHTTPClient(
        url=["http://host-a.invalid/single_step", "http://host-d.invalid/single_step"],
        timeout_s=5,
        retries=1,
        backoff_base=0,
    )
    with patch(
        "requests.Session.post",
        side_effect=[requests.Timeout("slow"), _resp(200, body)],
    ) as p:
        out = client.predict("CCO", model_name="RxNGormer (USPTO-Full)", topk=1)
    assert out[0]["reactants"] == "A"
    assert p.call_args_list[0].args[0] == "http://host-a.invalid/single_step"
    assert p.call_args_list[1].args[0] == "http://host-d.invalid/single_step"


def test_dmpnnfp_model_alias_primary_then_graphfp_fallback():
    empty = {"status": "success", "one_step_results": {}}
    body = {
        "status": "success",
        "one_step_results": {
            "reactants": ["A"],
            "scores": [0.5],
            "template": [""],
        },
    }
    client = RetroplannerHTTPClient(url=URL, timeout_s=5, retries=0)
    with patch(
        "requests.Session.post",
        side_effect=[_resp(200, empty), _resp(200, body)],
    ) as p:
        out = client.predict("CCO", model_name="GraphFP (USPTO-Full)", topk=1)
    assert out[0]["reactants"] == "A"
    # host-d's canonical name is tried first (primary host); the legacy
    # host-a name is the fallback alias.
    assert p.call_args_list[0].kwargs["json"]["savedOptions"]["oneStepModel"] == [
        "DMPNNFP (USPTO-Full)",
    ]
    assert p.call_args_list[1].kwargs["json"]["savedOptions"]["oneStepModel"] == [
        "GraphFP (USPTO-Full)",
    ]


# ---------- score normalisation ----------


def test_normalise_log_space_rxngormer():
    """All-negative scores → exp + max-normalise so top entry → 1.0."""
    body = {
        "status": "success",
        "one_step_results": {
            "reactants": ["A", "B", "C"],
            "scores": [-0.45, -1.02, -9.52],
            "template": ["", "", ""],
        },
    }
    client = RetroplannerHTTPClient(url=URL, timeout_s=5, retries=0)
    with patch("requests.Session.post", return_value=_resp(200, body)):
        out = client.predict("X", model_name="RxNGormer", topk=3)
    import math
    expected_top = 1.0
    expected_mid = math.exp(-1.02) / math.exp(-0.45)
    assert out[0]["score"] == pytest.approx(expected_top, abs=1e-6)
    assert out[1]["score"] == pytest.approx(expected_mid, abs=1e-6)
    # Order preserved (rank-1 stays rank-1).
    assert out[0]["score"] > out[1]["score"] > out[2]["score"]


def test_normalise_already_in_unit_interval():
    body = {
        "status": "success",
        "one_step_results": {
            "reactants": ["A", "B"], "scores": [0.7, 0.3], "template": ["", ""],
        },
    }
    client = RetroplannerHTTPClient(url=URL, timeout_s=5, retries=0)
    with patch("requests.Session.post", return_value=_resp(200, body)):
        out = client.predict("X", model_name="GraphFP (USPTO-Full)", topk=2)
    assert out[0]["score"] == pytest.approx(0.7)
    assert out[1]["score"] == pytest.approx(0.3)


def test_normalise_mixed_sign_min_max():
    body = {
        "status": "success",
        "one_step_results": {
            "reactants": ["A", "B", "C"], "scores": [-1.0, 0.5, 2.0],
            "template": ["", "", ""],
        },
    }
    client = RetroplannerHTTPClient(url=URL, timeout_s=5, retries=0)
    with patch("requests.Session.post", return_value=_resp(200, body)):
        out = client.predict("X", model_name="X", topk=3)
    # min-max: -1 → 0.0, 0.5 → 0.5, 2.0 → 1.0
    assert out[0]["score"] == pytest.approx(0.0)
    assert out[1]["score"] == pytest.approx(0.5)
    assert out[2]["score"] == pytest.approx(1.0)


def test_normalise_already_in_unit_interval_with_ties():
    body = {
        "status": "success",
        "one_step_results": {
            "reactants": ["A", "B"], "scores": [0.7, 0.7], "template": ["", ""],
        },
    }
    client = RetroplannerHTTPClient(url=URL, timeout_s=5, retries=0)
    with patch("requests.Session.post", return_value=_resp(200, body)):
        out = client.predict("X", model_name="X", topk=2)
    # already-in-[0,1] branch — both 0.7. Pass-through, no min-max.
    assert out[0]["score"] == pytest.approx(0.7)
    assert out[1]["score"] == pytest.approx(0.7)


def test_normalise_degenerate_min_max_returns_half():
    """All-equal values > 1 (escape both other branches) → 0.5 each."""
    body = {
        "status": "success",
        "one_step_results": {
            "reactants": ["A", "B"], "scores": [5.0, 5.0], "template": ["", ""],
        },
    }
    client = RetroplannerHTTPClient(url=URL, timeout_s=5, retries=0)
    with patch("requests.Session.post", return_value=_resp(200, body)):
        out = client.predict("X", model_name="X", topk=2)
    assert out[0]["score"] == pytest.approx(0.5)
    assert out[1]["score"] == pytest.approx(0.5)


# ---------- failure classification ----------


def test_empty_one_step_results_raises():
    body = {"status": "success", "one_step_results": {}}
    client = RetroplannerHTTPClient(url=URL, timeout_s=5, retries=0)
    with patch("requests.Session.post", return_value=_resp(200, body)):
        with pytest.raises(RetroplannerHTTPError, match="empty"):
            client.predict("X", model_name="RxNGormer", topk=10)


def test_empty_reactants_raises():
    body = {
        "status": "success",
        "one_step_results": {"reactants": [], "scores": [], "template": []},
    }
    client = RetroplannerHTTPClient(url=URL, timeout_s=5, retries=0)
    with patch("requests.Session.post", return_value=_resp(200, body)):
        with pytest.raises(RetroplannerHTTPError, match="empty"):
            client.predict("X", model_name="RxNGormer", topk=10)


def test_non_success_status_raises():
    body = {"status": "error", "message": "oops"}
    client = RetroplannerHTTPClient(url=URL, timeout_s=5, retries=0)
    with patch("requests.Session.post", return_value=_resp(200, body)):
        with pytest.raises(RetroplannerHTTPError):
            client.predict("X", model_name="RxNGormer", topk=10)


def test_4xx_no_retry():
    err = requests.HTTPError("400 client")
    client = RetroplannerHTTPClient(url=URL, timeout_s=5, retries=3)
    with patch("requests.Session.post", return_value=_resp(400, raises=err)) as p:
        with pytest.raises(RetroplannerHTTPError):
            client.predict("X", model_name="RxNGormer", topk=10)
    assert p.call_count == 1, "4xx must not retry"


def test_5xx_retried_then_raises():
    err = requests.HTTPError("500 server")
    client = RetroplannerHTTPClient(url=URL, timeout_s=5, retries=3, backoff_base=0)
    with patch("requests.Session.post", return_value=_resp(500, raises=err)) as p:
        with pytest.raises(RetroplannerHTTPError):
            client.predict("X", model_name="RxNGormer", topk=10)
    assert p.call_count == 1 + 3, "first attempt + 3 retries"


def test_connection_error_retried():
    client = RetroplannerHTTPClient(url=URL, timeout_s=5, retries=2, backoff_base=0)
    with patch(
        "requests.Session.post", side_effect=requests.ConnectionError("net down"),
    ) as p:
        with pytest.raises(RetroplannerHTTPError):
            client.predict("X", model_name="RxNGormer", topk=10)
    assert p.call_count == 1 + 2


def test_json_parse_failure_raises():
    m = MagicMock()
    m.status_code = 200
    m.raise_for_status.return_value = None
    m.json.side_effect = ValueError("not json")
    client = RetroplannerHTTPClient(url=URL, timeout_s=5, retries=0)
    with patch("requests.Session.post", return_value=m):
        with pytest.raises(RetroplannerHTTPError):
            client.predict("X", model_name="RxNGormer", topk=10)


def test_exponential_backoff_observed():
    sleeps: list[float] = []

    def fake_sleep(s):
        sleeps.append(s)

    client = RetroplannerHTTPClient(url=URL, timeout_s=5, retries=3, backoff_base=1.0)
    with patch("requests.Session.post", side_effect=requests.ConnectionError("x")), \
         patch("time.sleep", side_effect=fake_sleep):
        with pytest.raises(RetroplannerHTTPError):
            client.predict("X", model_name="RxNGormer", topk=10)
    # Three retries → three sleeps with 1, 2, 4.
    assert sleeps == [1.0, 2.0, 4.0]


# ---------- ChemBART overload / deadline policy ----------


def _success_body():
    return {
        "status": "success",
        "one_step_results": {
            "reactants": ["A.B"],
            "scores": [0.8],
            "template": ["[ChemBart]"],
        },
    }


def test_chembart_defaults_to_two_total_attempts_even_when_generic_retries_are_three():
    err = requests.HTTPError("503 busy")
    client = RetroplannerHTTPClient(
        url=URL,
        timeout_s=5,
        retries=3,
        backoff_base=0,
        retry_jitter_s=0,
    )
    busy = _resp(503, raises=err, headers={"Retry-After": "0"})
    with patch("requests.Session.post", return_value=busy) as p:
        with pytest.raises(RetroplannerHTTPError, match="after 2 attempts"):
            client.predict("X", model_name="ChemBart (USPTO-Full)", topk=10)
    assert p.call_count == 2


def test_chembart_alias_fallback_cannot_exceed_two_actual_posts():
    bad_name = requests.HTTPError("400 unknown model")
    busy = requests.HTTPError("503 busy")
    client = RetroplannerHTTPClient(
        url=URL,
        timeout_s=5,
        retries=9,
        backoff_base=0,
        retry_jitter_s=0,
    )
    with patch(
        "requests.Session.post",
        side_effect=[
            _resp(400, raises=bad_name),
            _resp(503, raises=busy, headers={"Retry-After": "0"}),
        ],
    ) as p:
        with pytest.raises(RetroplannerHTTPError, match="after 2 attempts"):
            client.predict("X", model_name="ChemBart (USPTO-Full)", topk=10)
    assert p.call_count == 2


def test_chembart_503_honours_retry_after_and_adds_positive_jitter():
    err = requests.HTTPError("503 busy")
    client = RetroplannerHTTPClient(
        url=URL,
        timeout_s=5,
        retries=9,  # ChemBART's model-specific cap still wins.
        backoff_base=0.1,
        retry_jitter_s=0.25,
    )
    sleeps: list[float] = []
    with patch(
        "requests.Session.post",
        side_effect=[
            _resp(503, raises=err, headers={"Retry-After": "2"}),
            _resp(200, _success_body()),
        ],
    ) as p, patch(
        "tools._retroplanner_http.random.uniform", return_value=0.125,
    ), patch(
        "tools._retroplanner_http.time.sleep", side_effect=sleeps.append,
    ):
        out = client.predict("X", model_name="ChemBart (USPTO-Full)", topk=10)
    assert out[0]["reactants"] == "A.B"
    assert p.call_count == 2
    assert sleeps == [pytest.approx(2.125)]


def test_chembart_second_attempt_timeout_is_clamped_to_remaining_deadline():
    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = Clock()
    err = requests.HTTPError("503 busy")
    client = RetroplannerHTTPClient(
        url=URL,
        timeout_s=120,
        retries=3,
        backoff_base=0,
        chembart_deadline_s=10,
        chembart_retry_min_budget_s=1,
        retry_jitter_s=0,
    )
    with patch(
        "requests.Session.post",
        side_effect=[
            _resp(503, raises=err, headers={"Retry-After": "3"}),
            _resp(200, _success_body()),
        ],
    ) as p, patch(
        "tools._retroplanner_http.time.monotonic", side_effect=clock.monotonic,
    ), patch(
        "tools._retroplanner_http.time.sleep", side_effect=clock.sleep,
    ):
        client.predict("X", model_name="ChemBart (USPTO-Full)", topk=10)
    assert p.call_args_list[0].kwargs["timeout"] == pytest.approx(10.0)
    assert p.call_args_list[1].kwargs["timeout"] == pytest.approx(7.0)


def test_chembart_does_not_sleep_past_total_deadline():
    err = requests.HTTPError("503 busy")
    client = RetroplannerHTTPClient(
        url=URL,
        timeout_s=120,
        retries=3,
        backoff_base=0,
        chembart_deadline_s=2,
        chembart_retry_min_budget_s=1,
        retry_jitter_s=0,
    )
    with patch(
        "requests.Session.post",
        return_value=_resp(503, raises=err, headers={"Retry-After": "3"}),
    ) as p, patch("tools._retroplanner_http.time.sleep") as sleep:
        with pytest.raises(RetroplannerHTTPError, match="insufficient remaining budget"):
            client.predict("X", model_name="ChemBart (USPTO-Full)", topk=10)
    assert p.call_count == 1
    sleep.assert_not_called()


def test_slow_503_is_terminal_when_no_full_retry_budget_remains():
    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

    clock = Clock()
    err = requests.HTTPError("503 after server queue deadline")
    client = RetroplannerHTTPClient(
        url=URL,
        timeout_s=120,
        retries=3,
        backoff_base=0,
        chembart_deadline_s=130,
        chembart_retry_min_budget_s=75,
        retry_jitter_s=0,
    )

    def slow_busy(*args, **kwargs):
        clock.now = 110.0
        return _resp(503, raises=err, headers={"Retry-After": "1"})

    with patch(
        "requests.Session.post", side_effect=slow_busy,
    ) as p, patch(
        "tools._retroplanner_http.time.monotonic", side_effect=clock.monotonic,
    ), patch("tools._retroplanner_http.time.sleep") as sleep:
        with pytest.raises(RetroplannerHTTPError, match="insufficient remaining budget"):
            client.predict("X", model_name="ChemBart (USPTO-Full)", topk=10)
    assert p.call_count == 1
    sleep.assert_not_called()


def test_read_timeout_single_endpoint_is_not_retried():
    client = RetroplannerHTTPClient(url=URL, timeout_s=5, retries=3, backoff_base=0)
    with patch(
        "requests.Session.post", side_effect=requests.ReadTimeout("ambiguous"),
    ) as p:
        with pytest.raises(RetroplannerHTTPError, match="single endpoint not retried"):
            client.predict("X", model_name="RxNGormer", topk=10)
    assert p.call_count == 1


def test_read_timeout_retains_failover_for_multiple_endpoints():
    a = "http://a.invalid/single_step"
    b = "http://b.invalid/single_step"
    client = RetroplannerHTTPClient(
        url=[a, b], timeout_s=5, retries=1, backoff_base=0,
    )
    with patch(
        "requests.Session.post",
        side_effect=[requests.ReadTimeout("ambiguous"), _resp(200, _success_body())],
    ) as p:
        out = client.predict("X", model_name="RxNGormer", topk=10)
    assert out[0]["reactants"] == "A.B"
    assert [call.args[0] for call in p.call_args_list] == [a, b]
