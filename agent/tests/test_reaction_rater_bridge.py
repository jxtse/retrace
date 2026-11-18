"""Tests for the reaction_rater HTTP bridge.

Offline only — `requests.Session.post` is monkeypatched, no network.
Covers: payload shape, success parse, and fail-soft None on every
terminal failure (5xx, timeout, 4xx, bad JSON, non-success body).
"""
import json

import pytest
import requests

from tools.verifier._reaction_rater_bridge import ReactionRaterClient


class _Resp:
    def __init__(self, status_code=200, body=None, raise_json=False):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self._raise_json = raise_json

    def json(self):
        if self._raise_json:
            raise ValueError("not json")
        return self._body


def _client(monkeypatch, responder):
    c = ReactionRaterClient("http://rater.test/api", retries=1, backoff_base=0)
    monkeypatch.setattr(c._session, "post", responder)
    return c


def test_payload_and_success(monkeypatch):
    captured = {}

    def post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _Resp(200, {"status": "success",
                           "results": {"reaction_is_feasible": True, "confidence": 0.87}})

    c = _client(monkeypatch, post)
    out = c.rate(["CC(=O)O", "CCO"], "CCOC(C)=O")

    assert captured["url"] == "http://rater.test/api"
    assert captured["json"] == {"reaction": "CC(=O)O.CCO>>CCOC(C)=O"}
    assert out == {"confidence": 0.87, "feasible": True}


def test_confidence_clamped(monkeypatch):
    def post(url, json=None, headers=None, timeout=None):
        return _Resp(200, {"status": "success",
                           "results": {"reaction_is_feasible": True, "confidence": 1.4}})
    c = _client(monkeypatch, post)
    assert c.rate(["A"], "B")["confidence"] == 1.0


def test_empty_inputs_return_none(monkeypatch):
    c = _client(monkeypatch, lambda *a, **k: pytest.fail("should not POST"))
    assert c.rate([], "P") is None
    assert c.rate(["R"], "") is None


def test_5xx_returns_none(monkeypatch):
    c = _client(monkeypatch, lambda *a, **k: _Resp(503, {}))
    assert c.rate(["R"], "P") is None


def test_4xx_returns_none(monkeypatch):
    c = _client(monkeypatch, lambda *a, **k: _Resp(422, {"error": "bad shape"}))
    assert c.rate(["R"], "P") is None


def test_timeout_returns_none(monkeypatch):
    def post(*a, **k):
        raise requests.Timeout("slow")
    c = _client(monkeypatch, post)
    assert c.rate(["R"], "P") is None


def test_connection_error_returns_none(monkeypatch):
    def post(*a, **k):
        raise requests.ConnectionError("refused")
    c = _client(monkeypatch, post)
    assert c.rate(["R"], "P") is None


def test_bad_json_returns_none(monkeypatch):
    c = _client(monkeypatch, lambda *a, **k: _Resp(200, raise_json=True))
    assert c.rate(["R"], "P") is None


def test_non_success_body_returns_none(monkeypatch):
    c = _client(monkeypatch, lambda *a, **k: _Resp(200, {"status": "error"}))
    assert c.rate(["R"], "P") is None


def test_missing_confidence_returns_none(monkeypatch):
    c = _client(monkeypatch, lambda *a, **k: _Resp(
        200, {"status": "success", "results": {"reaction_is_feasible": True}}))
    assert c.rate(["R"], "P") is None


def test_retry_then_success(monkeypatch):
    calls = {"n": 0}

    def post(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.ConnectionError("blip")
        return _Resp(200, {"status": "success",
                           "results": {"reaction_is_feasible": False, "confidence": 0.2}})

    c = _client(monkeypatch, post)
    out = c.rate(["R"], "P")
    assert calls["n"] == 2
    assert out == {"confidence": 0.2, "feasible": False}
