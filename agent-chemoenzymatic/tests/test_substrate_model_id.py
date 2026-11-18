from __future__ import annotations

import pytest

from core.substrate_client import (
    SUBSTRATE_BASE_URL,
    SubstrateLLMClient,
    _substrate_model_id,
    make_llm_client,
)
from core.llm_client import LLMClient


@pytest.mark.parametrize(
    "given,expected",
    [
        # Dotted Claude ids -> hyphenated (the substrate-authorized wire form).
        ("claude-opus-4.8", "claude-opus-4-8"),
        ("claude-opus-4.8-xhigh", "claude-opus-4-8-xhigh"),
        ("claude-opus-4.8-high", "claude-opus-4-8-high"),
        ("claude-sonnet-4.6", "claude-sonnet-4-6"),
        ("claude-opus-4.7", "claude-opus-4-7"),
        # Already-hyphenated Claude id -> unchanged (idempotent).
        ("claude-opus-4-8", "claude-opus-4-8"),
        # gpt-5.5 stays dotted (it is authorized as-is on /v1/responses).
        ("gpt-5.5", "gpt-5.5"),
        ("gpt-5-5", "gpt-5-5"),
        # Non-Claude / empty -> untouched.
        ("some-other-model", "some-other-model"),
        ("", ""),
    ],
)
def test_substrate_model_id(given, expected):
    assert _substrate_model_id(given) == expected


def test_substrate_client_normalises_model_and_pins_base_url():
    c = SubstrateLLMClient(model="claude-opus-4.8")
    assert c.model == "claude-opus-4-8"
    assert c.base_url == SUBSTRATE_BASE_URL.rstrip("/")


def test_substrate_client_explicit_base_url_kept():
    c = SubstrateLLMClient(base_url="http://localhost:3000/v1", model="claude-opus-4.8")
    assert c.base_url == "http://localhost:3000/v1"
    assert c.model == "claude-opus-4-8"


def test_make_llm_client_selects_substrate_on_port_3000():
    c = make_llm_client(base_url="http://localhost:3000/v1", model="claude-opus-4.8")
    assert isinstance(c, SubstrateLLMClient)
    assert c.model == "claude-opus-4-8"


def test_make_llm_client_selects_substrate_on_port_4141():
    # The substrate moved :3000 -> :4141 (2026-06-22); :4141 now routes to the
    # SubstrateLLMClient (model-id normalisation + 128k clamp + unwrapped SSE).
    c = make_llm_client(base_url="http://localhost:4141/v1", model="claude-opus-4.8")
    assert isinstance(c, SubstrateLLMClient)
    assert c.model == "claude-opus-4-8"


def test_make_llm_client_uses_plain_client_off_substrate():
    # A non-substrate endpoint (e.g. a raw vllm/OpenAI-compatible server) gets the
    # plain client, which does NOT normalise the id.
    c = make_llm_client(base_url="http://localhost:8000/v1", model="qwen3.6-35b-a3b")
    assert isinstance(c, LLMClient)
    assert not isinstance(c, SubstrateLLMClient)
    assert c.model == "qwen3.6-35b-a3b"


def test_make_llm_client_none_base_url_falls_back_to_default():
    c = make_llm_client(base_url=None, model="gpt-5.5")
    assert isinstance(c, LLMClient)
    assert c.base_url  # not None, no crash on .rstrip()
