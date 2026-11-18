"""Shared pytest fixtures for the retrosynthesis-agent test suite.

This file establishes the import path (so `import core...` and
`import tools...` resolve) and a couple of small fixtures used by tests.
"""
import os
import sys

# Ensure the repo root is on sys.path even when pytest is invoked from a
# subdirectory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


import pytest


class FakeLLMClient:
    """Minimal stand-in for `LLMClient` exposing only `chat(messages)`.

    Used by tests that exercise LLM-dependent helpers without burning
    tokens. Configure `responses` as a list of strings; each call to
    `chat` pops the next response. If the list is empty, raises
    RuntimeError so an unexpected extra call fails loudly.
    """

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        if not self.responses:
            raise RuntimeError("FakeLLMClient ran out of canned responses")
        return self.responses.pop(0)


@pytest.fixture
def fake_llm():
    return FakeLLMClient()
