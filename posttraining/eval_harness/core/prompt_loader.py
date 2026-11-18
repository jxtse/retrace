"""Prompt template loading utilities."""

from __future__ import annotations

import os


class PromptLoader:

    """Loads `.md` templates from a base directory and renders them with
    str.format(**kwargs). Templates are cached on first read.

    Failure modes:
    - Template file missing → FileNotFoundError with the resolved path.
    - format() KeyError on render → re-raised as KeyError with template
      name included in the message.

    All LLM-facing user/system text in core.agent MUST come through this
    loader. New inline f-strings aimed at the LLM are a regression.
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self._cache: dict[str, str] = {}

    def render(self, name: str, **kwargs) -> str:
        if name not in self._cache:
            path = os.path.join(self.base_dir, name + ".md")
            try:
                with open(path) as f:
                    self._cache[name] = f.read()
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Prompt template not found: {path} "
                    f"(loader base_dir={self.base_dir})"
                ) from e
        body = self._cache[name]
        if not kwargs:
            return body
        try:
            return body.format(**kwargs)
        except KeyError as e:
            raise KeyError(
                f"Prompt template '{name}' references unknown placeholder {e}"
            ) from None

