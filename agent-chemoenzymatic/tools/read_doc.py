"""Just-in-time prompt-section loader.

Mirrors Claude Code's SkillTool listing/loading split: the system prompt
ships only a short discovery listing; the model calls ``read_doc(topic=...)``
to pull the full content for a section into context only when it needs it.

Sections live under ``prompts/sections/<topic>.md``. Topic names use ``/`` as
a path separator (e.g. ``tools/neural_retro``).
"""

from __future__ import annotations

import json
import os

from tools.base import Tool


_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "prompts", "sections"))


def list_topics(base_dir: str | None = None) -> list[str]:
    base = base_dir or _BASE_DIR
    out = []
    if not os.path.isdir(base):
        return out
    for root, _, files in os.walk(base):
        for f in files:
            if f.endswith(".md"):
                rel = os.path.relpath(os.path.join(root, f), base)
                out.append(rel[:-3])  # strip .md
    return sorted(out)


class ReadDocTool(Tool):
    name = "read_doc"
    description = (
        "Load detailed prompt sections (per-tool usage, scoring schema, "
        "completion checklist) into context on demand. "
        "Use whenever you're about to call a tool you haven't read about this "
        "session, or you want the full scoring rubric. Topics use '/' as "
        "separator, e.g. 'tools/neural_retro', 'gates/completion'. Pass no "
        "argument to list available topics."
    )
    parameters = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Topic identifier (e.g. 'tools/neural_retro').",
            },
        },
    }
    tier = 1

    def __init__(self, base_dir: str | None = None):
        self.base_dir = base_dir or _BASE_DIR

    def execute(self, topic: str | None = None, **_) -> str:  # type: ignore[override]
        topics = list_topics(self.base_dir)
        if not topic:
            return json.dumps({"available_topics": topics})
        # Strip a leading slash and a trailing .md so the model can be sloppy.
        t = topic.strip().lstrip("/")
        if t.endswith(".md"):
            t = t[:-3]
        path = os.path.join(self.base_dir, t + ".md")
        if not os.path.isfile(path):
            return json.dumps({
                "error": f"unknown topic: {topic}",
                "available_topics": topics,
            })
        try:
            with open(path) as f:
                content = f.read()
        except Exception as e:
            return json.dumps({"error": f"read failed: {e}"})
        return json.dumps({"topic": t, "content": content})
