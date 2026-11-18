"""
Session management for the retrosynthesis agent.
Supports saving/loading conversation state, resuming sessions.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any

logger = logging.getLogger(__name__)

SESSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".sessions")


@dataclass
class Session:
    """A conversation session with the retrosynthesis agent."""
    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    target_smiles: str = ""
    messages: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    route_found: bool = False
    route_text: str = ""
    iterations: int = 0
    metadata: dict = field(default_factory=dict)

    def add_message(self, role: str, content: str, **kwargs):
        msg = {"role": role, "content": content, **kwargs}
        self.messages.append(msg)
        self.updated_at = time.time()

    def add_tool_result(self, tool_name: str, args: dict, result: str):
        self.tool_results.append({
            "tool": tool_name,
            "args": args,
            "result": result,
            "timestamp": time.time(),
        })
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SessionManager:
    """Manage multiple sessions with persistence."""

    def __init__(self, sessions_dir: str = SESSIONS_DIR):
        self.sessions_dir = sessions_dir
        os.makedirs(sessions_dir, exist_ok=True)

    def create_session(self, target_smiles: str = "") -> Session:
        session_id = f"session_{int(time.time())}_{os.getpid()}"
        session = Session(session_id=session_id, target_smiles=target_smiles)
        self.save(session)
        logger.info("Created session: %s", session_id)
        return session

    def save(self, session: Session):
        path = os.path.join(self.sessions_dir, f"{session.session_id}.json")
        with open(path, "w") as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)

    def load(self, session_id: str) -> Session | None:
        path = os.path.join(self.sessions_dir, f"{session_id}.json")
        if not os.path.exists(path):
            logger.warning("Session not found: %s", session_id)
            return None
        with open(path) as f:
            return Session.from_dict(json.load(f))

    def list_sessions(self) -> list[dict]:
        """List all sessions with summary info."""
        sessions = []
        for fname in sorted(os.listdir(self.sessions_dir)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self.sessions_dir, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data["session_id"],
                    "target_smiles": data.get("target_smiles", ""),
                    "route_found": data.get("route_found", False),
                    "iterations": data.get("iterations", 0),
                    "created_at": data.get("created_at", 0),
                    "messages_count": len(data.get("messages", [])),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions

    def delete(self, session_id: str):
        path = os.path.join(self.sessions_dir, f"{session_id}.json")
        if os.path.exists(path):
            os.remove(path)
