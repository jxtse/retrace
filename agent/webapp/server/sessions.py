"""In-process session registry. Single-user / localhost only."""

from __future__ import annotations

import time
import uuid
from threading import RLock
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from webapp.server.runner import RunSession


_LOCK = RLock()
_SESSIONS: dict[str, "RunSession"] = {}


def new_session_id() -> str:
    return f"web-{int(time.time())}-{uuid.uuid4().hex[:6]}"


def register(session: "RunSession") -> None:
    with _LOCK:
        _SESSIONS[session.session_id] = session


def get(session_id: str) -> Optional["RunSession"]:
    with _LOCK:
        return _SESSIONS.get(session_id)


def all_sessions() -> list["RunSession"]:
    with _LOCK:
        return list(_SESSIONS.values())
