"""SSE response helper. One ``data:`` line per JSON event, ``\\n\\n``
separator, no SSE ``event:`` field (the JSON payload carries the event
type)."""

from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import Request
from fastapi.responses import StreamingResponse


SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",          # disable nginx proxy buffering if behind one
    "Connection": "keep-alive",
}


def sse_response(generator: AsyncIterator[dict], request: Request | None = None) -> StreamingResponse:
    async def _wrap():
        try:
            async for evt in generator:
                if request is not None and await request.is_disconnected():
                    break
                yield "data: " + json.dumps(evt, ensure_ascii=False) + "\n\n"
        except Exception as e:
            yield "data: " + json.dumps({"event": "_stream_error", "payload": {"error": str(e)}, "ts": 0}, ensure_ascii=False) + "\n\n"

    return StreamingResponse(_wrap(), media_type="text/event-stream", headers=SSE_HEADERS)
