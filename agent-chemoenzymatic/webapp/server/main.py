"""FastAPI entry point for the retrosynthesis agent webapp.

Run with:
    uvicorn webapp.server.main:app --reload --port 8000

The frontend dev server runs at :5173 (Vite) and proxies /api here, or
talks to :8000 directly with CORS.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Make the project root importable when running via `uvicorn webapp.server.main:app`.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402

from webapp.server import presets as preset_mod  # noqa: E402
from webapp.server import sessions  # noqa: E402
from webapp.server.runner import RunSession  # noqa: E402
from webapp.server.schemas import RunCreated, RunRequest  # noqa: E402
from webapp.server.sse import sse_response  # noqa: E402

logger = logging.getLogger("webapp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="Retrosynthesis Agent Web UI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Config bootstrap (load once, deep-copied per run) ──────────────────────

_DEFAULT_CONFIG_PATH = os.environ.get("AGENT_CONFIG")
_DEFAULT_THINKING = os.environ.get("AGENT_THINKING", "fast")
_BASE_CONFIG: dict[str, Any] | None = None


def _get_base_config() -> dict[str, Any]:
    """Lazily load the YAML config. Cached after first read."""
    global _BASE_CONFIG
    if _BASE_CONFIG is None:
        from core.config import load_config
        from core.llm_client import _load_dotenv_once
        _load_dotenv_once()
        _BASE_CONFIG = load_config(_DEFAULT_CONFIG_PATH, thinking=_DEFAULT_THINKING)
        logger.info(
            "loaded base config (thinking=%s, override=%s)",
            _DEFAULT_THINKING,
            _DEFAULT_CONFIG_PATH or "-",
        )
    return _BASE_CONFIG


@app.on_event("startup")
async def _startup() -> None:
    # Pre-warm config so the first /api/runs doesn't pay the YAML parse cost.
    _get_base_config()
    (Path(_PROJECT_ROOT) / "runs").mkdir(exist_ok=True)


# ── Health / presets ───────────────────────────────────────────────────────

@app.get("/api/health")
async def health() -> dict:
    return {"ok": True, "config_path": _DEFAULT_CONFIG_PATH, "thinking": _DEFAULT_THINKING}


@app.get("/api/presets")
async def list_presets() -> dict:
    return {"presets": preset_mod.list_presets()}


# ── Run lifecycle ──────────────────────────────────────────────────────────

@app.post("/api/runs", response_model=RunCreated)
async def create_run(req: RunRequest) -> RunCreated:
    if not req.smiles.strip():
        raise HTTPException(status_code=400, detail="smiles is empty")
    try:
        preset_mod.PRESETS[(req.model_family, req.preset)]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"unknown preset: {req.model_family}/{req.preset}")

    sid = sessions.new_session_id()
    run = RunSession(session_id=sid, base_config=_get_base_config())
    run.bind_loop(asyncio.get_running_loop())

    overrides = req.overrides.model_dump(exclude_none=True) if req.overrides else None
    try:
        run.start(req.smiles, req.model_family, req.preset, overrides)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.exception("failed to start run")
        raise HTTPException(status_code=500, detail=f"start failed: {e}")

    sessions.register(run)
    return RunCreated(session_id=sid, effective=run.effective_config)


@app.get("/api/runs/{sid}/events")
async def stream_events(sid: str, request: Request):
    run = sessions.get(sid)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return sse_response(run.iter_events(), request=request)


@app.get("/api/runs/{sid}/result")
async def get_result(sid: str, timeout: float = Query(default=0.0, ge=0.0, le=86400.0)):
    run = sessions.get(sid)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown session")
    if timeout > 0:
        try:
            result = await run.wait_result(timeout=timeout)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="timeout waiting for result")
    else:
        result = run.result
    return {
        "session_id": sid,
        "status": run.status,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "effective": run.effective_config,
        "result": result,
        "error": run.error,
    }


@app.post("/api/runs/{sid}/cancel")
async def cancel_run(sid: str):
    run = sessions.get(sid)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown session")
    ok = run.cancel()
    return {"cancelled": ok, "status": run.status}


# ── Routes (scratchpad) ────────────────────────────────────────────────────

@app.get("/api/runs/{sid}/routes")
async def list_routes(sid: str):
    run = sessions.get(sid)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown session")

    routes: list[dict] = []
    # Mid-run: peek into the live scratchpad if the agent is up.
    if run.agent is not None:
        store = getattr(run.agent, "scratchpad_store", None)
        if store is not None:
            try:
                routes = [e.to_dict() for e in store.list()]
            except Exception as e:
                logger.warning("scratchpad list failed: %s", e)

    # Post-run: prefer the ranked result from RetrosynthesisAgent.run() output, since
    # the agent's own ranking is what becomes the visible final.
    if run.result is not None:
        result_routes = run.result.get("all_routes") or []
        if result_routes:
            routes = result_routes

    return {"session_id": sid, "status": run.status, "routes": routes}


# ── Result store (heavy tool payload rehydration) ──────────────────────────

@app.get("/api/runs/{sid}/results/{tcid}")
async def load_tool_result(sid: str, tcid: str):
    run = sessions.get(sid)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown session")

    # Direct disk read — does NOT touch the agent's in-memory ResultStore
    # (it lives in the worker thread). The on-disk layout is the agent's
    # contract (services/result_store.py persists every heavy tool).
    base = Path(_PROJECT_ROOT) / "runs" / f"_session_{sid}" / "results" / f"{tcid}.json"
    if not base.exists():
        # Try alternative path used by the agent when AGENT_SESSION_ID is set.
        for alt in (
            Path(_PROJECT_ROOT) / "runs" / f"_session_{sid}" / "results_root" / f"{tcid}.json",
            Path(_PROJECT_ROOT) / "runs" / f"_session_{sid}" / f"{tcid}.json",
        ):
            if alt.exists():
                base = alt
                break
        else:
            raise HTTPException(status_code=404, detail="result not found")
    try:
        blob = json.loads(base.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read result: {e}")
    return blob


# ── Molecule SVG / PNG fallback (server-side RDKit) ────────────────────────

@app.get("/api/mol/png")
async def mol_png(smiles: str = Query(..., min_length=1), w: int = 240, h: int = 180):
    try:
        from scripts.show_smiles import smiles_to_image_base64
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"renderer unavailable: {e}")
    data_uri = smiles_to_image_base64(smiles, size=(w, h))
    if not data_uri:
        raise HTTPException(status_code=422, detail="invalid SMILES")
    # data_uri = "data:image/png;base64,..."
    import base64
    b64 = data_uri.split(",", 1)[1]
    return Response(content=base64.b64decode(b64), media_type="image/png")


@app.get("/api/rxn/png")
async def rxn_png(
    reactants: str = Query("", description="dot-separated SMILES"),
    products: str = Query("", description="dot-separated SMILES"),
    reagents: str = Query(""),
    w: int = 480,
    h: int = 220,
):
    try:
        from scripts.show_smiles import reaction_smiles_to_image_base64
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"renderer unavailable: {e}")
    r = [s for s in reactants.split(".") if s]
    p = [s for s in products.split(".") if s]
    rg = [s for s in reagents.split(".") if s]
    data_uri = reaction_smiles_to_image_base64(r, rg, p, size=(w, h))
    if not data_uri:
        raise HTTPException(status_code=422, detail="invalid reaction smiles")
    import base64
    b64 = data_uri.split(",", 1)[1]
    return Response(content=base64.b64decode(b64), media_type="image/png")


# ── Top-level static file serving for production builds ────────────────────

_CLIENT_DIST = _PROJECT_ROOT / "webapp" / "client" / "dist"
if _CLIENT_DIST.exists():
    from fastapi.staticfiles import StaticFiles  # noqa: E402
    app.mount("/", StaticFiles(directory=str(_CLIENT_DIST), html=True), name="client")
else:
    @app.get("/")
    async def _no_client():
        return JSONResponse(
            {"hint": "client not built; run `cd webapp/client && npm run dev` for the dev UI"},
            status_code=200,
        )
