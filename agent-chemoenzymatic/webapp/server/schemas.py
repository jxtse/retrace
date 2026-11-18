"""Pydantic request / response models for the agent webapp."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class RunOverrides(BaseModel):
    max_iterations: Optional[int] = None
    stop_after_distinct_routes: Optional[int] = None
    max_depth: Optional[int] = None
    wallclock_s: Optional[float] = None


class RunRequest(BaseModel):
    smiles: str = Field(..., description="Target SMILES")
    model_family: Literal["claude", "gpt5_5"] = "claude"
    preset: Literal["fast", "standard", "thorough"] = "fast"
    overrides: Optional[RunOverrides] = None


class RunCreated(BaseModel):
    session_id: str
    effective: dict[str, Any]


class TraceEvent(BaseModel):
    event: str
    payload: dict[str, Any]
    ts: float


class PresetInfo(BaseModel):
    family: str
    preset: str
    model: str
    reasoning_effort: str
    max_iterations: int
    stop_after_distinct_routes: int


class RouteSummary(BaseModel):
    label: str
    total_score: float
    steps: list[str]
    sub_scores: dict[str, float]
    starting_materials: list[str]
    notes: str = ""
    provenance: str = "unknown"


class RunStatus(BaseModel):
    session_id: str
    status: Literal["running", "done", "error", "cancelled"]
    started_at: float
    finished_at: Optional[float] = None
    error: Optional[str] = None
