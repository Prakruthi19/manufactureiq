"""
ManufactureIQ — FastAPI Backend
================================
Thin REST layer over the Celery pipeline.

Run (from project root, with Redis up and a Celery worker running):
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /api/health                     — liveness
    POST /api/inspections                — submit an inspection, returns task_id
    GET  /api/inspections/{task_id}      — poll status + final result
    POST /api/inspections/{task_id}/approve  — (full pipeline only, not yet active)
"""

from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from celery.result import AsyncResult

from celery_app import celery_app
from tasks.inspection_task import run_pipeline

app = FastAPI(title="ManufactureIQ API", version="0.1.0")

# CORS — allow the Vite/React dev servers to call us during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default
        "http://localhost:3000",  # CRA / Next default
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ─────────────────────────────────────────────────
class InspectionRequest(BaseModel):
    """Payload the frontend submits."""
    inspection: dict = Field(..., description="Raw inspection record")
    sensors: list[dict] = Field(default_factory=list, description="Sensor readings")
    full: bool = Field(True, description="True = full 7-agent pipeline, False = dry (3 agents)")


class SubmitResponse(BaseModel):
    task_id: str
    status: str


class StatusResponse(BaseModel):
    task_id: str
    state: str                      # PENDING | PROGRESS | SUCCESS | FAILURE
    progress: Optional[dict] = None
    result: Optional[dict] = None   # final PipelineState dict on SUCCESS
    error: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/inspections", response_model=SubmitResponse)
def submit_inspection(req: InspectionRequest) -> SubmitResponse:
    """Enqueue an inspection for pipeline processing."""
    async_result = run_pipeline.delay(req.inspection, req.sensors, req.full)
    return SubmitResponse(task_id=async_result.id, status="queued")


@app.get("/api/inspections/{task_id}", response_model=StatusResponse)
def get_inspection(task_id: str) -> StatusResponse:
    """Poll a submitted inspection by its task id."""
    res: AsyncResult = celery_app.AsyncResult(task_id)
    out = StatusResponse(task_id=task_id, state=res.state)

    if res.state == "PROGRESS":
        out.progress = res.info if isinstance(res.info, dict) else {"info": str(res.info)}
    elif res.state == "SUCCESS":
        out.result = res.result
    elif res.state == "FAILURE":
        out.error = str(res.info)

    return out


# ── Serve the built React dashboard ───────────────────────────────────────────
# In the single-container deploy the frontend is built into ../frontend/dist and
# served here, so the whole app lives at one origin (no CORS, no separate host).
# Mounted last so every /api/* route above takes precedence. No-op in dev when
# dist/ doesn't exist (you run `npm run dev` instead).
from pathlib import Path as _Path
from fastapi.staticfiles import StaticFiles

_dist = _Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="dashboard")
