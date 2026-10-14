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
    GET  /api/samples                    — held-out demo wafers + batch scenarios
    GET  /api/model-card                 — evaluation metrics for both models
    DELETE /api/history                  — clear inspection history (re-run demos)
    POST /api/inspections/{task_id}/approve  — (full pipeline only, not yet active)
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
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


BASE_DIR = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def _demo_samples() -> dict:
    path = BASE_DIR / "data" / "processed" / "demo_samples.json"
    if not path.exists():
        raise HTTPException(503, "Demo samples not built — run scripts/build_demo_samples.py")
    return json.loads(path.read_text())


@app.get("/api/samples")
def samples() -> dict:
    """Held-out wafers (with their true label) and batch scenarios for the dashboard."""
    return _demo_samples()


@app.get("/api/model-card")
def model_card() -> dict:
    """Test-set metrics written by the training scripts — shown in the UI as-is."""
    out = {}
    for key, name in (("wafer_classifier", "wafer_classifier_metrics.json"),
                      ("process_monitor", "process_monitor_metrics.json")):
        path = BASE_DIR / "models" / name
        out[key] = json.loads(path.read_text()) if path.exists() else None
    return out


@app.get("/api/drift")
def drift_report() -> dict:
    """How far recent live wafers' features have moved from the training data (PSI)."""
    from core import drift
    from core.wafer_features import FEATURE_NAMES
    return drift.live_report(FEATURE_NAMES)


@app.delete("/api/history")
def clear_history() -> dict:
    from core import drift, history
    history.clear()
    drift.clear()
    return {"status": "cleared"}


# ── Catch stray writes to unknown paths ───────────────────────────────────────
# Without this, a POST/PUT/etc. to any non-API path (e.g. bots probing /submit)
# falls through to the StaticFiles mount below, which only serves GET/HEAD and so
# returns a bare 405. Answer with an explicit JSON 404 instead. GET is untouched,
# so the SPA is still served normally.
from fastapi import Request
from fastapi.responses import JSONResponse


@app.api_route("/{full_path:path}", methods=["POST", "PUT", "PATCH", "DELETE"])
def unknown_write(full_path: str, request: Request) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"detail": f"No such endpoint: {request.method} /{full_path}"},
    )


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
