"""
ManufactureIQ — Celery Task
============================
Each inspection record runs through the pipeline as an independent async task.

- full=True  → the full 7-agent LangGraph pipeline (run_graph), non-interactive.
- full=False → the fast dry pipeline (ingestor → sensor_parser → classifier),
               used for bulk CSV batches so they don't fire N× LLM calls.
"""

import uuid
from datetime import datetime

from celery_app import celery_app
from core.state import PipelineState
from agents.orchestrator.dry_pipeline import run_dry_pipeline

# Map LangGraph node names → the agent keys the dashboard uses.
# Nodes not in this map (human_review, finalize, failed) are ignored for progress.
_NODE_TO_AGENT = {
    "ingestor": "ingestor",
    "sensor_parser": "sensor_parser",
    "classifier": "classifier",
    "root_cause": "root_cause",
    "trend_detector": "trend",
    "supplier_alert": "supplier_alert",
    "compliance_report": "compliance",
}


@celery_app.task(
    bind=True,
    name="tasks.inspection_task.run_pipeline",
    max_retries=3,
    default_retry_delay=5,
)
def run_pipeline(self, inspection_payload: dict, sensor_payload: list, full: bool = True) -> dict:
    """
    Run the ManufactureIQ pipeline for one inspection record.

    Args:
        inspection_payload: raw inspection JSON
        sensor_payload:     list of sensor reading dicts
        full:               True = full 7-agent pipeline, False = dry (3 agents)

    Returns:
        Final PipelineState as a JSON-safe dict.
    """
    state = PipelineState(
        run_id=str(uuid.uuid4()),
        status="received",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        raw_inspection_payload=inspection_payload,
        raw_sensor_payload=sensor_payload or [],
    )

    self.update_state(
        state="PROGRESS",
        meta={"run_id": state.run_id, "status": state.status, "current": None, "completed": []},
    )

    completed: list[str] = []

    def _on_step(name: str, payload) -> None:
        """Stream progress after each stage/node so the dashboard lights up agents."""
        # The concurrent "actions" node covers both supplier_alert + compliance.
        if name == "actions":
            keys = ["supplier_alert", "compliance"]
        else:
            k = _NODE_TO_AGENT.get(name)
            if k is None:
                return  # non-agent node (human_review / finalize)
            keys = [k]
        for k in keys:
            if k not in completed:
                completed.append(k)
        status = payload.get("status") if isinstance(payload, dict) else getattr(payload, "status", "running")
        self.update_state(
            state="PROGRESS",
            meta={"run_id": state.run_id, "status": status, "current": keys[-1], "completed": list(completed)},
        )

    if full:
        from agents.orchestrator.graph import run_graph
        final_state = run_graph(state, interrupt=False, on_step=_on_step)
    else:
        final_state = run_dry_pipeline(state, on_step=_on_step)

    return final_state.model_dump(mode="json")
