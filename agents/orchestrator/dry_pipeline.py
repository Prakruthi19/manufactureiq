"""
ManufactureIQ — Dry Pipeline Runner
====================================
A lightweight, LLM-free execution path used for the first end-to-end
milestone (backend → worker → frontend).

Runs only the deterministic Tier-1 + ML agents:
    ingestor → sensor_parser → classifier → trend_detector

No LLM calls, so a batch of wafers costs nothing and finishes in seconds. The
trend detector is included so batch runs build the lot/tool history that
pattern detection relies on. Single inspections use the full graph
(`agents.orchestrator.graph.run_graph`).
"""

import logging
from typing import Callable

from core.state import PipelineState
from agents.ingestor.ingestor_agent import ingestor_node
from agents.ingestor.sensor_parser_agent import sensor_parser_node
from agents.classifier.defect_classifier_agent import defect_classifier_node
from agents.trend.trend_detector_agent import trend_detector_node

logger = logging.getLogger(__name__)

# Ordered stages of the dry pipeline. Each is a (state) -> state function.
DRY_STAGES: list[tuple[str, Callable[[PipelineState], PipelineState]]] = [
    ("ingestor",       ingestor_node),
    ("sensor_parser",  sensor_parser_node),
    ("classifier",     defect_classifier_node),
    ("trend_detector", trend_detector_node),
]


def run_dry_pipeline(
    state: PipelineState,
    on_step: Callable[[str, PipelineState], None] | None = None,
) -> PipelineState:
    """
    Run the dry pipeline synchronously.

    Args:
        state:   initial PipelineState with raw payloads.
        on_step: optional callback(stage_name, state) invoked after each stage —
                 used by the Celery task to publish progress updates.

    Returns:
        Final PipelineState (status == "trend_detected", or "failed" if a stage halted).
    """
    for name, node in DRY_STAGES:
        state = node(state)
        logger.info(f"[DryPipeline] {name} -> {state.status}")
        if on_step is not None:
            on_step(name, state)
        if state.status == "failed":
            logger.warning(f"[DryPipeline] halted at '{name}' — {state.errors}")
            break
    return state
