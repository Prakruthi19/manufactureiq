"""
ManufactureIQ — Dry Pipeline Runner
====================================
A lightweight, LLM-free execution path used for the first end-to-end
milestone (backend → worker → frontend).

Runs only the deterministic Tier-1 + ML agents:
    ingestor → sensor_parser → classifier

No OpenAI, no ChromaDB. Once the full LLM pipeline (root_cause, supplier,
compliance) is wired and its import paths fixed, swap callers over to
`agents.orchestrator.graph.run_graph`.
"""

import logging
from typing import Callable

from core.state import PipelineState
from agents.ingestor.ingestor_agent import ingestor_node
from agents.ingestor.sensor_parser_agent import sensor_parser_node
from agents.classifier.defect_classifier_agent import defect_classifier_node

logger = logging.getLogger(__name__)

# Ordered stages of the dry pipeline. Each is a (state) -> state function.
DRY_STAGES: list[tuple[str, Callable[[PipelineState], PipelineState]]] = [
    ("ingestor",       ingestor_node),
    ("sensor_parser",  sensor_parser_node),
    ("classifier",     defect_classifier_node),
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
        Final PipelineState (status == "classified", or "failed" if a stage halted).
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
