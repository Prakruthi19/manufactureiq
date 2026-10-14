"""
ManufactureIQ — LangGraph Orchestrator
========================================
Wires all 7 agents into a single execution graph.

Graph flow:
    ingestor → sensor_parser → classifier
             → root_cause → trend_detector
             → [conditional] human_review? → pause
             → [parallel] supplier_alert + compliance_report
             → done

Features:
    - Sequential routing through Tiers 1 and 2
    - Conditional edge: high/critical defects pause for human review
    - Parallel execution: Agents 6 + 7 run simultaneously
    - Human-in-the-loop: pipeline pauses, waits for approval
    - Full state preserved at every step via LangGraph checkpointing
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from core.state import PipelineState

# ── Import agent node functions ───────────────────────────────────────────────
from agents.ingestor.ingestor_agent       import ingestor_node
from agents.ingestor.sensor_parser_agent  import sensor_parser_node
from agents.classifier.defect_classifier_agent import defect_classifier_node
from agents.trend.trend_detector_agent    import trend_detector_node

logger = logging.getLogger(__name__)

# ── Lazy imports for LLM agents (avoid loading if not needed) ─────────────────
def _root_cause_node(state: PipelineState) -> PipelineState:
    from agents.root_cause.root_cause_agent import root_cause_node
    return root_cause_node(state)

def _supplier_alert_node(state: PipelineState) -> PipelineState:
    from agents.supplier.supplier_alert_agent import supplier_alert_node
    return supplier_alert_node(state)

def _compliance_report_node(state: PipelineState) -> PipelineState:
    from agents.complaince.complaince_agent import compliance_report_node
    return compliance_report_node(state)


# ── Human review node ─────────────────────────────────────────────────────────
def human_review_node(state: PipelineState) -> PipelineState:
    """
    Pause point for human-in-the-loop review.
    LangGraph interrupts here when requires_human_review=True.
    The pipeline resumes when the manager calls resume_pipeline().
    """
    logger.info(f"[HumanReview] Paused — run_id: {state.run_id} — awaiting approval")
    # State is preserved by LangGraph checkpointer
    # Manager reviews via API: GET /pipeline/{run_id}/review
    # Manager approves via API: POST /pipeline/{run_id}/approve
    return state


# ── Final node ────────────────────────────────────────────────────────────────
def finalize_node(state: PipelineState) -> PipelineState:
    """Mark pipeline as completed."""
    from datetime import datetime
    state.status     = "completed"
    state.updated_at = datetime.utcnow()
    logger.info(
        f"[Orchestrator] Pipeline complete — run_id: {state.run_id} | "
        f"defect: {state.classification.defect_type if state.classification else 'N/A'} | "
        f"pattern: {state.trend.pattern if state.trend else 'N/A'}"
    )
    return state


# ── Conditional routing functions ─────────────────────────────────────────────

def route_after_classifier(
    state: PipelineState,
) -> Literal["root_cause", "healthy", "failed"]:
    """After classification — continue, skip (healthy), or fail."""
    if state.status == "failed":
        return "failed"
    # No defect → skip the LLM agents, but still run the (cheap, ML-only) trend
    # detector so the Isolation Forest can flag unusual-but-unclassified readings.
    if state.classification and state.classification.defect_type == "no_defect":
        logger.info("[Orchestrator] No defect detected — skipping LLM analysis, running anomaly check")
        return "healthy"
    return "root_cause"


def route_after_trend(
    state: PipelineState,
) -> Literal["human_review", "parallel_actions", "healthy"]:
    """
    After trend detection — decide if human review is needed.

    Human review triggered if:
    - Classifier flagged requires_human_review = True
    - Defect severity is critical
    - Trend says escalate = True AND severity is high/critical
    """
    if state.status == "failed":
        return "parallel_actions"   # still run actions even if trend failed

    # Healthy record — anomaly check done, nothing for the action tier to do.
    if state.classification and state.classification.defect_type == "no_defect":
        return "healthy"

    clf   = state.classification
    trend = state.trend

    needs_review = (
        (clf and clf.requires_human_review) or
        (clf and clf.severity == "critical") or
        (trend and trend.escalate and clf and clf.severity in ["high", "critical"])
    )

    if needs_review:
        logger.info(
            f"[Orchestrator] Human review required — "
            f"severity: {clf.severity if clf else 'N/A'} | "
            f"escalate: {trend.escalate if trend else False}"
        )
        return "human_review"

    return "parallel_actions"


def route_after_human_review(
    state: PipelineState,
) -> Literal["parallel_actions", "failed"]:
    """After human review — approved or rejected."""
    if state.human_approved is False:
        logger.info(f"[Orchestrator] Human rejected — halting pipeline")
        state.status = "failed"
        state.errors.append("Pipeline halted: rejected by human reviewer")
        return "failed"
    return "parallel_actions"


# ── Parallel action node ──────────────────────────────────────────────────────
# Supplier Alert (Agent 6) and Compliance Report (Agent 7) are independent, so we
# run them CONCURRENTLY in one node (on state copies) and merge — avoids LangGraph's
# concurrent-write-to-root error while still overlapping the two LLM calls.

def run_actions_node(state: PipelineState) -> PipelineState:
    """Run Supplier Alert + Compliance Report concurrently, then merge results."""
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime as _dt

    s_sup  = state.model_copy(deep=True)
    s_comp = state.model_copy(deep=True)
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_sup  = ex.submit(_supplier_alert_node, s_sup)
        f_comp = ex.submit(_compliance_report_node, s_comp)
        r_sup, r_comp = f_sup.result(), f_comp.result()

    state.supplier_alert    = r_sup.supplier_alert
    state.compliance_report = r_comp.compliance_report
    for e in (r_sup.errors + r_comp.errors):
        if e not in state.errors:
            state.errors.append(e)
    state.updated_at = _dt.utcnow()
    return state


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct the ManufactureIQ LangGraph execution graph.
    Returns a compiled graph ready to run.
    """

    # LangGraph requires state to be a dict or TypedDict
    # We use PipelineState.model_dump() / model_validate() for conversion
    from typing import Any

    def to_dict(state): return state.model_dump(mode="json") if hasattr(state, "model_dump") else state
    def from_dict(d):   return PipelineState.model_validate(d) if isinstance(d, dict) else d

    # Wrap each node to handle Pydantic ↔ dict conversion
    def wrap(fn):
        def wrapped(state):
            s = from_dict(state) if isinstance(state, dict) else state
            r = fn(s)
            return r.model_dump(mode="json") if hasattr(r, "model_dump") else r
        wrapped.__name__ = fn.__name__
        return wrapped

    # ── Graph definition ──────────────────────────────────────────────────────
    graph = StateGraph(dict)

    # Add nodes
    graph.add_node("ingestor",          wrap(ingestor_node))
    graph.add_node("sensor_parser",     wrap(sensor_parser_node))
    graph.add_node("classifier",        wrap(defect_classifier_node))
    graph.add_node("root_cause",        wrap(_root_cause_node))
    graph.add_node("trend_detector",    wrap(trend_detector_node))
    graph.add_node("human_review",      wrap(human_review_node))
    graph.add_node("actions",           wrap(run_actions_node))
    graph.add_node("finalize",          wrap(finalize_node))
    graph.add_node("failed",            wrap(lambda s: s))

    # ── Sequential edges — Tier 1 + Tier 2 ───────────────────────────────────
    graph.set_entry_point("ingestor")
    graph.add_edge("ingestor",       "sensor_parser")
    graph.add_edge("sensor_parser",  "classifier")

    # Conditional after classifier
    graph.add_conditional_edges(
        "classifier",
        lambda s: route_after_classifier(from_dict(s) if isinstance(s, dict) else s),
        {"root_cause": "root_cause", "healthy": "trend_detector", "failed": "failed"}
    )

    graph.add_edge("root_cause",     "trend_detector")

    # Conditional after trend — human review or actions
    graph.add_conditional_edges(
        "trend_detector",
        lambda s: route_after_trend(from_dict(s) if isinstance(s, dict) else s),
        {
            "human_review":    "human_review",
            "parallel_actions": "actions",
            "healthy":          "finalize",
        }
    )

    # Human review → conditional resume
    graph.add_conditional_edges(
        "human_review",
        lambda s: route_after_human_review(from_dict(s) if isinstance(s, dict) else s),
        {
            "parallel_actions": "actions",
            "failed":           "failed",
        }
    )

    # ── Actions (supplier + compliance run concurrently) → finalize ──────────
    graph.add_edge("actions",   "finalize")
    graph.add_edge("finalize",  END)
    graph.add_edge("failed",    END)

    return graph


def compile_graph(checkpointer=None, interrupt: bool = True):
    """Compile graph with optional checkpointer for state persistence.

    Args:
        interrupt: if True, pause before the human_review node (interactive mode).
                   if False, the human_review node passes through and the pipeline
                   runs all agents end-to-end without stopping.
    """
    graph = build_graph()
    memory = checkpointer or MemorySaver()
    return graph.compile(
        checkpointer=memory,
        interrupt_before=["human_review"] if interrupt else [],
    )


# ── Public API ────────────────────────────────────────────────────────────────

def run_graph(state: PipelineState, thread_id: str = None, interrupt: bool = True,
              on_step=None) -> PipelineState:
    """
    Run the full pipeline for one inspection record.
    Returns final PipelineState.

    Args:
        state:     Initial PipelineState with raw payloads
        thread_id: Optional thread ID for checkpointing (enables resume)
        interrupt: pause before human_review (True) or run straight through (False)
        on_step:   optional callback(node_name, state_dict) after each node — used
                   by the Celery task to stream progress to the dashboard.
    """
    import uuid
    app       = compile_graph(interrupt=interrupt)
    thread_id = thread_id or str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    input_dict = state.model_dump(mode="json")

    # Run the graph
    final = None
    for step in app.stream(input_dict, config=config):
        node_name = list(step.keys())[0]
        node_out  = step[node_name]
        status    = node_out.get("status", "unknown") if isinstance(node_out, dict) else "unknown"
        logger.info(f"[Graph] ✓ {node_name} → {status}")
        final = node_out
        if on_step is not None and isinstance(node_out, dict):
            on_step(node_name, node_out)

    return PipelineState.model_validate(final) if final else state


def resume_graph(thread_id: str, approved: bool, notes: str = "") -> PipelineState:
    """
    Resume a paused pipeline after human review.

    Args:
        thread_id: The pipeline thread to resume
        approved:  True = approve and continue, False = reject and halt
        notes:     Optional reviewer notes
    """
    app    = compile_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # Update state with human decision
    current = app.get_state(config)
    update  = {
        "human_approved":      approved,
        "human_review_notes":  notes,
    }
    app.update_state(config, update)

    # Resume from interrupt
    final = None
    for step in app.stream(None, config=config):
        node_name = list(step.keys())[0]
        logger.info(f"[Graph] ✓ {node_name} (resumed)")
        final = step[node_name]

    return PipelineState.model_validate(final) if final else None