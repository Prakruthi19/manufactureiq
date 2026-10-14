"""
Tests for LangGraph Orchestrator — routing logic only.
Run: python agents/orchestrator/test_graph.py

Full end-to-end test requires OPENAI_API_KEY in .env.
"""
import sys, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import datetime
from core.state import (
    PipelineState, IngestedData, InspectionRecord,
    DefectClassification, TrendReport
)
from agents.orchestrator.graph import route_after_classifier, route_after_trend, route_after_human_review

def make_state(severity="high", escalate=True, requires_review=False,
               human_approved=None, status="trend_detected"):
    inspection = InspectionRecord(
        record_id="REC-001", batch_id="BATCH-0146",
        product_id="M014860", machine_id="MCH-002",
        timestamp=datetime.utcnow(),
    )
    return PipelineState(
        run_id=str(uuid.uuid4()),
        status=status,
        raw_inspection_payload={},
        raw_sensor_payload=[],
        ingested=IngestedData(inspection=inspection, sensors=[]),
        classification=DefectClassification(
            defect_type="assembly_misalignment",
            severity=severity,
            confidence_score=0.94,
            location_description="Power system",
            recommended_action="Inspect motor",
            requires_human_review=requires_review,
        ),
        trend=TrendReport(
            pattern="machine_pattern",
            affected_batch_ids=["BATCH-0146"],
            affected_machine_ids=["MCH-002"],
            defect_frequency_last_7d=9,
            escalate=escalate,
            summary="Machine pattern detected.",
        ),
        human_approved=human_approved,
    )

# ── Test 1: Classifier routes to root_cause on success ────────────────────────
print("=" * 55)
print("TEST 1: Classifier → root_cause (normal flow)")
print("=" * 55)
state  = make_state(status="classified")
result = route_after_classifier(state)
print(f"Route: {result}")
assert result == "root_cause"
print("PASSED ✅\n")

# ── Test 2: Classifier routes to failed on failure ────────────────────────────
print("=" * 55)
print("TEST 2: Classifier → failed (pipeline failed)")
print("=" * 55)
state        = make_state(status="failed")
state.status = "failed"
result       = route_after_classifier(state)
print(f"Route: {result}")
assert result == "failed"
print("PASSED ✅\n")

# ── Test 3: High severity + escalate → human_review ─────────────────────────
print("=" * 55)
print("TEST 3: High severity + escalate → human_review")
print("=" * 55)
state  = make_state(severity="high", escalate=True)
result = route_after_trend(state)
print(f"Route: {result}")
assert result == "human_review"
print("PASSED ✅\n")

# ── Test 4: Low severity, no escalate → parallel_actions ─────────────────────
print("=" * 55)
print("TEST 4: Low severity, no escalate → parallel_actions")
print("=" * 55)
state  = make_state(severity="low", escalate=False, requires_review=False)
result = route_after_trend(state)
print(f"Route: {result}")
assert result == "parallel_actions"
print("PASSED ✅\n")

# ── Test 5: Critical severity → human_review regardless ──────────────────────
print("=" * 55)
print("TEST 5: Critical severity → always human_review")
print("=" * 55)
state  = make_state(severity="critical", escalate=False)
result = route_after_trend(state)
print(f"Route: {result}")
assert result == "human_review"
print("PASSED ✅\n")

# ── Test 6: Human approved → parallel_actions ────────────────────────────────
print("=" * 55)
print("TEST 6: Human approved → parallel_actions")
print("=" * 55)
state  = make_state(human_approved=True)
result = route_after_human_review(state)
print(f"Route: {result}")
assert result == "parallel_actions"
print("PASSED ✅\n")

# ── Test 7: Human rejected → failed ──────────────────────────────────────────
print("=" * 55)
print("TEST 7: Human rejected → failed")
print("=" * 55)
state  = make_state(human_approved=False)
result = route_after_human_review(state)
print(f"Route: {result}")
assert result == "failed"
print("PASSED ✅\n")

# ── Test 8: Full graph builds without error ───────────────────────────────────
print("=" * 55)
print("TEST 8: Graph compiles successfully")
print("=" * 55)
from agents.orchestrator.graph import compile_graph, build_graph
app   = compile_graph()
graph = build_graph()
nodes = list(graph.nodes.keys())
print(f"Graph nodes: {nodes}")
assert "ingestor"          in nodes
assert "sensor_parser"     in nodes
assert "classifier"        in nodes
assert "root_cause"        in nodes
assert "trend_detector"    in nodes
assert "human_review"      in nodes
assert "supplier_alert"    in nodes
assert "compliance_report" in nodes
assert "finalize"          in nodes
print("PASSED ✅\n")

print("=" * 55)
print("All orchestrator tests passed.")
print("Full pipeline test: python scripts/run_pipeline.py")
print("=" * 55)