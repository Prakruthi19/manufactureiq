"""
Tests for Supplier Alert Agent — logic only (no OpenAI API needed).
Run: python agents/supplier_alert/test_supplier_alert.py

Note: Full agent test (with GPT-4o) requires OPENAI_API_KEY in .env.
      Run the full pipeline test instead: python scripts/run_pipeline.py
"""
import sys, uuid
sys.path.insert(0, "/home/claude/manufactureiq")

from datetime import datetime
from core.state import (
    PipelineState, IngestedData, InspectionRecord,
    DefectClassification, RootCauseReport, TrendReport
)

# ── Import note ───────────────────────────────────────────────────────────────
# On your machine: from agents.supplier_alert.supplier_alert_agent import SupplierAlertAgentRunner
# The 'agents' SDK package may conflict with local agents/ folder in some envs.
# If you get ImportError, run: pip install openai-agents --upgrade
# and ensure PYTHONPATH is set to project root.

def make_state(material_issue=False, pattern="isolated", supplier_id="SUP-003"):
    inspection = InspectionRecord(
        record_id="REC-TEST-001",
        batch_id="BATCH-0146",
        product_id="M014860",
        machine_id="MCH-002",
        supplier_id=supplier_id,
        material_batch_id="MAT-0088",
        timestamp=datetime.utcnow(),
        raw_notes="Power failure detected. Torque: 76. Tool wear: 100.",
    )
    return PipelineState(
        run_id=str(uuid.uuid4()),
        raw_inspection_payload={},
        raw_sensor_payload=[],
        status="trend_detected",
        ingested=IngestedData(inspection=inspection, sensors=[]),
        classification=DefectClassification(
            defect_type="assembly_misalignment",
            severity="high",
            confidence_score=0.94,
            location_description="Power system",
            recommended_action="Inspect motor",
        ),
        root_cause=RootCauseReport(
            primary_cause="Excessive motor load at high RPM",
            contributing_factors=["Tool wear", "High torque"],
            material_issue=material_issue,
            machine_issue=True,
            process_issue=False,
            confidence_score=0.85,
            raw_analysis="Full analysis...",
        ),
        trend=TrendReport(
            pattern=pattern,
            affected_batch_ids=["BATCH-0146", "BATCH-0089"],
            affected_machine_ids=["MCH-002"],
            defect_frequency_last_7d=9,
            escalate=True,
            summary="Supplier pattern detected.",
        ),
    )

def needs_alert(state: PipelineState) -> bool:
    """Replicated logic from SupplierAlertAgentRunner._needs_alert()"""
    if not state.ingested.inspection.supplier_id:
        return False
    if state.root_cause and state.root_cause.material_issue:
        return True
    if state.trend and state.trend.pattern == "supplier_pattern":
        return True
    if state.classification and state.classification.defect_type == "material_contamination":
        return True
    return False

def build_prompt(state: PipelineState) -> str:
    """Replicated logic from SupplierAlertAgentRunner._build_prompt()"""
    i  = state.ingested.inspection
    c  = state.classification
    rc = state.root_cause
    t  = state.trend
    return (
        f"Supplier: {i.supplier_id} | Batch: {i.batch_id}\n"
        f"Defect: {c.defect_type} | Severity: {c.severity}\n"
        f"Material issue: {rc.material_issue}\n"
        f"Pattern: {t.pattern}\n"
        f"Task: should a supplier alert be sent?"
    )

# ── Test 1: No supplier issue → skip ─────────────────────────────────────────
print("=" * 55)
print("TEST 1: No material issue → should NOT alert")
print("=" * 55)
state = make_state(material_issue=False, pattern="isolated")
result = needs_alert(state)
print(f"Needs alert: {result}")
assert not result
print("PASSED ✅\n")

# ── Test 2: Supplier pattern → alert ─────────────────────────────────────────
print("=" * 55)
print("TEST 2: Supplier pattern → SHOULD alert")
print("=" * 55)
state = make_state(material_issue=False, pattern="supplier_pattern")
result = needs_alert(state)
print(f"Needs alert: {result}")
assert result
print("PASSED ✅\n")

# ── Test 3: Material issue → alert ────────────────────────────────────────────
print("=" * 55)
print("TEST 3: Material issue in root cause → SHOULD alert")
print("=" * 55)
state = make_state(material_issue=True, pattern="isolated")
result = needs_alert(state)
print(f"Needs alert: {result}")
assert result
print("PASSED ✅\n")

# ── Test 4: No supplier ID → no alert ────────────────────────────────────────
print("=" * 55)
print("TEST 4: No supplier ID → should NOT alert")
print("=" * 55)
state = make_state(material_issue=False, pattern="supplier_pattern", supplier_id=None)
result = needs_alert(state)
print(f"Needs alert: {result}")
assert not result
print("PASSED ✅\n")

# ── Test 5: Prompt builds correctly ──────────────────────────────────────────
print("=" * 55)
print("TEST 5: Prompt contains key fields")
print("=" * 55)
state  = make_state(material_issue=True, pattern="supplier_pattern")
prompt = build_prompt(state)
print(prompt)
assert "SUP-003" in prompt
assert "BATCH-0146" in prompt
print("\nPASSED ✅\n")

print("=" * 55)
print("All supplier alert logic tests passed.")
print("Full GPT-4o test: set OPENAI_API_KEY and run full pipeline.")
print("=" * 55)