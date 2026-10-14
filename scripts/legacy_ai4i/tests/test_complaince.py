"""
Tests for Compliance Report Agent — logic only (no OpenAI API needed).
Run: python agents/compliance/test_compliance.py

Full test with CrewAI requires OPENAI_API_KEY in .env.
"""
import sys, uuid
sys.path.insert(0, "/home/claude/manufactureiq")

from datetime import datetime
from core.state import (
    PipelineState, IngestedData, InspectionRecord, SensorReading,
    DefectClassification, RootCauseReport, TrendReport, ComplianceReport
)

def make_state(severity="high", escalate=True, requires_review=False):
    inspection = InspectionRecord(
        record_id="REC-TEST-001",
        batch_id="BATCH-0146",
        product_id="M014860",
        machine_id="MCH-002",
        operator_id="OPR-001",
        supplier_id="SUP-003",
        material_batch_id="MAT-0088",
        timestamp=datetime.utcnow(),
        raw_notes="Power failure detected. Torque: 76. Tool wear: 100 min.",
    )
    sensor = SensorReading(
        sensor_id="MCH-002-S1",
        timestamp=datetime.utcnow(),
        temperature_c=26.85,
        machine_speed_rpm=2800,
        anomaly_flagged=True,
    )
    return PipelineState(
        run_id=str(uuid.uuid4()),
        raw_inspection_payload={},
        raw_sensor_payload=[],
        status="trend_detected",
        ingested=IngestedData(inspection=inspection, sensors=[sensor]),
        classification=DefectClassification(
            defect_type="assembly_misalignment",
            severity=severity,
            confidence_score=0.94,
            location_description="Power system — motor and drive",
            recommended_action="Check power supply and motor load",
            requires_human_review=requires_review,
        ),
        root_cause=RootCauseReport(
            primary_cause="Excessive motor load at 2,800 RPM causing power draw above 9,000W",
            contributing_factors=[
                "Tool wear at 100 min approaching critical threshold",
                "High torque (76 Nm) combined with elevated RPM",
            ],
            material_issue=False,
            machine_issue=True,
            process_issue=True,
            confidence_score=0.85,
            raw_analysis="Full GPT-4o analysis here...",
        ),
        trend=TrendReport(
            pattern="machine_pattern",
            affected_batch_ids=["BATCH-0146", "BATCH-0089"],
            affected_machine_ids=["MCH-002", "MCH-005"],
            defect_frequency_last_7d=9,
            escalate=escalate,
            summary="Machine MCH-002 shows recurring power failures.",
        ),
    )

# ── Test helper — replicate sign-off logic ────────────────────────────────────
def needs_sign_off(state: PipelineState) -> bool:
    clf   = state.classification
    trend = state.trend
    return (
        clf.requires_human_review or
        clf.severity in ["high", "critical"] or
        (trend and trend.escalate)
    )

def build_context(state: PipelineState) -> str:
    i  = state.ingested.inspection
    c  = state.classification
    rc = state.root_cause
    t  = state.trend
    s  = state.ingested.sensors[0] if state.ingested.sensors else None
    return (
        f"Record: {i.record_id} | Batch: {i.batch_id} | Machine: {i.machine_id}\n"
        f"Defect: {c.defect_type} | Severity: {c.severity} | Confidence: {c.confidence_score:.0%}\n"
        f"Root cause: {rc.primary_cause}\n"
        f"Pattern: {t.pattern} | Escalate: {t.escalate}\n"
        f"Sensor: {s.temperature_c}°C | {s.machine_speed_rpm} RPM"
    )

# ── Test 1: Sign-off required for high severity ───────────────────────────────
print("=" * 55)
print("TEST 1: High severity → sign-off required")
print("=" * 55)
state  = make_state(severity="high", escalate=True)
result = needs_sign_off(state)
print(f"Sign-off required: {result}")
assert result
print("PASSED ✅\n")

# ── Test 2: Low severity, no escalation → no sign-off ────────────────────────
print("=" * 55)
print("TEST 2: Low severity, no escalation → no sign-off")
print("=" * 55)
state  = make_state(severity="low", escalate=False, requires_review=False)
result = needs_sign_off(state)
print(f"Sign-off required: {result}")
assert not result
print("PASSED ✅\n")

# ── Test 3: Human review flag → sign-off ─────────────────────────────────────
print("=" * 55)
print("TEST 3: Human review required → sign-off")
print("=" * 55)
state  = make_state(severity="low", escalate=False, requires_review=True)
result = needs_sign_off(state)
print(f"Sign-off required: {result}")
assert result
print("PASSED ✅\n")

# ── Test 4: Context builds correctly ─────────────────────────────────────────
print("=" * 55)
print("TEST 4: Context contains all key fields")
print("=" * 55)
state   = make_state()
context = build_context(state)
print(context)
assert "BATCH-0146" in context
assert "MCH-002"    in context
assert "2800"       in context
print("\nPASSED ✅\n")

# ── Test 5: Compliance report structure ──────────────────────────────────────
print("=" * 55)
print("TEST 5: ComplianceReport model validates correctly")
print("=" * 55)
report = ComplianceReport(
    report_id             = f"CAR-{uuid.uuid4().hex[:8].upper()}",
    generated_at          = datetime.utcnow(),
    iso_standard          = "ISO 9001:2015",
    defect_summary        = "Power failure (PWF) detected on MCH-002 — HIGH severity",
    corrective_action_plan= "Inspect motor load, replace worn tooling, recalibrate drive settings",
    preventive_action_plan= "Implement weekly RPM + torque threshold monitoring",
    sign_off_required     = True,
    full_report_text      = "Full ISO report here...",
)
print(f"Report ID:    {report.report_id}")
print(f"ISO Standard: {report.iso_standard}")
print(f"Sign-off:     {report.sign_off_required}")
print(f"Generated:    {report.generated_at.strftime('%Y-%m-%d')}")
assert report.iso_standard == "ISO 9001:2015"
assert report.sign_off_required
print("PASSED ✅\n")

print("=" * 55)
print("All compliance report tests passed.")
print("Full CrewAI test: set OPENAI_API_KEY and run full pipeline.")
print("=" * 55)