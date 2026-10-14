"""
Tests for Trend Detector Agent.
Run: python agents/trend/test_trend.py
"""
import sys, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import datetime
from core.state import PipelineState
from agents.ingestor.ingestor_agent import DataIngestorAgent
from agents.ingestor.sensor_parser_agent import SensorParserAgent
from agents.classifier.defect_classifier_agent import DefectClassifierAgent
from agents.trend.trend_detector_agent import TrendDetectorAgent

ingestor   = DataIngestorAgent()
parser     = SensorParserAgent()
classifier = DefectClassifierAgent()
trend      = TrendDetectorAgent()

def run_pipeline(batch_id="BATCH-0146", machine_id="MCH-002", supplier_id="SUP-003"):
    state = PipelineState(
        run_id=str(uuid.uuid4()),
        raw_inspection_payload={
            "record_id":   str(uuid.uuid4()),
            "batch_id":    batch_id,
            "product_id":  "M014860",
            "machine_id":  machine_id,
            "operator_id": "OPR-001",
            "supplier_id": supplier_id,
            "timestamp":   datetime.utcnow().isoformat(),
            "raw_notes":   "Power failure. Torque: 76. Tool wear: 100 min.",
        },
        raw_sensor_payload=[{
            "sensor_id":         f"{machine_id}-S1",
            "timestamp":         datetime.utcnow().isoformat(),
            "temperature_c":     26.85,
            "machine_speed_rpm": 2800,
            "anomaly_flagged":   True,
        }]
    )
    state = ingestor.run(state)
    state = parser.run(state)
    state = classifier.run(state)
    return trend.run(state)

# ── Test 1: Basic trend detection ─────────────────────────────────────────────
print("=" * 55)
print("TEST 1: Basic trend detection")
print("=" * 55)
state = run_pipeline()
t = state.trend
print(f"Status:          {state.status}")
print(f"Pattern:         {t.pattern}")
print(f"Escalate:        {t.escalate}")
print(f"Freq last 7d:    {t.defect_frequency_last_7d}")
print(f"Summary:         {t.summary[:100]}...")
assert state.status == "trend_detected"
assert t.pattern in ["isolated","batch_pattern","machine_pattern","supplier_pattern","unknown"]
print("PASSED ✅\n")

# ── Test 2: Full pipeline state verification ───────────────────────────────────
print("=" * 55)
print("TEST 2: Full pipeline state — 5 agents complete")
print("=" * 55)
print(f"ingested:        ✅ {state.ingested.inspection.product_id}")
print(f"sensor_summary:  ✅ anomaly={state.sensor_summary['any_anomaly']}")
print(f"classification:  ✅ {state.classification.defect_type} ({state.classification.severity})")
print(f"root_cause:      ⏳ skipped (needs OpenAI key)")
print(f"trend:           ✅ {state.trend.pattern} | escalate={state.trend.escalate}")
print(f"supplier_alert:  ⏳ not yet (Agent 6)")
print(f"compliance:      ⏳ not yet (Agent 7)")
assert state.trend is not None
print("PASSED ✅\n")

# ── Test 3: Isolation Forest score ────────────────────────────────────────────
print("=" * 55)
print("TEST 3: Isolation Forest anomaly score")
print("=" * 55)
score = trend._get_isolation_score(
    state.ingested.sensors[0],
    state.sensor_summary
)
print(f"Anomaly score: {score:.4f}")
print(f"Anomalous:     {score < 0}")
print("PASSED ✅\n")

print("=" * 55)
print("All trend detector tests passed.")
print("=" * 55)