"""
Tests for Defect Classifier Agent.
Run: python agents/classifier/test_classifier.py
"""
import sys, uuid, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import datetime
from core.state import PipelineState
from agents.ingestor.ingestor_agent import DataIngestorAgent
from agents.ingestor.sensor_parser_agent import SensorParserAgent
from agents.classifier.defect_classifier_agent import DefectClassifierAgent

ingestor   = DataIngestorAgent()
parser     = SensorParserAgent()
classifier = DefectClassifierAgent()

def run_pipeline(inspection: dict, sensors: list) -> PipelineState:
    """Run first 3 agents in sequence."""
    state = PipelineState(
        run_id=str(uuid.uuid4()),
        raw_inspection_payload=inspection,
        raw_sensor_payload=sensors,
    )
    state = ingestor.run(state)
    state = parser.run(state)
    state = classifier.run(state)
    return state

# ── Test 1: Power failure conditions (PWF) ────────────────────────────────────
print("=" * 55)
print("TEST 1: Power Failure conditions → expect PWF")
print("=" * 55)
state = run_pipeline(
    inspection={
        "record_id":  "REC-PWF-001",
        "batch_id":   "BATCH-0146",
        "product_id": "M014860",
        "machine_id": "MCH-002",
        "timestamp":  datetime.utcnow().isoformat(),
        "raw_notes":  "High power draw detected. Torque: 76.5. Tool wear: 100 min.",
    },
    sensors=[{
        "sensor_id":         "MCH-002-S1",
        "timestamp":         datetime.utcnow().isoformat(),
        "temperature_c":     26.85,   # 300K
        "machine_speed_rpm": 2800,    # very high → high power
        "anomaly_flagged":   True,
    }]
)
c = state.classification
print(f"Status:          {state.status}")
print(f"Defect type:     {c.defect_type}")
print(f"Severity:        {c.severity}")
print(f"Confidence:      {c.confidence_score:.2%}")
print(f"Action:          {c.recommended_action}")
print(f"Human review:    {c.requires_human_review}")
assert state.status == "classified"
print("PASSED ✅\n")

# ── Test 2: Normal conditions → expect none ───────────────────────────────────
print("=" * 55)
print("TEST 2: Normal conditions → expect no defect")
print("=" * 55)
state = run_pipeline(
    inspection={
        "record_id":  "REC-NORM-001",
        "batch_id":   "BATCH-0001",
        "product_id": "L047181",
        "machine_id": "MCH-001",
        "timestamp":  datetime.utcnow().isoformat(),
        "raw_notes":  "Routine inspection. Tool wear: 3 min. Torque: 46.",
    },
    sensors=[{
        "sensor_id":         "MCH-001-S1",
        "timestamp":         datetime.utcnow().isoformat(),
        "temperature_c":     25.05,   # 298.2K — normal
        "machine_speed_rpm": 1408,    # normal
        "anomaly_flagged":   False,
    }]
)
c = state.classification
print(f"Status:       {state.status}")
print(f"Defect type:  {c.defect_type}")
print(f"Severity:     {c.severity}")
print(f"Confidence:   {c.confidence_score:.2%}")
assert state.status == "classified"
print("PASSED ✅\n")

# ── Test 3: Real sample from dataset ─────────────────────────────────────────
print("=" * 55)
print("TEST 3: Real sample payload from dataset")
print("=" * 55)
from pathlib import Path
data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "sample_payloads.json"
with open(data_path) as f:
    samples = json.load(f)

# Pick a failure sample
sample = samples[2]
state = PipelineState(
    run_id=str(uuid.uuid4()),
    raw_inspection_payload=sample["inspection"],
    raw_sensor_payload=sample["sensors"],
)
state = ingestor.run(state)
state = parser.run(state)
state = classifier.run(state)

c = state.classification
print(f"Status:       {state.status}")
print(f"Defect type:  {c.defect_type}")
print(f"Severity:     {c.severity}")
print(f"Confidence:   {c.confidence_score:.2%}")
print(f"Action:       {c.recommended_action}")
assert state.status == "classified"
print("PASSED ✅\n")

# ── Test 4: Pipeline state after 3 agents ─────────────────────────────────────
print("=" * 55)
print("TEST 4: Verify full state after 3 agents")
print("=" * 55)
print(f"run_id:              {state.run_id}")
print(f"status:              {state.status}")
print(f"ingested:            ✅ {state.ingested.inspection.product_id}")
print(f"sensor_summary:      ✅ anomaly={state.sensor_summary['any_anomaly']}")
print(f"classification:      ✅ {state.classification.defect_type} ({state.classification.severity})")
print(f"root_cause:          ⏳ not yet (Agent 4)")
print(f"trend:               ⏳ not yet (Agent 5)")
print(f"supplier_alert:      ⏳ not yet (Agent 6)")
print(f"compliance_report:   ⏳ not yet (Agent 7)")
assert state.ingested is not None
assert state.sensor_summary is not None
assert state.classification is not None
print("PASSED ✅\n")

print("=" * 55)
print("All classifier tests passed.")
print("=" * 55)