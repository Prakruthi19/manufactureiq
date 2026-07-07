"""
Quick test for the Data Ingestor Agent.
Run: python agents/ingestor/test_ingestor.py
"""
import sys, json, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Windows consoles default to cp1252, which can't encode the ✅ emojis below.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from datetime import datetime
from core.state import PipelineState
from agents.ingestor.ingestor_agent import DataIngestorAgent

agent = DataIngestorAgent()

# ── Test 1: Valid payload ─────────────────────────────────────────────────────
print("=" * 50)
print("TEST 1: Valid payload")
print("=" * 50)

state = PipelineState(
    run_id=str(uuid.uuid4()),
    raw_inspection_payload={
        "record_id":        "REC-001",
        "batch_id":         "BATCH-0146",
        "product_id":       "M014860",
        "machine_id":       "MCH-002",
        "operator_id":      "OPR-003",
        "timestamp":        datetime.utcnow().isoformat(),
        "supplier_id":      "SUP-003",
        "raw_notes":        "Power failure detected. Torque spike at 1,741 rpm.",
        "material_batch_id": "BATCH-0146",
    },
    raw_sensor_payload=[{
        "sensor_id":         "MCH-002-S1",
        "timestamp":         datetime.utcnow().isoformat(),
        "temperature_c":     27.65,
        "machine_speed_rpm": 1741,
        "anomaly_flagged":   True,
    }]
)

result = agent.run(state)
print(f"Status:   {result.status}")
print(f"Product:  {result.ingested.inspection.product_id}")
print(f"Batch:    {result.ingested.inspection.batch_id}")
print(f"Sensors:  {len(result.ingested.sensors)}")
print(f"Warnings: {result.ingested.ingest_warnings}")
assert result.status == "ingested"
print("PASSED ✅\n")

# ── Test 2: Missing critical field ────────────────────────────────────────────
print("=" * 50)
print("TEST 2: Missing critical field (machine_id)")
print("=" * 50)

state2 = PipelineState(
    run_id=str(uuid.uuid4()),
    raw_inspection_payload={
        "record_id":  "REC-002",
        "batch_id":   "BATCH-0001",
        "product_id": "L047181",
        "timestamp":  datetime.utcnow().isoformat(),
        # machine_id missing
    },
    raw_sensor_payload=[]
)

result2 = agent.run(state2)
print(f"Status: {result2.status}")
print(f"Errors: {result2.errors}")
assert result2.status == "failed"
print("PASSED ✅\n")

# ── Test 3: Missing sensor_id (auto-assigned) ─────────────────────────────────
print("=" * 50)
print("TEST 3: Sensor missing sensor_id — should auto-assign")
print("=" * 50)

state3 = PipelineState(
    run_id=str(uuid.uuid4()),
    raw_inspection_payload={
        "record_id":  "REC-003",
        "batch_id":   "BATCH-0002",
        "product_id": "H029424",
        "machine_id": "MCH-005",
        "timestamp":  datetime.utcnow().isoformat(),
    },
    raw_sensor_payload=[{
        # no sensor_id
        "timestamp":         datetime.utcnow().isoformat(),
        "temperature_c":     25.0,
        "machine_speed_rpm": 1500,
        "anomaly_flagged":   False,
    }]
)

result3 = agent.run(state3)
print(f"Status:    {result3.status}")
print(f"Sensor ID: {result3.ingested.sensors[0].sensor_id}")
print(f"Warnings:  {result3.ingested.ingest_warnings}")
assert result3.status == "ingested"
assert "MCH-005" in result3.ingested.sensors[0].sensor_id
print("PASSED ✅\n")

# ── Test 4: Load from real sample_payloads.json ────────────────────────────────
print("=" * 50)
print("TEST 4: Real sample payload from dataset")
print("=" * 50)

_ROOT = Path(__file__).resolve().parent.parent.parent
with open(_ROOT / "data" / "raw" / "sample_payloads.json") as f:
    samples = json.load(f)

sample = samples[0]
state4 = PipelineState(
    run_id=str(uuid.uuid4()),
    raw_inspection_payload=sample["inspection"],
    raw_sensor_payload=sample["sensors"],
)

result4 = agent.run(state4)
print(f"Status:   {result4.status}")
print(f"Product:  {result4.ingested.inspection.product_id}")
print(f"Notes:    {result4.ingested.inspection.raw_notes}")
print(f"Sensors:  {len(result4.ingested.sensors)}")
assert result4.status == "ingested"
print("PASSED ✅\n")

print("=" * 50)
print("All tests passed.")
print("=" * 50)