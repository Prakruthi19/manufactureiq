"""
Tests for Sensor Parser Agent.
Run: python agents/ingestor/test_sensor_parser.py
"""
import sys, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Windows consoles default to cp1252, which can't encode emojis in the output.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from datetime import datetime
from core.state import PipelineState, IngestedData, InspectionRecord, SensorReading
from agents.ingestor.ingestor_agent import DataIngestorAgent
from agents.ingestor.sensor_parser_agent import SensorParserAgent

ingestor = DataIngestorAgent()
parser   = SensorParserAgent()

def make_state(sensors: list[dict], notes: str = "") -> PipelineState:
    state = PipelineState(
        run_id=str(uuid.uuid4()),
        raw_inspection_payload={
            "record_id":  str(uuid.uuid4()),
            "batch_id":   "BATCH-0146",
            "product_id": "M014860",
            "machine_id": "MCH-002",
            "timestamp":  datetime.utcnow().isoformat(),
            "raw_notes":  notes,
        },
        raw_sensor_payload=sensors
    )
    return ingestor.run(state)  # run ingestor first

# ── Test 1: Normal sensor readings — no anomalies ─────────────────────────────
print("=" * 50)
print("TEST 1: Normal readings — no anomalies expected")
print("=" * 50)
state = make_state([{
    "sensor_id":         "MCH-002-S1",
    "timestamp":         datetime.utcnow().isoformat(),
    "temperature_c":     26.5,
    "machine_speed_rpm": 1500,
    "anomaly_flagged":   False,
}])
result = parser.run(state)
summary = result.sensor_summary
print(f"Any anomaly:   {summary['any_anomaly']}")
print(f"Anomaly count: {summary['anomaly_count']}")
print(f"Risk flags:    {summary['risk_flags']}")
assert not summary["any_anomaly"]
print("PASSED ✅\n")

# ── Test 2: Low RPM — HDF + PWF risk ─────────────────────────────────────────
print("=" * 50)
print("TEST 2: Low RPM → HDF + PWF risk")
print("=" * 50)
state = make_state([{
    "sensor_id":         "MCH-002-S1",
    "timestamp":         datetime.utcnow().isoformat(),
    "temperature_c":     27.0,
    "machine_speed_rpm": 1100,   # below 1168 threshold
    "anomaly_flagged":   True,
}])
result = parser.run(state)
summary = result.sensor_summary
print(f"Any anomaly:   {summary['any_anomaly']}")
print(f"Risk flags:    {summary['risk_flags']}")
print(f"HDF risk:      {summary['hdf_risk']}")
print(f"PWF risk:      {summary['pwf_risk']}")
assert summary["any_anomaly"]
assert summary["hdf_risk"]
print("PASSED ✅\n")

# ── Test 3: High temperature — out of range ───────────────────────────────────
print("=" * 50)
print("TEST 3: High temperature → out of range warning")
print("=" * 50)
state = make_state([{
    "sensor_id":         "MCH-003-S1",
    "timestamp":         datetime.utcnow().isoformat(),
    "temperature_c":     35.0,   # above 30°C threshold
    "machine_speed_rpm": 1500,
    "anomaly_flagged":   False,
}])
result = parser.run(state)
summary = result.sensor_summary
print(f"Temp out of range: {summary['temp_out_of_range']}")
print(f"Warnings: {summary['warnings']}")
assert summary["temp_out_of_range"]
print("PASSED ✅\n")

# ── Test 4: No sensors — pipeline continues with warning ──────────────────────
print("=" * 50)
print("TEST 4: No sensors — should continue with warning")
print("=" * 50)
state = make_state([])
result = parser.run(state)
summary = result.sensor_summary
print(f"Status:   {result.status}")
print(f"Warnings: {summary['warnings']}")
assert result.status == "ingested"
assert not summary["any_anomaly"]
print("PASSED ✅\n")

print("=" * 50)
print("All sensor parser tests passed.")
print("=" * 50)