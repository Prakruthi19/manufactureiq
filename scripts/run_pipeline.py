import json, sys, uuid, logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(message)s",
    datefmt="%H:%M:%S"
)

from core.state import PipelineState
from agents.orchestrator.graph import run_graph

# A held-out wafer from the demo set (never seen in training).
_demo = json.loads((Path(__file__).resolve().parent.parent / "data" / "processed" / "demo_samples.json").read_text())
_pick = next(s for s in _demo["singles"] if s["true_pattern"] != "none")
INSPECTION = _pick["inspection"]
SENSORS = []
print(f"Wafer {INSPECTION['wafer_id']} — expert label: {_pick['true_pattern']}")

print("\n" + "=" * 60)
print("ManufactureIQ — Running Full 7-Agent Pipeline")
print("=" * 60)

start = datetime.utcnow()

state = PipelineState(
    run_id=str(uuid.uuid4()),
    raw_inspection_payload=INSPECTION,
    raw_sensor_payload=SENSORS,
)

print(f"\nRun ID:  {state.run_id}")
print(f"Lot:     {INSPECTION['batch_id']}")
print(f"Tool:    {INSPECTION['machine_id']}")
print("\nStarting pipeline...\n")

final = run_graph(state, interrupt=False)

elapsed = (datetime.utcnow() - start).total_seconds()

print("\n" + "=" * 60)
print("PIPELINE RESULTS")
print("=" * 60)
print(f"\nStatus:   {final.status}")
print(f"Duration: {elapsed:.1f}s")
print(f"Errors:   {final.errors or 'None'}")

if final.classification:
    c = final.classification
    print(f"\n── Defect Classification ──────────────────")
    print(f"  Pattern:    {c.pattern}  (expert: {_pick['true_pattern']})")
    print(f"  Severity:   {c.severity}")
    print(f"  Confidence: {c.confidence_score:.0%}")
    print(f"  Action:     {c.recommended_action}")
    print(f"  Review:     {c.requires_human_review}")

if final.root_cause:
    r = final.root_cause
    print(f"\n── Root Cause ─────────────────────────────")
    print(f"  Primary:    {r.primary_cause}")
    print(f"  Material:   {r.material_issue}")
    print(f"  Machine:    {r.machine_issue}")

if final.trend:
    t = final.trend
    print(f"\n── Trend Report ───────────────────────────")
    print(f"  Pattern:    {t.pattern}")
    print(f"  Escalate:   {t.escalate}")
    print(f"  Summary:    {t.summary}")

if final.supplier_alert:
    a = final.supplier_alert
    print(f"\n── Supplier Alert ─────────────────────────")
    print(f"  Triggered:  {a.triggered}")
    print(f"  Supplier:   {a.supplier_id}")

if final.compliance_report:
    r = final.compliance_report
    print(f"\n── Compliance Report ──────────────────────")
    print(f"  Report ID:  {r.report_id}")
    print(f"  Sign-off:   {r.sign_off_required}")

print(f"\nPipeline complete in {elapsed:.1f}s\n")