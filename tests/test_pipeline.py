"""
ManufactureIQ — test suite
===========================
    python -m pytest tests -q

Needs the trained models (scripts/train_models.py). No LLM calls, no network:
the LLM agents are only tested for their deterministic decision logic.
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Isolated history DB — tests must never touch the real one.
os.environ["HISTORY_DB"] = str(Path(tempfile.mkdtemp()) / "history.sqlite")

from core import history                                              # noqa: E402
from core.secom import SensorCleaner                                  # noqa: E402
from core.state import (DefectClassification, IngestedData,           # noqa: E402
                        InspectionRecord, PipelineState, RootCauseReport, TrendReport)
from core.wafer_features import (FEATURE_NAMES, features_from_map,    # noqa: E402
                                 features_from_points, map_from_points)


# ── helpers ───────────────────────────────────────────────────────────────────
def blank_wafer(n=40):
    """n×n grid with an inscribed circle of passing die."""
    return np.array(map_from_points(np.zeros((0, 2), int), n, n))


def edge_ring(n=40):
    g = blank_wafer(n)
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt(((xx - (n - 1) / 2) / (n / 2)) ** 2 + ((yy - (n - 1) / 2) / (n / 2)) ** 2)
    g[(r > 0.88) & (r <= 1.0)] = 2
    return g


def centre_blob(n=40):
    g = blank_wafer(n)
    c = n // 2
    g[c - 4:c + 4, c - 4:c + 4] = 2
    return g


def payload(grid, **kw):
    base = {"record_id": f"T-{uuid.uuid4().hex[:6]}", "batch_id": "LOT-T", "product_id": "DEV",
            "machine_id": "TOOL-1", "timestamp": "2026-10-08T10:00:00",
            "wafer_map": grid.tolist() if hasattr(grid, "tolist") else grid}
    base.update(kw)
    return base


def state_for(inspection: dict) -> PipelineState:
    return PipelineState(run_id=str(uuid.uuid4()), raw_inspection_payload=inspection)


# ── wafer features ────────────────────────────────────────────────────────────
def test_feature_vector_shape_and_finite():
    f = features_from_map(edge_ring())
    assert f.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(f).all()


def test_features_capture_the_physics():
    """Each pattern should light up the feature designed for it."""
    ring, centre = dict(zip(FEATURE_NAMES, features_from_map(edge_ring()))), \
                   dict(zip(FEATURE_NAMES, features_from_map(centre_blob())))
    assert ring["edge_vs_wafer"] > 2 > centre["edge_vs_wafer"]
    assert centre["center_vs_wafer"] > 2 > ring["center_vs_wafer"]
    assert ring["edge_sector_entropy"] > 0.95          # spread evenly all the way round


def test_clean_wafer_has_zero_failure_features():
    f = dict(zip(FEATURE_NAMES, features_from_map(blank_wafer())))
    assert f["fail_density"] == 0 and f["n_clusters"] == 0


def test_points_and_map_paths_agree():
    """Training (points) and serving (map) must compute the same features."""
    g = edge_ring()
    ys, xs = np.nonzero(g == 2)
    a = features_from_points(np.column_stack([xs, ys]), *g.shape, int((g >= 1).sum()))
    assert np.allclose(a, features_from_map(g))


# ── ingestor validation ───────────────────────────────────────────────────────
from agents.ingestor.ingestor_agent import DataIngestorAgent          # noqa: E402
ingestor = DataIngestorAgent()


@pytest.mark.parametrize("bad, why", [
    ([[0, 1], [1]], "ragged"),
    ([[0, 3, 1]] * 10, "value 3"),
    ([[0] * 10] * 10, "no die"),
    ([[1, 1], [1, 1]], "too small"),
])
def test_ingestor_rejects_bad_wafer_maps(bad, why):
    s = ingestor.run(state_for(payload(bad)))
    assert s.status == "failed", why


def test_ingestor_requires_wafer_map():
    p = payload(edge_ring()); del p["wafer_map"]
    assert ingestor.run(state_for(p)).status == "failed"


def test_ingestor_drops_wrong_length_sensors_with_warning():
    s = ingestor.run(state_for(payload(edge_ring(), process_sensors=[1.0] * 10)))
    assert s.status == "ingested"
    assert s.ingested.inspection.process_sensors is None
    assert any("590" in w for w in s.ingested.ingest_warnings)


# ── SECOM cleaning ────────────────────────────────────────────────────────────
def test_sensor_cleaner_learns_from_train_only():
    X = np.random.default_rng(0).normal(size=(50, 4))
    X[:, 1] = 5.0                       # constant → dropped
    X[:40, 2] = np.nan                  # 80% missing → dropped
    c = SensorCleaner().fit(X)
    assert list(c.keep_) == [0, 3]
    out = c.transform(np.array([[np.nan, 1, 1, 2.0]]))
    assert np.isfinite(out).all() and out[0, 0] == pytest.approx(np.median(X[:, 0]))


# ── end-to-end ML path (no LLM) ───────────────────────────────────────────────
def test_dry_pipeline_classifies_a_wafer():
    from agents.orchestrator.dry_pipeline import run_dry_pipeline
    s = run_dry_pipeline(state_for(payload(edge_ring(), process_sensors=[None] * 590)))
    assert s.status == "trend_detected", s.errors
    clf = s.classification
    assert clf.pattern in clf.probabilities
    assert sum(clf.probabilities.values()) == pytest.approx(1, abs=1e-3)
    assert s.sensor_summary["spc"]["sensors_checked"] > 0
    assert s.trend.process_risk is not None


# ── trend rules (history) ─────────────────────────────────────────────────────
from agents.trend.trend_detector_agent import TrendDetectorAgent      # noqa: E402
trend = TrendDetectorAgent()


def classified(pattern, lot, tool, material="MAT-1", supplier="SUP-1"):
    insp = InspectionRecord(record_id=uuid.uuid4().hex, batch_id=lot, product_id="DEV",
                            machine_id=tool, timestamp="2026-10-08T10:00:00",
                            supplier_id=supplier, material_batch_id=material,
                            wafer_map=blank_wafer().tolist())
    st = PipelineState(run_id="t", ingested=IngestedData(inspection=insp), status="classified")
    st.classification = DefectClassification(defect_type=pattern.lower(), pattern=pattern)
    return trend.run(st).trend


def test_lot_pattern_needs_three_wafers():
    history.clear()
    assert not classified("Loc", "LOT-1", "T1").escalate
    assert not classified("Loc", "LOT-1", "T1").escalate
    t = classified("Loc", "LOT-1", "T1")
    assert t.escalate and t.pattern == "batch_pattern" and t.lot_same_pattern == 3


def test_tool_pattern_needs_two_lots():
    history.clear()
    for _ in range(3):
        t = classified("Edge-Ring", "LOT-A", "ETCH-9", material="M-A")
    assert t.pattern == "batch_pattern"                  # one lot only → lot, not tool
    t = classified("Edge-Ring", "LOT-B", "ETCH-9", material="M-B")
    assert t.pattern == "machine_pattern" and t.affected_machine_ids == ["ETCH-9"]


def test_material_pattern_needs_several_tools_and_lots():
    history.clear()
    classified("Random", "LOT-X", "CVD-1", material="M-BAD")
    t = classified("Random", "LOT-Y", "CVD-1", material="M-BAD")
    assert t.pattern != "supplier_pattern"               # one tool → could be the tool
    t = classified("Random", "LOT-Z", "CVD-2", material="M-BAD")
    assert t.pattern == "supplier_pattern" and t.material_tools_affected == 2


def test_one_lot_on_two_tools_is_not_a_material_pattern():
    """Same lot on two tools: the LOT is a common factor too, so the material isn't singled out."""
    history.clear()
    classified("Random", "LOT-1", "CVD-1", material="M-1")
    t = classified("Random", "LOT-1", "CVD-2", material="M-1")
    assert t.pattern != "supplier_pattern"


def test_material_rate_checked_against_baseline_when_history_allows():
    history.clear()
    for i in range(60):                                  # baseline: 50% defective elsewhere
        classified("Random" if i % 2 else "none", f"LOT-B{i}", f"T{i % 3}", material="M-OTHER")
    classified("none", "LOT-P", "CVD-1", material="M-NEW")
    classified("Random", "LOT-Q", "CVD-1", material="M-NEW")
    t = classified("Random", "LOT-R", "CVD-2", material="M-NEW")   # 67% vs 50%: under 2x
    assert t.pattern != "supplier_pattern"


def test_clean_wafer_never_escalates():
    history.clear()
    t = classified("none", "LOT-1", "T1")
    assert not t.escalate


# ── supplier alert gate ───────────────────────────────────────────────────────
from agents.supplier.supplier_alert_agent import SupplierAlertAgentRunner  # noqa: E402
gate = SupplierAlertAgentRunner()._needs_alert


def alert_state(pattern, material_issue=False, trend_pattern="isolated"):
    insp = InspectionRecord(record_id="r", batch_id="L", product_id="P", machine_id="T",
                            timestamp="2026-10-08T10:00:00", supplier_id="SUP-1",
                            wafer_map=blank_wafer().tolist())
    st = PipelineState(run_id="t", ingested=IngestedData(inspection=insp))
    st.classification = DefectClassification(defect_type=pattern.lower(), pattern=pattern)
    st.root_cause = RootCauseReport(material_issue=material_issue)
    st.trend = TrendReport(pattern=trend_pattern)
    return st


def test_supplier_not_blamed_for_tool_patterns():
    assert not gate(alert_state("Edge-Ring", material_issue=True))


def test_supplier_alert_for_material_linked_pattern():
    assert gate(alert_state("Random", material_issue=True))
    assert not gate(alert_state("Random", material_issue=False))


def test_supplier_alert_on_cross_tool_material_pattern():
    assert gate(alert_state("Center", trend_pattern="supplier_pattern"))


# ── API ───────────────────────────────────────────────────────────────────────
def test_api_read_endpoints():
    from fastapi.testclient import TestClient
    from api.main import app
    c = TestClient(app)
    assert c.get("/api/health").json() == {"status": "ok"}
    card = c.get("/api/model-card").json()
    assert card["wafer_classifier"]["selected_strategy"]
    assert card["process_monitor"]["test"]["pr_auc"] is not None
    samples = c.get("/api/samples").json()
    assert samples["singles"] and samples["scenarios"]
