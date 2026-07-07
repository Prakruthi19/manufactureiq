"""
ManufactureIQ — Trend Detector Agent
======================================
Agent 5 of 7 — Tier 2: Analysis

Framework:  Pure Python + Pandas + Scikit-learn (Isolation Forest)
Role:       Detects whether the current defect is isolated or
            part of a systemic pattern across batches, machines,
            or suppliers using the trained Isolation Forest model.

Input:      state.classification  (from Agent 3)
            state.root_cause      (from Agent 4)
            state.ingested        (from Agent 1)
Output:     state.trend           (TrendReport)

Why ML not LLM:
    Pattern detection across thousands of historical records
    is a data problem not a language problem.
    Isolation Forest detects anomalous clusters without
    needing labeled examples of "what a bad trend looks like".
"""

import logging
import joblib
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from core.state import PipelineState, TrendReport

logger   = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent.parent.parent

PATTERN_THRESHOLDS = {
    "batch_defect_rate":    0.10,
    "machine_defect_rate":  0.08,
    "supplier_defect_rate": 0.12,
    "frequency_spike":      3,
    "isolation_score":      -0.05,
}


class TrendDetectorAgent:

    def __init__(self):
        self.name    = "TrendDetectorAgent"
        self.model   = None
        self.scaler  = None
        self.history = None
        self._load_models()
        self._load_history()

    def _load_models(self):
        models_dir  = BASE_DIR / "models"
        self.model  = joblib.load(models_dir / "trend_detector.pkl")
        self.scaler = joblib.load(models_dir / "trend_scaler.pkl")
        logger.info(f"[{self.name}] Isolation Forest loaded")

    def _load_history(self):
        try:
            data_path    = BASE_DIR / "data" / "raw" / "ai4i2020.csv"
            self.history = pd.read_csv(data_path)
            if "power_W" not in self.history.columns:
                self.history["power_W"] = (
                    self.history["Torque [Nm]"] *
                    self.history["Rotational speed [rpm]"] *
                    (2 * math.pi / 60)
                )
            if "temp_diff_K" not in self.history.columns:
                self.history["temp_diff_K"] = (
                    self.history["Process temperature [K]"] -
                    self.history["Air temperature [K]"]
                )
            if "wear_torque" not in self.history.columns:
                self.history["wear_torque"] = (
                    self.history["Tool wear [min]"] *
                    self.history["Torque [Nm]"]
                )
            logger.info(f"[{self.name}] History loaded — {len(self.history):,} rows")
        except FileNotFoundError:
            logger.warning(f"[{self.name}] Dataset not found — pattern analysis limited")
            self.history = None

    def run(self, state: PipelineState) -> PipelineState:
        logger.info(f"[{self.name}] Starting — run_id: {state.run_id}")

        if state.status == "failed":
            return state

        if not state.classification:
            state.errors.append(f"[{self.name}] No classification — skipping")
            return state

        try:
            inspection = state.ingested.inspection
            clf        = state.classification
            sensors    = state.ingested.sensors
            sensor     = sensors[0] if sensors else None

            iso_score                        = self._get_isolation_score(sensor, state.sensor_summary)
            batch_pattern, batch_ids         = self._check_batch_pattern(inspection.batch_id)
            machine_pattern, machine_ids     = self._check_machine_pattern(inspection.machine_id)
            supplier_pattern                 = self._check_supplier_pattern(inspection.supplier_id)
            freq_7d, freq_spike              = self._check_frequency(inspection.machine_id, clf.defect_type)
            pattern                          = self._determine_pattern(iso_score, batch_pattern, machine_pattern, supplier_pattern)

            escalate = freq_spike or machine_pattern or supplier_pattern or iso_score < PATTERN_THRESHOLDS["isolation_score"]

            parts = []
            if pattern == "isolated":
                parts.append("Defect appears isolated to this inspection record.")
            if batch_pattern:
                parts.append(f"Batch pattern detected across {len(batch_ids)} batches.")
            if machine_pattern:
                parts.append(f"Machine {inspection.machine_id} shows recurring failures.")
            if supplier_pattern and inspection.supplier_id:
                parts.append(f"Supplier {inspection.supplier_id} material linked to pattern.")
            if freq_spike:
                parts.append(f"Frequency spike: {freq_7d} occurrences in last 7 days.")
            parts.append(f"Isolation score: {iso_score:.4f} ({'anomalous' if iso_score < 0 else 'normal'}).")

            state.trend = TrendReport(
                pattern=pattern,
                affected_batch_ids=batch_ids[:5],
                affected_machine_ids=machine_ids[:5],
                defect_frequency_last_7d=freq_7d,
                escalate=escalate,
                summary=" ".join(parts),
            )
            state.status     = "trend_detected"
            state.updated_at = datetime.utcnow()

            logger.info(f"[{self.name}] Done — pattern: {pattern} | escalate: {escalate}")

        except Exception as e:
            state.errors.append(f"[{self.name}] Error: {str(e)}")
            logger.exception(f"[{self.name}] Unexpected error")

        return state

    def _get_isolation_score(self, sensor, summary: Optional[dict]) -> float:
        summary  = summary or {}
        temp_c   = (sensor.temperature_c or 27.0) if sensor else 27.0
        rpm      = (sensor.machine_speed_rpm or 1500) if sensor else 1500
        torque   = 40.0
        tool_wear= 50.0
        power_w  = summary.get("power_w") or (torque * rpm * (2 * math.pi / 60))
        temp_diff= summary.get("temp_diff_k") or 10.0
        wear_t   = summary.get("wear_torque") or (tool_wear * torque)
        features = np.array([[rpm, torque, tool_wear, power_w, temp_diff, wear_t, 1]])
        scaled   = self.scaler.transform(features)
        return float(self.model.decision_function(scaled)[0])

    def _check_batch_pattern(self, batch_id: Optional[str]) -> tuple[bool, list]:
        if self.history is None or not batch_id or "batch_id" not in self.history.columns:
            return False, []
        batch_data  = self.history[self.history["batch_id"] == batch_id]
        if len(batch_data) == 0:
            return False, []
        defect_rate = batch_data["Machine failure"].mean()
        affected    = list(self.history[self.history["Machine failure"] == 1]["batch_id"].unique()[:5])
        return defect_rate > PATTERN_THRESHOLDS["batch_defect_rate"], affected

    def _check_machine_pattern(self, machine_id: Optional[str]) -> tuple[bool, list]:
        if self.history is None or not machine_id or "machine_id" not in self.history.columns:
            return False, []
        machine_data = self.history[self.history["machine_id"] == machine_id]
        if len(machine_data) == 0:
            return False, []
        defect_rate  = machine_data["Machine failure"].mean()
        affected     = list(self.history[self.history["Machine failure"] == 1]["machine_id"].unique()[:5])
        return defect_rate > PATTERN_THRESHOLDS["machine_defect_rate"], affected

    def _check_supplier_pattern(self, supplier_id: Optional[str]) -> bool:
        if self.history is None or not supplier_id or "supplier_id" not in self.history.columns:
            return False
        supplier_data = self.history[self.history["supplier_id"] == supplier_id]
        if len(supplier_data) == 0:
            return False
        return supplier_data["Machine failure"].mean() > PATTERN_THRESHOLDS["supplier_defect_rate"]

    def _check_frequency(self, machine_id: Optional[str], defect_type: str) -> tuple[int, bool]:
        if self.history is None or "timestamp" not in self.history.columns:
            return 0, False
        try:
            self.history["timestamp"] = pd.to_datetime(self.history["timestamp"])
            cutoff  = self.history["timestamp"].max() - timedelta(days=7)
            recent  = self.history[(self.history["timestamp"] >= cutoff) & (self.history["Machine failure"] == 1)]
            if machine_id and "machine_id" in self.history.columns:
                recent = recent[recent["machine_id"] == machine_id]
            freq        = len(recent)
            avg_weekly  = (self.history["Machine failure"].sum() / 90) * 7
            return freq, freq > avg_weekly * PATTERN_THRESHOLDS["frequency_spike"]
        except Exception:
            return 0, False

    def _determine_pattern(self, iso_score, batch_pattern, machine_pattern, supplier_pattern) -> str:
        if supplier_pattern: return "supplier_pattern"
        if machine_pattern:  return "machine_pattern"
        if batch_pattern:    return "batch_pattern"
        if iso_score < PATTERN_THRESHOLDS["isolation_score"]: return "batch_pattern"
        return "isolated"


# ── LangGraph node function ───────────────────────────────────────────────────
_agent = TrendDetectorAgent()

def trend_detector_node(state: PipelineState) -> PipelineState:
    return _agent.run(state)