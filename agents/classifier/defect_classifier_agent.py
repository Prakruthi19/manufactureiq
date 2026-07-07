"""
ManufactureIQ — Defect Classifier Agent
=========================================
Agent 3 of 7 — Tier 2: Analysis

Framework:  Scikit-learn (Random Forest)
Role:       Loads trained ML model, builds feature vector from
            ingested data + sensor summary, predicts defect type,
            severity, and confidence score.

Input:      state.ingested       (from Agent 1)
            state.sensor_summary (from Agent 2)
Output:     state.classification (DefectClassification)

Why ML not LLM:
    This is a structured tabular classification problem with labeled
    training data. Random Forest achieves 98% accuracy here.
    An LLM would be slower, more expensive, and less accurate on
    numeric sensor data.
"""

import json
import logging
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

from core.state import (
    PipelineState,
    DefectClassification,
    Severity,
)

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# ── Severity mapping ──────────────────────────────────────────────────────────
# Maps defect type + confidence to severity level
SEVERITY_MAP = {
    "none": "low",
    "TWF":  "medium",   # tool wear — scheduled maintenance
    "HDF":  "medium",   # heat dissipation — inspect cooling
    "PWF":  "high",     # power failure — most common, immediate action
    "OSF":  "high",     # overstrain — risk of tool breakage
    "RNF":  "critical", # random — unpredictable, escalate immediately
}

# ── Recommended actions per defect type ───────────────────────────────────────
ACTIONS = {
    "none": "No action required — continue production",
    "TWF":  "Schedule tool replacement within next maintenance window",
    "HDF":  "Inspect cooling system and heat dissipation components immediately",
    "PWF":  "Check power supply, motor load, and rotational drive system",
    "OSF":  "Reduce torque settings and inspect tool for signs of overstrain",
    "RNF":  "Halt production on this machine — escalate to engineering team",
}

# ── Location descriptions per defect type ─────────────────────────────────────
LOCATIONS = {
    "none": "No defect detected",
    "TWF":  "Cutting tool — wear zone",
    "HDF":  "Thermal management system — cooling circuit",
    "PWF":  "Power system — motor and drive",
    "OSF":  "Tool-workpiece interface — overstrain zone",
    "RNF":  "Unknown — requires physical inspection",
}

# ── Human review threshold ────────────────────────────────────────────────────
# If confidence below this OR defect is critical → pause for human review
CONFIDENCE_THRESHOLD = 0.75
HUMAN_REVIEW_DEFECTS = {"RNF", "OSF"}


class DefectClassifierAgent:
    """
    Loads the trained Random Forest model and classifies defects
    from sensor + inspection data.
    """

    def __init__(self):
        self.name        = "DefectClassifierAgent"
        self.model       = None
        self.le_type     = None   # LabelEncoder for product type
        self.le_label    = None   # LabelEncoder for defect labels
        self.metadata    = None
        self._load_models()

    def _load_models(self):
        """Load trained model and encoders from disk."""
        try:
            self.model    = joblib.load(MODELS_DIR / "defect_classifier.pkl")
            self.le_type  = joblib.load(MODELS_DIR / "label_encoder_type.pkl")
            self.le_label = joblib.load(MODELS_DIR / "label_encoder_label.pkl")

            with open(MODELS_DIR / "metadata.json") as f:
                self.metadata = json.load(f)

            logger.info(
                f"[{self.name}] Model loaded — "
                f"{self.metadata['best_classifier']} | "
                f"accuracy: {self.metadata['rf_accuracy']}"
            )
        except FileNotFoundError as e:
            logger.error(
                f"[{self.name}] Model files not found: {e}. "
                f"Run scripts/train_models.py first."
            )
            raise

    def run(self, state: PipelineState) -> PipelineState:
        """
        Main entry point — called by LangGraph orchestrator.
        """
        logger.info(f"[{self.name}] Starting — run_id: {state.run_id}")

        # Guards
        if state.status == "failed":
            logger.warning(f"[{self.name}] Pipeline failed — skipping")
            return state

        if not state.ingested:
            state.errors.append(f"[{self.name}] No ingested data — cannot classify")
            state.status = "failed"
            return state

        try:
            # ── Step 1: Build feature vector ──────────────────────────────────
            features, warnings = self._build_features(state)

            if features is None:
                state.errors.append(f"[{self.name}] Could not build feature vector")
                state.status = "failed"
                return state

            # ── Step 2: Run ML model ──────────────────────────────────────────
            import pandas as pd
            feature_names  = self.metadata["features"]
            features_df    = pd.DataFrame([features], columns=feature_names)
            prediction_encoded = self.model.predict(features_df)[0]
            prediction_proba   = self.model.predict_proba(features_df)[0]

            # Decode prediction
            defect_type  = self.le_label.inverse_transform([prediction_encoded])[0]
            confidence   = float(prediction_proba.max())

            # ── Step 3: Map to severity + action ─────────────────────────────
            severity     = self._get_severity(defect_type, confidence)
            action       = ACTIONS.get(defect_type, "Inspect machine")
            location     = LOCATIONS.get(defect_type, "Unknown")
            needs_review = (
                confidence < CONFIDENCE_THRESHOLD or
                defect_type in HUMAN_REVIEW_DEFECTS
            )

            # ── Step 4: Build DefectClassification ───────────────────────────
            classification = DefectClassification(
                defect_type           = self._map_defect_type(defect_type),
                severity              = severity,
                confidence_score      = round(confidence, 4),
                location_description  = location,
                recommended_action    = action,
                requires_human_review = needs_review,
            )

            # ── Step 5: Update state ──────────────────────────────────────────
            state.classification = classification
            state.status         = "classified"
            state.updated_at     = datetime.utcnow()

            # Add ingest warnings from feature building
            state.ingested.ingest_warnings.extend(warnings)

            logger.info(
                f"[{self.name}] Done — "
                f"defect: {defect_type} | "
                f"severity: {severity} | "
                f"confidence: {confidence:.2%} | "
                f"human_review: {needs_review}"
            )

        except Exception as e:
            state.errors.append(f"[{self.name}] Error: {str(e)}")
            state.status = "failed"
            logger.exception(f"[{self.name}] Unexpected error")

        return state

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_features(
        self, state: PipelineState
    ) -> tuple[Optional[list], list]:
        """
        Build the 9-feature vector the ML model expects.

        Features (must match training order exactly):
            0. Air temperature [K]
            1. Process temperature [K]
            2. Rotational speed [rpm]
            3. Torque [Nm]
            4. Tool wear [min]
            5. Type_encoded
            6. power_W
            7. temp_diff_K
            8. wear_torque
        """
        warnings  = []
        inspection = state.ingested.inspection
        sensors    = state.ingested.sensors
        summary    = state.sensor_summary or {}

        # ── Extract from sensors ──────────────────────────────────────────────
        sensor      = sensors[0] if sensors else None
        temp_c      = sensor.temperature_c     if sensor else None
        rpm         = sensor.machine_speed_rpm if sensor else None

        # Convert °C to Kelvin (model was trained on Kelvin).
        # Prefer structured fields (e.g. from a CSV upload) over sensor/defaults.
        air_k_struct  = _num(getattr(inspection, "air_temperature_k", None))
        proc_k_struct = _num(getattr(inspection, "process_temperature_k", None))
        air_temp_k  = air_k_struct if air_k_struct is not None else (
            (temp_c + 273.15) if temp_c is not None else 300.0
        )
        proc_temp_k = proc_k_struct if proc_k_struct is not None else air_temp_k + 10.0

        if temp_c is None:
            warnings.append("Air temperature missing — using default 300K")
        if rpm is None:
            warnings.append("RPM missing — using default 1500")
            rpm = 1500

        # ── Extract from inspection notes (tool wear, torque, type) ──────────
        # In production these come from structured fields
        # For now parse from notes or use sensor summary defaults
        # Structured fields win; otherwise fall back to parsing free-text notes.
        tw_struct = _num(getattr(inspection, "tool_wear_min", None))
        tq_struct = _num(getattr(inspection, "torque_nm", None))
        tool_wear = tw_struct if tw_struct is not None else self._extract_tool_wear(inspection.raw_notes, warnings)
        torque    = tq_struct if tq_struct is not None else self._extract_torque(inspection.raw_notes, warnings)

        # Product type encoding
        product_id  = inspection.product_id or "M000000"
        type_letter = product_id[0] if product_id[0] in ["L", "M", "H"] else "M"
        try:
            type_encoded = self.le_type.transform([type_letter])[0]
        except Exception:
            type_encoded = 1   # default to M
            warnings.append(f"Unknown product type '{type_letter}' — defaulting to M")

        # ── Derived features from sensor summary ─────────────────────────────
        power_w     = summary.get("power_w")
        temp_diff_k = summary.get("temp_diff_k")
        wear_torque = summary.get("wear_torque")

        # Calculate if not in summary
        import math
        if power_w is None or power_w == rpm:   # sensor parser stored rpm as placeholder
            power_w = torque * rpm * (2 * math.pi / 60)

        if temp_diff_k is None:
            temp_diff_k = proc_temp_k - air_temp_k

        if wear_torque is None:
            wear_torque = tool_wear * torque

        # ── Build final feature vector ────────────────────────────────────────
        features = [
            air_temp_k,    # 0. Air temperature [K]
            proc_temp_k,   # 1. Process temperature [K]
            rpm,           # 2. Rotational speed [rpm]
            torque,        # 3. Torque [Nm]
            tool_wear,     # 4. Tool wear [min]
            type_encoded,  # 5. Type_encoded
            power_w,       # 6. power_W
            temp_diff_k,   # 7. temp_diff_K
            wear_torque,   # 8. wear_torque
        ]

        logger.debug(f"[{self.name}] Features: {dict(zip(self.metadata['features'], features))}")
        return features, warnings

    def _extract_tool_wear(self, notes: Optional[str], warnings: list) -> float:
        """Extract tool wear minutes from inspection notes."""
        if not notes:
            warnings.append("Tool wear not in notes — using default 50 min")
            return 50.0
        import re
        match = re.search(r"wear[:\s]+(\d+)", notes, re.IGNORECASE)
        if match:
            return float(match.group(1))
        warnings.append("Tool wear not parsed from notes — using default 50 min")
        return 50.0

    def _extract_torque(self, notes: Optional[str], warnings: list) -> float:
        """Extract torque from inspection notes."""
        if not notes:
            warnings.append("Torque not in notes — using default 40 Nm")
            return 40.0
        import re
        match = re.search(r"torque[:\s]+([\d]+\.?[\d]*)", notes, re.IGNORECASE)
        if match:
            return float(match.group(1))
        warnings.append("Torque not parsed — using default 40 Nm")
        return 40.0

    def _get_severity(self, defect_type: str, confidence: float) -> Severity:
        """Determine severity — escalate if confidence is low."""
        base = SEVERITY_MAP.get(defect_type, "medium")
        if confidence < CONFIDENCE_THRESHOLD and base == "low":
            return "medium"   # low confidence → bump up severity
        return base

    def _map_defect_type(self, raw: str):
        """Map model output to state schema DefectType."""
        mapping = {
            "none": "no_defect",          # healthy machine — NOT "unknown"
            "TWF":  "structural_crack",
            "HDF":  "material_contamination",
            "PWF":  "assembly_misalignment",
            "OSF":  "dimensional_deviation",
            "RNF":  "unknown",            # random failure — genuinely unclassifiable
        }
        return mapping.get(raw, "unknown")


def _num(v):
    """Coerce a value to float, or None if missing/blank/non-numeric."""
    try:
        return float(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


# ── LangGraph node function ───────────────────────────────────────────────────
_agent = DefectClassifierAgent()

def defect_classifier_node(state: PipelineState) -> PipelineState:
    """LangGraph node — wraps DefectClassifierAgent.run()"""
    return _agent.run(state)