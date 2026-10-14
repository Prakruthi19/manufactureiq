"""
ManufactureIQ — Defect Classifier Agent
=========================================
Agent 3 of 7 — Tier 2: Analysis

Framework:  XGBoost on engineered wafer-map features (core/wafer_features.py)
Role:       Classifies the wafer map into one of WM-811K's 9 classes:
            none, Center, Donut, Edge-Ring, Edge-Loc, Loc, Random, Scratch, Near-full

Input:      state.ingested.inspection.wafer_map   (from Agent 1)
Output:     state.classification (DefectClassification)

Why ML not LLM:
    172,950 expert-labelled examples exist. A trained model is faster, cheaper,
    reproducible and measurably accurate (models/wafer_classifier_metrics.json).
    An LLM would have to "look at" a grid of numbers and guess.

Human review:
    If the model's top probability is below the threshold chosen on validation
    data (auto-classified wafers ≥95% correct), the wafer goes to a human.
    Near-full is always reviewed — it means quarantining a whole lot.
    Probabilities are temperature-calibrated (core/calibration.py) so the
    confidence shown, and the threshold, mean what they say.

Drift:
    Every wafer's features are logged so core/drift.py can compare recent
    inputs with the training data (GET /api/drift).
"""

import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from core import drift
from core.calibration import apply_temperature
from core.defect_knowledge import KNOWLEDGE, slug
from core.state import DefectClassification, PipelineState
from core.wafer_features import describe, features_from_map

logger     = logging.getLogger(__name__)
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

ALWAYS_REVIEW   = {"Near-full"}
SEVERITY_ORDER  = ["low", "medium", "high", "critical"]
HIGH_YIELD_LOSS = 0.30      # >30% of die failing → at least "high" whatever the pattern


class DefectClassifierAgent:

    def __init__(self):
        self.name = "DefectClassifierAgent"
        bundle = joblib.load(MODELS_DIR / "wafer_classifier.pkl")
        self.model     = bundle["model"]
        self.classes   = bundle["classes"]
        self.threshold = bundle["review_threshold"]
        self.temperature = bundle.get("temperature", 1.0)
        logger.info(f"[{self.name}] Loaded {bundle['strategy']} model, "
                    f"review threshold {self.threshold:.2f}, temperature {self.temperature:.2f}")

    def run(self, state: PipelineState) -> PipelineState:
        logger.info(f"[{self.name}] Starting — run_id: {state.run_id}")
        if state.status == "failed":
            return state
        if not state.ingested:
            state.errors.append(f"[{self.name}] No ingested data — cannot classify")
            state.status = "failed"
            return state

        try:
            feats   = features_from_map(state.ingested.inspection.wafer_map)
            proba   = apply_temperature(self.model.predict_proba(feats.reshape(1, -1)), self.temperature)[0]
            order   = np.argsort(-proba)
            pattern = self.classes[order[0]]
            conf    = float(proba[order[0]])
            info    = KNOWLEDGE[pattern]

            severity = info["severity"]
            if pattern != "none" and feats[0] > HIGH_YIELD_LOSS:
                severity = max(severity, "high", key=SEVERITY_ORDER.index)

            review = conf < self.threshold or pattern in ALWAYS_REVIEW
            state.classification = DefectClassification(
                defect_type           = slug(pattern),
                severity              = severity,
                confidence_score      = round(conf, 4),
                location_description  = info["location"],
                recommended_action    = info["action"],
                requires_human_review = review,
                # extra fields (model_config extra="allow") — evidence for later agents + UI
                pattern               = pattern,
                runner_up             = {"pattern": self.classes[order[1]],
                                         "probability": round(float(proba[order[1]]), 4)},
                probabilities         = {c: round(float(p), 4) for c, p in zip(self.classes, proba)},
                review_threshold      = self.threshold,
                evidence              = describe(feats),
            )
            try:
                drift.log(feats)
            except Exception:                       # monitoring must never break inspection
                logger.warning(f"[{self.name}] Could not log features for drift monitoring")
            state.status     = "classified"
            state.updated_at = datetime.utcnow()
            logger.info(f"[{self.name}] Done — {pattern} ({conf:.1%}) | severity {severity} | review {review}")

        except Exception as e:
            state.errors.append(f"[{self.name}] Error: {e}")
            state.status = "failed"
            logger.exception(f"[{self.name}] Unexpected error")

        return state


_agent = DefectClassifierAgent()

def defect_classifier_node(state: PipelineState) -> PipelineState:
    """LangGraph node — wraps DefectClassifierAgent.run()"""
    return _agent.run(state)
