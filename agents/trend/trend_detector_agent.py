"""
ManufactureIQ — Trend Detector Agent
======================================
Agent 5 of 7 — Tier 2: Analysis

Framework:  SQLite inspection history + scikit-learn (SECOM process monitor)
Role:       Decides whether this wafer is a one-off or part of a pattern, and
            whether the process run itself looked risky.

    1. Records the wafer in the inspection history (core/history.py)
    2. Lot pattern      — ≥3 wafers in this lot share the defect pattern
    3. Tool pattern     — this tool produced the same SYSTEMATIC pattern on
                          ≥3 recent wafers from ≥2 different lots (one lot alone
                          could just be a bad lot; two lots on one tool points
                          at the tool)
    4. Material pattern — defective wafers that share one material lot span
                          ≥2 tools AND ≥2 production lots. Lot and tool both
                          vary; the material is the only common factor. Once
                          there is enough history (≥50 wafers on other
                          material lots), its defect rate must also be ≥2× theirs.
    5. Process risk     — the SECOM model's failure-risk score for this run.
                          On held-out future months this model was barely better
                          than chance (test ROC-AUC ≈ 0.58), so it is REPORTED
                          but never escalates anything on its own.

Input:      state.classification, state.ingested, state.sensor_summary
Output:     state.trend (TrendReport)
"""

import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from core import history
from core.defect_knowledge import SYSTEMATIC
from core.secom import risk_score
from core.state import PipelineState, TrendReport

logger     = logging.getLogger(__name__)
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

LOT_MIN_SAME        = 3
TOOL_MIN_SAME       = 3
TOOL_MIN_LOTS       = 2
MATERIAL_MIN_TOOLS  = 2
MATERIAL_MIN_LOTS   = 2
MATERIAL_RATE_RATIO = 2.0
MATERIAL_BASELINE_N = 50      # below this, too little history to compare rates


class TrendDetectorAgent:

    def __init__(self):
        self.name = "TrendDetectorAgent"
        bundle = joblib.load(MODELS_DIR / "process_monitor.pkl")
        self.process_model     = bundle["model"]
        self.process_threshold = bundle["threshold"]
        logger.info(f"[{self.name}] Process monitor loaded ({bundle['kind']})")

    def run(self, state: PipelineState) -> PipelineState:
        logger.info(f"[{self.name}] Starting — run_id: {state.run_id}")
        if state.status == "failed" or not state.classification:
            return state

        try:
            insp    = state.ingested.inspection
            pattern = state.classification.pattern
            history.record(insp.record_id, insp.batch_id, insp.machine_id,
                           insp.supplier_id, insp.material_batch_id, pattern)

            report = TrendReport(pattern="isolated")
            parts: list[str] = []

            # ── Process risk (SECOM) — informational only ─────────────────────
            if insp.process_sensors:
                x = np.array([[np.nan if v is None else float(v) for v in insp.process_sensors]])
                report.process_risk  = round(float(risk_score(self.process_model, x)[0]), 4)
                report.process_alert = report.process_risk >= self.process_threshold

            if pattern == "none":
                report.summary = "No defect pattern on this wafer."
                if report.process_alert:
                    report.summary += (" The process run scored above the risk threshold, but that "
                                       "model is a weak predictor — treat as a prompt to check SPC, not a finding.")
                state.trend = report
                state.status = "trend_detected"
                return state

            # ── Lot ───────────────────────────────────────────────────────────
            seen, same = history.lot_stats(insp.batch_id, pattern)
            report.lot_wafers_seen, report.lot_same_pattern = seen, same
            lot_hit = same >= LOT_MIN_SAME
            if lot_hit:
                report.pattern = "batch_pattern"
                report.affected_batch_ids = [insp.batch_id]
                parts.append(f"{same} of {seen} inspected wafers in lot {insp.batch_id} show {pattern}.")

            # ── Tool ──────────────────────────────────────────────────────────
            tool_hit = False
            if pattern in SYSTEMATIC:
                n, n_lots, lots = history.tool_recent(insp.machine_id, pattern)
                report.tool_same_pattern_recent = n
                tool_hit = n >= TOOL_MIN_SAME and n_lots >= TOOL_MIN_LOTS
                if tool_hit:
                    report.pattern = "machine_pattern"
                    report.affected_machine_ids = [insp.machine_id]
                    report.affected_batch_ids = lots[:5]
                    parts.append(f"Tool {insp.machine_id} produced {pattern} on {n} recent wafers "
                                 f"across {n_lots} lots — points at the tool, not one lot.")

            # ── Material ──────────────────────────────────────────────────────
            mat_hit = False
            if insp.material_batch_id:
                m = history.material_stats(insp.material_batch_id)
                report.material_tools_affected = m["tools_with_defects"]
                has_baseline = m["baseline_wafers"] >= MATERIAL_BASELINE_N
                mat_hit = (m["tools_with_defects"] >= MATERIAL_MIN_TOOLS and
                           m["lots_with_defects"] >= MATERIAL_MIN_LOTS and
                           (not has_baseline or
                            m["defect_rate"] >= MATERIAL_RATE_RATIO * max(m["baseline_defect_rate"], 0.01)))
                if mat_hit:
                    report.pattern = "supplier_pattern"
                    base = (f"{m['defect_rate']:.0%} vs {m['baseline_defect_rate']:.0%} on other material lots"
                            if has_baseline else "too little history yet for a rate comparison")
                    parts.append(f"Material lot {insp.material_batch_id}: {m['defects']}/{m['wafers']} wafers "
                                 f"defective across {m['lots_with_defects']} lots and "
                                 f"{m['tools_with_defects']} tools — the material is the common factor ({base}).")

            if not parts:
                parts.append(f"No repeat of {pattern} yet in this lot, on this tool, or on this material "
                             f"({seen} wafer(s) from the lot inspected so far).")
            if report.process_risk is not None:
                parts.append(f"Process-run risk score {report.process_risk:.2f} "
                             f"({'above' if report.process_alert else 'below'} threshold; low-confidence model).")

            report.escalate = lot_hit or tool_hit or mat_hit
            report.summary  = " ".join(parts)
            state.trend      = report
            state.status     = "trend_detected"
            state.updated_at = datetime.utcnow()
            logger.info(f"[{self.name}] Done — {report.pattern} | escalate {report.escalate}")

        except Exception as e:
            state.errors.append(f"[{self.name}] Error: {e}")
            logger.exception(f"[{self.name}] Unexpected error")

        return state


_agent = TrendDetectorAgent()

def trend_detector_node(state: PipelineState) -> PipelineState:
    return _agent.run(state)
