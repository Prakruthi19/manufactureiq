"""
ManufactureIQ — Sensor Parser Agent
=====================================
Agent 2 of 7 — Tier 1: Intake

Framework:  Pure Python + NumPy
Role:       Turns the raw inspection into numbers the later agents use:
              1. Wafer summary — die count, failing die, yield
              2. SPC check     — which of the run's process sensors are outside
                                 their normal operating band (mean ± 3σ of good
                                 runs in the SECOM training period)

Input:      state.ingested (from Agent 1)
Output:     state.sensor_summary (dict)

Why no ML / LLM:
    Statistical process control is the standard, explainable way fabs watch
    sensors. It needs no labels and its output ("sensor S059 is 4.2σ high")
    is exactly the evidence the Root Cause agent needs.

A caveat the summary carries forward: with ~440 sensors, about one sensor is
outside 3σ on a perfectly normal run by chance alone. So a run is called
unusual only when MORE sensors are out than on 95% of good runs.
"""

import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from core.state import PipelineState

logger     = logging.getLogger(__name__)
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class SensorParserAgent:

    def __init__(self):
        self.name    = "SensorParserAgent"
        bundle       = joblib.load(MODELS_DIR / "process_monitor.pkl")
        self.cleaner = bundle["model"].named_steps["clean"]
        self.spc     = bundle["spc"]

    def run(self, state: PipelineState) -> PipelineState:
        logger.info(f"[{self.name}] Starting — run_id: {state.run_id}")

        if not state.ingested:
            state.errors.append(f"[{self.name}] No ingested data found — skipping")
            return state
        if state.status == "failed":
            return state

        inspection = state.ingested.inspection
        summary: dict = {"warnings": []}

        # ── 1. Wafer summary ─────────────────────────────────────────────────
        grid    = np.asarray(inspection.wafer_map)
        n_die   = int((grid >= 1).sum())
        n_fail  = int((grid == 2).sum())
        summary["wafer"] = {
            "rows": int(grid.shape[0]), "cols": int(grid.shape[1]),
            "die": n_die, "failing_die": n_fail,
            "yield_pct": round(100 * (1 - n_fail / max(n_die, 1)), 1),
        }

        # ── 2. SPC on process sensors ────────────────────────────────────────
        if inspection.process_sensors:
            raw   = np.array([[np.nan if v is None else float(v) for v in inspection.process_sensors]])
            n_missing = int(np.isnan(raw[0, self.cleaner.keep_]).sum())
            clean = self.cleaner.transform(raw)[0]          # drop dead sensors, fill gaps
            spc   = self.spc.check(clean)
            spc["missing_filled"] = n_missing
            summary["spc"] = spc
            if spc["unusual_vs_normal"]:
                summary["warnings"].append(
                    f"{spc['out_of_control']} process sensors outside 3σ "
                    f"(a normal run has ≤{spc['typical_out_of_control_p95']:.0f})")
        else:
            summary["spc"] = None
            summary["warnings"].append("No process-sensor data — SPC check skipped")

        summary["any_anomaly"] = bool(summary["spc"] and summary["spc"]["unusual_vs_normal"])

        state.ingested.ingest_warnings.extend(summary["warnings"])
        state.sensor_summary = summary
        state.status         = "ingested"
        state.updated_at     = datetime.utcnow()
        logger.info(f"[{self.name}] Done — yield {summary['wafer']['yield_pct']}% | "
                    f"SPC out-of-control: {summary['spc']['out_of_control'] if summary['spc'] else 'n/a'}")
        return state


_agent = SensorParserAgent()

def sensor_parser_node(state: PipelineState) -> PipelineState:
    """LangGraph node — wraps SensorParserAgent.run()"""
    return _agent.run(state)
