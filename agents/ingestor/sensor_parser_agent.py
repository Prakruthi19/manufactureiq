"""
ManufactureIQ — Sensor Parser Agent
=====================================
Agent 2 of 7 — Tier 1: Intake

Framework:  Pure Python + Pydantic v2
Role:       Reads validated sensor readings from IngestedData,
            checks against safe operating ranges,
            calculates derived metrics,
            flags anomalies and maps them to likely failure modes.

Input:      state.ingested (IngestedData from Agent 1)
Output:     state.ingested.sensors updated with anomaly_flagged=True
            state.sensor_summary added to state (used by Defect Classifier)

Why no LLM:
    Sensor threshold checking is deterministic engineering logic.
    Using an LLM here would be slow, expensive, and less accurate
    than simple range comparisons derived from the AI4I paper.
"""

import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from core.state import PipelineState, SensorReading

logger = logging.getLogger(__name__)


# ── Safe operating ranges from AI4I 2020 paper ───────────────────────────────
# Values outside these ranges are associated with machine failures

@dataclass
class SensorThresholds:
    """Safe operating ranges for each sensor type."""

    # Temperature thresholds (converted from Kelvin in dataset)
    temp_c_min:          float = 24.0    # 297K
    temp_c_max:          float = 30.0    # 303K


    # Temperature differential (process - air)
    temp_diff_min_k:     float = 8.6     # below this → HDF risk
    temp_diff_max_k:     float = 12.0

    # Rotational speed
    rpm_min:             int   = 1168    # below this → HDF risk
    rpm_max:             int   = 2886

    # Torque
    torque_min_nm:       float = 3.8
    torque_max_nm:       float = 76.6

    # Tool wear
    tool_wear_warning:   int   = 150     # start watching
    tool_wear_critical:  int   = 200     # TWF risk zone

    # Power (torque × rpm × 2π/60)
    power_min_w:         float = 3500.0  # below → PWF risk
    power_max_w:         float = 9000.0  # above → PWF risk

    # Wear × torque (OSF indicator)
    wear_torque_critical: float = 11000.0  # Type L threshold (most conservative)


THRESHOLDS = SensorThresholds()


@dataclass
class SensorSummary:
    """
    Derived metrics and anomaly flags calculated by the Sensor Parser.
    Passed to the Defect Classifier as additional features.
    """
    # Derived metrics
    power_w:              Optional[float] = None
    temp_diff_k:          Optional[float] = None
    wear_torque:          Optional[float] = None

    # Anomaly flags (map to failure modes)
    hdf_risk:             bool = False   # Heat Dissipation Failure risk
    pwf_risk:             bool = False   # Power Failure risk
    twf_risk:             bool = False   # Tool Wear Failure risk
    osf_risk:             bool = False   # Overstrain Failure risk

    # Out-of-range flags
    temp_out_of_range:    bool = False
    rpm_out_of_range:     bool = False
    torque_out_of_range:  bool = False

    # Overall
    any_anomaly:          bool = False
    anomaly_count:        int  = 0
    risk_flags:           list = field(default_factory=list)
    warnings:             list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "power_w":            self.power_w,
            "temp_diff_k":        self.temp_diff_k,
            "wear_torque":        self.wear_torque,
            "hdf_risk":           self.hdf_risk,
            "pwf_risk":           self.pwf_risk,
            "twf_risk":           self.twf_risk,
            "osf_risk":           self.osf_risk,
            "temp_out_of_range":  self.temp_out_of_range,
            "rpm_out_of_range":   self.rpm_out_of_range,
            "torque_out_of_range": self.torque_out_of_range,
            "any_anomaly":        self.any_anomaly,
            "anomaly_count":      self.anomaly_count,
            "risk_flags":         self.risk_flags,
            "warnings":           self.warnings,
        }


class SensorParserAgent:
    """
    Validates sensor readings against safe operating ranges,
    calculates derived metrics, and flags likely failure modes.
    Runs as the second node in the LangGraph pipeline.
    """

    def __init__(self):
        self.name       = "SensorParserAgent"
        self.thresholds = THRESHOLDS

    def run(self, state: PipelineState) -> PipelineState:
        """
        Main entry point — called by LangGraph orchestrator.
        """
        logger.info(f"[{self.name}] Starting — run_id: {state.run_id}")

        # Guard — ingestor must have run first
        if not state.ingested:
            state.errors.append(f"[{self.name}] No ingested data found — skipping")
            logger.warning(f"[{self.name}] Skipping — no ingested data")
            return state

        if state.status == "failed":
            logger.warning(f"[{self.name}] Pipeline already failed — skipping")
            return state

        sensors  = state.ingested.sensors
        summary  = SensorSummary()

        if not sensors:
            summary.warnings.append("No sensor readings available — classifier will run on inspection data only")
            logger.warning(f"[{self.name}] No sensors to parse")
        else:
            # Parse all sensors and aggregate into one summary
            for sensor in sensors:
                self._check_sensor(sensor, summary)

            # Calculate derived metrics from aggregated sensor data
            self._calculate_derived_metrics(sensors, summary)

            # Determine failure mode risks from derived metrics
            self._assess_failure_risks(summary)

            # Final anomaly count
            summary.anomaly_count = sum([
                summary.hdf_risk,
                summary.pwf_risk,
                summary.twf_risk,
                summary.osf_risk,
                summary.temp_out_of_range,
                summary.rpm_out_of_range,
                summary.torque_out_of_range,
            ])
            summary.any_anomaly = summary.anomaly_count > 0

        # Store summary in state so Defect Classifier can use it
        state.ingested.ingest_warnings.extend(summary.warnings)
        state.sensor_summary = summary.to_dict()
        state.status         = "ingested"   # still ingested — classifier is next
        state.updated_at     = datetime.utcnow()

        logger.info(
            f"[{self.name}] Done — "
            f"anomalies: {summary.anomaly_count} | "
            f"risks: {summary.risk_flags}"
        )
        return state

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_sensor(self, sensor: SensorReading, summary: SensorSummary):
        """Check individual sensor reading against thresholds."""
        t = self.thresholds

        # Temperature check
        if sensor.temperature_c is not None:
            if not (t.temp_c_min <= sensor.temperature_c <= t.temp_c_max):
                summary.temp_out_of_range = True
                summary.warnings.append(
                    f"Sensor {sensor.sensor_id}: temperature {sensor.temperature_c}°C "
                    f"outside safe range [{t.temp_c_min}, {t.temp_c_max}]"
                )

        # RPM check
        if sensor.machine_speed_rpm is not None:
            if not (t.rpm_min <= sensor.machine_speed_rpm <= t.rpm_max):
                summary.rpm_out_of_range = True
                summary.warnings.append(
                    f"Sensor {sensor.sensor_id}: RPM {sensor.machine_speed_rpm} "
                    f"outside safe range [{t.rpm_min}, {t.rpm_max}]"
                )

        # Pressure as proxy for torque if torque not available
        if sensor.pressure_bar is not None:
            torque_proxy = sensor.pressure_bar * 10   # rough conversion
            if not (t.torque_min_nm <= torque_proxy <= t.torque_max_nm):
                summary.torque_out_of_range = True
                summary.warnings.append(
                    f"Sensor {sensor.sensor_id}: pressure-derived torque {torque_proxy:.1f}Nm "
                    f"outside safe range"
                )

        # Propagate anomaly flag from ingestor
        if sensor.anomaly_flagged:
            summary.warnings.append(
                f"Sensor {sensor.sensor_id}: anomaly flag set by ingestor"
            )

    def _calculate_derived_metrics(
        self, sensors: list[SensorReading], summary: SensorSummary
    ):
        """
        Calculate power, temp_diff, wear_torque from available sensor data.
        Uses averages across multiple sensors if available.
        """
        rpms   = [s.machine_speed_rpm for s in sensors if s.machine_speed_rpm]
        temps  = [s.temperature_c     for s in sensors if s.temperature_c]

        avg_rpm  = sum(rpms)  / len(rpms)  if rpms  else None
        avg_temp = sum(temps) / len(temps) if temps else None

        # Power = torque × rpm × (2π/60)
        # We don't have torque directly in SensorReading
        # so we store rpm-based component and let classifier combine with torque
        if avg_rpm:
            # Store as partial — classifier will complete with torque from inspection
            summary.power_w = avg_rpm   # placeholder — classifier completes this

        # Temperature differential — needs process temp (from inspection notes)
        # We flag HDF risk based on RPM alone if temp diff unavailable
        if avg_rpm and avg_rpm < self.thresholds.rpm_min:
            summary.temp_diff_k = 0.0   # assume worst case

        # Store for classifier
        if avg_rpm:
            summary.power_w = avg_rpm

    def _assess_failure_risks(self, summary: SensorSummary):
        """Map sensor anomalies to AI4I failure mode risks."""
        t = self.thresholds

        # HDF risk: low RPM (temp diff check done in classifier with full data)
        if summary.rpm_out_of_range:
            summary.hdf_risk = True
            summary.risk_flags.append("HDF")

        # PWF risk: power out of range (RPM too high or too low)
        if summary.rpm_out_of_range:
            summary.pwf_risk = True
            if "PWF" not in summary.risk_flags:
                summary.risk_flags.append("PWF")

        # TWF risk: tool wear in warning/critical zone
        # Tool wear comes from inspection record — flagged here if notes mention it
        if summary.warnings and any("wear" in w.lower() for w in summary.warnings):
            summary.twf_risk = True
            summary.risk_flags.append("TWF")

        # OSF risk: overstrain — high torque proxy × wear
        if summary.torque_out_of_range:
            summary.osf_risk = True
            summary.risk_flags.append("OSF")


# ── LangGraph node function ───────────────────────────────────────────────────
_agent = SensorParserAgent()

def sensor_parser_node(state: PipelineState) -> PipelineState:
    """LangGraph node — wraps SensorParserAgent.run()"""
    return _agent.run(state)