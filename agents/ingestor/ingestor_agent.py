"""
ManufactureIQ — Data Ingestor Agent
=====================================
Agent 1 of 7 — Tier 1: Intake

Framework:  Pure Python + Pydantic v2
Role:       Validates and structures a raw wafer inspection: identifiers,
            the wafer map (WM-811K grid) and the run's process sensors (SECOM)
Input:      raw_inspection_payload (dict), raw_sensor_payload (list of dicts)
Output:     IngestedData written to PipelineState.ingested

Rules:
- No LLM calls — this is pure validation and parsing
- Never mutates fields it doesn't own
- Logs all warnings but never silently drops data
- If critical fields are missing → sets status to "failed"
"""

import logging
from datetime import datetime
from typing import Optional
from pydantic import ValidationError

from core.state import (
    PipelineState,
    IngestedData,
    InspectionRecord,
    SensorReading,
    PipelineStatus,
)

logger = logging.getLogger(__name__)


# ── Field requirements ────────────────────────────────────────────────────────
# These fields MUST exist for the pipeline to continue.
# batch_id = production lot, machine_id = process tool. No wafer map → nothing
# to classify, so it is critical too.
CRITICAL_INSPECTION_FIELDS = [
    "record_id",
    "batch_id",
    "product_id",
    "machine_id",
    "timestamp",
    "wafer_map",
]

WAFER_MAP_MIN, WAFER_MAP_MAX = 5, 300     # grid side length, in die
N_PROCESS_SENSORS            = 590        # SECOM signals per run

# These fields are optional but logged as warnings if missing
OPTIONAL_INSPECTION_FIELDS = [
    "operator_id",
    "supplier_id",
    "material_batch_id",
    "raw_notes",
    "process_sensors",
]


class DataIngestorAgent:
    """
    Validates and structures raw payload into typed Pydantic models.
    Runs as the first node in the LangGraph pipeline.
    """

    def __init__(self):
        self.name = "DataIngestorAgent"

    def run(self, state: PipelineState) -> PipelineState:
        """
        Main entry point — called by LangGraph orchestrator.
        Takes PipelineState, returns updated PipelineState.
        """
        logger.info(f"[{self.name}] Starting ingestion — run_id: {state.run_id}")
        warnings = []

        try:
            # ── Step 1: Validate inspection payload ───────────────────────────
            inspection, insp_warnings = self._parse_inspection(
                state.raw_inspection_payload
            )
            warnings.extend(insp_warnings)

            if inspection is None:
                # Critical failure — cannot continue without inspection record
                state.status = "failed"
                state.errors.append(
                    f"[{self.name}] Critical: inspection payload invalid — pipeline halted"
                )
                logger.error(f"[{self.name}] Failed — invalid inspection payload")
                return state

            # ── Step 2: Validate sensor readings ─────────────────────────────
            sensors, sensor_warnings = self._parse_sensors(
                state.raw_sensor_payload,
                inspection.machine_id,
            )
            warnings.extend(sensor_warnings)

            # ── Step 3: Build IngestedData ────────────────────────────────────
            ingested = IngestedData(
                inspection=inspection,
                sensors=sensors,
                ingest_warnings=warnings,
            )

            # ── Step 4: Update state ──────────────────────────────────────────
            state.ingested  = ingested
            state.status    = "ingested"
            state.updated_at = datetime.utcnow()

            logger.info(
                f"[{self.name}] Done — "
                f"product: {inspection.product_id} | "
                f"batch: {inspection.batch_id} | "
                f"sensors: {len(sensors)} | "
                f"warnings: {len(warnings)}"
            )

        except Exception as e:
            state.status = "failed"
            state.errors.append(f"[{self.name}] Unexpected error: {str(e)}")
            logger.exception(f"[{self.name}] Unexpected error")

        return state

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_inspection(
        self, payload: dict
    ) -> tuple[Optional[InspectionRecord], list[str]]:
        """
        Validate inspection payload.
        Returns (InspectionRecord, warnings) or (None, errors) on failure.
        """
        warnings = []

        if not payload:
            return None, ["Inspection payload is empty"]

        # Check critical fields
        missing_critical = [
            f for f in CRITICAL_INSPECTION_FIELDS if not payload.get(f)
        ]
        if missing_critical:
            return None, [f"Missing critical fields: {missing_critical}"]

        # Check optional fields — warn but continue
        missing_optional = [
            f for f in OPTIONAL_INSPECTION_FIELDS if not payload.get(f)
        ]
        if missing_optional:
            warnings.append(f"Missing optional fields: {missing_optional}")

        # Parse and validate timestamp
        payload = self._normalize_timestamp(payload, "timestamp", warnings)

        # Wafer map must be a clean 0/1/2 grid — a malformed map would silently
        # produce garbage features, so reject it here.
        map_error = self._check_wafer_map(payload.get("wafer_map"))
        if map_error:
            return None, [f"Invalid wafer_map: {map_error}"]

        sensors = payload.get("process_sensors")
        if sensors is not None and (not isinstance(sensors, list) or len(sensors) != N_PROCESS_SENSORS):
            warnings.append(f"process_sensors must have {N_PROCESS_SENSORS} values "
                            f"(got {len(sensors) if isinstance(sensors, list) else type(sensors).__name__}) — ignored")
            payload = {**payload, "process_sensors": None}

        # Validate with Pydantic
        try:
            record = InspectionRecord(**payload)
            return record, warnings
        except ValidationError as e:
            errors = [f"Validation error: {err['msg']}" for err in e.errors()]
            return None, errors

    @staticmethod
    def _check_wafer_map(grid) -> Optional[str]:
        if not isinstance(grid, list) or not grid or not all(isinstance(r, list) for r in grid):
            return "must be a 2-D list"
        h, w = len(grid), len(grid[0])
        if not (WAFER_MAP_MIN <= h <= WAFER_MAP_MAX and WAFER_MAP_MIN <= w <= WAFER_MAP_MAX):
            return f"size {h}x{w} outside {WAFER_MAP_MIN}-{WAFER_MAP_MAX}"
        if any(len(r) != w for r in grid):
            return "rows have different lengths"
        if any(v not in (0, 1, 2) for r in grid for v in r):
            return "values must be 0 (no die), 1 (pass) or 2 (fail)"
        if not any(v for r in grid for v in r):
            return "no die on the wafer"
        return None

    def _parse_sensors(
        self, payload: list[dict], machine_id: str
    ) -> tuple[list[SensorReading], list[str]]:
        """
        Validate list of sensor readings.
        Skips invalid readings but logs them — never drops silently.
        """
        warnings  = []
        sensors   = []

        if not payload:
            return sensors, warnings      # optional: process data lives in inspection.process_sensors

        for i, raw in enumerate(payload):
            raw = self._normalize_timestamp(raw, "timestamp", warnings)

            # Auto-fill sensor_id if missing
            if not raw.get("sensor_id"):
                raw["sensor_id"] = f"{machine_id}-S{i+1}"
                warnings.append(f"Sensor {i+1}: sensor_id missing — auto-assigned {raw['sensor_id']}")

            try:
                sensor = SensorReading(**raw)
                sensors.append(sensor)
            except ValidationError as e:
                warnings.append(
                    f"Sensor {i+1} skipped — validation error: {e.errors()[0]['msg']}"
                )

        if not sensors:
            warnings.append("All sensor readings failed validation — running on inspection data only")

        return sensors, warnings

    def _normalize_timestamp(
        self, payload: dict, field: str, warnings: list
    ) -> dict:
        """
        Converts string timestamps to datetime objects.
        Handles ISO format strings gracefully.
        """
        payload = payload.copy()
        val = payload.get(field)

        if isinstance(val, str):
            try:
                payload[field] = datetime.fromisoformat(val.replace("Z", "+00:00"))
            except ValueError:
                warnings.append(f"Could not parse {field}: '{val}' — using current time")
                payload[field] = datetime.utcnow()
        elif val is None:
            warnings.append(f"Field '{field}' is None — using current time")
            payload[field] = datetime.utcnow()

        return payload


# ── LangGraph node function ───────────────────────────────────────────────────
# This is what LangGraph calls — a plain function that takes and returns state

_agent = DataIngestorAgent()

def ingestor_node(state: PipelineState) -> PipelineState:
    """LangGraph node — wraps DataIngestorAgent.run()"""
    return _agent.run(state)