"""
ManufactureIQ — Shared Pipeline State & Data Models
====================================================
Single source of truth for all typed objects that flow through the
LangGraph pipeline. Every agent reads/writes fields on PipelineState
and never invents its own schema.

Pydantic v2.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Type aliases ──────────────────────────────────────────────────────────────
PipelineStatus = Literal[
    "received",
    "ingested",
    "classified",
    "root_cause_analyzed",
    "trend_detected",
    "actions_dispatched",
    "completed",
    "human_review_required",
    "failed",
]

Severity = Literal["low", "medium", "high", "critical"]

DefectType = Literal[
    "surface_scratch",
    "structural_crack",
    "dimensional_deviation",
    "material_contamination",
    "assembly_misalignment",
    "unknown",
]

DefectPattern = Literal[
    "isolated",
    "batch_pattern",
    "machine_pattern",
    "supplier_pattern",
    "unknown",
]


# ── Tier 1: Intake models ─────────────────────────────────────────────────────

class SensorReading(BaseModel):
    """A single validated sensor reading."""
    model_config = ConfigDict(extra="allow")

    sensor_id:         str
    timestamp:         datetime
    temperature_c:     Optional[float] = None
    vibration_hz:      Optional[float] = None
    pressure_bar:      Optional[float] = None
    machine_speed_rpm: Optional[float] = None
    anomaly_flagged:   bool = False


class InspectionRecord(BaseModel):
    """Validated inspection payload produced by the Data Ingestor."""
    model_config = ConfigDict(extra="allow")

    # Critical fields
    record_id:  str
    batch_id:   str
    product_id: str
    machine_id: str
    timestamp:  datetime

    # Optional fields
    operator_id:       Optional[str] = None
    supplier_id:       Optional[str] = None
    material_batch_id: Optional[str] = None
    raw_notes:         Optional[str] = None
    image_url:         Optional[str] = None
    defect_coords:     Optional[Any] = None


class IngestedData(BaseModel):
    """Structured output of the Data Ingestor + Sensor Parser."""
    inspection:      InspectionRecord
    sensors:         list[SensorReading] = Field(default_factory=list)
    ingest_warnings: list[str]           = Field(default_factory=list)


# ── Tier 2: Analysis models ───────────────────────────────────────────────────

class DefectClassification(BaseModel):
    """Output of the Defect Classifier agent."""
    model_config = ConfigDict(extra="allow")

    defect_type:           str               = "unknown"
    severity:              Severity          = "low"
    confidence_score:      float             = 0.0
    location_description:  str               = ""
    recommended_action:    str               = ""
    requires_human_review: bool              = False


class RootCauseReport(BaseModel):
    """Output of the Root Cause Analyzer agent."""
    model_config = ConfigDict(extra="allow")

    primary_cause:        str        = ""
    contributing_factors: list[str]  = Field(default_factory=list)
    similar_past_defects: list[str]  = Field(default_factory=list)
    material_issue:       bool       = False
    machine_issue:        bool       = False
    process_issue:        bool       = False
    confidence_score:     float      = 0.0
    raw_analysis:         str        = ""


class TrendReport(BaseModel):
    """Output of the Trend Detector agent."""
    model_config = ConfigDict(extra="allow")

    pattern:                  DefectPattern = "unknown"
    affected_batch_ids:       list[str]     = Field(default_factory=list)
    affected_machine_ids:     list[str]     = Field(default_factory=list)
    defect_frequency_last_7d: int           = 0
    escalate:                 bool          = False
    summary:                  str           = ""


class SupplierAlert(BaseModel):
    """Output of the Supplier Alert agent."""
    model_config = ConfigDict(extra="allow")

    triggered:         bool              = False
    supplier_id:       Optional[str]     = None
    material_batch_id: Optional[str]     = None
    alert_message:     str               = ""
    severity:          Severity          = "low"
    sent_at:           Optional[datetime]= None


class ComplianceReport(BaseModel):
    """Output of the Compliance Report agent."""
    model_config = ConfigDict(extra="allow")

    report_id:              str      = ""
    generated_at:           datetime = Field(default_factory=datetime.utcnow)
    iso_standard:           str      = "ISO 9001:2015"
    defect_summary:         str      = ""
    corrective_action_plan: str      = ""
    preventive_action_plan: str      = ""
    sign_off_required:      bool     = False
    full_report_text:       str      = ""


# ── Master Pipeline State ─────────────────────────────────────────────────────

class PipelineState(BaseModel):
    """
    Flows through every node in the LangGraph pipeline.
    Each agent mutates only the fields it owns.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Identity
    run_id: str

    # Raw inputs (what the user uploads)
    raw_inspection_payload: dict      = Field(default_factory=dict)
    raw_sensor_payload:     list[dict]= Field(default_factory=list)

    # Tier 1 outputs
    ingested:       Optional[IngestedData] = None

    # Sensor Parser output (between Tier 1 and 2)
    sensor_summary: Optional[dict]         = None

    # Tier 2 outputs
    classification: Optional[DefectClassification] = None
    root_cause:     Optional[RootCauseReport]      = None
    trend:          Optional[TrendReport]           = None

    # Tier 3 outputs
    supplier_alert:     Optional[SupplierAlert]    = None
    compliance_report:  Optional[ComplianceReport] = None

    # Human-in-the-loop
    human_review_notes: Optional[str]  = None
    human_approved:     Optional[bool] = None

    # Bookkeeping
    status:     PipelineStatus = "received"
    errors:     list[str]      = Field(default_factory=list)
    created_at: datetime       = Field(default_factory=datetime.utcnow)
    updated_at: datetime       = Field(default_factory=datetime.utcnow)