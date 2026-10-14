"""
ManufactureIQ — Compliance Report Agent
=========================================
Agent 7 of 7 — Tier 3: Action

Framework:  LangChain + OpenAI (single gpt-4o-mini call)
Role:       Generates an ISO 9001:2015 compliant corrective action
            report from the full pipeline analysis.
            Runs CONCURRENTLY with Agent 6 (Supplier Alert) via the
            orchestrator's `actions` node.

Input:      state.classification  (from Agent 3)
            state.root_cause      (from Agent 4)
            state.trend           (from Agent 5)
            state.ingested        (from Agent 1)
Output:     state.compliance_report (ComplianceReport)

Design note:
    Previously a 2-agent CrewAI crew (QC Analyst + Compliance Writer) = 2 LLM
    calls. Collapsed to a single gpt-4o-mini prompt that plays both roles — half
    the calls, ~20x cheaper, and no CrewAI/Redis/telemetry dependency.
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from core.state import PipelineState, ComplianceReport

logger = logging.getLogger(__name__)


# Combines the two former CrewAI roles (QC Analyst + Compliance Writer) into a
# single prompt — one LLM call instead of two.
COMPLIANCE_SYSTEM_PROMPT = """You are both a senior Quality Control Analyst and an
ISO 9001:2015 certified documentation specialist. You interpret manufacturing defect
data (sensor readings, root cause, trends, supplier info) and write formal Corrective
Action Reports (CARs) used in audits, supplier reviews, and regulatory submissions.
Your reports are precise, complete, follow the 8D methodology, and reference
ISO 9001:2015 clause 10.2 (Nonconformity and corrective action)."""

COMPLIANCE_USER_TEMPLATE = """Write a complete ISO 9001:2015 Corrective Action Report (CAR)
for the manufacturing defect below.

{context}

The report MUST include these sections, in order, with clear headers:
1. DEFECT SUMMARY — what happened, when, where (use exact IDs/values)
2. ROOT CAUSE ANALYSIS — primary cause and contributing factors
3. SCOPE OF IMPACT — affected batches, machines, suppliers
4. IMMEDIATE CORRECTIVE ACTIONS — steps taken or ordered
5. PREVENTIVE ACTION PLAN — steps to prevent recurrence
6. VERIFICATION & SIGN-OFF — who must approve and when

Reference ISO 9001:2015 clause 10.2. Use professional, audit-ready language.
End with: "Prepared by: ManufactureIQ AI System | Date: {date}"."""


class ComplianceReportAgent:
    """
    Generates an ISO 9001:2015 corrective action report with a single
    GPT-4o-mini call (previously a 2-agent CrewAI crew = 2 calls).
    """

    def __init__(self):
        self.name = "ComplianceReportAgent"
        self.llm  = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.2,
            timeout=30,        # fail fast instead of hanging the solo worker
            max_retries=1,
        )

    def run(self, state: PipelineState) -> PipelineState:
        """Main entry point — called by LangGraph orchestrator."""
        logger.info(f"[{self.name}] Starting — run_id: {state.run_id}")

        if state.status == "failed":
            logger.warning(f"[{self.name}] Pipeline failed — skipping")
            return state

        if not state.classification:
            state.errors.append(f"[{self.name}] No classification — cannot generate report")
            return state

        try:
            # ── Build context from pipeline state ─────────────────────────────
            context = self._build_context(state)

            # ── Single LLM call — QC analysis + ISO report in one prompt ──────
            logger.info(f"[{self.name}] Generating ISO 9001 report (single LLM call)...")
            response = self.llm.invoke([
                SystemMessage(content=COMPLIANCE_SYSTEM_PROMPT),
                HumanMessage(content=COMPLIANCE_USER_TEMPLATE.format(
                    context=context,
                    date=datetime.utcnow().strftime("%Y-%m-%d"),
                )),
            ])
            report_text = response.content

            # ── Parse into ComplianceReport ───────────────────────────────────
            report = self._parse_report(report_text, state)

            state.compliance_report = report
            state.updated_at        = datetime.utcnow()

            logger.info(
                f"[{self.name}] Done — "
                f"report_id: {report.report_id} | "
                f"sign_off: {report.sign_off_required}"
            )

        except Exception as e:
            logger.exception(f"[{self.name}] Error generating report")
            state.errors.append(f"[{self.name}] {str(e)}")
            # Don't fail pipeline — report failure is non-blocking
            state.compliance_report = ComplianceReport(
                report_id       = f"CAR-{uuid.uuid4().hex[:8].upper()}",
                generated_at    = datetime.utcnow(),
                defect_summary  = "Report generation failed — see errors",
                full_report_text= f"Error: {str(e)}",
                sign_off_required = True,   # always require sign-off on failure
            )

        return state

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_context(self, state: PipelineState) -> str:
        """Build context string from pipeline state for CrewAI agents."""
        inspection = state.ingested.inspection
        clf        = state.classification
        root_cause = state.root_cause
        trend      = state.trend
        summary    = state.sensor_summary or {}
        wafer      = summary.get("wafer") or {}
        spc        = summary.get("spc") or {}

        return f"""
INSPECTION RECORD
-----------------
Record ID:        {inspection.record_id}
Product ID:       {inspection.product_id}
Wafer ID:         {inspection.wafer_id or inspection.record_id}
Lot ID:           {inspection.batch_id}
Process tool:     {inspection.machine_id}
Operator ID:      {inspection.operator_id or 'Unknown'}
Supplier ID:      {inspection.supplier_id or 'Unknown'}
Material Batch:   {inspection.material_batch_id or 'Unknown'}
Timestamp:        {inspection.timestamp}
Notes:            {inspection.raw_notes or 'None'}

WAFER & PROCESS DATA
--------------------
Die / failing:    {wafer.get('die', 'N/A')} / {wafer.get('failing_die', 'N/A')}  (yield {wafer.get('yield_pct', 'N/A')}%)
SPC:              {spc.get('out_of_control', 'N/A')} process sensors outside 3σ (normal ≤{spc.get('typical_out_of_control_p95', 'N/A')})
Process risk:     {trend.process_risk if trend and trend.process_risk is not None else 'N/A'} (low-confidence model)

DEFECT CLASSIFICATION
---------------------
Defect pattern:   {getattr(clf, 'pattern', clf.defect_type)} (WM-811K wafer-map pattern)
Severity:         {clf.severity}
Confidence:       {clf.confidence_score:.0%}
Location:         {clf.location_description}
Recommended:      {clf.recommended_action}
Human review:     {clf.requires_human_review}

ROOT CAUSE ANALYSIS
-------------------
Primary cause:    {root_cause.primary_cause if root_cause else 'Not analyzed'}
Contributing:     {', '.join(root_cause.contributing_factors) if root_cause else 'N/A'}
Material issue:   {root_cause.material_issue if root_cause else False}
Machine issue:    {root_cause.machine_issue if root_cause else False}
Process issue:    {root_cause.process_issue if root_cause else False}
Confidence:       {root_cause.confidence_score if root_cause else 0:.0%}

TREND REPORT
------------
Pattern:          {trend.pattern if trend else 'Unknown'}
Escalate:         {trend.escalate if trend else False}
Same pattern in lot: {trend.lot_same_pattern if trend else 0} of {trend.lot_wafers_seen if trend else 0} wafers inspected
Affected batches: {trend.affected_batch_ids if trend else []}
Affected machines:{trend.affected_machine_ids if trend else []}
Summary:          {trend.summary if trend else 'N/A'}
"""

    def _parse_report(
        self, report_text: str, state: PipelineState
    ) -> ComplianceReport:
        """Parse the CrewAI output into a ComplianceReport."""
        clf   = state.classification
        trend = state.trend

        # Extract sections using simple text parsing
        def extract_section(text: str, header: str) -> str:
            lines   = text.split("\n")
            capture = False
            result  = []
            for line in lines:
                if header.upper() in line.upper():
                    capture = True
                    continue
                if capture:
                    # Stop at next section header (all caps or numbered)
                    if line.strip() and any(
                        kw in line.upper() for kw in [
                            "ROOT CAUSE", "SCOPE", "IMMEDIATE", "PREVENTIVE",
                            "VERIFICATION", "DEFECT SUMMARY", "SIGN-OFF"
                        ]
                    ) and line.strip() != line.strip().lower():
                        break
                    result.append(line)
            return "\n".join(result).strip()

        # Determine if sign-off is required
        sign_off = (
            clf.requires_human_review or
            clf.severity in ["high", "critical"] or
            (trend and trend.escalate)
        )

        return ComplianceReport(
            report_id             = f"CAR-{uuid.uuid4().hex[:8].upper()}",
            generated_at          = datetime.utcnow(),
            iso_standard          = "ISO 9001:2015",
            defect_summary        = extract_section(report_text, "DEFECT SUMMARY") or
                                    f"{clf.defect_type} defect detected — {clf.severity} severity",
            corrective_action_plan= extract_section(report_text, "IMMEDIATE CORRECTIVE") or
                                    clf.recommended_action,
            preventive_action_plan= extract_section(report_text, "PREVENTIVE ACTION") or
                                    "Preventive actions to be defined by engineering team.",
            sign_off_required     = sign_off,
            full_report_text      = report_text,
        )


# ── LangGraph node function ───────────────────────────────────────────────────
_agent = ComplianceReportAgent()

def compliance_report_node(state: PipelineState) -> PipelineState:
    """LangGraph node — wraps ComplianceReportAgent.run()"""
    return _agent.run(state)