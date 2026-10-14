"""
ManufactureIQ — Supplier Alert Agent
======================================
Agent 6 of 7 — Tier 3: Action

Framework:  OpenAI Agents SDK
Role:       If the defect traces back to a material or supplier issue,
            generates a structured supplier alert with severity,
            affected batches, and recommended action.
            Runs in PARALLEL with Agent 7 (Compliance Report).

Input:      state.root_cause     (from Agent 4)
            state.trend          (from Agent 5)
            state.ingested       (from Agent 1)
            state.classification (from Agent 3)
Output:     state.supplier_alert (SupplierAlert)

Why OpenAI Agents SDK:
    Single-purpose structured output task — the Agents SDK's
    tool calling + structured output handling is cleaner than
    full LangGraph for a focused generation task like this.
    Also demonstrates multi-framework architecture.
"""

import os
import logging
from datetime import datetime
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

from core.defect_knowledge import KNOWLEDGE
from core.state import PipelineState, SupplierAlert

logger = logging.getLogger(__name__)


# ── Structured output schema for the agent ────────────────────────────────────
class SupplierAlertOutput(BaseModel):
    """Structured output the OpenAI Agent must return."""
    should_alert:       bool
    supplier_id:        Optional[str]
    material_batch_id:  Optional[str]
    severity:           str           # low | medium | high | critical
    alert_message:      str
    recommended_action: str
    affected_batches:   list[str]


# ── Agent instructions ────────────────────────────────────────────────────────
SUPPLIER_ALERT_INSTRUCTIONS = """You are a supply chain quality engineer responsible for
alerting suppliers when manufacturing defects are traced back to raw materials.

Your job:
1. Review the defect analysis and trend report provided
2. Decide if the defect is linked to a supplier material issue
3. If YES — write a clear, professional supplier alert
4. If NO — set should_alert to false and leave other fields empty

You are only called when the pipeline already has evidence of a material link
(the same material lot failing on several tools, or root-cause analysis naming
incoming material for a pattern that material can cause). Decide whether the
evidence is strong enough to contact the supplier; if it is thin, set
should_alert to false and say why.

Alert message rules:
- Professional and factual — no accusations
- Include specific batch IDs and defect type
- State the impact clearly
- Request corrective action with timeline"""


# Lazily-built structured LLM (avoids needing the API key at import time).
_structured_llm = None

def _get_structured_llm():
    global _structured_llm
    if _structured_llm is None:
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"), timeout=30, max_retries=1)
        _structured_llm = llm.with_structured_output(SupplierAlertOutput)
    return _structured_llm


class SupplierAlertAgentRunner:
    """
    Wraps the OpenAI Agents SDK agent and integrates
    it with the ManufactureIQ pipeline state.
    """

    def __init__(self):
        self.name = "SupplierAlertAgent"

    def run(self, state: PipelineState) -> PipelineState:
        """Main entry point — called by LangGraph orchestrator."""
        logger.info(f"[{self.name}] Starting — run_id: {state.run_id}")

        if state.status == "failed":
            logger.warning(f"[{self.name}] Pipeline failed — skipping")
            return state

        # ── Check if supplier alert is needed ─────────────────────────────────
        needs_alert = self._needs_alert(state)

        if not needs_alert:
            logger.info(f"[{self.name}] No supplier issue detected — skipping alert")
            state.supplier_alert = SupplierAlert(
                triggered=False,
                alert_message="No supplier issue identified for this defect.",
            )
            return state

        try:
            # ── Build prompt context ──────────────────────────────────────────
            prompt = self._build_prompt(state)

            # ── Run structured LLM ────────────────────────────────────────────
            logger.info(f"[{self.name}] Running LLM (structured output)...")
            output: SupplierAlertOutput = _get_structured_llm().invoke([
                SystemMessage(content=SUPPLIER_ALERT_INSTRUCTIONS),
                HumanMessage(content=prompt),
            ])

            # ── Map to PipelineState ──────────────────────────────────────────
            state.supplier_alert = SupplierAlert(
                triggered         = output.should_alert,
                supplier_id       = output.supplier_id or state.ingested.inspection.supplier_id,
                material_batch_id = output.material_batch_id or state.ingested.inspection.material_batch_id,
                alert_message     = output.alert_message,
                severity          = output.severity,
                sent_at           = datetime.utcnow() if output.should_alert else None,
            )

            state.updated_at = datetime.utcnow()

            logger.info(
                f"[{self.name}] Done — "
                f"triggered: {output.should_alert} | "
                f"severity: {output.severity} | "
                f"supplier: {output.supplier_id}"
            )

        except Exception as e:
            # Don't fail pipeline — alert is important but not blocking
            logger.exception(f"[{self.name}] Error generating alert")
            state.supplier_alert = SupplierAlert(
                triggered=False,
                alert_message=f"Alert generation failed: {str(e)}",
            )
            state.errors.append(f"[{self.name}] {str(e)}")

        return state

    # ── Private helpers ───────────────────────────────────────────────────────

    def _needs_alert(self, state: PipelineState) -> bool:
        """
        Decide if we even need to run the agent.
        Saves an API call if no supplier issue exists.
        """
        # No supplier on record — nothing to alert
        if not state.ingested.inspection.supplier_id:
            return False

        # Strongest evidence: the same material lot is failing on several tools.
        if state.trend and state.trend.pattern == "supplier_pattern":
            return True

        # Otherwise only if the pattern CAN come from incoming material
        # (Random, Near-full) AND root-cause analysis concluded it did.
        # One wafer with an Edge-Ring is a tool problem — never the supplier's.
        pattern = getattr(state.classification, "pattern", None) if state.classification else None
        material_possible = bool(pattern and KNOWLEDGE.get(pattern, {}).get("material_link"))
        return material_possible and bool(state.root_cause and state.root_cause.material_issue)

    def _build_prompt(self, state: PipelineState) -> str:
        """Build the prompt string for the OpenAI Agent."""
        inspection = state.ingested.inspection
        clf        = state.classification
        root_cause = state.root_cause
        trend      = state.trend

        return f"""
DEFECT ANALYSIS REPORT
======================
Product ID:       {inspection.product_id}
Batch ID:         {inspection.batch_id}
Machine ID:       {inspection.machine_id}
Supplier ID:      {inspection.supplier_id or 'Unknown'}
Material Batch:   {inspection.material_batch_id or 'Unknown'}
Timestamp:        {inspection.timestamp}

DEFECT CLASSIFICATION
---------------------
Pattern:    {getattr(clf, 'pattern', clf.defect_type) if clf else 'Unknown'} (wafer-map failure pattern)
Severity:   {clf.severity if clf else 'Unknown'}
Confidence: {clf.confidence_score if clf else 0:.0%}
Action:     {clf.recommended_action if clf else 'N/A'}

ROOT CAUSE ANALYSIS
-------------------
Primary Cause:    {root_cause.primary_cause if root_cause else 'Not analyzed'}
Material Issue:   {root_cause.material_issue if root_cause else False}
Machine Issue:    {root_cause.machine_issue if root_cause else False}
Contributing:     {', '.join(root_cause.contributing_factors) if root_cause else 'N/A'}

TREND REPORT
------------
Pattern:          {trend.pattern if trend else 'Unknown'}
Escalate:         {trend.escalate if trend else False}
Material tools:   {trend.material_tools_affected if trend else 0} tool(s) with defects on this material lot
Affected Batches: {trend.affected_batch_ids if trend else []}
Summary:          {trend.summary if trend else 'N/A'}

TASK: Based on this analysis, should a supplier alert be sent?
If yes, generate a professional supplier alert.
Return your response as a SupplierAlertOutput JSON object.
"""


# ── LangGraph node function ───────────────────────────────────────────────────
_runner = SupplierAlertAgentRunner()

def supplier_alert_node(state: PipelineState) -> PipelineState:
    """LangGraph node — wraps SupplierAlertAgentRunner.run()"""
    return _runner.run(state)