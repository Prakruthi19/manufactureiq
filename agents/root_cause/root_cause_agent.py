"""
ManufactureIQ — Root Cause Analyzer Agent
==========================================
Agent 4 of 7 — Tier 2: Analysis

Framework:  LangChain + ChromaDB RAG + OpenAI GPT-4o
Role:       Retrieves similar past defects from ChromaDB,
            sends context to GPT-4o, returns structured
            root cause explanation.

Input:      state.classification  (from Agent 3)
            state.ingested        (from Agent 1)
            state.sensor_summary  (from Agent 2)
Output:     state.root_cause      (RootCauseReport)

Why LLM here:
    Root cause analysis requires language understanding —
    correlating sensor patterns, machine history, and material
    data into a human-readable causal explanation.
    ML cannot generate natural language explanations.
    RAG ensures the LLM grounds its answer in real past defects,
    not hallucinations.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
# Keep ChromaDB's built-in embedding model on the D: drive WITHOUT hijacking
# USERPROFILE/HOME. Setting those globally breaks other libraries that resolve
# Windows known-folders from %USERPROFILE% (e.g. CrewAI's appdirs → LocalAppData).
# Instead, point the ONNX embedder's download path straight at the seeded model.
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
# Portable cache location: honour CHROMA_CACHE_DIR when set (Windows dev points it
# at the pre-seeded model on D:), otherwise fall back to a repo-relative dir. On a
# fresh Linux host (e.g. Render) the ~80MB model downloads here on first use.
_ONNX_CACHE = os.getenv("CHROMA_CACHE_DIR") or str(
    Path(__file__).resolve().parent.parent.parent / ".cache_home"
)
ONNXMiniLM_L6_V2.DOWNLOAD_PATH = (
    Path(_ONNX_CACHE) / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2"
)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.state import PipelineState, RootCauseReport

logger   = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent.parent.parent

# ── Defect type to human label ────────────────────────────────────────────────
DEFECT_LABELS = {
    "structural_crack":      "Tool Wear Failure (TWF)",
    "heat_dissipation":      "Heat Dissipation Failure (HDF)",
    "assembly_misalignment": "Power Failure (PWF)",
    "dimensional_deviation": "Overstrain Failure (OSF)",
    "material_contamination":"Heat Dissipation Failure (HDF)",
    "unknown":               "Unknown / Random Failure",
    "surface_scratch":       "Surface Defect",
}

# ── Root cause prompt ─────────────────────────────────────────────────────────
ROOT_CAUSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert manufacturing quality engineer with 20 years of experience 
analyzing machine failures using sensor data and historical defect records.

Your job is to identify the ROOT CAUSE of a detected defect and explain it clearly 
to a quality control manager. Be specific, technical but understandable, and always 
ground your analysis in the sensor data and historical context provided.

Format your response EXACTLY as follows:
PRIMARY CAUSE: [one sentence — the main root cause]
CONTRIBUTING FACTORS: [2-3 bullet points of secondary factors]
MATERIAL ISSUE: [Yes/No — is the raw material or supplier involved?]
MACHINE ISSUE: [Yes/No — is the machine itself the problem?]
PROCESS ISSUE: [Yes/No — is the manufacturing process the problem?]
CONFIDENCE: [High/Medium/Low — how confident are you in this analysis?]
EXPLANATION: [2-3 sentences explaining the failure mechanism in plain language]"""),

    ("human", """A defect has been detected. Analyze the root cause.

DEFECT DETECTED:
- Type: {defect_type}
- Severity: {severity}
- Confidence: {confidence}
- Location: {location}

CURRENT SENSOR READINGS:
- Product type: {product_type}
- Air temperature: {air_temp}°C
- Machine speed: {rpm} RPM
- Anomaly flags: {anomaly_flags}
- Risk flags: {risk_flags}

INSPECTION NOTES:
{inspection_notes}

SIMILAR PAST DEFECTS (from historical database):
{similar_defects}

Based on this data, what is the root cause of this defect?""")
])


class RootCauseAnalyzerAgent:
    """
    Uses RAG to retrieve similar past defects from ChromaDB,
    then asks GPT-4o to explain the root cause.
    """

    def __init__(self):
        self.name       = "RootCauseAnalyzerAgent"
        self.collection = None
        self.llm        = None
        self.chain      = None
        self._initialize()

    def _initialize(self):
        """Load ChromaDB collection and initialize LLM chain."""
        try:
            # ── ChromaDB ──────────────────────────────────────────────────────
            chroma_dir = BASE_DIR / "data" / "chroma"
            client     = chromadb.PersistentClient(path=str(chroma_dir))
            self.collection = client.get_collection("defect_history")
            logger.info(
                f"[{self.name}] ChromaDB loaded — "
                f"{self.collection.count()} records"
            )

            # ── LLM chain ─────────────────────────────────────────────────────
            self.llm   = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,   # low temperature = consistent, factual answers
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=30,        # fail fast instead of hanging the solo worker
                max_retries=1,
            )
            self.chain = ROOT_CAUSE_PROMPT | self.llm | StrOutputParser()
            logger.info(f"[{self.name}] LLM chain initialized — gpt-4o")

        except Exception as e:
            logger.error(f"[{self.name}] Initialization failed: {e}")
            raise

    def run(self, state: PipelineState) -> PipelineState:
        """Main entry point — called by LangGraph orchestrator."""
        logger.info(f"[{self.name}] Starting — run_id: {state.run_id}")

        # ── Guards ────────────────────────────────────────────────────────────
        if state.status == "failed":
            logger.warning(f"[{self.name}] Pipeline failed — skipping")
            return state

        if not state.classification:
            state.errors.append(f"[{self.name}] No classification found — skipping")
            return state

        try:
            clf        = state.classification
            inspection = state.ingested.inspection
            summary    = state.sensor_summary or {}
            sensors    = state.ingested.sensors

            # ── Step 1: Retrieve similar past defects from ChromaDB ───────────
            similar_defects = self._retrieve_similar(clf.defect_type, summary)

            # ── Step 2: Build query context ───────────────────────────────────
            sensor      = sensors[0] if sensors else None
            air_temp    = round(sensor.temperature_c, 2) if sensor and sensor.temperature_c else "N/A"
            rpm         = sensor.machine_speed_rpm        if sensor else "N/A"

            context = {
                "defect_type":      DEFECT_LABELS.get(clf.defect_type, clf.defect_type),
                "severity":         clf.severity,
                "confidence":       f"{clf.confidence_score:.0%}",
                "location":         clf.location_description,
                "product_type":     inspection.product_id[0] if inspection.product_id else "M",
                "air_temp":         air_temp,
                "rpm":              rpm,
                "anomaly_flags":    summary.get("any_anomaly", False),
                "risk_flags":       summary.get("risk_flags", []),
                "inspection_notes": inspection.raw_notes or "No notes provided",
                "similar_defects":  similar_defects,
            }

            # ── Step 3: Call LLM ──────────────────────────────────────────────
            logger.info(f"[{self.name}] Calling GPT-4o for root cause analysis...")
            raw_response = self.chain.invoke(context)

            # ── Step 4: Parse structured response ────────────────────────────
            report = self._parse_response(
                raw_response,
                similar_defects,
            )

            # ── Step 5: Update state ──────────────────────────────────────────
            state.root_cause = report
            state.status     = "root_cause_analyzed"
            state.updated_at = datetime.utcnow()

            logger.info(
                f"[{self.name}] Done — "
                f"primary cause: {report.primary_cause[:60]}... | "
                f"material: {report.material_issue} | "
                f"machine: {report.machine_issue}"
            )

        except Exception as e:
            state.errors.append(f"[{self.name}] Error: {str(e)}")
            logger.exception(f"[{self.name}] Unexpected error")
            # Don't fail the pipeline — root cause is informational
            # Downstream agents can still run

        return state

    # ── Private helpers ───────────────────────────────────────────────────────

    def _retrieve_similar(
        self, defect_type: str, summary: dict, n: int = 5
    ) -> str:
        """
        Query ChromaDB for similar past defects.
        Returns formatted string for the LLM prompt.
        """
        try:
            query = (
                f"{defect_type} defect "
                f"{'high RPM' if summary.get('rpm_out_of_range') else ''} "
                f"{'power failure' if summary.get('pwf_risk') else ''} "
                f"{'heat dissipation' if summary.get('hdf_risk') else ''} "
                f"{'tool wear' if summary.get('twf_risk') else ''}"
            ).strip()

            results = self.collection.query(
                query_texts=[query],
                n_results=n,
                where={"machine_failure": 1}
            )

            if not results["documents"][0]:
                return "No similar historical defects found."

            lines = []
            for i, (doc, meta) in enumerate(
                zip(results["documents"][0], results["metadatas"][0])
            ):
                lines.append(
                    f"[Past defect {i+1}] {meta['defect_type']} | "
                    f"RPM: {meta['rpm']} | Torque: {meta['torque']} Nm | "
                    f"Tool wear: {meta['tool_wear']} min\n"
                    f"  Details: {doc[:200]}"
                )

            return "\n".join(lines)

        except Exception as e:
            logger.warning(f"[{self.name}] ChromaDB retrieval failed: {e}")
            return "Historical defect retrieval unavailable."

    def _parse_response(
        self, raw: str, similar_defects: str
    ) -> RootCauseReport:
        """
        Parse the LLM's structured text response into a RootCauseReport.
        Falls back gracefully if parsing fails.
        """
        def extract(label: str) -> str:
            for line in raw.split("\n"):
                if line.strip().upper().startswith(label.upper()):
                    return line.split(":", 1)[-1].strip()
            return ""

        def extract_bool(label: str) -> bool:
            val = extract(label).lower()
            return val.startswith("yes")

        def extract_bullets() -> list[str]:
            lines   = raw.split("\n")
            bullets = []
            capture = False
            for line in lines:
                if "CONTRIBUTING FACTORS" in line.upper():
                    capture = True
                    continue
                if capture and line.strip().startswith(("-", "•", "*")):
                    bullets.append(line.strip().lstrip("-•* "))
                elif capture and line.strip() and not line.strip().startswith(("-", "•", "*")):
                    if any(kw in line.upper() for kw in ["MATERIAL", "MACHINE", "PROCESS", "CONFIDENCE", "EXPLANATION"]):
                        break
            return bullets or ["See full analysis below"]

        # Extract similar defect record IDs for traceability
        similar_ids = [
            line.split("|")[0].replace("[Past defect", "").strip().strip("]")
            for line in similar_defects.split("\n")
            if line.startswith("[Past defect")
        ]

        return RootCauseReport(
            primary_cause        = extract("PRIMARY CAUSE") or "Unable to determine primary cause",
            contributing_factors = extract_bullets(),
            similar_past_defects = similar_ids,
            material_issue       = extract_bool("MATERIAL ISSUE"),
            machine_issue        = extract_bool("MACHINE ISSUE"),
            process_issue        = extract_bool("PROCESS ISSUE"),
            confidence_score     = 0.85 if "High" in extract("CONFIDENCE") else
                                   0.60 if "Medium" in extract("CONFIDENCE") else 0.40,
            raw_analysis         = raw,
        )


# ── LangGraph node function ───────────────────────────────────────────────────
_agent = RootCauseAnalyzerAgent()

def root_cause_node(state: PipelineState) -> PipelineState:
    """LangGraph node — wraps RootCauseAnalyzerAgent.run()"""
    return _agent.run(state)