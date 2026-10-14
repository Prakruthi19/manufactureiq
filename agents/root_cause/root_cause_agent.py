"""
ManufactureIQ — Root Cause Analyzer Agent
==========================================
Agent 4 of 7 — Tier 2: Analysis

Framework:  ChromaDB vector search + LangChain + OpenAI
Role:       Explains the most likely cause of the defect, using only evidence:

    1. Pattern knowledge — what this WM-811K pattern usually points to
                           (core/defect_knowledge.py)
    2. Similar wafers    — the 10 nearest historical wafers in feature space
                           (ChromaDB, ~40k expert-labelled wafers from the
                           TRAINING lots only — no test wafers in the index).
                           Their labels are an independent second opinion:
                           if neighbours disagree with the classifier, say so.
    3. Wafer evidence    — where failures sit (centre / edge / clusters / lines)
    4. Process evidence  — SPC out-of-control sensors from the Sensor Parser

The LLM may only choose among the documented causes for the pattern and must
state its confidence; it is told explicitly not to invent tools or events.

Why an LLM here:
    Turning four kinds of evidence into a short explanation a shift engineer
    can act on is a language task. The DECISIONS (pattern, escalation) are made
    by models and rules upstream; the LLM explains, it doesn't decide.
"""

import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import chromadb
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from core.defect_knowledge import KNOWLEDGE
from core.state import PipelineState, RootCauseReport
from core.wafer_features import features_from_map

logger   = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
N_SIMILAR = 10

ROOT_CAUSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a semiconductor yield engineer. You explain wafer-map failure
patterns to a shift engineer using ONLY the evidence given. Rules:
- Choose the primary cause from the KNOWN CAUSES list for the pattern. Do not invent
  tools, events, measurements or suppliers that are not in the evidence.
- If the similar-wafer vote disagrees with the classifier, or classifier confidence is
  low, say so and lower your confidence.
- MATERIAL ISSUE may be Yes only if the pattern's known causes include incoming
  material/chemicals AND the evidence does not point more strongly elsewhere.

Format EXACTLY:
PRIMARY CAUSE: [one sentence]
CONTRIBUTING FACTORS:
- [evidence-backed factor]
- [evidence-backed factor]
MATERIAL ISSUE: [Yes/No]
MACHINE ISSUE: [Yes/No]
PROCESS ISSUE: [Yes/No]
CONFIDENCE: [High/Medium/Low]
EXPLANATION: [2-3 plain sentences: what the map shows, why that points to the cause, what to check first]"""),
    ("human", """CLASSIFIER: {pattern} ({confidence} confidence; runner-up {runner_up})
PATTERN MEANING: {location}
KNOWN CAUSES for {pattern}:
{known_causes}

WAFER EVIDENCE:
{wafer_evidence}

SIMILAR PAST WAFERS ({n_similar} nearest, expert-labelled): {similar_vote}

PROCESS SENSORS (SPC, mean±3σ of good runs): {spc}

CONTEXT: lot {lot}, tool {tool}, material lot {material}
NOTES: {notes}""")
])


class RootCauseAnalyzerAgent:

    def __init__(self):
        self.name = "RootCauseAnalyzerAgent"
        client = chromadb.PersistentClient(path=str(BASE_DIR / "data" / "chroma"))
        # embedding_function=None: vectors are our own wafer features, not text
        self.collection = client.get_collection("wafer_history", embedding_function=None)
        scaler = np.load(BASE_DIR / "models" / "wafer_index_scaler.npz")
        self.mean, self.std = scaler["mean"], scaler["std"]
        self.chain = ROOT_CAUSE_PROMPT | ChatOpenAI(
            model=LLM_MODEL, temperature=0.1, api_key=os.getenv("OPENAI_API_KEY"),
            timeout=30, max_retries=1,
        ) | StrOutputParser()
        logger.info(f"[{self.name}] {self.collection.count():,} historical wafers indexed; LLM {LLM_MODEL}")

    def run(self, state: PipelineState) -> PipelineState:
        logger.info(f"[{self.name}] Starting — run_id: {state.run_id}")
        if state.status == "failed" or not state.classification:
            return state

        try:
            clf, insp = state.classification, state.ingested.inspection
            pattern   = clf.pattern
            info      = KNOWLEDGE[pattern]

            vote, neighbours = self._similar_wafers(insp.wafer_map)
            agree = vote.get(pattern, 0)

            spc = (state.sensor_summary or {}).get("spc")
            spc_text = "not available" if not spc else (
                f"{spc['out_of_control']} of {spc['sensors_checked']} sensors outside limits "
                f"(normal runs: ≤{spc['typical_out_of_control_p95']:.0f}); worst: " +
                (", ".join(f"{w['sensor']} {w['z_score']:+.1f}σ" for w in spc["worst_sensors"]) or "none") +
                ". Sensor names are anonymised in the source data, so they cannot be mapped to tools.")

            raw = self.chain.invoke({
                "pattern":        pattern,
                "confidence":     f"{clf.confidence_score:.0%}",
                "runner_up":      f"{clf.runner_up['pattern']} {clf.runner_up['probability']:.0%}",
                "location":       info["location"],
                "known_causes":   "\n".join(f"- {c}" for c in info["likely_causes"]),
                "wafer_evidence": "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in clf.evidence.items()),
                "n_similar":      N_SIMILAR,
                "similar_vote":   ", ".join(f"{k} {v}" for k, v in vote.most_common()) +
                                  f" → {agree}/{N_SIMILAR} agree with the classifier",
                "spc":            spc_text,
                "lot":            insp.batch_id,
                "tool":           insp.machine_id,
                "material":       insp.material_batch_id or "unknown",
                "notes":          insp.raw_notes or "none",
            })

            report = self._parse(raw)
            report.similar_past_defects = neighbours
            report.neighbour_agreement  = f"{agree}/{N_SIMILAR}"
            state.root_cause = report
            state.status     = "root_cause_analyzed"
            state.updated_at = datetime.utcnow()
            logger.info(f"[{self.name}] Done — {report.primary_cause[:70]}")

        except Exception as e:
            # Informational agent — downstream agents still run without it.
            state.errors.append(f"[{self.name}] Error: {e}")
            logger.exception(f"[{self.name}] Unexpected error")

        return state

    def _similar_wafers(self, wafer_map) -> tuple[Counter, list[str]]:
        q = (features_from_map(wafer_map) - self.mean) / self.std
        res = self.collection.query(query_embeddings=[q.tolist()], n_results=N_SIMILAR)
        metas = res["metadatas"][0]
        vote  = Counter(m["pattern"] for m in metas)
        return vote, [f"{m['wafer']} ({m['pattern']})" for m in metas]

    @staticmethod
    def _parse(raw: str) -> RootCauseReport:
        def field(label):
            for line in raw.splitlines():
                if line.strip().upper().startswith(label):
                    return line.split(":", 1)[-1].strip()
            return ""

        bullets, capture = [], False
        for line in raw.splitlines():
            s = line.strip()
            if s.upper().startswith("CONTRIBUTING FACTORS"):
                capture = True
                continue
            if capture and s.startswith(("-", "•", "*")):
                bullets.append(s.lstrip("-•* "))
            elif capture and s:
                break

        conf = field("CONFIDENCE").lower()
        return RootCauseReport(
            primary_cause        = field("PRIMARY CAUSE") or "Unable to determine primary cause",
            contributing_factors = bullets,
            material_issue       = field("MATERIAL ISSUE").lower().startswith("yes"),
            machine_issue        = field("MACHINE ISSUE").lower().startswith("yes"),
            process_issue        = field("PROCESS ISSUE").lower().startswith("yes"),
            confidence_score     = 0.85 if conf.startswith("high") else 0.6 if conf.startswith("medium") else 0.4,
            raw_analysis         = raw,
            explanation          = field("EXPLANATION"),
        )


_agent = None

def root_cause_node(state: PipelineState) -> PipelineState:
    """LangGraph node — built on first use so the API key is only needed when called."""
    global _agent
    if _agent is None:
        _agent = RootCauseAnalyzerAgent()
    return _agent.run(state)
