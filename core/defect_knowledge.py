"""
ManufactureIQ — Wafer defect-pattern knowledge base
====================================================
What each WM-811K failure pattern usually points to on a fab floor. Used by:
    • Defect Classifier  → severity, location text, recommended action
    • Root Cause agent   → grounding facts for the LLM (it may only reason
                           from these + the evidence, not invent causes)
    • Supplier Alert     → `material_link`: can incoming material plausibly
                           cause this pattern at all?

Why a lookup table and not vector search?
    There are 9 patterns. Retrieving 1 of 9 documents by embedding similarity
    adds latency (an embedding model) and a failure mode (retrieving the wrong
    one) for no benefit — the classifier already tells us which pattern it is.
    Vector search IS used where it pays off: finding similar past wafers among
    ~40k historical ones (see agents/root_cause/root_cause_agent.py).

Content reflects causes commonly cited in wafer-map literature (e.g. Wu et al.
2015 and the defect-pattern surveys that followed). A pattern suggests where to
look; it does not prove a cause — the wording keeps that distinction.
"""

KNOWLEDGE: dict[str, dict] = {
    "none": {
        "label": "No pattern",
        "severity": "low",
        "location": "No spatial failure pattern",
        "likely_causes": [],
        "material_link": False,
        "action": "No action — failures (if any) are at the normal background level.",
    },
    "Center": {
        "label": "Center",
        "severity": "high",
        "location": "Wafer centre",
        "likely_causes": [
            "Non-uniform film deposition (thicker/thinner at the centre)",
            "CMP (polishing) pressure or slurry-distribution problems at the centre",
            "Photoresist spin-coating non-uniformity",
        ],
        "material_link": False,
        "action": "Check deposition and CMP uniformity on the tool that processed this lot; "
                  "compare thickness maps against the last good lot.",
    },
    "Donut": {
        "label": "Donut",
        "severity": "high",
        "location": "Ring between centre and edge",
        "likely_causes": [
            "Radial non-uniformity in a deposition or etch step",
            "CMP pad / retaining-ring wear producing a ring-shaped removal profile",
            "Heater-zone imbalance in a thermal step",
        ],
        "material_link": False,
        "action": "Review radial uniformity data for deposition, etch and CMP tools; "
                  "check pad and heater-zone maintenance records.",
    },
    "Edge-Ring": {
        "label": "Edge-Ring",
        "severity": "high",
        "location": "Entire wafer edge",
        "likely_causes": [
            "Etch rate non-uniformity at the wafer edge",
            "Edge-bead removal / photoresist edge problems",
            "Temperature gradient at the edge during thermal processing",
        ],
        "material_link": False,
        "action": "Inspect etch and lithography edge settings (edge exclusion, edge-bead removal); "
                  "check thermal edge uniformity.",
    },
    "Edge-Loc": {
        "label": "Edge-Loc",
        "severity": "medium",
        "location": "One section of the wafer edge",
        "likely_causes": [
            "Wafer handling or clamping damage at one point on the edge",
            "Localised edge contamination",
            "Chuck / clamp-ring misalignment",
        ],
        "material_link": False,
        "action": "Check wafer handling robots, clamps and chuck alignment on the tools in this lot's route.",
    },
    "Loc": {
        "label": "Loc",
        "severity": "medium",
        "location": "A localised cluster away from the edge",
        "likely_causes": [
            "Particle or contamination event at one spot",
            "Localised process excursion (e.g. a hot spot)",
            "Defect on the photomask / reticle (repeats at the same spot across wafers)",
        ],
        "material_link": False,
        "action": "Run a particle check on the suspect tools; if the cluster repeats at the same "
                  "position across wafers, inspect the reticle.",
    },
    "Random": {
        "label": "Random",
        "severity": "medium",
        "location": "Spread across the whole wafer with no structure",
        "likely_causes": [
            "Airborne or chemical particle contamination",
            "Contaminated process chemicals or gases",
            "Incoming substrate (raw wafer) quality",
        ],
        "material_link": True,
        "action": "Check cleanroom particle counts and chemical/gas lot changes; "
                  "if several tools show it on the same material lot, involve the supplier.",
    },
    "Scratch": {
        "label": "Scratch",
        "severity": "high",
        "location": "A line of failures across the wafer",
        "likely_causes": [
            "Mechanical scratch during wafer transfer (robot arm, cassette)",
            "CMP scratch from slurry agglomerates or pad debris",
        ],
        "material_link": False,
        "action": "Inspect wafer transfer robots and cassettes; check CMP slurry filtration and pad condition.",
    },
    "Near-full": {
        "label": "Near-full",
        "severity": "critical",
        "location": "Almost the entire wafer",
        "likely_causes": [
            "Gross process failure: wrong recipe, skipped or repeated step",
            "Major equipment malfunction during processing",
            "Defective incoming substrate or material",
        ],
        "material_link": True,
        "action": "Quarantine the lot. Verify the recipe and step history for this wafer and "
                  "hold the tool until it is checked.",
    },
}

# Patterns that are systematic (same tool hits every wafer the same way) — a
# repeat on one tool is a strong signal; vs. scattered patterns.
SYSTEMATIC = {"Center", "Donut", "Edge-Ring", "Edge-Loc", "Scratch", "Near-full"}


def slug(pattern: str) -> str:
    """'Edge-Ring' → 'edge_ring', 'none' → 'no_defect' (value stored in state)."""
    return "no_defect" if pattern == "none" else pattern.lower().replace("-", "_")


def from_slug(defect_type: str) -> str:
    for p in KNOWLEDGE:
        if slug(p) == defect_type:
            return p
    return defect_type
