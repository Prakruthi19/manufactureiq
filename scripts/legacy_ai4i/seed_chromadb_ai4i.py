"""
ManufactureIQ — Seed ChromaDB with Defect History
===================================================
Run ONCE before starting the pipeline:
    python scripts/seed_chromadb.py

Loads failure records from ai4i2020.csv into ChromaDB
so the Root Cause Analyzer can retrieve similar past defects.
"""

import os
from pathlib import Path

# ── Redirect ChromaDB's embedding-model download OFF the C: drive ─────────────
# ChromaDB's built-in embedder downloads all-MiniLM-L6-v2 to Path.home()/.cache.
# Path.home() reads USERPROFILE/HOME, and chromadb resolves that download path at
# IMPORT TIME — so these MUST be set BEFORE `import chromadb`.
_CACHE_HOME = Path(r"D:\manufactureiq\.cache_home")
_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ["USERPROFILE"] = str(_CACHE_HOME)
os.environ["HOME"]        = str(_CACHE_HOME)

import pandas as pd
import chromadb

BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "data" / "raw" / "ai4i2020.csv"
CHROMA_DIR = BASE_DIR / "data" / "chroma"   # must match root_cause_agent.py read path
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


DEFECT_LABELS = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}

def get_defect_type(row) -> str:
    for code in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        if row.get(code, 0): return code
    return "none"

def build_document(row) -> str:
    """Convert a dataset row into readable text for embedding."""
    import math
    defect_code = get_defect_type(row)
    defect_name = DEFECT_LABELS.get(defect_code, "No Failure")
    power       = round(row["Torque [Nm]"] * row["Rotational speed [rpm]"] * (2 * math.pi / 60), 1)
    temp_diff   = round(row["Process temperature [K]"] - row["Air temperature [K]"], 2)
    wear_torque = round(row["Tool wear [min]"] * row["Torque [Nm]"], 1)

    return (
        f"Defect type: {defect_name} ({defect_code}). "
        f"Product type: {row['Type']}. "
        f"Air temperature: {row['Air temperature [K]']}K. "
        f"Process temperature: {row['Process temperature [K]']}K. "
        f"Temperature differential: {temp_diff}K. "
        f"Rotational speed: {row['Rotational speed [rpm]']} RPM. "
        f"Torque: {row['Torque [Nm]']} Nm. "
        f"Tool wear: {row['Tool wear [min]']} minutes. "
        f"Power: {power} W. "
        f"Wear-torque product: {wear_torque}. "
        f"Machine failure: {'Yes' if row['Machine failure'] else 'No'}."
    )

def build_metadata(row, doc_id: str) -> dict:
    return {
        "record_id":       doc_id,
        "product_type":    str(row["Type"]),
        "defect_type":     get_defect_type(row),
        "machine_failure": int(row["Machine failure"]),
        "tool_wear":       int(row["Tool wear [min]"]),
        "rpm":             int(row["Rotational speed [rpm]"]),
        "torque":          float(row["Torque [Nm]"]),
    }

print("=" * 55)
print("ManufactureIQ — Seeding ChromaDB")
print("=" * 55)

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n[1/4] Loading dataset...")
df       = pd.read_csv(DATA_PATH)
failures = df[df["Machine failure"] == 1].copy()
normals  = df[df["Machine failure"] == 0].sample(200, random_state=42)
combined = pd.concat([failures, normals]).reset_index(drop=True)
print(f"      Failure records: {len(failures):,}")
print(f"      Normal records:  200 (for contrast)")
print(f"      Total to embed:  {len(combined):,}")

# ── Init ChromaDB ─────────────────────────────────────────────────────────────
print("\n[2/4] Initializing ChromaDB...")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

try:
    client.delete_collection("defect_history")
    print("      Deleted existing collection")
except Exception:
    pass

# Uses ChromaDB's built-in embedding function (all-MiniLM-L6-v2 via ONNX)
collection = client.create_collection(
    name="defect_history",
    metadata={"hnsw:space": "cosine"}
)
print("      Collection created: defect_history")

# ── Build documents ───────────────────────────────────────────────────────────
print("\n[3/4] Building documents...")
documents, metadatas, ids = [], [], []

for idx, (_, row) in enumerate(combined.iterrows()):
    prefix  = "F" if row["Machine failure"] else "N"
    doc_id  = f"{prefix}-{idx:05d}"
    documents.append(build_document(row))
    metadatas.append(build_metadata(row, doc_id))
    ids.append(doc_id)

print(f"      Built {len(documents):,} documents")

# ── Load in batches ───────────────────────────────────────────────────────────
print("\n[4/4] Loading into ChromaDB...")
BATCH = 100
for i in range(0, len(documents), BATCH):
    collection.upsert(
        documents=documents[i:i+BATCH],
        metadatas=metadatas[i:i+BATCH],
        ids=ids[i:i+BATCH],
    )
    print(f"      {min(i+BATCH, len(documents))}/{len(documents)} loaded", end="\r")

print(f"\n      Total in ChromaDB: {collection.count()}")
# ── Test query ────────────────────────────────────────────────────────────────
print("\nTest query: 'power failure high rpm torque'")
results = collection.query(
    query_texts=["power failure high rotational speed high torque"],
    n_results=3,
    where={"machine_failure": 1}
)
for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"  [{i+1}] {meta['defect_type']} | RPM: {meta['rpm']} | Torque: {meta['torque']}")
    print(f"       {doc[:100]}...")

print("\n" + "=" * 55)
print(f"Done. ChromaDB at: {CHROMA_DIR}")
print("=" * 55)