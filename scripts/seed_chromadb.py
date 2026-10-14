"""
ManufactureIQ — Build the similar-wafer index (ChromaDB)
=========================================================
Run after scripts/train_wafer_classifier.py:
    python scripts/seed_chromadb.py

Stores historical wafers as vectors so the Root Cause agent can ask
"which past wafers looked most like this one, and what did experts call them?"

    vector   = the 26 wafer features, standardised (so no feature dominates
               just because of its units)
    metadata = expert label, lot-block, original row id

What goes in:
    • every DEFECTIVE wafer from the training/validation lots (~20k)
    • 20,000 randomly sampled "none" wafers from the same lots
      (all 117k would mostly add more of the same and slow the index)
What stays out:
    • every wafer from the TEST lots — demo wafers come from there, and a
      wafer must never find itself or its lot-mates as "history".

No text-embedding model is involved, which also removed the ~4-minute model
load the old text-based index needed on first use.
"""

import sys
from pathlib import Path

import chromadb
import numpy as np

BASE_DIR   = Path(__file__).resolve().parent.parent
FEATURES   = BASE_DIR / "data" / "processed" / "wm811k_features.npz"
TEST_IDX   = BASE_DIR / "models" / "wafer_test_idx.npy"
SCALER_OUT = BASE_DIR / "models" / "wafer_index_scaler.npz"
CHROMA_DIR = BASE_DIR / "data" / "chroma"
N_NONE     = 20_000
SEED       = 42


def main():
    if not TEST_IDX.exists():
        sys.exit("Run scripts/train_wafer_classifier.py first (it defines the test lots).")
    d = np.load(FEATURES)
    X, y, sid, grp = d["X"], d["y"], d["sample_id"], d["group"]

    dev = np.ones(len(y), bool)
    dev[np.load(TEST_IDX)] = False
    rng = np.random.default_rng(SEED)
    none_dev = np.flatnonzero(dev & (y == "none"))
    keep = np.concatenate([np.flatnonzero(dev & (y != "none")),
                           rng.choice(none_dev, size=min(N_NONE, len(none_dev)), replace=False)])
    print(f"Indexing {len(keep):,} training-lot wafers "
          f"({(y[keep] != 'none').sum():,} defective, {(y[keep] == 'none').sum():,} none); "
          f"{(~dev).sum():,} test-lot wafers excluded")

    mean, std = X[dev].mean(0), X[dev].std(0) + 1e-9
    np.savez(SCALER_OUT, mean=mean, std=std)
    Z = (X[keep] - mean) / std

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    for old in ("defect_history", "wafer_history"):     # defect_history = old AI4I index
        try:
            client.delete_collection(old)
        except Exception:
            pass
    col = client.create_collection("wafer_history", embedding_function=None,
                                   metadata={"hnsw:space": "l2"})
    B = 2000
    for i in range(0, len(keep), B):
        k = keep[i:i + B]
        col.add(
            ids=[f"WM{s:06d}" for s in sid[k]],
            embeddings=Z[i:i + B].tolist(),
            metadatas=[{"pattern": str(p), "wafer": f"WM{s:06d}", "lot_block": int(g)}
                       for p, s, g in zip(y[k], sid[k], grp[k])],
        )
        print(f"  {min(i + B, len(keep)):,}/{len(keep):,}", end="\r", flush=True)
    print(f"\nwafer_history: {col.count():,} vectors → {CHROMA_DIR}")

    # Sanity check: a training wafer's nearest neighbour should be itself.
    probe = keep[0]
    res = col.query(query_embeddings=[((X[probe] - mean) / std).tolist()], n_results=3)
    print(f"Self-check: WM{sid[probe]:06d} ({y[probe]}) → nearest {res['ids'][0]}")


if __name__ == "__main__":
    main()
