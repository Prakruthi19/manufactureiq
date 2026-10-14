"""
ManufactureIQ — Download the two real semiconductor datasets
=============================================================
1. WM-811K (wafer maps)  → Defect Classifier
   811,457 wafer maps from 46,393 production lots in a real fab
   (Wu, Jang & Chen, IEEE Trans. Semiconductor Manufacturing, 2015).
   172,950 of them were labelled by domain experts into 9 classes.

   Source used here: a sharded mirror on Hugging Face
   (datasets/khaubeil/wm811k-demo-data), because the original MIR-Lab link is
   dead and Kaggle needs an account. We keep ONLY the raw fields — wafer
   size, number of valid die, the (x, y) of every failing die, and the expert
   label. The mirror's pre-computed features are ignored: every feature the
   model uses is computed by core/wafer_features.py so it can be explained.
   The class counts are checked against the published distribution below.

2. SECOM (process sensors) → Sensor Parser + Trend Detector
   1,567 production runs from a real semiconductor line, 590 sensor signals
   each, pass/fail at in-house line testing, with timestamps (UCI ML repo).

Run once:
    python scripts/download_data.py

Output (git-ignored, re-creatable):
    data/raw/secom/secom.data, secom_labels.data
    data/raw/wm811k/shard-XXXX.npz   (labelled wafers only, compact)

Memory: shards are processed one at a time (~1 MB each), so this runs fine on a
machine with little free RAM.
"""

import gzip
import json
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BASE_DIR  = Path(__file__).resolve().parent.parent
RAW_DIR   = BASE_DIR / "data" / "raw"
WM_DIR    = RAW_DIR / "wm811k"
SECOM_DIR = RAW_DIR / "secom"

SECOM_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
WM_URL    = "https://huggingface.co/datasets/khaubeil/wm811k-demo-data/resolve/main/analysis/"
N_SHARDS  = 2048

# Mirror's label ids → our class names (WM-811K's 8 patterns + "none").
WM_CLASSES = {
    0: "Center", 1: "Donut", 2: "Edge-Ring", 3: "Edge-Loc", 4: "Loc",
    5: "Near-full", 6: "none", 7: "Random", 8: "Scratch",
}

# Published labelled-class counts for WM-811K (Wu et al. 2015 / LSWMD.pkl).
# If the mirror were truncated or relabelled, these would not match.
PUBLISHED_COUNTS = {
    "none": 147_431, "Edge-Ring": 9_680, "Edge-Loc": 5_189, "Center": 4_294,
    "Loc": 3_593, "Scratch": 1_193, "Random": 866, "Donut": 555, "Near-full": 149,
}


def _get(url: str, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
    return b""


# ── SECOM ─────────────────────────────────────────────────────────────────────
def download_secom() -> None:
    SECOM_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("secom.data", "secom_labels.data", "secom.names"):
        out = SECOM_DIR / name
        if out.exists():
            continue
        out.write_bytes(_get(SECOM_URL + name))
    n = sum(1 for _ in open(SECOM_DIR / "secom.data"))
    print(f"SECOM: {n} runs → {SECOM_DIR}")


# ── WM-811K ───────────────────────────────────────────────────────────────────
def _fetch_shard(i: int) -> Counter:
    """Download one shard, keep labelled wafers only, save compact .npz."""
    out = WM_DIR / f"shard-{i:04d}.npz"
    if out.exists():
        with np.load(out) as z:
            return Counter(WM_CLASSES[int(l)] for l in z["label"])

    records = json.loads(gzip.decompress(_get(f"{WM_URL}shard-{i}.json.gz")))
    ids, labels, hs, ws, n_valid, offsets, points = [], [], [], [], [], [0], []
    for r in records.values():
        if r.get("ground_truth") is None:
            continue                              # unlabelled — not usable
        xy = np.asarray(r["mask"]["fail_xy"], dtype=np.int16).reshape(-1, 2)
        ids.append(r["sample_id"])
        labels.append(r["ground_truth"])
        hs.append(r["shape"]["h"])
        ws.append(r["shape"]["w"])
        n_valid.append(r["n_valid"])
        points.append(xy)
        offsets.append(offsets[-1] + len(xy))

    np.savez_compressed(
        out,
        sample_id=np.asarray(ids, dtype=np.int32),
        label=np.asarray(labels, dtype=np.int8),
        h=np.asarray(hs, dtype=np.int16),
        w=np.asarray(ws, dtype=np.int16),
        n_valid=np.asarray(n_valid, dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.int64),
        fail_xy=np.concatenate(points) if points else np.zeros((0, 2), np.int16),
    )
    return Counter(WM_CLASSES[l] for l in labels)


def download_wm811k(workers: int = 8) -> None:
    WM_DIR.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(_fetch_shard, i) for i in range(N_SHARDS)]
        for f in as_completed(futures):
            counts += f.result()
            done += 1
            if done % 128 == 0 or done == N_SHARDS:
                print(f"WM-811K: {done}/{N_SHARDS} shards", flush=True)

    print(f"\nWM-811K labelled wafers: {sum(counts.values()):,}")
    print(f"  {'class':<10} {'ours':>8} {'published':>10}")
    ok = True
    for cls, pub in PUBLISHED_COUNTS.items():
        mark = "" if counts[cls] == pub else "  ← MISMATCH"
        ok &= counts[cls] == pub
        print(f"  {cls:<10} {counts[cls]:>8,} {pub:>10,}{mark}")
    print("  ✓ matches the published distribution" if ok else
          "  ✗ does NOT match the published distribution — investigate before training")


if __name__ == "__main__":
    download_secom()
    download_wm811k()
    sys.exit(0)
