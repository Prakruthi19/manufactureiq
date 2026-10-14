"""
ManufactureIQ — Build the WM-811K feature table
=================================================
Reads data/raw/wm811k/shard-*.npz (from scripts/download_data.py), computes the
26 wafer-map features for every labelled wafer with core/wafer_features.py, and
saves one compact table:

    data/processed/wm811k_features.npz
        X          (172,950 × 26) float32
        y          class name per wafer
        sample_id  row index in the original LSWMD file
        group      lot-block id used to split train/test (see below)
        h, w       wafer-map size (needed only to redraw demo wafers)

    data/processed/wm811k_maps48.npz   — input for the CNN comparison
        bits       failing-die map per wafer, resized to 48×48, bit-packed
                   (288 bytes per wafer, same row order as the feature table)

Why 48×48, and why only the failure map
    Wafer grids range from ~10×10 to ~200×200 depending on the product. A CNN
    needs one fixed size, and resizing to a common size also hides the product
    size from the CNN — the same shortcut the features avoid. Every wafer then
    sits in the same 48×48 circle, so the "is there a die here" channel is
    identical for all wafers and carries no information; only failures are kept.
    A cell is marked failing if ANY die inside it failed (max-pooling), so thin
    scratches survive downsizing of big wafers.

Lot-aware grouping
    Wafers from the same lot went through the same tools at the same time, so
    they look alike. If lot-mates land on both sides of a random train/test
    split, the test score is inflated. The original file is ordered lot by lot
    (≤25 wafers each), so blocks of 1,000 consecutive rows (~40 lots) are used
    as split groups — at most one lot per block boundary can straddle a split.

Runtime: ~15-20 min on a laptop CPU, one shard in memory at a time.
"""

import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from core.wafer_features import FEATURE_NAMES, features_from_points  # noqa: E402
from scripts.download_data import WM_CLASSES, WM_DIR                  # noqa: E402

OUT        = BASE_DIR / "data" / "processed" / "wm811k_features.npz"
MAPS_OUT   = BASE_DIR / "data" / "processed" / "wm811k_maps48.npz"
BLOCK_SIZE = 1000
MAP_SIZE   = 48


def failure_bitmap(fail_xy: np.ndarray, h: int, w: int, size: int = MAP_SIZE) -> np.ndarray:
    """size×size bool map: True where any die in that cell failed."""
    out = np.zeros((size, size), dtype=bool)
    if len(fail_xy):
        xs = np.minimum(fail_xy[:, 0].astype(int) * size // w, size - 1)
        ys = np.minimum(fail_xy[:, 1].astype(int) * size // h, size - 1)
        out[ys, xs] = True
    return out


def main():
    shards = sorted(WM_DIR.glob("shard-*.npz"))
    if len(shards) != 2048:
        sys.exit(f"Expected 2048 shards in {WM_DIR}, found {len(shards)} — run scripts/download_data.py")

    X, y, sid, hs, ws, maps = [], [], [], [], [], []
    t0 = time.time()
    for k, path in enumerate(shards, 1):
        with np.load(path) as z:
            off = z["offsets"]
            for i in range(len(z["label"])):
                pts = z["fail_xy"][off[i]:off[i + 1]]
                X.append(features_from_points(pts, int(z["h"][i]), int(z["w"][i]), int(z["n_valid"][i])))
                maps.append(np.packbits(failure_bitmap(pts, int(z["h"][i]), int(z["w"][i]))))
            y += [WM_CLASSES[int(l)] for l in z["label"]]
            sid += z["sample_id"].tolist()
            hs += z["h"].tolist()
            ws += z["w"].tolist()
        if k % 128 == 0:
            el = time.time() - t0
            print(f"  {k}/2048 shards, {len(y):,} wafers, {el/60:.1f} min "
                  f"(~{el/k*(2048-k)/60:.0f} min left)", flush=True)

    order = np.argsort(sid)
    sid   = np.asarray(sid)[order]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        X=np.vstack(X)[order].astype(np.float32),
        y=np.asarray(y)[order],
        sample_id=sid,
        group=(sid // BLOCK_SIZE).astype(np.int32),
        h=np.asarray(hs, dtype=np.int16)[order],
        w=np.asarray(ws, dtype=np.int16)[order],
        feature_names=np.asarray(FEATURE_NAMES),
    )
    print(f"Saved {OUT}  ({len(sid):,} wafers × {len(FEATURE_NAMES)} features)")
    np.savez_compressed(MAPS_OUT, bits=np.vstack(maps)[order], size=MAP_SIZE, sample_id=sid)
    print(f"Saved {MAPS_OUT}  ({len(sid):,} wafers × {MAP_SIZE}×{MAP_SIZE} failure maps)")


if __name__ == "__main__":
    main()
