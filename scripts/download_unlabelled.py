"""
ManufactureIQ — Download the UNLABELLED WM-811K wafers (for CNN pre-training)
==============================================================================
Of WM-811K's 811,457 wafers only 172,950 were labelled by experts. The other
~638k carry no label but are real wafers from the same fab, so a CNN can learn
what wafer-map patterns look like from them before it ever sees a label
(scripts/pretrain_wafer_cnn.py).

What is kept
    Only the 48×48 failure map per wafer (same resizing as
    build_wafer_features.py, bit-packed, 288 bytes each) and its sample_id.
    Wafers with no failing die are dropped: two views of an empty map are
    identical to every other empty map, so they teach a contrastive model
    nothing.

Output (git-ignored)
    data/raw/wm811k_unlabelled/shard-XXXX.npz      per-shard cache (resumable)
    data/processed/wm811k_unlabelled_maps48.npz    bits, size, sample_id, group

Which wafers pre-training may use is decided later, in pretrain_wafer_cnn.py
(no wafer from a validation or test lot-block), not here.

Downloads ~1.3 GB of compressed JSON. Meant for Colab
(notebooks/wafer_cnn_colab.ipynb); ~5-10 min there.
    python scripts/download_unlabelled.py
"""

import gzip
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from scripts.build_wafer_features import BLOCK_SIZE, MAP_SIZE, failure_bitmap  # noqa: E402
from scripts.download_data import N_SHARDS, RAW_DIR, WM_URL, _get              # noqa: E402

CACHE = RAW_DIR / "wm811k_unlabelled"
OUT   = BASE_DIR / "data" / "processed" / "wm811k_unlabelled_maps48.npz"


def _fetch_shard(i: int) -> Path:
    out = CACHE / f"shard-{i:04d}.npz"
    if out.exists():
        return out
    records = json.loads(gzip.decompress(_get(f"{WM_URL}shard-{i}.json.gz")))
    ids, maps = [], []
    for r in records.values():
        if r.get("ground_truth") is not None:
            continue                              # labelled — already in wm811k_maps48.npz
        xy = np.asarray(r["mask"]["fail_xy"], dtype=np.int16).reshape(-1, 2)
        if not len(xy):
            continue                              # empty map — nothing to learn from
        ids.append(r["sample_id"])
        maps.append(np.packbits(failure_bitmap(xy, r["shape"]["h"], r["shape"]["w"])))
    np.savez_compressed(out, sample_id=np.asarray(ids, dtype=np.int32),
                        bits=np.vstack(maps) if maps else np.zeros((0, MAP_SIZE * MAP_SIZE // 8), np.uint8))
    return out


def main(workers: int = 16) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for f in as_completed([pool.submit(_fetch_shard, i) for i in range(N_SHARDS)]):
            f.result()
            done += 1
            if done % 128 == 0 or done == N_SHARDS:
                print(f"unlabelled: {done}/{N_SHARDS} shards", flush=True)

    sid, bits = [], []
    for path in sorted(CACHE.glob("shard-*.npz")):
        with np.load(path) as z:
            sid.append(z["sample_id"])
            bits.append(z["bits"])
    sid, bits = np.concatenate(sid), np.vstack(bits)
    order = np.argsort(sid)
    sid, bits = sid[order], bits[order]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, bits=bits, size=MAP_SIZE, sample_id=sid,
                        group=(sid // BLOCK_SIZE).astype(np.int32))
    print(f"Saved {OUT}  ({len(sid):,} unlabelled wafers with ≥1 failing die, {MAP_SIZE}×{MAP_SIZE})")


if __name__ == "__main__":
    main()
