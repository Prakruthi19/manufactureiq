"""
ManufactureIQ — Build demo inspections from HELD-OUT data
==========================================================
Run after training:
    python scripts/build_demo_samples.py
Output: data/processed/demo_samples.json (served by GET /api/samples)

Everything the demo shows is data the models never trained on:
    wafer maps      — wafers from the TEST lots of WM-811K (with their true
                      expert label, so the UI can show predicted vs actual)
    process sensors — runs from the TEST period of SECOM (last 30%, by time)

Single inspections: 2 wafers per class, drawn at random with a fixed seed —
not hand-picked, so the demo shows the model's real hit rate, mistakes included.

Batch scenarios: real consecutive wafers from test lots that contain a
recurring pattern. WM-811K and SECOM have no tool or supplier IDs, so those
are SIMULATED assignments (marked `simulated_metadata`), chosen to show
how lot / tool / material pattern detection behaves.

The two datasets come from different fabs: pairing a wafer map with a SECOM
run is a demo convenience, not a real link. Docs say so.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from core.secom import load_secom                       # noqa: E402
from core.wafer_features import map_from_points         # noqa: E402
from scripts.download_data import WM_DIR                # noqa: E402

OUT        = BASE_DIR / "data" / "processed" / "demo_samples.json"
FEATURES   = BASE_DIR / "data" / "processed" / "wm811k_features.npz"
SEED       = 7
PER_CLASS  = 2
LOT_SIZE   = 20
TOOLS      = ["ETCH-01", "ETCH-02", "CMP-01", "CVD-03", "LITHO-02"]
SUPPLIERS  = ["SUP-SILTRONIX", "SUP-WAFERCO"]


def wafer_points(sample_id: int):
    with np.load(WM_DIR / f"shard-{sample_id % 2048:04d}.npz") as z:
        i = int(np.flatnonzero(z["sample_id"] == sample_id)[0])
        return z["fail_xy"][z["offsets"][i]:z["offsets"][i + 1]], int(z["h"][i]), int(z["w"][i])


def make_inspection(sid, lot, tool, supplier, material, ts, sensors, idx):
    pts, h, w = wafer_points(int(sid))
    return {
        "record_id": f"INSP-{sid:06d}-{idx:02d}",
        "wafer_id": f"WM{sid:06d}",
        "batch_id": lot,
        "product_id": f"DEV-{h}x{w}",
        "machine_id": tool,
        "supplier_id": supplier,
        "material_batch_id": material,
        "timestamp": ts.isoformat(timespec="seconds"),
        "raw_notes": "",
        "wafer_map": map_from_points(pts, h, w),
        "process_sensors": [None if np.isnan(v) else round(float(v), 6) for v in sensors],
    }


def main():
    d = np.load(FEATURES)
    y, sid, grp = d["y"], d["sample_id"], d["group"]
    test_idx = np.load(BASE_DIR / "models" / "wafer_test_idx.npy")
    X_secom, y_secom, _ = load_secom()
    secom_test = np.arange(int(len(y_secom) * 0.7), len(y_secom))   # same cut as training
    rng = np.random.default_rng(SEED)
    t0 = datetime(2026, 10, 5, 8, 0)

    # ── Single inspections: random held-out wafers, 2 per class ───────────────
    singles = []
    for cls in sorted(set(y)):
        pool = test_idx[y[test_idx] == cls]
        for j, i in enumerate(rng.choice(pool, size=min(PER_CLASS, len(pool)), replace=False)):
            run = int(rng.choice(secom_test))
            lot = f"LOT-{grp[i]:04d}"
            singles.append({
                "label": f"{d['sample_id'][i]:06d} · {lot}",
                "true_pattern": str(cls),
                "secom_run": {"row": run, "failed_line_test": bool(y_secom[run])},
                "inspection": make_inspection(sid[i], lot, TOOLS[int(grp[i]) % len(TOOLS)],
                                              SUPPLIERS[int(grp[i]) % 2], f"MAT-{grp[i] % 97:03d}",
                                              t0 + timedelta(minutes=len(singles) * 7),
                                              X_secom[run], 0),
            })
    rng.shuffle(singles)     # don't list them grouped by answer

    # ── Batch scenarios: real consecutive test-lot wafers ─────────────────────
    order = test_idx[np.argsort(sid[test_idx])]
    def best_window(pattern, exclude=(), size=LOT_SIZE):
        best, best_n = None, -1
        for s in range(0, len(order) - size):
            w = order[s:s + size]
            if grp[w[0]] != grp[w[-1]] or grp[w[0]] in exclude:
                continue                                    # stay inside one lot-block
            n = int((y[w] == pattern).sum())
            if n > best_n:
                best, best_n = w, n
        return best, best_n

    scenarios = []
    def add(name, story, window, tool, material, supplier, lots=None, tools=None):
        lots  = lots or [f"LOT-{grp[window[0]]:04d}"] * len(window)
        tools = tools or [tool] * len(window)
        runs  = rng.choice(secom_test, size=len(window), replace=False)
        items = []
        for k, (i, run) in enumerate(zip(window, runs)):
            items.append({"true_pattern": str(y[i]),
                          "inspection": make_inspection(sid[i], lots[k], tools[k], supplier,
                                                        material, t0 + timedelta(hours=2, minutes=3 * k),
                                                        X_secom[run], k)})
        scenarios.append({"name": name, "story": story, "lot": " + ".join(dict.fromkeys(lots)),
                          "simulated_metadata": True,
                          "counts": {p: int((y[window] == p).sum()) for p in sorted(set(y[window]))},
                          "items": items})

    w1, _ = best_window("Edge-Ring")
    add("Lot A — recurring Edge-Ring",
        "A real test lot where many wafers show Edge-Ring. Expect a LOT pattern once ≥3 are seen.",
        w1, "ETCH-02", "MAT-A01", SUPPLIERS[0])
    w2, _ = best_window("Edge-Ring", exclude={grp[w1[0]]})
    add("Lot B — same pattern, same tool",
        "A different real lot with Edge-Ring, routed (simulated) through the SAME etch tool as Lot A. "
        "Run after Lot A: expect a TOOL pattern — two lots, one tool.",
        w2, "ETCH-02", "MAT-B02", SUPPLIERS[0])
    # Scenario C: two DIFFERENT lots on two DIFFERENT tools, one shared material
    # lot. Lot and tool both vary, so the material is the only common factor.
    used = {grp[w1[0]], grp[w2[0]]}
    c1, _ = best_window("Random", exclude=used, size=10)
    c2, _ = best_window("Random", exclude=used | {grp[c1[0]]}, size=10)
    add("Lot C — Random defects: two lots, two tools, one material lot",
        "Two different real lots with scattered (Random) failures, run (simulated) on two different "
        "CVD tools but cut from the SAME material lot. Expect a MATERIAL pattern: lot and tool both "
        "vary, only the material is shared.",
        np.concatenate([c1, c2]), None, "MAT-C07", SUPPLIERS[1],
        lots=[f"LOT-{grp[c1[0]]:04d}"] * len(c1) + [f"LOT-{grp[c2[0]]:04d}"] * len(c2),
        tools=["CVD-01"] * len(c1) + ["CVD-02"] * len(c2))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Held-out data only. Tool/supplier/material IDs are simulated; "
                "WM-811K and SECOM come from different fabs and are paired for the demo.",
        "singles": singles, "scenarios": scenarios,
    }))
    print(f"{len(singles)} single inspections, {len(scenarios)} batch scenarios → {OUT} "
          f"({OUT.stat().st_size / 1e6:.1f} MB)")
    for s in scenarios:
        print(f"  {s['name']}: {s['counts']}")


if __name__ == "__main__":
    main()
