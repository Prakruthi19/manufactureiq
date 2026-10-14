"""
ManufactureIQ — Wafer-map feature engineering
==============================================
Turns a wafer map into a fixed-length vector of features, each with a physical
meaning. Used by BOTH scripts/train_models.py and the Defect Classifier agent,
so training and inference can never compute features differently
(train/serve skew).

A wafer map (WM-811K format) is a 2-D grid:
    0 = no die here (outside the wafer)   1 = die passed   2 = die failed

Why hand-built features and not a CNN?
    The WM-811K paper (Wu et al. 2015) used density + geometry features.
    They train in minutes on a CPU, and every prediction can be explained
    ("72% of failures sit in the outer ring, spread evenly around it" →
    Edge-Ring). scripts/train_wafer_cnn.py trains a CNN on the raw maps as a
    comparison, on the same lot-aware split.

Feature groups (26 total):
    density   — overall failure rate
    radial    — failure density in 6 concentric rings  → Center / Donut / Edge-Ring
    angular   — how unevenly failures spread around the edge → Edge-Loc vs Edge-Ring
    cluster   — connected groups of failing die        → Loc / Scratch / Random
    line      — do failures line up along one direction → Scratch
Wafer size / die count is deliberately NOT a feature: it identifies the product,
and a model could learn "product X usually has pattern Y" instead of the pattern.
Every feature is therefore a ratio or a 0-1 position. The raw failure COUNT is
excluded for the same reason: count ÷ failure rate = die count, so the two
together would hand the model the product size by the back door.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage

RING_EDGES   = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, np.inf]   # normalised radius
N_SECTORS    = 8                                        # 45° each
LINE_ANGLES  = np.deg2rad(np.arange(0, 180, 10))        # directions tried for "line"
EIGHT_CONN   = np.ones((3, 3), dtype=int)               # diagonal neighbours count

FEATURE_NAMES = (
    ["fail_density"]
    + [f"ring{i}_density" for i in range(len(RING_EDGES) - 1)]
    + ["center_vs_wafer", "edge_vs_wafer", "mid_vs_center_edge",
       "edge_sector_max", "edge_sector_cv", "edge_sector_entropy", "edge_sectors_hit",
       "n_clusters", "largest_cluster_frac", "largest_cluster_area",
       "largest_cluster_radius", "largest_cluster_elongation", "largest_cluster_fill",
       "isolated_frac", "clustered_frac",
       "line_score", "line_score_largest",
       "mean_fail_radius", "std_fail_radius"]
)


def _radius(x: np.ndarray, y: np.ndarray, h: int, w: int) -> np.ndarray:
    """Normalised distance from wafer centre: 0 = centre, 1 = wafer edge."""
    cx, cy = (w - 1) / 2, (h - 1) / 2
    return np.sqrt(((x - cx) / (w / 2)) ** 2 + ((y - cy) / (h / 2)) ** 2)


def _angle(x: np.ndarray, y: np.ndarray, h: int, w: int) -> np.ndarray:
    cx, cy = (w - 1) / 2, (h - 1) / 2
    return np.arctan2(y - cy, x - cx)


@lru_cache(maxsize=256)
def _die_layout(h: int, w: int):
    """Radius/angle of every die position for a wafer size (cached: sizes repeat)."""
    gy, gx   = np.mgrid[0:h, 0:w]
    gr       = _radius(gx, gy, h, w)
    on_wafer = gr <= 1.0
    die_r    = gr[on_wafer]
    die_a    = _angle(gx, gy, h, w)[on_wafer]
    ring_die = np.array([((die_r >= lo) & (die_r < hi)).sum()
                         for lo, hi in zip(RING_EDGES[:-1], RING_EDGES[1:])])
    edge_die = die_r >= 0.6
    sec_die  = np.bincount(((die_a[edge_die] + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS,
                           minlength=N_SECTORS)
    return ring_die, sec_die


def _line_score(x: np.ndarray, y: np.ndarray) -> float:
    """
    Largest share of points lying in one 2-die-wide band, over 18 directions.
    A scratch puts many failures on one line → high. Random noise → low.
    (A cheap stand-in for the Radon transform used in the original paper.)
    """
    n = len(x)
    if n < 5:
        return 0.0
    best = 0
    for t in LINE_ANGLES:
        proj = x * np.cos(t) + y * np.sin(t)
        bins = np.floor((proj - proj.min()) / 2.0).astype(int)
        best = max(best, np.bincount(bins).max())
    return best / n


def features_from_points(fail_xy: np.ndarray, h: int, w: int, n_valid: int) -> np.ndarray:
    """
    fail_xy : (N, 2) int array of failing-die (x=column, y=row) positions
    h, w    : wafer-map grid size
    n_valid : number of real die on the wafer (pass + fail)
    """
    fail_xy = np.asarray(fail_xy, dtype=float).reshape(-1, 2)
    fx, fy  = fail_xy[:, 0], fail_xy[:, 1]
    n_fail  = len(fail_xy)
    eps     = 1e-9

    # Die positions: the wafer is approximated as the ellipse inscribed in the grid
    # (the raw data gives failing die exactly but not every passing die's position).
    ring_die, sec_d = _die_layout(int(h), int(w))

    fr = _radius(fx, fy, h, w)
    fa = _angle(fx, fy, h, w)

    # ── density ──────────────────────────────────────────────────────────────
    density = n_fail / max(n_valid, 1)
    feats = [density]

    # ── radial: failure density per ring ─────────────────────────────────────
    ring_d = []
    for (lo, hi), die_in in zip(zip(RING_EDGES[:-1], RING_EDGES[1:]), ring_die):
        fail_in = ((fr >= lo) & (fr < hi)).sum()
        ring_d.append(fail_in / die_in if die_in else 0.0)
    feats += ring_d
    centre, edge = ring_d[0], max(ring_d[4], ring_d[5])
    mid          = np.mean(ring_d[1:4])
    feats += [
        centre / (density + eps),               # Center: centre much worse than average
        edge / (density + eps),                 # Edge-*: edge much worse than average
        mid / (max(centre, edge) + eps),        # Donut: middle worse than centre AND edge
    ]

    # ── angular: are edge failures in one place or all the way round? ────────
    edge_fail = fr >= 0.6
    sec_f = np.bincount(((fa[edge_fail] + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS,
                        minlength=N_SECTORS)
    sec_density = sec_f / np.maximum(sec_d, 1)
    if sec_f.sum():
        p = sec_f / sec_f.sum()
        entropy = -(p[p > 0] * np.log(p[p > 0])).sum() / np.log(N_SECTORS)   # 1 = even
        cv = sec_density.std() / (sec_density.mean() + eps)
    else:
        entropy, cv = 0.0, 0.0
    feats += [sec_density.max(), cv, entropy, (sec_f > 0).sum() / N_SECTORS]

    # ── clusters: connected groups of failing die ────────────────────────────
    grid = np.zeros((h, w), dtype=bool)
    if n_fail:
        grid[np.clip(fy.astype(int), 0, h - 1), np.clip(fx.astype(int), 0, w - 1)] = True
    labels, n_clusters = ndimage.label(grid, structure=EIGHT_CONN)
    if n_clusters:
        sizes   = np.bincount(labels.ravel())[1:]
        big     = int(sizes.argmax()) + 1
        ly, lx  = np.nonzero(labels == big)
        lsize   = len(lx)
        l_r     = _radius(lx, ly, h, w).mean()
        if lsize >= 5:                       # shape is meaningless for 1-4 die
            ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([lx, ly]))))
            elong = np.sqrt((ev[1] + eps) / (ev[0] + eps))      # long thin → large
            # fill: share of the cluster's covering ellipse actually filled
            ell_area = np.pi * 2 * np.sqrt(max(ev[0], 0.25)) * 2 * np.sqrt(max(ev[1], 0.25))
            fill = min(lsize / ell_area, 1.0)
        else:
            elong, fill = 1.0, 1.0
        isolated  = sizes[sizes <= 2].sum() / n_fail
        clustered = sizes[sizes >= 10].sum() / n_fail
        line_big  = _line_score(lx.astype(float), ly.astype(float))
        feats += [n_clusters / max(n_valid, 1) * 100, lsize / n_fail, lsize / max(n_valid, 1),
                  l_r, min(elong, 50.0), fill, isolated, clustered]
    else:
        line_big = 0.0
        feats += [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    # ── lines (scratches) ────────────────────────────────────────────────────
    feats += [_line_score(fx, fy), line_big]

    # ── where failures sit, on average ───────────────────────────────────────
    feats += [fr.mean() if n_fail else 0.0, fr.std() if n_fail else 0.0]

    out = np.asarray(feats, dtype=np.float32)
    assert len(out) == len(FEATURE_NAMES)
    return out


def features_from_map(wafer_map) -> np.ndarray:
    """Same features from a WM-811K style grid (0 = no die, 1 = pass, 2 = fail)."""
    grid = np.asarray(wafer_map)
    h, w = grid.shape
    ys, xs = np.nonzero(grid == 2)
    return features_from_points(np.column_stack([xs, ys]), h, w, int((grid >= 1).sum()))


def map_from_points(fail_xy: np.ndarray, h: int, w: int) -> list[list[int]]:
    """
    Rebuild a 0/1/2 grid for display / demo payloads. Die positions use the same
    inscribed-ellipse approximation as the features; failing die are exact.
    """
    gy, gx = np.mgrid[0:h, 0:w]
    grid = (_radius(gx, gy, h, w) <= 1.0).astype(np.int8)
    fail_xy = np.asarray(fail_xy, dtype=int).reshape(-1, 2)
    grid[fail_xy[:, 1], fail_xy[:, 0]] = 2
    return grid.tolist()


def describe(features: np.ndarray) -> dict:
    """Human-readable summary of the strongest signals (for the LLM + UI)."""
    f = dict(zip(FEATURE_NAMES, map(float, features)))
    return {
        "failure_rate_pct":        round(f["fail_density"] * 100, 2),
        "centre_ring_fail_pct":    round(f["ring0_density"] * 100, 2),
        "outer_ring_fail_pct":     round(max(f["ring4_density"], f["ring5_density"]) * 100, 2),
        "mid_rings_fail_pct":      round(np.mean([f["ring1_density"], f["ring2_density"], f["ring3_density"]]) * 100, 2),
        "edge_spread_evenness":    round(f["edge_sector_entropy"], 2),
        "largest_cluster_share":   round(f["largest_cluster_frac"], 2),
        "largest_cluster_elongation": round(f["largest_cluster_elongation"], 1),
        "isolated_fail_share":     round(f["isolated_frac"], 2),
        "line_score":              round(f["line_score_largest"], 2),
    }
