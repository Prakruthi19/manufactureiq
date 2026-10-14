"""
ManufactureIQ — Probability calibration
========================================
"90% confident" should mean "right about 90% of the time". Out of the box it
often doesn't: SMOTE and boosting push probabilities towards 0 and 1, so the
model sounds surer than it is. The human-review rule ("send it to an engineer
below X% confidence") is only meaningful if the confidence is honest.

Temperature scaling (Guo et al. 2017)
    One number T, fitted on VALIDATION data (never test):
        calibrated p  ∝  p ** (1 / T)
    T > 1 softens over-confident probabilities, T < 1 sharpens timid ones.
    It never changes which class is on top, so macro-F1, recall and the
    confusion matrix are untouched; only the confidence numbers move.
    One parameter can't overfit 28,000 validation wafers, which is why it's
    preferred to per-class isotonic regression on the rare classes (40 test
    Near-full wafers).

Expected calibration error (ECE)
    Bucket predictions by confidence (0.0-0.1, …, 0.9-1.0). In each bucket,
    compare average confidence with the share actually correct. ECE is the
    average gap, weighted by bucket size. 0 = perfectly honest.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale class probabilities (rows sum to 1) by temperature T."""
    logp = np.log(np.clip(proba, EPS, 1.0)) / temperature
    logp -= logp.max(axis=-1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=-1, keepdims=True)


def fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
    """T minimising negative log-likelihood of the true labels (grid search: robust, no tuning)."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 200))
    nll = [-np.log(np.clip(apply_temperature(proba, t)[np.arange(len(y)), y], EPS, 1.0)).mean()
           for t in grid]
    return float(grid[int(np.argmin(nll))])


def reliability(proba: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    """Confidence-vs-accuracy buckets + ECE for the model's top prediction."""
    conf, pred = proba.max(1), proba.argmax(1)
    correct = pred == y
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        gap = abs(conf[m].mean() - correct[m].mean())
        ece += m.mean() * gap
        bins.append({"lo": round(float(edges[b]), 2), "hi": round(float(edges[b + 1]), 2),
                     "confidence": round(float(conf[m].mean()), 4),
                     "accuracy": round(float(correct[m].mean()), 4), "n": int(m.sum())})
    return {"ece": round(float(ece), 4), "bins": bins}
