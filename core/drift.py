"""
ManufactureIQ — Input drift monitor
====================================
A model is only as good as the match between today's wafers and the wafers it
was trained on. A new product, a new tester or a process change shifts the
inputs, and the test-set metrics silently stop applying. Nobody gets an error.

What this does
    • At training time, each of the 26 features is cut into 10 buckets at the
      training set's deciles, and the share of training wafers per bucket is
      saved (models/drift_reference.json).
    • Every wafer the live classifier sees is logged (features only).
    • The recent wafers are compared with the reference per feature using the
      Population Stability Index:
          PSI = Σ (live% − train%) · ln(live% / train%)
      Industry rule of thumb (credit-risk origin, widely used in MLOps):
          < 0.10 stable   0.10–0.25 moderate shift   > 0.25 major shift

Why PSI and not a KS test
    With hundreds of wafers a KS test flags tiny, harmless differences as
    "significant". PSI measures how big the shift is, which is the question.

It measures the INPUTS, so it needs no labels — drift shows up the day it
starts, long before anyone could measure accuracy on the new wafers.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT       = Path(__file__).resolve().parent.parent
REFERENCE  = ROOT / "models" / "drift_reference.json"
DB_PATH    = Path(os.getenv("HISTORY_DB") or ROOT / "data" / "history.sqlite")
N_BINS     = 10
MIN_WAFERS = 50          # fewer than this → too noisy to judge
WINDOW     = 500         # compare the most recent N wafers
STABLE, MAJOR = 0.10, 0.25
EPS        = 1e-4        # floor for empty buckets (ln(0) is undefined)


# ── reference (training time) ─────────────────────────────────────────────────
def build_reference(X: np.ndarray, names: list[str]) -> dict:
    """Per feature: inner bucket edges at the training deciles + training share per bucket."""
    ref = {}
    for j, name in enumerate(names):
        edges = np.unique(np.quantile(X[:, j], np.linspace(0, 1, N_BINS + 1)[1:-1]))
        share = np.bincount(np.searchsorted(edges, X[:, j], side="right"),
                            minlength=len(edges) + 1) / len(X)
        ref[name] = {"edges": edges.round(6).tolist(), "share": share.round(6).tolist()}
    return ref


def psi(ref: dict, values: np.ndarray) -> float:
    edges, expected = np.asarray(ref["edges"]), np.asarray(ref["share"])
    actual = np.bincount(np.searchsorted(edges, values, side="right"),
                         minlength=len(expected)) / max(len(values), 1)
    e, a = np.clip(expected, EPS, None), np.clip(actual, EPS, None)
    return float(((a - e) * np.log(a / e)).sum())


def compare(reference: dict, X: np.ndarray, names: list[str]) -> dict:
    """PSI per feature, worst first, with an overall status."""
    scores = sorted(((n, round(psi(reference[n], X[:, j]), 4)) for j, n in enumerate(names)),
                    key=lambda t: -t[1])
    worst = scores[0][1] if scores else 0.0
    status = "stable" if worst < STABLE else "moderate" if worst < MAJOR else "major"
    return {"status": status, "max_psi": worst,
            "n_moderate": sum(STABLE <= s < MAJOR for _, s in scores),
            "n_major": sum(s >= MAJOR for _, s in scores),
            "features": [{"feature": n, "psi": s} for n, s in scores]}


# ── live log (serving time) ───────────────────────────────────────────────────
def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.execute("CREATE TABLE IF NOT EXISTS feature_log (inspected_at TEXT NOT NULL, features TEXT NOT NULL)")
    return c


def log(features: np.ndarray) -> None:
    with closing(_conn()) as c, c:
        c.execute("INSERT INTO feature_log VALUES (?, ?)",
                  (datetime.utcnow().isoformat(), json.dumps([round(float(v), 6) for v in features])))


def clear() -> None:
    with closing(_conn()) as c, c:
        c.execute("DELETE FROM feature_log")


def live_report(names: list[str], window: int = WINDOW) -> dict:
    """Drift of the most recent `window` live wafers against the training reference."""
    if not REFERENCE.exists():
        return {"status": "unavailable", "detail": "models/drift_reference.json missing — retrain"}
    with closing(_conn()) as c:
        rows = c.execute("SELECT features FROM feature_log ORDER BY inspected_at DESC LIMIT ?",
                         (window,)).fetchall()
    out = {"n_wafers": len(rows), "window": window, "min_wafers": MIN_WAFERS,
           "thresholds": {"stable_below": STABLE, "major_above": MAJOR}}
    if len(rows) < MIN_WAFERS:
        return {**out, "status": "not_enough_data"}
    X = np.array([json.loads(r[0]) for r in rows], dtype=float)
    return {**out, **compare(json.loads(REFERENCE.read_text()), X, names)}
