"""
ManufactureIQ — SECOM process-sensor helpers
=============================================
SECOM = 590 anonymised sensor signals per production run from a real
semiconductor line (UCI ML repository). This module holds the pieces that
must be identical at training and inference time:

    load_secom()        raw data + labels + timestamps, in time order
    SensorCleaner       drop useless sensors, fill gaps (fit on TRAINING runs only)
    SPCLimits           statistical-process-control limits (mean ± 3σ of good runs)

Kept in core/ (not in the training script) because joblib pickles reference
these classes by import path — the agent must be able to import them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

BASE_DIR  = Path(__file__).resolve().parent.parent
SECOM_DIR = BASE_DIR / "data" / "raw" / "secom"
N_SENSORS = 590


def load_secom() -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Returns X (runs × 590, NaN = missing), y (1 = failed line test), timestamps."""
    X = pd.read_csv(SECOM_DIR / "secom.data", sep=" ", header=None).to_numpy(dtype=float)
    lab = pd.read_csv(SECOM_DIR / "secom_labels.data", sep=" ", header=None, names=["y", "ts"])
    ts = pd.to_datetime(lab["ts"], format="%d/%m/%Y %H:%M:%S")
    y = (lab["y"] == 1).astype(int).to_numpy()
    order = np.argsort(ts.to_numpy(), kind="stable")          # already sorted; be explicit
    return X[order], y[order], ts.iloc[order].reset_index(drop=True)


class SensorCleaner(BaseEstimator, TransformerMixin):
    """
    1. Drop sensors missing in more than `max_missing` of training runs.
    2. Drop sensors that never change in training runs (carry no information).
    3. Fill remaining gaps with the training median.
    Everything is learned in fit() from training rows only — no test leakage.
    """

    def __init__(self, max_missing: float = 0.5):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        missing = np.isnan(X).mean(axis=0)
        std     = np.nanstd(X, axis=0)
        self.keep_    = np.flatnonzero((missing <= self.max_missing) & (np.nan_to_num(std) > 1e-12))
        self.medians_ = np.nanmedian(X[:, self.keep_], axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)[:, self.keep_]
        gaps = np.isnan(X)
        X[gaps] = np.take(self.medians_, np.nonzero(gaps)[1])
        return X


def risk_score(model, X) -> np.ndarray:
    """Higher = more likely to fail. Works for classifiers and Isolation Forest."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return -model.decision_function(X)          # Isolation Forest: low score = anomalous


class SPCLimits:
    """
    Classic statistical process control: a sensor is "out of control" when it
    is more than 3 standard deviations from its mean on GOOD (passing) runs.

    With ~470 sensors, ~1 false alarm per run is expected by pure chance
    (0.27% × 470), so the useful signal is how many sensors are out compared
    with a normal run — `typical_ooc_p95` records that baseline.
    """

    def __init__(self, sensor_ids: np.ndarray, mean: np.ndarray, std: np.ndarray,
                 typical_ooc_p95: float):
        self.sensor_ids      = sensor_ids
        self.mean            = mean
        self.std             = std
        self.typical_ooc_p95 = typical_ooc_p95

    @classmethod
    def fit(cls, X_clean_good: np.ndarray, sensor_ids: np.ndarray) -> "SPCLimits":
        mean = X_clean_good.mean(axis=0)
        std  = X_clean_good.std(axis=0) + 1e-9
        ooc  = (np.abs((X_clean_good - mean) / std) > 3).sum(axis=1)
        return cls(sensor_ids, mean, std, float(np.percentile(ooc, 95)))

    def check(self, x_clean: np.ndarray, top: int = 5) -> dict:
        z   = (np.asarray(x_clean, dtype=float) - self.mean) / self.std
        ooc = np.flatnonzero(np.abs(z) > 3)
        worst = ooc[np.argsort(-np.abs(z[ooc]))][:top]
        return {
            "sensors_checked":      int(len(z)),
            "out_of_control":       int(len(ooc)),
            "typical_out_of_control_p95": self.typical_ooc_p95,
            "unusual_vs_normal":    bool(len(ooc) > self.typical_ooc_p95),
            "worst_sensors": [
                {"sensor": f"S{int(self.sensor_ids[i]):03d}", "z_score": round(float(z[i]), 1)}
                for i in worst
            ],
        }
