"""
ManufactureIQ — Defect Classifier Evaluation (class-imbalance study)
=====================================================================
AI4I 2020 is ~96.6% healthy, so plain accuracy is meaningless (always
predicting "none" scores ~96.6%). This script measures what matters:
per-class recall (how many real failures are caught), precision (how many
alarms are real) and macro-F1, using stratified 5-fold cross-validation.

Usage:
    python scripts/evaluate_models.py baseline     # current training recipe
    python scripts/evaluate_models.py weighted     # + inverse-frequency class weights
    python scripts/evaluate_models.py threshold    # + tuned "any failure" alarm threshold
    python scripts/evaluate_models.py two_stage    # failure-vs-healthy, then failure type
    python scripts/evaluate_models.py no_rnf       # step 5: RNF excluded from training
    python scripts/evaluate_models.py smote        # step 6: SMOTE on training folds only
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

BASE_DIR  = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "ai4i2020.csv"

FEATURES = [
    "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
    "Torque [Nm]", "Tool wear [min]", "Type_encoded", "power_W", "temp_diff_K", "wear_torque",
]
N_JOBS = 2   # keep memory/CPU low on a laptop


def load():
    """Same features + label rule as scripts/train_models.py."""
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.replace("﻿", "") for c in df.columns]
    df["Type_encoded"] = LabelEncoder().fit_transform(df["Type"])
    df["power_W"]     = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
    df["temp_diff_K"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["wear_torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    def label(r):
        for t in ("TWF", "HDF", "PWF", "OSF", "RNF"):
            if r[t]:
                return t
        return "none"

    df["label"] = df.apply(label, axis=1)
    X = df[FEATURES].copy()
    X.columns = [c.replace("[", "(").replace("]", ")") for c in X.columns]
    return X, df["label"].to_numpy()


def xgb(num_class):
    # Same hyper-parameters as train_models.py.
    return XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, eval_metric="mlogloss", random_state=42,
        n_jobs=N_JOBS, objective="multi:softprob", num_class=num_class,
    )


def cv_predict(X, y, fit_predict):
    """Out-of-fold predictions with stratified 5-fold CV."""
    pred = np.empty(len(y), dtype=object)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in skf.split(X, y):
        pred[te] = fit_predict(X.iloc[tr], y[tr], X.iloc[te])
    return pred


# ── Variants ──────────────────────────────────────────────────────────────────

def fp_baseline(Xtr, ytr, Xte, weighted=False):
    le = LabelEncoder().fit(ytr)
    m = xgb(len(le.classes_))
    w = compute_sample_weight("balanced", ytr) if weighted else None
    m.fit(Xtr, le.transform(ytr), sample_weight=w)
    return le.inverse_transform(m.predict(Xte))


def _fit_weighted(Xtr, ytr):
    le = LabelEncoder().fit(ytr)
    m = xgb(len(le.classes_))
    m.fit(Xtr, le.transform(ytr), sample_weight=compute_sample_weight("balanced", ytr))
    return m, le


def _apply_threshold(proba, classes, t):
    """Alarm if P(any failure) >= t, then pick the most likely failure type."""
    none_i = list(classes).index("none")
    p_fail = 1.0 - proba[:, none_i]
    masked = proba.copy()
    masked[:, none_i] = -1
    out = np.where(p_fail >= t, np.asarray(classes)[masked.argmax(1)], "none")
    return out


def _tune_threshold(Xtr, ytr, fit=_fit_weighted):
    """Pick t on an inner validation split of the TRAINING fold only (no test leakage)."""
    from sklearn.model_selection import train_test_split
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=0)
    m, le = fit(Xa, ya)
    proba = m.predict_proba(Xv)
    fails = [c for c in le.classes_ if c != "none"]
    grid = np.arange(0.05, 0.96, 0.05)
    scores = [f1_score(yv, _apply_threshold(proba, le.classes_, t), labels=fails,
                       average="macro", zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def fp_threshold(Xtr, ytr, Xte):
    t = _tune_threshold(Xtr, ytr)
    m, le = _fit_weighted(Xtr, ytr)
    THRESHOLDS.append(t)
    return _apply_threshold(m.predict_proba(Xte), le.classes_, t)


def fp_two_stage(Xtr, ytr, Xte):
    from sklearn.model_selection import train_test_split
    # Stage 1: failure vs healthy (weighted), threshold tuned on inner split.
    yb = (ytr != "none").astype(int)
    def s1():
        return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", random_state=42,
                             n_jobs=N_JOBS, scale_pos_weight=(yb == 0).sum() / max((yb == 1).sum(), 1))
    Xa, Xv, ya, yv = train_test_split(Xtr, yb, test_size=0.25, stratify=yb, random_state=0)
    p = s1().fit(Xa, ya).predict_proba(Xv)[:, 1]
    grid = np.arange(0.05, 0.96, 0.05)
    t = float(grid[int(np.argmax([f1_score(yv, (p >= g).astype(int)) for g in grid]))])
    THRESHOLDS.append(t)
    is_fail = s1().fit(Xtr, yb).predict_proba(Xte)[:, 1] >= t
    # Stage 2: failure type, trained on failure rows only.
    f = ytr != "none"
    m2, le2 = _fit_weighted(Xtr[f], ytr[f])
    out = np.full(len(Xte), "none", dtype=object)
    if is_fail.any():
        out[is_fail] = le2.inverse_transform(m2.predict(Xte[is_fail]))
    return out


def fp_no_rnf(Xtr, ytr, Xte):
    # Builds on the weighted model (step 2) — threshold tuning (step 3) was rejected.
    keep = ytr != "RNF"
    return fp_baseline(Xtr[keep], ytr[keep], Xte, weighted=True)


def fp_smote(Xtr, ytr, Xte):
    # Synthetic minority rows on the TRAINING fold only (never the test fold);
    # replaces class weights. RNF excluded as in step 5.
    from imblearn.over_sampling import SMOTE
    keep = ytr != "RNF"
    Xs, ys = SMOTE(random_state=42, k_neighbors=5).fit_resample(Xtr[keep], ytr[keep])
    return fp_baseline(Xs, ys, Xte, weighted=False)


THRESHOLDS = []


def report(name, y, pred, labels):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    print(classification_report(y, pred, labels=labels, digits=3, zero_division=0))
    fails = [l for l in labels if l != "none"]
    print("macro-F1 (failure classes only):",
          round(f1_score(y, pred, labels=fails, average="macro", zero_division=0), 3))
    is_f, said_f = (y != "none"), (pred != "none")
    print(f"Any-failure detection: caught {int((is_f & said_f).sum())}/{int(is_f.sum())} "
          f"real failures, {int((~is_f & said_f).sum())} false alarms on "
          f"{int((~is_f).sum())} healthy rows")
    print("\nConfusion (rows = truth, cols = predicted):")
    print(pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=labels, columns=labels))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    X, y = load()
    labels = ["none", "TWF", "HDF", "PWF", "OSF", "RNF"]
    print("Class counts:", dict(pd.Series(y).value_counts()))

    if step == "baseline":
        report("BASELINE — current recipe (XGBoost, no class weights)", y,
               cv_predict(X, y, fp_baseline), labels)
    elif step == "weighted":
        report("WEIGHTED — inverse-frequency sample weights", y,
               cv_predict(X, y, lambda a, b, c: fp_baseline(a, b, c, weighted=True)), labels)
    elif step == "threshold":
        report("THRESHOLD — weighted + tuned any-failure threshold", y,
               cv_predict(X, y, fp_threshold), labels)
        print("Chosen thresholds per fold:", THRESHOLDS)
    elif step == "two_stage":
        report("TWO-STAGE — failure vs healthy, then failure type", y,
               cv_predict(X, y, fp_two_stage), labels)
        print("Stage-1 thresholds per fold:", THRESHOLDS)
    elif step == "no_rnf":
        pred = cv_predict(X, y, fp_no_rnf)
        keep = y != "RNF"
        report("NO-RNF — weighted, RNF excluded from training (scored on non-RNF rows)",
               y[keep], pred[keep], [l for l in labels if l != "RNF"])
        r = y == "RNF"
        print(f"\nRNF rows (reported separately): {int((pred[r] != 'none').sum())}/{int(r.sum())} raised an alarm")
    elif step == "smote":
        pred = cv_predict(X, y, fp_smote)
        keep = y != "RNF"
        report("SMOTE — oversampled training folds, RNF excluded (scored on non-RNF rows)",
               y[keep], pred[keep], [l for l in labels if l != "RNF"])
    else:
        sys.exit(f"Unknown step '{step}'")
