"""
ManufactureIQ — Train the wafer-map Defect Classifier (WM-811K)
================================================================
Input : data/processed/wm811k_features.npz  (scripts/build_wafer_features.py)
Output: models/wafer_classifier.pkl, models/wafer_classifier_metrics.json

The imbalance problem
    "none" is 85% of wafers; Near-full is 149 wafers (0.09%). A model that
    always says "none" scores 85% accuracy and is useless. So:
      • accuracy is never used to choose anything
      • model selection = macro-F1 over all 9 classes (every class counts equally)
      • we also report defect-detection recall (did we catch defective wafers
        at all?) — the number a fab cares about most.

Leakage control
    Split by lot-block (see build_wafer_features.py): wafers from the same lots
    never sit on both sides of train / validation / test.
      test       ≈ 20% of lot-blocks — touched ONCE at the very end
      validation ≈ 20% of the rest  — used to compare strategies + set thresholds
      train      = remainder

Strategies compared (all XGBoost on the same 26 features, + one RandomForest)
    unweighted        — baseline: ignore imbalance
    balanced          — weight = 1 / class frequency (rare classes count a lot)
    sqrt_balanced     — weight = 1 / sqrt(frequency): a softer middle ground
    undersample_none  — keep 25% of "none" wafers + sqrt weights
    smote             — synthesise extra minority examples (training fold only)
    rf_balanced       — RandomForest, class_weight="balanced_subsample"

Calibration (core/calibration.py)
    SMOTE + boosting make probabilities over-confident. One temperature T is
    fitted on VALIDATION so that "90% confident" means right ~90% of the time.
    It never changes the predicted class, only the confidence. ECE before/after
    is reported on test. The review threshold below is chosen on calibrated
    confidences, and the live agent applies the same T.

Drift reference (core/drift.py)
    The training features' decile buckets are saved so the live system can
    measure how far incoming wafers have moved from what the model learned on.

Human review threshold
    If the model's top probability is below a threshold, the wafer goes to a
    human instead of being auto-classified. Chosen on VALIDATION so that
    auto-classified wafers are ≥95% correct — measured only on wafers where the
    model OR the expert says "defect". (Measured on all wafers, the easy "none"
    majority makes any threshold look fine and nothing ever reaches a human —
    the same accuracy trap as above.) We report the share sent to humans.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

BASE_DIR   = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from core import drift                                                       # noqa: E402
from core.calibration import apply_temperature, fit_temperature, reliability  # noqa: E402

FEATURES   = BASE_DIR / "data" / "processed" / "wm811k_features.npz"
MODELS_DIR = BASE_DIR / "models"
SEED       = 42
AUTO_ACCURACY_TARGET = 0.95


def xgb(n_classes: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        objective="multi:softprob", num_class=n_classes,
        eval_metric="mlogloss", random_state=SEED, n_jobs=2,
    )


def class_weights(y: np.ndarray, power: float) -> np.ndarray:
    counts = np.bincount(y)
    w = (counts.sum() / (len(counts) * counts)) ** power
    return w[y]


def fit_strategy(name, X, y, n_classes, none_id):
    rng = np.random.default_rng(SEED)
    if name == "unweighted":
        return xgb(n_classes).fit(X, y)
    if name == "balanced":
        return xgb(n_classes).fit(X, y, sample_weight=class_weights(y, 1.0))
    if name == "sqrt_balanced":
        return xgb(n_classes).fit(X, y, sample_weight=class_weights(y, 0.5))
    if name == "undersample_none":
        keep = (y != none_id) | (rng.random(len(y)) < 0.25)
        return xgb(n_classes).fit(X[keep], y[keep], sample_weight=class_weights(y[keep], 0.5))
    if name == "smote":
        counts = np.bincount(y)
        target = {c: max(n, 3000) for c, n in enumerate(counts)}          # lift rare classes to 3k
        Xs, ys = SMOTE(sampling_strategy=target, random_state=SEED).fit_resample(X, y)
        return xgb(n_classes).fit(Xs, ys)
    if name == "rf_balanced":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                      class_weight="balanced_subsample",
                                      random_state=SEED, n_jobs=2).fit(X, y)
    raise ValueError(name)


def lot_split(y: np.ndarray, groups: np.ndarray):
    """Train / validation / test indices; no lot-block appears in two of them.
    Shared with train_wafer_cnn.py so both models are judged on the same wafers."""
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    dev_idx, test_idx = next(outer.split(np.zeros(len(y)), y, groups))
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    tr_rel, val_rel = next(inner.split(np.zeros(len(dev_idx)), y[dev_idx], groups[dev_idx]))
    tr_idx, val_idx = dev_idx[tr_rel], dev_idx[val_rel]
    assert not set(groups[tr_idx]) & set(groups[test_idx])
    assert not set(groups[val_idx]) & (set(groups[test_idx]) | set(groups[tr_idx]))
    return tr_idx, val_idx, test_idx, dev_idx


def scores(y_true, y_pred, none_id, n_classes):
    defect = y_true != none_id
    return {
        "macro_f1":            round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "macro_f1_patterns":   round(float(f1_score(y_true, y_pred, average="macro",
                                                    labels=[c for c in range(n_classes) if c != none_id])), 4),
        "defect_recall":       round(float((y_pred[defect] != none_id).mean()), 4),
        "false_alarm_rate":    round(float((y_pred[~defect] != none_id).mean()), 4),
        "accuracy":            round(float((y_true == y_pred).mean()), 4),
    }


def main():
    print("=" * 66)
    print("Defect Classifier — WM-811K wafer maps")
    print("=" * 66)
    d = np.load(FEATURES, allow_pickle=False)
    X, groups, names = d["X"], d["group"], list(d["feature_names"])
    le = LabelEncoder().fit(d["y"])
    y = le.transform(d["y"])
    classes, n_classes = list(le.classes_), len(le.classes_)
    none_id = classes.index("none")
    print(f"{len(y):,} wafers, {len(names)} features, {len(np.unique(groups))} lot-blocks")

    # ── Lot-aware splits ──────────────────────────────────────────────────────
    tr_idx, val_idx, test_idx, dev_idx = lot_split(y, groups)
    print(f"train {len(tr_idx):,} | validation {len(val_idx):,} | test {len(test_idx):,}  "
          f"(no lot-block shared)")
    print("  test class counts:", {classes[c]: int(n) for c, n in enumerate(np.bincount(y[test_idx], minlength=n_classes))})

    # ── 1. Compare imbalance strategies on validation ─────────────────────────
    print("\n[1/3] Imbalance strategies — validation set")
    print(f"      {'strategy':<18}{'macroF1':>9}{'patterns':>10}{'defect rec':>11}{'false alarm':>12}{'accuracy':>10}{'min':>6}")
    comparison = {}
    for name in ["unweighted", "balanced", "sqrt_balanced", "undersample_none", "smote", "rf_balanced"]:
        t0 = time.time()
        m = fit_strategy(name, X[tr_idx], y[tr_idx], n_classes, none_id)
        s = scores(y[val_idx], m.predict(X[val_idx]), none_id, n_classes)
        s["per_class_recall"] = {classes[c]: round(float(r), 3) for c, r in enumerate(
            precision_recall_fscore_support(y[val_idx], m.predict(X[val_idx]), labels=range(n_classes),
                                            zero_division=0)[1])}
        comparison[name] = s
        print(f"      {name:<18}{s['macro_f1']:>9.3f}{s['macro_f1_patterns']:>10.3f}{s['defect_recall']:>11.3f}"
              f"{s['false_alarm_rate']:>12.3f}{s['accuracy']:>10.3f}{(time.time()-t0)/60:>6.1f}", flush=True)

    best = max(comparison, key=lambda k: comparison[k]["macro_f1"])
    print(f"      → selected by macro-F1: {best}")

    # ── 2. Human-review threshold from validation (train-only model) ──────────
    m_val = fit_strategy(best, X[tr_idx], y[tr_idx], n_classes, none_id)
    p_val_raw = m_val.predict_proba(X[val_idx])
    y_val     = y[val_idx]
    temperature = fit_temperature(p_val_raw, y_val)
    p_val = apply_temperature(p_val_raw, temperature)
    print(f"\n[2/3] Calibration: temperature T = {temperature:.2f} (fitted on validation); "
          f"validation ECE {reliability(p_val_raw, y_val)['ece']:.4f} → {reliability(p_val, y_val)['ece']:.4f}")
    conf, pred = p_val.max(1), p_val.argmax(1)
    relevant = (pred != none_id) | (y_val != none_id)          # wafers where a defect is in play
    threshold = 0.99
    for t in np.round(np.arange(0.30, 0.99, 0.01), 2):
        auto = (conf >= t) & relevant
        if auto.any() and (pred[auto] == y_val[auto]).mean() >= AUTO_ACCURACY_TARGET:
            threshold = float(t)
            break
    print(f"      Human-review threshold: calibrated confidence < {threshold:.2f} → human "
          f"({(conf < threshold).mean():.1%} of validation wafers; auto-classified defect-relevant "
          f"wafers ≥{AUTO_ACCURACY_TARGET:.0%} correct)")

    # ── 3. Final model on train+validation, ONE evaluation on test ────────────
    print("\n[3/3] Refit on train+validation, evaluate on held-out test lots")
    model = fit_strategy(best, X[dev_idx], y[dev_idx], n_classes, none_id)
    p_te_raw = model.predict_proba(X[test_idx])
    p_te  = apply_temperature(p_te_raw, temperature)
    y_te, pred_te, conf_te = y[test_idx], p_te.argmax(1), p_te.max(1)
    calib_before, calib_after = reliability(p_te_raw, y_te), reliability(p_te, y_te)
    defect_rel = (pred_te != none_id) | (y_te != none_id)
    calibration = {
        "method": "temperature scaling, T fitted on validation (NLL)",
        "temperature": round(temperature, 4),
        "test_ece_before": calib_before["ece"],
        "test_ece_after": calib_after["ece"],
        "test_ece_defect_relevant_before": reliability(p_te_raw[defect_rel], y_te[defect_rel])["ece"],
        "test_ece_defect_relevant_after": reliability(p_te[defect_rel], y_te[defect_rel])["ece"],
        "bins_before": calib_before["bins"],
        "bins_after": calib_after["bins"],
    }
    print(f"      Calibration on test: ECE {calibration['test_ece_before']:.4f} → {calibration['test_ece_after']:.4f} "
          f"(defect-relevant wafers {calibration['test_ece_defect_relevant_before']:.4f} → "
          f"{calibration['test_ece_defect_relevant_after']:.4f})")

    # Drift reference from the final model's training wafers + two sanity checks
    reference = drift.build_reference(X[dev_idx], names)
    drift_checks = {
        "test_lots_vs_training": drift.compare(reference, X[test_idx], names),
        "defective_wafers_only_vs_training": drift.compare(reference, X[test_idx][y_te != none_id], names),
    }
    for k, v in drift_checks.items():
        print(f"      Drift check — {k}: {v['status']} (max PSI {v['max_psi']:.3f}, "
              f"worst feature {v['features'][0]['feature']})")
    test = scores(y_te, pred_te, none_id, n_classes)
    auto = conf_te >= threshold
    rel  = (pred_te != none_id) | (y_te != none_id)
    test["human_review_share"]     = round(float((~auto).mean()), 4)
    test["auto_classified_accuracy"] = round(float((pred_te[auto & rel] == y_te[auto & rel]).mean()), 4)
    test["defects_auto_passed_as_none"] = int(((y_te != none_id) & (pred_te == none_id) & auto).sum())
    test["defects_total"]          = int((y_te != none_id).sum())
    report = classification_report(y_te, pred_te, target_names=classes, digits=3, zero_division=0)
    cm = confusion_matrix(y_te, pred_te, labels=range(n_classes))
    print(report)
    print("      Confusion matrix (rows = true, cols = predicted):")
    print("      " + "".join(f"{c[:6]:>8}" for c in classes))
    for c, row in zip(classes, cm):
        print(f"      {c[:9]:<9}" + "".join(f"{v:>8}" for v in row)[3:])
    print(f"\n      macro-F1 {test['macro_f1']:.3f} | defect recall {test['defect_recall']:.3f} | "
          f"false alarms on good wafers {test['false_alarm_rate']:.3f} | accuracy {test['accuracy']:.3f}")
    print(f"      {test['human_review_share']:.1%} of wafers routed to human review; auto-classified "
          f"defect-relevant wafers are {test['auto_classified_accuracy']:.1%} correct; "
          f"{test['defects_auto_passed_as_none']}/{test['defects_total']} defective wafers auto-passed as none")

    importance = sorted(zip(names, map(float, getattr(model, "feature_importances_", np.zeros(len(names))))),
                        key=lambda t: -t[1])

    joblib.dump({"model": model, "classes": classes, "feature_names": names,
                 "review_threshold": threshold, "strategy": best, "temperature": temperature},
                MODELS_DIR / "wafer_classifier.pkl")
    prf = precision_recall_fscore_support(y_te, pred_te, labels=range(n_classes), zero_division=0)
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "WM-811K (Wu et al. 2015) — 172,950 expert-labelled wafer maps from a real fab",
        "split": {"method": "StratifiedGroupKFold by lot-block (1,000 consecutive wafers)",
                  "train": int(len(tr_idx)), "validation": int(len(val_idx)), "test": int(len(test_idx))},
        "classes": classes,
        "strategy_comparison_validation": comparison,
        "selected_strategy": best,
        "review_threshold": threshold,
        "calibration": calibration,
        "drift_checks": {k: {**v, "features": v["features"][:5]} for k, v in drift_checks.items()},
        "test": test,
        "test_per_class": {c: {"precision": round(float(prf[0][i]), 3), "recall": round(float(prf[1][i]), 3),
                               "f1": round(float(prf[2][i]), 3), "support": int(prf[3][i])}
                           for i, c in enumerate(classes)},
        "test_confusion_matrix": cm.tolist(),
        "feature_importance": [[n, round(v, 4)] for n, v in importance],
    }
    (MODELS_DIR / "wafer_classifier_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(MODELS_DIR / "wafer_test_idx.npy", test_idx)
    drift.REFERENCE.write_text(json.dumps(reference), encoding="utf-8")
    print("\nSaved models/wafer_classifier.pkl + wafer_classifier_metrics.json")


if __name__ == "__main__":
    main()
