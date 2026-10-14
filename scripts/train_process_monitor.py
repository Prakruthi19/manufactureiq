"""
ManufactureIQ — Train the process monitor on SECOM
====================================================
Model used by the Trend Detector: given one production run's 590 sensor
readings, how likely is it to FAIL the line test?

Why this is hard (and why that's fine to say in an interview):
    104 failures in 1,567 runs (6.6%), 590 anonymised sensors, lots of missing
    values, and the failure rate drifts month to month. Published results on
    SECOM are modest. The goal here is an HONEST estimate, not a big number.

Method
    1. Chronological split: first 70% of runs (by time) = train, last 30% = test.
       A random split would let the model learn from the future.
    2. Every imbalance strategy is compared with time-series cross-validation
       INSIDE the training period. The test period is used exactly once.
    3. Metric = PR-AUC (average precision). With 6% positives, accuracy is
       meaningless (always "pass" = 94%) and ROC-AUC flatters the model.
       Baseline PR-AUC of a random model = the failure rate.
    4. Alert threshold chosen on training data for a fixed false-alarm budget
       (10% of good runs flagged) — that's how a fab would actually run it.
    5. Test metrics come with 95% bootstrap confidence intervals, because with
       ~26 test failures a single number is misleading.
    6. For contrast, the script also scores the common tutorial recipe (SMOTE on
       the whole dataset, THEN a random split). It reports ~0.99 because SMOTE's
       synthetic failures are blends of real ones, so near-copies of test rows
       end up in training. That number is leakage, not skill.

Output: models/process_monitor.pkl + models/process_monitor_metrics.json
"""

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.secom import SensorCleaner, SPCLimits, load_secom, risk_score  # noqa: E402

warnings.filterwarnings("ignore")
SEED        = 42
MODELS_DIR  = Path(__file__).resolve().parent.parent / "models"
FALSE_ALARM = 0.10          # alert budget: flag at most ~10% of good runs


def candidates(pos_weight: float) -> dict:
    """Each entry is a different answer to 'how do we handle 6% failures?'"""
    clean = lambda: [("clean", SensorCleaner()), ("scale", StandardScaler())]
    return {
        # Baseline: ignore the imbalance entirely.
        "logreg_unweighted": Pipeline(clean() + [
            ("model", LogisticRegression(C=0.01, max_iter=5000))]),
        # Re-weight: a missed failure costs ~14x a false alarm.
        "logreg_balanced": Pipeline(clean() + [
            ("model", LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000))]),
        # Oversample: synthesise extra failures (SMOTE) — only inside training folds.
        "logreg_smote": ImbPipeline(clean() + [
            ("smote", SMOTE(random_state=SEED, k_neighbors=5)),
            ("model", LogisticRegression(C=0.01, max_iter=5000))]),
        # Non-linear, re-weighted per bootstrap sample.
        "random_forest_balanced": Pipeline(clean() + [
            ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                             class_weight="balanced_subsample",
                                             random_state=SEED, n_jobs=2))]),
        # Boosting with the positive class up-weighted.
        "xgboost_weighted": Pipeline(clean() + [
            ("model", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.3,
                                    scale_pos_weight=pos_weight, eval_metric="aucpr",
                                    random_state=SEED, n_jobs=2))]),
        # Unsupervised: learn what a GOOD run looks like, flag anything unlike it.
        # Needs no failure labels at all — fit on passing runs only (see fit_model).
        "isolation_forest": Pipeline(clean() + [
            ("model", IsolationForest(n_estimators=400, random_state=SEED, n_jobs=2))]),
    }


def fit_model(name: str, model, X, y):
    if name == "isolation_forest":
        return model.fit(X[y == 0])
    return model.fit(X, y)


def bootstrap_ci(y, s, metric, n=1000):
    rng, vals = np.random.default_rng(SEED), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() and (1 - y[i]).sum():
            vals.append(metric(y[i], s[i]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def main():
    print("=" * 64)
    print("Process monitor — SECOM")
    print("=" * 64)

    X, y, ts = load_secom()
    cut = int(len(y) * 0.7)
    X_tr, y_tr, X_te, y_te = X[:cut], y[:cut], X[cut:], y[cut:]
    print(f"Train: {ts.iloc[0]:%Y-%m-%d} → {ts.iloc[cut-1]:%Y-%m-%d}  "
          f"{len(y_tr)} runs, {y_tr.sum()} failures ({y_tr.mean():.1%})")
    print(f"Test:  {ts.iloc[cut]:%Y-%m-%d} → {ts.iloc[-1]:%Y-%m-%d}  "
          f"{len(y_te)} runs, {y_te.sum()} failures ({y_te.mean():.1%})")

    # ── 1. Compare strategies with time-series CV inside the training period ──
    tscv = TimeSeriesSplit(n_splits=4)
    pos_weight = (y_tr == 0).sum() / max(y_tr.sum(), 1)
    cv = {}
    print(f"\n[1/3] Time-series CV (4 folds, train period only) — PR-AUC, "
          f"random baseline ≈ {y_tr.mean():.3f}")
    for name in candidates(pos_weight):
        pr, roc = [], []
        for a, b in tscv.split(X_tr):
            if y_tr[b].sum() == 0:
                continue
            m = fit_model(name, candidates(pos_weight)[name], X_tr[a], y_tr[a])
            s = risk_score(m, X_tr[b])
            pr.append(average_precision_score(y_tr[b], s))
            roc.append(roc_auc_score(y_tr[b], s))
        cv[name] = {"pr_auc": round(float(np.mean(pr)), 3), "pr_auc_std": round(float(np.std(pr)), 3),
                    "roc_auc": round(float(np.mean(roc)), 3)}
        print(f"      {name:<24} PR-AUC {cv[name]['pr_auc']:.3f} ± {cv[name]['pr_auc_std']:.3f}   "
              f"ROC-AUC {cv[name]['roc_auc']:.3f}")

    best = max(cv, key=lambda k: cv[k]["pr_auc"])
    print(f"      → selected: {best}")

    # ── 2. Refit on the full training period, pick the alert threshold ────────
    print("\n[2/3] Refit on all training runs; set alert threshold")
    model = fit_model(best, candidates(pos_weight)[best], X_tr, y_tr)
    s_tr = risk_score(model, X_tr)
    threshold = float(np.quantile(s_tr[y_tr == 0], 1 - FALSE_ALARM))
    print(f"      threshold = {threshold:.4f}  (flags {FALSE_ALARM:.0%} of good training runs)")

    # ── 3. One look at the held-out future ────────────────────────────────────
    print("\n[3/3] Held-out test period (used once)")
    s_te = risk_score(model, X_te)
    alert = s_te >= threshold
    test = {
        "pr_auc":        round(float(average_precision_score(y_te, s_te)), 3),
        "pr_auc_ci95":   bootstrap_ci(y_te, s_te, average_precision_score),
        "roc_auc":       round(float(roc_auc_score(y_te, s_te)), 3),
        "roc_auc_ci95":  bootstrap_ci(y_te, s_te, roc_auc_score),
        "random_baseline_pr_auc": round(float(y_te.mean()), 3),
        "at_threshold": {
            "failures_caught":  f"{int(alert[y_te == 1].sum())}/{int(y_te.sum())}",
            "recall":           round(float(alert[y_te == 1].mean()), 3),
            "false_alarm_rate": round(float(alert[y_te == 0].mean()), 3),
            "precision":        round(float(y_te[alert].mean()) if alert.any() else 0.0, 3),
        },
    }
    print(f"      PR-AUC  {test['pr_auc']:.3f}  95% CI {test['pr_auc_ci95']}  "
          f"(random = {test['random_baseline_pr_auc']})")
    print(f"      ROC-AUC {test['roc_auc']:.3f}  95% CI {test['roc_auc_ci95']}")
    t = test["at_threshold"]
    print(f"      At threshold: caught {t['failures_caught']} failures (recall {t['recall']:.0%}), "
          f"false alarms {t['false_alarm_rate']:.0%}, precision {t['precision']:.0%}")

    # ── Why published SECOM numbers look so much better ───────────────────────
    print("\n[contrast] Same model family, different evaluation:")
    leak = {}
    Xc = SensorCleaner().fit_transform(X)
    Xs, ys = SMOTE(random_state=SEED).fit_resample(Xc, y)
    a, b, c, d = train_test_split(Xs, ys, test_size=0.3, random_state=SEED, stratify=ys)
    m = RandomForestClassifier(300, random_state=SEED, n_jobs=2).fit(a, c)
    leak["smote_before_random_split"] = round(float(roc_auc_score(d, m.predict_proba(b)[:, 1])), 3)
    a, b, c, d = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    m = fit_model(best, candidates(pos_weight)[best], a, c)
    leak["random_split_no_leak"] = round(float(roc_auc_score(d, risk_score(m, b))), 3)
    leak["chronological_split"] = test["roc_auc"]
    for k, v in leak.items():
        print(f"      ROC-AUC {v:.3f}  {k}")

    # ── SPC limits from good TRAINING runs (used by the Sensor Parser) ────────
    cleaner = model.named_steps["clean"]
    spc = SPCLimits.fit(cleaner.transform(X_tr[y_tr == 0]), cleaner.keep_)

    # Held-out runs the demo can replay (never seen in training).
    demo_idx = np.flatnonzero(np.arange(len(y)) >= cut)

    joblib.dump({
        "model": model, "kind": best, "threshold": threshold, "spc": spc,
        "n_sensors": X.shape[1],
    }, MODELS_DIR / "process_monitor.pkl")

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "SECOM (UCI) — 1,567 runs, 590 sensors, real semiconductor line",
        "split": {"type": "chronological 70/30",
                  "train": [str(ts.iloc[0]), str(ts.iloc[cut - 1])],
                  "test": [str(ts.iloc[cut]), str(ts.iloc[-1])],
                  "train_failures": int(y_tr.sum()), "test_failures": int(y_te.sum())},
        "sensors_kept": int(len(cleaner.keep_)),
        "cv_comparison": cv,
        "selected": best,
        "alert_threshold": threshold,
        "false_alarm_budget": FALSE_ALARM,
        "test": test,
        "evaluation_contrast_roc_auc": leak,
        "decision": ("Weak signal on future runs: shown as a low-confidence process-risk score; "
                     "it never escalates on its own. SPC out-of-control sensors are passed to "
                     "root-cause analysis as evidence."),
        "spc_typical_out_of_control_p95": spc.typical_ooc_p95,
        "demo_test_rows": [int(demo_idx[0]), int(demo_idx[-1])],
    }
    (MODELS_DIR / "process_monitor_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved models/process_monitor.pkl  (sensors kept: {len(cleaner.keep_)})")


if __name__ == "__main__":
    main()
