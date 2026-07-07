"""
ManufactureIQ — ML Model Training
===================================
Trains two models on the AI4I 2020 dataset:

1. Defect Classifier  — Random Forest + XGBoost (multiclass)
   Input:  sensor readings (temp, rpm, torque, tool_wear, type)
   Output: defect type (TWF, HDF, PWF, OSF, RNF, none)

2. Trend Detector     — Isolation Forest (anomaly detection)
   Input:  aggregated defect stats per machine/batch/supplier
   Output: anomaly score (-1 = anomalous pattern, 1 = normal)

Models saved to: models/
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "data" / "raw" / "ai4i2020.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("ManufactureIQ — ML Training Pipeline")
print("=" * 60)

# ── Load Data ─────────────────────────────────────────────────────────────────
print("\n[1/6] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"      Loaded {len(df):,} rows, {df.shape[1]} columns")
print(f"      Columns: {list(df.columns)}")

# ── Feature Engineering ───────────────────────────────────────────────────────
print("\n[2/6] Engineering features...")

# Encode product type L/M/H → 0/1/2
le_type = LabelEncoder()
df["Type_encoded"] = le_type.fit_transform(df["Type"])

# Derived features — these are the physics-based interactions from the paper
df["power_W"]      = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
df["temp_diff_K"]  = df["Process temperature [K]"] - df["Air temperature [K]"]
df["wear_torque"]  = df["Tool wear [min]"] * df["Torque [Nm]"]   # OSF indicator

# ── Defect Label ──────────────────────────────────────────────────────────────
# Map failure modes to a single multiclass label
def get_defect_label(row):
    if row.get("TWF", 0): return "TWF"
    if row.get("HDF", 0): return "HDF"
    if row.get("PWF", 0): return "PWF"
    if row.get("OSF", 0): return "OSF"
    if row.get("RNF", 0): return "RNF"
    return "none"

df["defect_label"] = df.apply(get_defect_label, axis=1)
print(f"      Label distribution:\n{df['defect_label'].value_counts().to_string()}")

# ── Features & Target ─────────────────────────────────────────────────────────
FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_encoded",
    "power_W",
    "temp_diff_K",
    "wear_torque",
]

X = df[FEATURES]
y = df["defect_label"]

# Encode target labels
le_label = LabelEncoder()
y_encoded = le_label.fit_transform(y)

# Train/test split — stratified to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\n      Train: {len(X_train):,} | Test: {len(X_test):,}")

# =============================================================================
# MODEL 1: RANDOM FOREST CLASSIFIER
# =============================================================================
print("\n[3/6] Training Random Forest Classifier...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",   # handles class imbalance (few RNF samples)
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)

rf_preds  = rf_model.predict(X_test)
rf_acc    = accuracy_score(y_test, rf_preds)
rf_f1     = f1_score(y_test, rf_preds, average="weighted")

print(f"      Accuracy:    {rf_acc:.4f}")
print(f"      F1 (weighted): {rf_f1:.4f}")
print(f"\n      Classification Report:")
print(classification_report(
    y_test, rf_preds,
    target_names=le_label.classes_,
    zero_division=0
))

# Feature importance
importance_df = pd.DataFrame({
    "feature":   FEATURES,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)
print("      Feature Importance:")
print(importance_df.to_string(index=False))

# =============================================================================
# MODEL 2: XGBOOST CLASSIFIER
# =============================================================================
print("\n[4/6] Training XGBoost Classifier...")

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)
# XGBoost requires clean column names (no brackets)
# Rename ALL features to remove brackets — consistent across RF and XGBoost
X_train.columns = [c.replace("[","(").replace("]",")") for c in X_train.columns]
X_test.columns  = [c.replace("[","(").replace("]",")") for c in X_test.columns]
X.columns       = [c.replace("[","(").replace("]",")") for c in X.columns]
FEATURES        = [f.replace("[","(").replace("]",")") for f in FEATURES]
xgb_model.fit(X_train, y_train, verbose=False)

xgb_preds = xgb_model.predict(X_test)
xgb_acc   = accuracy_score(y_test, xgb_preds)
xgb_f1    = f1_score(y_test, xgb_preds, average="weighted")

print(f"      Accuracy:      {xgb_acc:.4f}")
print(f"      F1 (weighted): {xgb_f1:.4f}")

# Pick the best classifier
best_name  = "random_forest" if rf_f1 >= xgb_f1 else "xgboost"
best_model = rf_model         if rf_f1 >= xgb_f1 else xgb_model
print(f"\n      Winner: {best_name.upper()} (F1: {max(rf_f1, xgb_f1):.4f})")

# =============================================================================
# MODEL 3: ISOLATION FOREST (TREND DETECTOR)
# =============================================================================
print("\n[5/6] Training Isolation Forest Trend Detector...")

# Aggregate features per machine over rolling window
# We use the full dataset to build the anomaly baseline
trend_features = df[[
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "power_W",
    "temp_diff_K",
    "wear_torque",
    "Machine failure",
]].copy()

# Scale features for Isolation Forest
scaler = StandardScaler()
trend_scaled = scaler.fit_transform(trend_features)

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,    # expect ~5% anomalous patterns
    random_state=42,
    n_jobs=-1,
)
iso_forest.fit(trend_scaled)

# Quick validation — failure rows should score lower (more anomalous)
scores         = iso_forest.decision_function(trend_scaled)
failure_scores = scores[df["Machine failure"] == 1].mean()
normal_scores  = scores[df["Machine failure"] == 0].mean()
print(f"      Avg anomaly score — failures: {failure_scores:.4f}")
print(f"      Avg anomaly score — normal:   {normal_scores:.4f}")
print(f"      Separation: {normal_scores - failure_scores:.4f} (higher = better)")

# =============================================================================
# SAVE MODELS
# =============================================================================
print("\n[6/6] Saving models...")

# Save best classifier
joblib.dump(best_model, MODELS_DIR / "defect_classifier.pkl")
joblib.dump(rf_model,   MODELS_DIR / "defect_classifier_rf.pkl")
joblib.dump(xgb_model,  MODELS_DIR / "defect_classifier_xgb.pkl")

# Save trend detector + scaler (always save both together)
joblib.dump(iso_forest, MODELS_DIR / "trend_detector.pkl")
joblib.dump(scaler,     MODELS_DIR / "trend_scaler.pkl")

# Save label encoders (needed at inference time)
joblib.dump(le_type,    MODELS_DIR / "label_encoder_type.pkl")
joblib.dump(le_label,   MODELS_DIR / "label_encoder_label.pkl")

# Save metadata — agents read this to know what model is loaded
metadata = {
    "trained_at":       datetime.utcnow().isoformat(),
    "dataset_rows":     len(df),
    "features":         FEATURES,
    "classes":          list(le_label.classes_),
    "best_classifier":  best_name,
    "rf_accuracy":      round(rf_acc, 4),
    "rf_f1":            round(rf_f1, 4),
    "xgb_accuracy":     round(xgb_acc, 4),
    "xgb_f1":           round(xgb_f1, 4),
    "isolation_forest": {
        "contamination":    0.05,
        "n_estimators":     200,
        "failure_score":    round(float(failure_scores), 4),
        "normal_score":     round(float(normal_scores), 4),
    }
}
with open(MODELS_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n      Models saved to: {MODELS_DIR}/")
print(f"      defect_classifier.pkl   ({best_name})")
print(f"      defect_classifier_rf.pkl")
print(f"      defect_classifier_xgb.pkl")
print(f"      trend_detector.pkl")
print(f"      trend_scaler.pkl")
print(f"      label_encoder_type.pkl")
print(f"      label_encoder_label.pkl")
print(f"      metadata.json")

print("\n" + "=" * 60)
print("Training complete.")
print("=" * 60)