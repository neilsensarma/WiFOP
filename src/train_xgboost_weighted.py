import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
import json
import joblib

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_PATH = "../data/processed/california_5km_training_panel_2023-07_clean.csv"
MODEL_PATH = "../models/xgb_weighted_model.pkl"
THRESHOLD_PATH = "../models/xgb_weighted_threshold.json"
META_PATH = "../models/xgb_weighted_metadata.json"

# ----------------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------------
print("Loading cleaned panel...")
df = pd.read_csv(DATA_PATH)

# Feature set
feature_cols = [
    "PRECTOTCORR", "PS", "RH2M", "T2M", "WS10M",
    "PRECTOT_7d_sum", "T2M_3d_mean", "RH2M_3d_min", "WS10M_3d_max",
    "ndvi", "ndvi_7d_mean",
    "fires_last_7d", "fires_last_14d",
    "pct_11","pct_12","pct_21","pct_22","pct_23","pct_24","pct_31",
    "pct_41","pct_42","pct_43","pct_52","pct_71","pct_81","pct_82","pct_90","pct_95"
]

target = "fire_today"

# Remove missing rows
before = len(df)
df = df.dropna(subset=feature_cols + [target])
print(f"Rows before: {before}, after dropping NaN: {len(df)}")

# ----------------------------------------------------------------------
# Train / Validation Split
# ----------------------------------------------------------------------
X = df[feature_cols]
y = df[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training rows: {len(X_train)} | Validation rows: {len(X_val)}")
print(f"Fire cases in train: {y_train.sum()} | Fire cases in val: {y_val.sum()}")

# ----------------------------------------------------------------------
# Compute Class Weight
# ----------------------------------------------------------------------
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale = neg / max(pos, 1)
print(f"Computed scale_pos_weight: {scale:.2f}")

# ----------------------------------------------------------------------
# Train XGBoost
# ----------------------------------------------------------------------
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale,
    learning_rate=0.05,
    max_depth=6,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

print("Training weighted XGBoost model...")
model.fit(X_train, y_train)

# ----------------------------------------------------------------------
# Evaluate ROC-AUC
# ----------------------------------------------------------------------
y_val_proba = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_val_proba)
print(f"ROC-AUC: {auc:.4f}")

# ----------------------------------------------------------------------
# Threshold Optimization (Precision-Recall)
# ----------------------------------------------------------------------
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

best_idx = np.argmax(f1_scores)
best_threshold = float(thresholds[best_idx])

print("\nOptimal threshold (max F1): {:.4f}".format(best_threshold))

# ----------------------------------------------------------------------
# Evaluate at Optimal Threshold
# ----------------------------------------------------------------------
y_pred_best = (y_val_proba >= best_threshold).astype(int)

print("\nClassification Report (optimal threshold):")
print(classification_report(y_val, y_pred_best, digits=3))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_best))

# ----------------------------------------------------------------------
# Save Model + Threshold + Metadata
# ----------------------------------------------------------------------
joblib.dump(model, MODEL_PATH)

with open(THRESHOLD_PATH, "w") as f:
    json.dump({"threshold": best_threshold}, f)

metadata = {
    "roc_auc": float(auc),
    "best_threshold": best_threshold,
    "num_train_rows": int(len(X_train)),
    "num_val_rows": int(len(X_val)),
    "train_fire_cases": int(y_train.sum()),
    "val_fire_cases": int(y_val.sum()),
    "features": feature_cols,
    "scale_pos_weight": float(scale)
}

with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print("Model, threshold, and metadata saved successfully.")
