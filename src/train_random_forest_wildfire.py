import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, classification_report, confusion_matrix
)
import json
import joblib

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_PATH = "../data/COMBINE_WIFOP_DATASET/data/processed/ca_5km_daily_panel_2020_2023_6_9.csv"
MODEL_PATH = "../models/2020_2023/rf_model.pkl"
THRESHOLD_PATH = "../models/2020_2023/rf_threshold.json"
META_PATH = "../models/2020_2023/rf_metadata.json"

# ----------------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------------
print("Loading combined panel...")
df = pd.read_csv(DATA_PATH, low_memory=False)

df["date"] = pd.to_datetime(df["date"])
df["grid_id"] = df["grid_id"].astype("category")

target = "fire_today"

# ----------------------------------------------------------------------
# Feature Engineering (same as XGBoost model)
# ----------------------------------------------------------------------
print("Engineering nonlinear interaction features...")

df["vpd_proxy"] = (1 - df["RH2M"] / 100) * df["T2M"]
df["hot_windy"] = df["T2M"] * df["WS10M"]
df["dry_windy"] = (100 - df["RH2M"]) * df["WS10M"]
df["fuel_moisture_proxy"] = df["ndvi"] * df["RH2M"]

optional_lags = ["fires_last_1d", "fires_last_3d"]
available_lags = [c for c in optional_lags if c in df.columns]

# ----------------------------------------------------------------------
# Final Feature List
# ----------------------------------------------------------------------
feature_cols = [
    # Base weather
    "T2M", "RH2M", "WS10M", "PS", "PRECTOTCORR",
    "T2M_3d_mean", "RH2M_3d_min", "WS10M_3d_max", "PRECTOT_7d_sum",

    # Vegetation
    "ndvi", "ndvi_7d_mean",

    # Lag-fire features
    "fires_last_7d", "fires_last_14d"
] + available_lags + [

    # Land cover
    "pct_Open Water", "pct_Perennial Ice/Snow",
    "pct_Developed, Open Space", "pct_Developed, Low Intensity",
    "pct_Developed, Medium Intensity", "pct_Developed, High Intensity",
    "pct_Barren Land (Rock/Sand/Clay)", "pct_Deciduous Forest",
    "pct_Evergreen Forest", "pct_Mixed Forest", "pct_Shrub/Scrub",
    "pct_Grassland/Herbaceous", "pct_Pasture/Hay",
    "pct_Cultivated Crops", "pct_Woody Wetlands",
    "pct_Emergent Herbaceous Wetlands",

    # Nonlinear interaction features
    "vpd_proxy", "hot_windy", "dry_windy", "fuel_moisture_proxy"
]

print(f"Using {len(feature_cols)} features.")

# ----------------------------------------------------------------------
# Train / Validation Split (random)
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
# Train Random Forest
# ----------------------------------------------------------------------
print("Training Random Forest model...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",   # handles rare fire events
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)

# ----------------------------------------------------------------------
# Evaluate Model
# ----------------------------------------------------------------------
y_val_proba = rf.predict_proba(X_val)[:, 1]

roc_auc = roc_auc_score(y_val, y_val_proba)
auc_pr = average_precision_score(y_val, y_val_proba)

print(f"\nROC-AUC: {roc_auc:.4f}")
print(f"AUC-PR:  {auc_pr:.4f}")

# ----------------------------------------------------------------------
# Optimal Threshold Selection (Max F1)
# ----------------------------------------------------------------------
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

best_idx = np.argmax(f1_scores)
best_threshold = float(thresholds[best_idx])

print(f"\nOptimal threshold (max F1): {best_threshold:.4f}")

y_pred_best = (y_val_proba >= best_threshold).astype(int)

print("\nClassification Report (optimal threshold):")
print(classification_report(y_val, y_pred_best, digits=3))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_best))

# ----------------------------------------------------------------------
# Save Model + Threshold + Metadata
# ----------------------------------------------------------------------
joblib.dump(rf, MODEL_PATH)

with open(THRESHOLD_PATH, "w") as f:
    json.dump({"threshold": best_threshold}, f)

metadata = {
    "roc_auc": float(roc_auc),
    "auc_pr": float(auc_pr),
    "best_threshold": best_threshold,
    "num_train_rows": int(len(X_train)),
    "num_val_rows": int(len(X_val)),
    "train_fires": int(y_train.sum()),
    "val_fires": int(y_val.sum()),
    "features": feature_cols
}

with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print("\nRandom Forest model, threshold, and metadata saved successfully.")
