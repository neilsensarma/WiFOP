import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier
import json
import joblib

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_PATH = "../data/COMBINE_WIFOP_DATASET/data/processed/ca_5km_daily_panel_2020_2023_6_9.csv"
MODEL_PATH = "../models/2020_2023/xgb_final_model.pkl"
THRESHOLD_PATH = "../models/2020_2023/xgb_final_threshold.json"
META_PATH = "../models/2020_2023/xgb_final_metadata.json"

# ----------------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------------
print("Loading combined panel (2020–2023)...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Convert dtypes
df["date"] = pd.to_datetime(df["date"])
df["grid_id"] = df["grid_id"].astype("category")

target = "fire_today"

# ----------------------------------------------------------------------
# Feature Engineering (Nonlinear & Spatiotemporal)
# ----------------------------------------------------------------------
print("Engineering nonlinear interaction features...")

df["vpd_proxy"] = (1 - df["RH2M"] / 100) * df["T2M"]  # Vapor Pressure Deficit (VPD): High VPD = dry fuels → much easier ignition
df["hot_windy"] = df["T2M"] * df["WS10M"]             # Combined “fire weather storm”: High temperature → dries fuels; High wind → moves flames, fuels ignition events.
df["dry_windy"] = (100 - df["RH2M"]) * df["WS10M"]    # Dryness–wind interaction factor: High wind + dry air = extreme risk
df["fuel_moisture_proxy"] = df["ndvi"] * df["RH2M"]   # Proxy for live fuel moisture: NDVI ≈ “how green and alive the vegetation is”; RH2M ≈ “how moist the air is”

# Add additional lag features if available
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
    "fires_last_7d",
    "fires_last_14d",
] + available_lags + [

    # Land cover (static)
    "pct_Open Water", "pct_Perennial Ice/Snow",
    "pct_Developed, Open Space", "pct_Developed, Low Intensity",
    "pct_Developed, Medium Intensity", "pct_Developed, High Intensity",
    "pct_Barren Land (Rock/Sand/Clay)", "pct_Deciduous Forest",
    "pct_Evergreen Forest", "pct_Mixed Forest", "pct_Shrub/Scrub",
    "pct_Grassland/Herbaceous", "pct_Pasture/Hay",
    "pct_Cultivated Crops", "pct_Woody Wetlands",
    "pct_Emergent Herbaceous Wetlands",

    # Newly engineered nonlinear features
    "vpd_proxy", "hot_windy", "dry_windy", "fuel_moisture_proxy"
]

print(f"Using {len(feature_cols)} features.")

# ----------------------------------------------------------------------
# Time-Based Split (Train: 2020–2022, Test: 2023)
# ----------------------------------------------------------------------
# We split by date (train = 2020–2022, val = 2023) to avoid leakage, preserve temporal order, 
# respect lag-based features, capture realistic fire-season dynamics, and evaluate whether the model can truly generalize to future years.
train = df[df["date"] < "2023-01-01"]
val   = df[df["date"] >= "2023-01-01"]

X_train = train[feature_cols]
y_train = train[target]

X_val = val[feature_cols]
y_val = val[target]

print(f"Train rows: {len(X_train)} | Val rows: {len(X_val)}")
print(f"Train fires: {y_train.sum()} | Val fires: {y_val.sum()}")

# ----------------------------------------------------------------------
# Compute Class Weight (Severe Imbalance)
# ----------------------------------------------------------------------
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale = neg / max(pos, 1)

print(f"scale_pos_weight = {scale:.2f}")

# ----------------------------------------------------------------------
# Train XGBoost (GPU Optimized)
# ----------------------------------------------------------------------
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",         # better metric for fire
    scale_pos_weight=scale,
    tree_method="hist",          
    predictor="auto",
    learning_rate=0.05,
    max_depth=10,
    n_estimators=700,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

print("\nTraining XGBoost model (GPU)...")
model.fit(X_train, y_train)

# ----------------------------------------------------------------------
# Evaluate Model
# ----------------------------------------------------------------------
y_val_proba = model.predict_proba(X_val)[:, 1]

auc_roc  = roc_auc_score(y_val, y_val_proba)
auc_pr   = average_precision_score(y_val, y_val_proba)

print(f"\nROC-AUC: {auc_roc:.4f}")
print(f"AUC-PR (recommended): {auc_pr:.4f}")

# ----------------------------------------------------------------------
# Optimal Threshold Selection (Max F1)
# ----------------------------------------------------------------------
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

best_idx = np.argmax(f1_scores)
best_threshold = float(thresholds[best_idx])

print(f"\nOptimal threshold (max F1): {best_threshold:.4f}")

y_pred_best = (y_val_proba >= best_threshold).astype(int)

print("\nClassification Report:")
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
    "roc_auc": float(auc_roc),
    "auc_pr": float(auc_pr),
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

print("\nModel, threshold, and metadata saved successfully.")