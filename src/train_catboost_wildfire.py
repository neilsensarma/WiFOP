import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    classification_report, confusion_matrix, average_precision_score
)
from sklearn.neighbors import BallTree
from catboost import CatBoostClassifier
import json
import joblib

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_PATH = "../data/COMBINE_WIFOP_DATASET/data/processed/ca_5km_daily_panel_2020_2023_6_9.csv"
MODEL_PATH = "../models/2020_2023/catboost_model.cbm"
THRESHOLD_PATH = "../models/2020_2023/catboost_threshold.json"
META_PATH = "../models/2020_2023/catboost_metadata.json"

# ----------------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------------
print("Loading combined panel...")
df = pd.read_csv(DATA_PATH, low_memory=False)

df["date"] = pd.to_datetime(df["date"])
df["grid_id"] = df["grid_id"].astype("category")

target = "fire_today"

# ----------------------------------------------------------------------
# Build BallTree for spatial operations
# ----------------------------------------------------------------------
print("Building BallTree for spatial neighbors...")

grid_meta = df[["grid_id", "lat", "lon"]].drop_duplicates().reset_index(drop=True)
coords_rad = np.radians(grid_meta[["lat", "lon"]].values)

tree = BallTree(coords_rad, metric="haversine")

# ----------------------------------------------------------------------
# 8-Nearest Neighbors (local fire propagation)
# ----------------------------------------------------------------------
print("Computing 8-nearest neighbors...")

K = 8 + 1   # +1 to include itself
dist_nn, idx_nn = tree.query(coords_rad, k=K)

# Map: grid_id → list of neighbor grid_ids
neighbor_map = {}
for i, g in enumerate(grid_meta["grid_id"]):
    neighbor_ids = grid_meta.iloc[idx_nn[i]]["grid_id"].tolist()
    neighbor_ids = [n for n in neighbor_ids if n != g]  # remove itself
    neighbor_map[g] = neighbor_ids

# ----------------------------------------------------------------------
# Compute neighbor fire lag features
# ----------------------------------------------------------------------
print("Computing neighbor fire lag features...")

def add_neighbor_lag(df, lag_col):
    out_col = "neighbor_" + lag_col
    df[out_col] = 0

    for date, group in df.groupby("date"):
        # fast lookup table: grid_id -> lag value
        lag_map = dict(zip(group["grid_id"], group[lag_col]))

        values = []
        for gid in group["grid_id"]:
            neigh = neighbor_map[gid]
            total = sum(lag_map.get(n, 0) for n in neigh)
            values.append(total)

        df.loc[group.index, out_col] = values


lag_features = ["fires_last_1d", "fires_last_3d", "fires_last_7d", "fires_last_14d"]
for lf in lag_features:
    add_neighbor_lag(df, lf)

# ----------------------------------------------------------------------
# Radius-Based (≤ 10 km) Spatial Smoothing
# ----------------------------------------------------------------------
print("Computing radius-based smoothing features (10 km)...")

radius_km = 10
radius_rad = radius_km / 6371.0  # Earth radius

# Query radius for all grid points
radius_neighbors = tree.query_radius(coords_rad, r=radius_rad)

# Smoothing function
def add_radius_mean(df, col, new_col):
    df[new_col] = 0
    for date, group in df.groupby("date"):
        # Build lookup for fast access
        value_map = dict(zip(group["grid_id"], group[col]))

        results = []
        for i, gid in enumerate(group["grid_id"]):
            neigh_idx = radius_neighbors[grid_meta.index[grid_meta["grid_id"] == gid][0]]
            neigh_ids = grid_meta.iloc[neigh_idx]["grid_id"]

            vals = [value_map.get(n, np.nan) for n in neigh_ids]
            results.append(np.nanmean(vals))

        df.loc[group.index, new_col] = results


radius_cols = {
    "T2M": "T2M_spatial_mean_10km",
    "RH2M": "RH2M_spatial_mean_10km",
    "WS10M": "WS10M_spatial_mean_10km",
    "ndvi": "ndvi_spatial_mean_10km"
}

# Add VPD proxy smoothing too
df["vpd_proxy"] = (1 - df["RH2M"] / 100) * df["T2M"]
radius_cols["vpd_proxy"] = "vpd_spatial_mean_10km"

for base, newname in radius_cols.items():
    add_radius_mean(df, base, newname)

# ----------------------------------------------------------------------
# Nonlinear Feature Engineering
# ----------------------------------------------------------------------
print("Engineering nonlinear features...")

df["hot_windy"] = df["T2M"] * df["WS10M"]
df["dry_windy"] = (100 - df["RH2M"]) * df["WS10M"]
df["fuel_moisture_proxy"] = df["ndvi"] * df["RH2M"]

optional_lags = ["fires_last_1d", "fires_last_3d"]
available_lags = [c for c in optional_lags if c in df.columns]

# Auto-detect NLCD columns
nlcd_cols = [c for c in df.columns if c.startswith("pct_")]

# ----------------------------------------------------------------------
# Final Feature List
# ----------------------------------------------------------------------
feature_cols = [
    # Weather
    "T2M", "RH2M", "WS10M", "PS", "PRECTOTCORR",
    "T2M_3d_mean", "RH2M_3d_min", "WS10M_3d_max", "PRECTOT_7d_sum",

    # Vegetation
    "ndvi", "ndvi_7d_mean",

    # Fires
    "fires_last_7d", "fires_last_14d"
] + available_lags + nlcd_cols + [

    # Nonlinear
    "vpd_proxy", "hot_windy", "dry_windy", "fuel_moisture_proxy",

    # Neighbor fire features
    "neighbor_fires_last_1d",
    "neighbor_fires_last_3d",
    "neighbor_fires_last_7d",
    "neighbor_fires_last_14d",

    # Radius-smoothed features
    "T2M_spatial_mean_10km",
    "RH2M_spatial_mean_10km",
    "WS10M_spatial_mean_10km",
    "ndvi_spatial_mean_10km",
    "vpd_spatial_mean_10km"
]

print(f"Using {len(feature_cols)} total features.")

# ----------------------------------------------------------------------
# Train / Validation Split (time-based)
# ----------------------------------------------------------------------
train = df[df["date"] < "2023-01-01"]
val   = df[df["date"] >= "2023-01-01"]

X_train = train[feature_cols]
y_train = train[target]

X_val = val[feature_cols]
y_val = val[target]

# ----------------------------------------------------------------------
# Class Weight for Imbalance
# ----------------------------------------------------------------------
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale = neg / max(pos, 1)

print(f"Using class_weight = {scale:.2f}")

# ----------------------------------------------------------------------
# Train CatBoost Model
# ----------------------------------------------------------------------
model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    learning_rate=0.05,
    depth=10,
    iterations=1500,
    l2_leaf_reg=3,
    random_state=42,
    thread_count=-1,
    class_weights=[1, scale],
    verbose=200
)

print("Training CatBoost model (with spatial features)...")
model.fit(X_train, y_train, eval_set=(X_val, y_val))


# ----------------------------------------------------------------------
# Evaluate Model
# ----------------------------------------------------------------------
y_val_proba = model.predict_proba(X_val)[:, 1]

auc_roc = roc_auc_score(y_val, y_val_proba)
auc_pr = average_precision_score(y_val, y_val_proba)

print(f"\nROC-AUC: {auc_roc:.4f}")
print(f"AUC-PR:  {auc_pr:.4f}")

# ----------------------------------------------------------------------
# Optimal Threshold (Max F1)
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
model.save_model(MODEL_PATH)

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
    "class_weight": float(scale)
}

with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print("\nModel, threshold, and metadata saved successfully.")
