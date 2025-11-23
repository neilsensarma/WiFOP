"""
clean_panel.py

Prepares the training-ready wildfire panel:
- Loads the July 2023 raw panel
- Filters grid cells with NASA + NLCD coverage
- Interpolates NDVI (per grid)
- Adds rolling window features (3d / 7d)
- Adds fire history features
- Outputs a clean ML-ready dataset
"""

import pandas as pd
import numpy as np

INPUT_FILE  = "../data/processed/california_5km_daily_panel_2023-07.csv"
OUTPUT_FILE = "../data/processed/california_5km_training_panel_2023-07_clean.csv"

print("Loading panel...")
df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

# ----------------------------------------------------------
# 1. FILTER VALID GRID CELLS (Option C: NASA + NLCD required)
# ----------------------------------------------------------

print("Filtering grid cells with NASA + NLCD coverage...")

# NASA coverage check
has_nasa = df.groupby("grid_id")["PRECTOTCORR"].apply(lambda s: s.notna().any())

# NLCD check
has_nlcd = df.groupby("grid_id")["pct_11"].apply(lambda s: s.notna().any())

valid_ids = has_nasa[has_nasa].index.intersection(has_nlcd[has_nlcd].index)
df = df[df["grid_id"].isin(valid_ids)]

print(f"Remaining grid cells: {len(valid_ids)}")
print(f"Remaining rows: {df.shape[0]}")

# ----------------------------------------------------------
# 2. NDVI INTERPOLATION (PER GRID)
# ----------------------------------------------------------

print("Interpolating NDVI per grid_id...")

df = df.sort_values(["grid_id", "date"])
df["ndvi"] = df.groupby("grid_id")["ndvi"].transform(
    lambda x: x.interpolate("linear").bfill().ffill()
)

# ----------------------------------------------------------
# 3. ROLLING WEATHER FEATURES
# ----------------------------------------------------------

def add_rolling_feature(df, col, window, agg, new_col):
    df[new_col] = (
        df.groupby("grid_id")[col]
          .transform(lambda s: getattr(s.rolling(window), agg)())
    )

print("Adding rolling weather features...")

add_rolling_feature(df, "T2M",          3, "mean", "T2M_3d_mean")
add_rolling_feature(df, "RH2M",         3, "min",  "RH2M_3d_min")
add_rolling_feature(df, "WS10M",        3, "max",  "WS10M_3d_max")
add_rolling_feature(df, "PRECTOTCORR",  7, "sum",  "PRECTOT_7d_sum")

# NDVI rolling
add_rolling_feature(df, "ndvi",         7, "mean", "ndvi_7d_mean")

# ----------------------------------------------------------
# 4. FIRE HISTORY FEATURES
# ----------------------------------------------------------

print("Adding fire history features...")

df["fires_last_7d"]  = df.groupby("grid_id")["fire_today"].transform(lambda s: s.rolling(7).sum())
df["fires_last_14d"] = df.groupby("grid_id")["fire_today"].transform(lambda s: s.rolling(14).sum())

df["fires_last_7d"] = df["fires_last_7d"].fillna(0)
df["fires_last_14d"] = df["fires_last_14d"].fillna(0)

# ----------------------------------------------------------
# 5. FINAL CLEANING
# ----------------------------------------------------------

print("Final cleaning...")

# Remove rows where NASA is missing (essential)
df = df[df["PRECTOTCORR"].notna()]

print("Saving clean panel...")
df.to_csv(OUTPUT_FILE, index=False)

print(f"\n🎉 DONE — Clean training panel saved to:\n{OUTPUT_FILE}")
print(f"Final shape: {df.shape}")
