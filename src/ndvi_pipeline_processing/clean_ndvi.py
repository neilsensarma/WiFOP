#!/usr/bin/env python
"""
clean_ndvi.py

Clean and feature-engineer NDVI data for wildfire prediction
from per-year CSVs: ndvi_2020_6_9.csv, ndvi_2021_6_9.csv, etc.

Pipeline:
1. Load all ndvi_????_6_9.csv files from an input folder.
2. Assume NDVI has already been correctly scaled and masked using
   product metadata (valid_range, _FillValue).
   => We do NOT do any extra range-based clipping here.
3. Small-gap interpolation per pixel into 'ndvi_interp'
   (fills up to 'limit_steps' consecutive missing time steps).git pull --rebase origin main

4. Add missingness features:
   - ndvi_missing_flag
   - ndvi_missing_streak (how many consecutive days a pixel has missing NDVI)
5. Add rolling vegetation features (per pixel) based on ndvi_interp:
   - ndvi_7d_mean  (7-day rolling mean)
   - ndvi_21d_mean (21-day rolling mean)
   - ndvi_drop = ndvi_7d_mean - ndvi_21d_mean
6. Save:
   - one merged CSV (e.g. cleaned_ndvi_2020_2023.csv)
   - one CSV per year: cleaned_ndvi_YYYY.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Load all yearly NDVI CSVs
# ---------------------------------------------------------------------

def load_all_ndvi(csv_folder: Path) -> pd.DataFrame:
    """
    Load all ndvi_????_6_9.csv files in csv_folder, concatenate into one DataFrame.
    Expected columns: date, longitude, latitude, ndvi
    """
    files = sorted(csv_folder.glob("ndvi_????_6_9.csv"))
    if not files:
        raise FileNotFoundError(f"No ndvi_????_6_9.csv files found in {csv_folder}")

    print("Loading NDVI CSV files:")
    for f in files:
        print(f"  - {f.name}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Basic sanity: ensure required columns
        required = {"date", "longitude", "latitude", "ndvi"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{f} is missing required columns: {missing}")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Parse date and enforce types
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all["longitude"] = df_all["longitude"].astype(float)
    df_all["latitude"] = df_all["latitude"].astype(float)
    df_all["ndvi"] = df_all["ndvi"].astype(float)

    print(f"Total rows loaded: {len(df_all)}")
    print(f"NaNs in ndvi after load: {df_all['ndvi'].isna().sum()}")
    return df_all


# ---------------------------------------------------------------------
# 2. Small-gap interpolation per pixel
# ---------------------------------------------------------------------

def interpolate_small_gaps(df: pd.DataFrame, limit_steps: int = 2) -> pd.DataFrame:
    """
    For each (latitude, longitude) pixel, perform time-based interpolation
    on 'ndvi' to fill small gaps, up to 'limit_steps' consecutive missing
    time steps.

    Result is stored in a new column 'ndvi_interp'.
    Larger gaps remain NaN to avoid fabricating vegetation.

    Note: We assume daily data; 'limit_steps=2' ≈ up to 2 consecutive days.
    """

    df = df.sort_values(["latitude", "longitude", "date"])

    def _interp_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").set_index("date")
        ndvi_interp = g["ndvi"].interpolate(
            method="time",
            limit=limit_steps,
            limit_direction="both"
        )
        g["ndvi_interp"] = ndvi_interp
        return g.reset_index()

    df = df.groupby(["latitude", "longitude"], group_keys=False).apply(_interp_group)

    n_orig_nan = df["ndvi"].isna().sum()
    n_interp_nan = df["ndvi_interp"].isna().sum()
    print(f"Interpolation: NaNs in ndvi={n_orig_nan}, NaNs in ndvi_interp={n_interp_nan}")
    return df


# ---------------------------------------------------------------------
# 3. Missingness features
# ---------------------------------------------------------------------

def add_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      - ndvi_missing_flag: 1 if ndvi is NaN, else 0
      - ndvi_missing_streak: length of current consecutive missing run (per pixel)
    """
    df["ndvi_missing_flag"] = df["ndvi"].isna().astype("int8")

    def _streak_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date")
        flag = g["ndvi_missing_flag"]

        # Identify runs where missing/valid state changes
        run_id = (flag != flag.shift()).cumsum()
        # Cumulative count within each run
        streak_raw = flag.groupby(run_id).cumsum()
        # Zero out streak where not missing
        g["ndvi_missing_streak"] = streak_raw.where(flag == 1, 0).astype("int32")
        return g

    df = df.groupby(["latitude", "longitude"], group_keys=False).apply(_streak_group)

    print("Missingness features added: ndvi_missing_flag, ndvi_missing_streak")
    return df


# ---------------------------------------------------------------------
# 4. Rolling vegetation features per pixel
# ---------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (latitude, longitude) pixel, compute rolling vegetation features
    using ndvi_interp:

      - ndvi_7d_mean:  7-day rolling mean
      - ndvi_21d_mean: 21-day rolling mean
      - ndvi_drop:     ndvi_7d_mean - ndvi_21d_mean

    Uses time-based rolling windows ("7D", "21D"), so irregular gaps are
    handled in calendar time.
    """

    def _rolling_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").set_index("date")
        nd = g["ndvi_interp"]

        g["ndvi_7d_mean"] = nd.rolling("7D", min_periods=3).mean()
        g["ndvi_21d_mean"] = nd.rolling("21D", min_periods=7).mean()
        g["ndvi_drop"] = g["ndvi_7d_mean"] - g["ndvi_21d_mean"]

        return g.reset_index()

    df = df.groupby(["latitude", "longitude"], group_keys=False).apply(_rolling_group)

    print("Rolling features added: ndvi_7d_mean, ndvi_21d_mean, ndvi_drop")
    return df


# ---------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------

def run_pipeline(input_dir: Path, output_csv: Path):
    print(f"Input folder: {input_dir}")
    print(f"Output (merged) file : {output_csv}")

    # 1) load all yearly NDVI
    df = load_all_ndvi(input_dir)

    # 2) interpolate small gaps (keep ndvi as original, create ndvi_interp)
    df = interpolate_small_gaps(df, limit_steps=2)

    # 3) missingness features based on original ndvi
    df = add_missingness_features(df)

    # 4) rolling features based on ndvi_interp
    df = add_rolling_features(df)

    # Final sort for tidiness
    df = df.sort_values(["date", "latitude", "longitude"]).reset_index(drop=True)

    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 5) Save merged file (all years)
    df.to_csv(output_csv, index=False)
    print(f"✔ Saved merged cleaned NDVI features to: {output_csv}")
    print(f"Final rows (merged): {len(df)}")
    print("Columns:", list(df.columns))

    # 6) Save per-year files: cleaned_ndvi_YYYY.csv in same folder as merged
    years = sorted(df["date"].dt.year.unique())
    out_dir = output_csv.parent
    print("\nSaving per-year cleaned NDVI files:")
    for y in years:
        df_y = df[df["date"].dt.year == y].copy()
        out_y = out_dir / f"cleaned_ndvi_{y}.csv"
        df_y.to_csv(out_y, index=False)
        print(f"  - Year {y}: {out_y} (rows={len(df_y)})")

    print("\nAll done.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean NDVI CSVs and create wildfire-ready NDVI features."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=".",
        help="Folder containing ndvi_????_6_9.csv files (default: current directory)."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="cleaned_ndvi_2020_2023.csv",
        help="Path to merged output cleaned NDVI CSV "
             "(default: cleaned_ndvi_2020_2023.csv). "
             "Per-year files cleaned_ndvi_YYYY.csv will be written "
             "to the same folder."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    run_pipeline(input_dir, output_csv)
