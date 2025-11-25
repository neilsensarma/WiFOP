"""
clean_ndvi.py

Purpose:
    Clean the raw NDVI dataset produced by the VIIRS NDVI pipeline.

What this script does:
    1. Load raw NDVI CSV (columns: date, longitude, latitude, ndvi).
    2. Ensure correct data types (date as datetime, ndvi as float).
    3. Enforce physical NDVI bounds [-1, 1] to remove sensor artifacts.
    4. Add an `ndvi_missing` flag column (1 if NDVI is NaN, else 0).
    5. Optionally drop rows with NaN NDVI (default: keep them).
    6. Save a cleaned CSV next to the original file.

How to run:
    conda activate wifop
    python clean_ndvi.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def clean_ndvi(
    input_path: str,
    output_path: str,
    drop_na_ndvi: bool = False,
) -> None:
    """
    Clean NDVI CSV and save a cleaned version.

    Parameters
    ----------
    input_path : str
        Path to the raw NDVI CSV file (e.g. 'data/raw/ndvi.csv').
    output_path : str
        Path where the cleaned CSV will be written.
    drop_na_ndvi : bool, default False
        If True, rows with NaN NDVI will be dropped.
        If False, NaNs are kept but flagged via ndvi_missing.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"📥 Loading raw NDVI data from: {input_path}")
    df = pd.read_csv(input_path)

    # --------------------------------------------------------------
    # 1. Basic info on the raw NDVI column
    # --------------------------------------------------------------
    print("\n🔎 Initial NDVI column summary:")
    print(df["ndvi"].describe())
    n_nan_raw = df["ndvi"].isna().sum()
    n_total = len(df)
    print(f"\nNaN count in NDVI (raw): {n_nan_raw} "
          f"({n_nan_raw / n_total * 100:.2f}%)")

    # --------------------------------------------------------------
    # 2. Ensure correct data types
    # --------------------------------------------------------------

    # Convert 'date' to datetime if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        print("⚠️ Warning: 'date' column not found in the input data.")

    # Make sure NDVI is float
    df["ndvi"] = pd.to_numeric(df["ndvi"], errors="coerce")

    # --------------------------------------------------------------
    # 3. Enforce physical NDVI bounds [-1, 1]
    #    Values outside this range are artifacts and clipped.
    # --------------------------------------------------------------
    before_min = df["ndvi"].min()
    before_max = df["ndvi"].max()

    df["ndvi"] = df["ndvi"].clip(lower=-1.0, upper=1.0)

    after_min = df["ndvi"].min()
    after_max = df["ndvi"].max()

    print("\n🌡 NDVI range before clipping: "
          f"[{before_min:.4f}, {before_max:.4f}]")
    print(f"🌡 NDVI range after  clipping: [{after_min:.4f}, {after_max:.4f}]")

    # --------------------------------------------------------------
    # 4. Add missingness flag
    #    ndvi_missing = 1 if NDVI is NaN, else 0
    # --------------------------------------------------------------
    df["ndvi_missing"] = df["ndvi"].isna().astype(int)

    n_nan_after_clip = df["ndvi"].isna().sum()
    print("\n🧮 Missing NDVI (NaN) count AFTER clipping: "
          f"{n_nan_after_clip} "
          f"({n_nan_after_clip / len(df) * 100:.2f}%)")

    # --------------------------------------------------------------
    # 5. Optionally drop rows where NDVI is NaN
    #    For now we keep them and let downstream code decide.
    # --------------------------------------------------------------
    if drop_na_ndvi:
        n_before = len(df)
        df = df.dropna(subset=["ndvi"])
        n_after = len(df)
        print(f"\n🧹 Dropped rows with NaN NDVI: {n_before - n_after} rows "
              f"(remaining: {n_after})")
    else:
        print("\nℹ️ Keeping rows with NaN NDVI (use ndvi_missing flag).")

    # --------------------------------------------------------------
    # 6. Final summary and save
    # --------------------------------------------------------------
    print("\n✅ Final NDVI summary (after cleaning):")
    print(df["ndvi"].describe())
    n_nan_final = df["ndvi"].isna().sum()
    print(f"\nFinal NaN count in NDVI: {n_nan_final} "
          f"({n_nan_final / len(df) * 100:.2f}%)")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save cleaned CSV
    df.to_csv(output_path, index=False)
    print(f"\n💾 Cleaned NDVI data written to: {output_path}")
    print(f"   Rows: {len(df)}  |  Columns: {len(df.columns)}")


if __name__ == "__main__":

    input_file = "/Users/pangdiyang/Desktop/data/raw/ndvi.csv"
    output_file = "/Users/pangdiyang/Desktop/data/raw/ndvi_cleaned.csv"

    # For now, we keep NaNs and use ndvi_missing flag.
    drop_na = False

    clean_ndvi(input_file, output_file, drop_na_ndvi=drop_na)
