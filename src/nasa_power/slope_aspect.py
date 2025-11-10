# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pyproj import Transformer

# -------- CONFIG (edit paths if needed) --------
input_csv  = "srtm30plus_california.csv"                      # headers row 1, units row 2
output_csv = "elev_1km_ca_with_slope_aspect.csv"
crs_proj   = "EPSG:5070"                            # projected CRS in meters (CONUS Albers)
round_xy   = -3                                      # 1km grid 
# ------------------------------------------------

# Load, skipping the units row (second physical row)
df_raw = pd.read_csv(input_csv, low_memory=False, skiprows=[1])

# Normalize headers and locate required columns (case-insensitive)
df_raw.columns = [c.strip() for c in df_raw.columns]
lower_map = {c.lower(): c for c in df_raw.columns}
need = ["longitude", "latitude", "elev"]
missing = [c for c in need if c not in lower_map]
if missing:
    raise ValueError(f"Missing required columns (case-insensitive): {missing}. "
                     f"Found: {list(df_raw.columns)}")

lon_col = lower_map["longitude"]
lat_col = lower_map["latitude"]
elev_col = lower_map["elev"]

# Clean and coerce to numeric; keep unparsable as NaN so we preserve rows
for c in (lon_col, lat_col, elev_col):
    df_raw[c] = (
        df_raw[c].astype(str)
        .str.replace(r"[A-Za-z_ ]+", "", regex=True)  # drop 'degrees east' etc.
        .str.replace(",", "", regex=False)
    )
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

# Prepare output columns (will fill where computable)
df_raw["slope_deg"] = np.nan
df_raw["aspect_deg"] = np.nan

# Keep only rows with valid lon/lat/elev for computation; others remain in output with NaNs
valid_mask = df_raw[[lon_col, lat_col, elev_col]].notna().all(axis=1)
df = df_raw.loc[valid_mask].copy()

if df.empty:
    df_raw.to_csv(output_csv, index=False)
    print(f"No valid rows. Wrote {len(df_raw):,} rows to {output_csv} with NaN slope/aspect.")
else:
    # Project lon/lat -> meters to get true dx,dy
    transformer = Transformer.from_crs("EPSG:4326", crs_proj, always_xy=True)
    x, y = transformer.transform(df[lon_col].to_numpy(float), df[lat_col].to_numpy(float))

    # Round to ~1km grid; then de-duplicate points falling in the same cell
    xr = np.round(x, round_xy)
    yr = np.round(y, round_xy)

    tmp = pd.DataFrame({
        "xr": xr,
        "yr": yr,
        "x": x,
        "y": y,
        "z": df[elev_col].to_numpy(float)
    })

    # If multiple samples land in the same rounded cell, average them
    tmp = tmp.groupby(["xr", "yr"], as_index=False).agg(
        x=("x", "mean"),
        y=("y", "mean"),
        z=("z", "mean")
    )

    # ---- Compute finite-difference gradients without forming a dense grid ----
    # p = dZ/dx along constant y (row-wise)
    tmp = tmp.sort_values(["yr", "xr"], kind="mergesort")  # stable
    g_row = tmp.groupby("yr", sort=False)

    # neighbor values along x within each y-row
    x_prev = g_row["x"].shift(1)
    x_next = g_row["x"].shift(-1)
    z_prev = g_row["z"].shift(1)
    z_next = g_row["z"].shift(-1)

    # central difference where possible; otherwise one-sided
    p_cent = (z_next - z_prev) / (x_next - x_prev)
    p_fwd  = (g_row["z"].transform("shift", -1) - g_row["z"].transform("shift", 0)) / \
             (g_row["x"].transform("shift", -1) - g_row["x"].transform("shift", 0))
    p_bwd  = (g_row["z"].transform("shift", 0) - g_row["z"].transform("shift", 1)) / \
             (g_row["x"].transform("shift", 0) - g_row["x"].transform("shift", 1))
    p = p_cent.where(p_cent.notna(), p_fwd).where(p_cent.notna() | p_fwd.notna(), p_bwd)

    # q = dZ/dy along constant x (column-wise)
    tmp = tmp.sort_values(["xr", "yr"], kind="mergesort")
    g_col = tmp.groupby("xr", sort=False)

    y_prev = g_col["y"].shift(1)
    y_next = g_col["y"].shift(-1)
    z_prev_c = g_col["z"].shift(1)
    z_next_c = g_col["z"].shift(-1)

    q_cent = (z_next_c - z_prev_c) / (y_next - y_prev)
    q_fwd  = (g_col["z"].transform("shift", -1) - g_col["z"].transform("shift", 0)) / \
             (g_col["y"].transform("shift", -1) - g_col["y"].transform("shift", 0))
    q_bwd  = (g_col["z"].transform("shift", 0) - g_col["z"].transform("shift", 1)) / \
             (g_col["y"].transform("shift", 0) - g_col["y"].transform("shift", 1))
    q = q_cent.where(q_cent.notna(), q_fwd).where(q_cent.notna() | q_fwd.notna(), q_bwd)

    # Collect gradients on the de-duplicated grid
    tmp = tmp.sort_values(["yr", "xr"], kind="mergesort")  # align with p order used first
    tmp["p"] = p.values
    # Recompute order to align q
    tmp = tmp.sort_values(["xr", "yr"], kind="mergesort")
    tmp["q"] = q.values
    # Restore a consistent order for downstream merge
    tmp = tmp.sort_values(["yr", "xr"], kind="mergesort")

    # ---- Slope & Aspect ----
    slope_rad = np.arctan(np.sqrt(tmp["p"]**2 + tmp["q"]**2))
    slope_deg = np.degrees(slope_rad)

    aspect_rad = np.arctan2(tmp["p"], -tmp["q"])  # 0 = North, clockwise
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360.0, aspect_deg)
    aspect_deg = np.where((tmp["p"] == 0) & (tmp["q"] == 0), np.nan, aspect_deg)

    tmp["slope_deg"] = slope_deg
    tmp["aspect_deg"] = aspect_deg

    # ---- Map results back to the original rows (by rounded xr/yr) ----
    # Build mapping for the valid subset
    key_valid = pd.DataFrame({"xr": np.round(x, round_xy),
                              "yr": np.round(y, round_xy)}, index=df.index)

    out = key_valid.merge(
        tmp[["xr", "yr", "slope_deg", "aspect_deg"]],
        on=["xr", "yr"], how="left"
    ).set_index(df.index)

    # Fill into df_raw for rows that had valid lon/lat/elev
    df_raw.loc[out.index, "slope_deg"]  = out["slope_deg"].values
    df_raw.loc[out.index, "aspect_deg"] = out["aspect_deg"].values

    # Write
    df_raw.to_csv(output_csv, index=False)
    computed = int(np.isfinite(out["slope_deg"]).sum())
    print(f"Wrote {len(df_raw):,} rows to {output_csv} "
          f"(computed slope/aspect for ~{computed:,} rows).")