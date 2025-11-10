# ============================================================
# viirs_ndvi_to_table_polygon.py
# Extracts daily NDVI for California from NOAA-20 VIIRS JP113C1 v001 (July 2023)
# Requires: xarray, netCDF4, pandas, numpy, geopandas, rioxarray, shapely
# Optional: pyarrow/fastparquet for Parquet output
# ============================================================

import os, re, glob
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray as rxr  # activates .rio accessor

# ===========================
# CONFIGURATION
# ===========================
# Folder containing the 31 daily .nc files
IN_FOLDER = "/Users/pangdiyang/Desktop/WiFOP/viirs_daily"

# Pattern for NOAA-20 JP113C1 v001 files in July 2023
GLOB_PATTERN = "VIIRS-Land_v001_JP113C1_NOAA-20_202307*.nc"

# Output folder (same as input)
OUT_FOLDER = IN_FOLDER

# California shapefile path
CA_BOUNDARY_PATH = "/Users/pangdiyang/Desktop/WiFOP/CA_boundary.shp"

# Write Parquet (True) or fallback to CSV
WRITE_PARQUET = True


# ===========================
# HELPER FUNCTIONS
# ===========================
def open_no_decode(path: str) -> xr.Dataset:
    """Open NetCDF file without automatic scale/offset decoding"""
    return xr.open_dataset(path, decode_cf=False, engine="netcdf4")


def parse_date_from_attrs_or_name(ds: xr.Dataset, path_str: str) -> str:
    """Get date string (YYYY-MM-DD) from attributes or filename"""
    for k in ("RangeBeginningDate", "time_coverage_start"):
        if k in ds.attrs:
            v = ds.attrs[k]
            if isinstance(v, str) and len(v) >= 10:
                return v[:10]
    m = re.search(r"_([12]\d{7})_", Path(path_str).name)
    if m:
        ymd = m.group(1)
        return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
    return "unknown-date"


def scale_ndvi(da: xr.DataArray) -> xr.DataArray:
    """Apply scale_factor and add_offset manually"""
    scale  = da.attrs.get("scale_factor", 1.0)
    offset = da.attrs.get("add_offset", 0.0)
    fill   = da.attrs.get("_FillValue", da.attrs.get("missing_value", None))

    out = da
    if fill is not None:
        out = out.where(out != fill)
    out = out.astype("float32", copy=False)
    out = out * np.float32(scale)
    if offset:
        out = out + np.float32(offset)
    return out


def polygon_clip_and_table(nc_path: str, ca_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Clip NDVI to CA polygon and export lon/lat/ndvi as DataFrame"""
    ds = open_no_decode(nc_path)

    # NDVI variable → rename to x/y for rioxarray compatibility
    da = ds["NDVI"].squeeze(drop=True)
    da = da.rename({"longitude": "x", "latitude": "y"})
    da = da.rio.write_crs("EPSG:4326", inplace=True)

    # Ensure polygon CRS matches dataset
    ca = ca_gdf.to_crs(da.rio.crs)

    # Clip NDVI to California boundary
    da_ca = da.rio.clip(ca.geometry, drop=True)

    # Apply scale factor (0.0001)
    ndvi = scale_ndvi(da_ca)

    # Convert to tidy DataFrame
    df = ndvi.to_dataframe(name="ndvi").reset_index().rename(columns={"x": "longitude", "y": "latitude"})
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ndvi"])

    # Add date column
    date_str = parse_date_from_attrs_or_name(ds, nc_path)
    df.insert(0, "date", date_str)

    # Keep essential columns
    df = df[["date", "longitude", "latitude", "ndvi"]]
    return df


def save_table(df: pd.DataFrame, out_base: str):
    """Save DataFrame as Parquet or CSV"""
    for c in df.select_dtypes(include="float64").columns:
        df[c] = df[c].astype("float32")

    try:
        if WRITE_PARQUET:
            df.to_parquet(f"{out_base}.parquet", index=False)
            print(f"✔ wrote {out_base}.parquet  rows={len(df)}")
            return
        raise RuntimeError("Parquet disabled")
    except Exception as e:
        print(f"(Parquet unavailable: {e}) → writing CSV")
        df.to_csv(f"{out_base}.csv", index=False)
        print(f"✔ wrote {out_base}.csv  rows={len(df)}")


# ===========================
# MAIN SCRIPT
# ===========================
if __name__ == "__main__":
    # Load CA polygon once
    ca_gdf = gpd.read_file(CA_BOUNDARY_PATH)

    files = sorted(glob.glob(os.path.join(IN_FOLDER, GLOB_PATTERN)))
    print(f"Found {len(files)} files")

    if not files:
        print("⚠️ No matching NetCDF files found — check IN_FOLDER or GLOB_PATTERN.")
        exit()

    all_chunks = []
    for fp in files:
        name = Path(fp).name
        print(f"Processing: {name}")
        try:
            df = polygon_clip_and_table(fp, ca_gdf)
            out_base = os.path.join(
                OUT_FOLDER,
                Path(fp).with_suffix("").name + "_CApoly_lonlat_ndvi"
            )
            save_table(df, out_base)
            all_chunks.append(df)
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")

    # Combine all days into one big file
    if all_chunks:
        big = pd.concat(all_chunks, ignore_index=True)
        out_base = os.path.join(OUT_FOLDER, "JP113C1_CApoly_202307_all_days_lonlat_ndvi")
        save_table(big, out_base)

    # Convert Parquet -> CSV

    import pandas as pd

