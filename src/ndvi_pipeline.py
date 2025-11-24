import os
import re
import json
import time
import pathlib
from datetime import datetime, date
from typing import List, Dict

import requests
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# 0. Small utilities
# ============================================================

def ensure_dir(path: str):
    """Create directory (and parents) if it does not exist."""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def robust_get(url: str, stream: bool = True, timeout: int = 60):
    """HTTP GET with a few retries."""
    tries = 3
    for i in range(tries):
        try:
            r = requests.get(url, stream=stream, timeout=timeout)
            if r.status_code == 200:
                return r
            print(f"[warn] HTTP {r.status_code} for {url}")
        except Exception as e:
            print(f"[warn] {e} ({i+1}/{tries})")
        time.sleep(2)
    raise RuntimeError(f"Failed to GET after {tries} tries: {url}")


def collect_nc_links(index_html: str, base_url: str) -> List[str]:
    """Collect absolute URLs of all .nc files in a directory index page."""
    soup = BeautifulSoup(index_html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.lower().endswith(".nc"):
            continue
        if href.startswith("http"):
            url = href
        else:
            url = base_url.rstrip("/") + "/" + href.lstrip("/")
        links.append(url)
    return links


def parse_settings(settings_path: str) -> Dict:
    """
    Load setting.json and parse:
      - time_period.start_date / end_date  (as date objects)
      - spatial_extent (lon/lat bounding box)
      - input_files.ndvi (output file for final NDVI table)
    """
    with open(settings_path, "r") as f:
        cfg = json.load(f)

    # Dates in setting.json are expected to be "YYYY-MM-DD"
    start = datetime.strptime(cfg["time_period"]["start_date"], "%Y-%m-%d").date()
    end = datetime.strptime(cfg["time_period"]["end_date"], "%Y-%m-%d").date()

    extent = cfg["spatial_extent"]
    lon_min = extent["min_longitude"]
    lon_max = extent["max_longitude"]
    lat_min = extent["min_latitude"]
    lat_max = extent["max_latitude"]

    ndvi_out_path = cfg["input_files"]["ndvi"]

    return {
        "start_date": start,
        "end_date": end,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "ndvi_out_path": ndvi_out_path,
    }


# ============================================================
# 1. Find & download VIIRS NDVI v001 NetCDF files
# ============================================================

# Only match v001 NDVI files; capture group is the date: YYYYMMDD
VIIRS_PATTERN = re.compile(
    r"VIIRS-Land_v001_JP113C1_NOAA-20_(\d{8})_.*\.nc$"
)


def find_viirs_urls_for_year(
    base_url: str,
    start_date: date,
    end_date: date,
) -> List[str]:
    """
    From a yearly index page (base_url), find all v001 NOAA-20 JP113C1 NDVI
    files whose date lies between start_date and end_date (inclusive).
    """
    print(f"Fetching index: {base_url}")
    html = robust_get(base_url, stream=False).text
    links = collect_nc_links(html, base_url)

    urls = []
    for u in links:
        m = VIIRS_PATTERN.search(u)
        if not m:
            continue

        yyyymmdd = m.group(1)
        file_date = datetime.strptime(yyyymmdd, "%Y%m%d").date()

        if start_date <= file_date <= end_date:
            urls.append(u)

    urls_sorted = sorted(urls)
    print(f"  Found {len(urls_sorted)} VIIRS v001 files in date range.")
    return urls_sorted


def download_file(url: str, out_dir: str) -> str:
    """Download a single NetCDF file to out_dir if not already present."""
    ensure_dir(out_dir)
    fname = url.rstrip("/").split("/")[-1]
    out_path = os.path.join(out_dir, fname)
    tmp_path = out_path + ".part"

    if os.path.exists(out_path):
        print(f"✔ exists: {fname}")
        return out_path

    r = robust_get(url, stream=True)
    total = int(r.headers.get("Content-Length", "0")) or None
    print(f"↓ downloading {fname} ...")

    downloaded = 0
    chunk_size = 1024 * 256

    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r   {downloaded/1e6:6.2f} MB ({pct:5.1f}%)", end="")
        print()  # newline

    os.replace(tmp_path, out_path)
    print(f"✔ saved {fname}")
    return out_path


# ============================================================
# 2. Read NetCDF, scale NDVI, clip to bbox, to table
# ============================================================

def open_no_decode(path: str) -> xr.Dataset:
    """Open NetCDF file without automatic scale/offset decoding."""
    return xr.open_dataset(path, decode_cf=False, engine="netcdf4")


def get_var(ds: xr.Dataset, candidates: List[str]) -> xr.DataArray:
    """Return the first variable in candidates that exists in ds."""
    for name in candidates:
        if name in ds.variables:
            return ds[name]
    raise KeyError(f"None of {candidates} found in dataset variables: {list(ds.variables)}")


def get_coord(ds: xr.Dataset, candidates: List[str]) -> str:
    """Return the first coordinate name in candidates that exists in ds."""
    for name in candidates:
        if name in ds.coords:
            return name
    raise KeyError(f"None of {candidates} found in dataset coords: {list(ds.coords)}")


def parse_date_from_attrs_or_name(ds: xr.Dataset, path_str: str) -> str:
    """
    Get date string (YYYY-MM-DD) from:
      1. 'time_coverage_start' attribute if present, or
      2. filename pattern '_YYYYMMDD_'.
    """
    if "time_coverage_start" in ds.attrs:
        v = ds.attrs["time_coverage_start"]
        if isinstance(v, str) and len(v) >= 10:
            return v[:10]

    m = re.search(r"_([12]\d{7})_", pathlib.Path(path_str).name)
    if m:
        ymd = m.group(1)
        return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"

    return "unknown-date"


def scale_ndvi(da: xr.DataArray) -> xr.DataArray:
    """Apply NDVI scale_factor and add_offset manually."""
    scale = da.attrs.get("scale_factor", 1.0)
    offset = da.attrs.get("add_offset", 0.0)

    out = da.astype("float32", copy=False)
    out = out * np.float32(scale)
    if offset:
        out = out + np.float32(offset)
    return out


def bbox_clip_and_table(
    nc_path: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> pd.DataFrame:
    """
    Open one VIIRS NetCDF file, clip NDVI by bounding box, and return a
    tidy table with columns: date, longitude, latitude, ndvi.
    """
    ds = open_no_decode(nc_path)

    # NDVI variable (e.g. "NDVI")
    da_raw = get_var(ds, ["NDVI", "ndvi"])

    # Apply scale factor / offset
    ndvi = scale_ndvi(da_raw).squeeze(drop=True)

    # Coordinate names (handle latitude vs lat, longitude vs lon)
    lat_name = get_coord(ndvi.to_dataset(name="tmp"), ["latitude", "lat"])
    lon_name = get_coord(ndvi.to_dataset(name="tmp"), ["longitude", "lon"])

    lat = ndvi.coords[lat_name]
    lon = ndvi.coords[lon_name]

    # Determine slice direction for latitude
    if lat[0] < lat[-1]:
        lat_slice = slice(lat_min, lat_max)
    else:
        lat_slice = slice(lat_max, lat_min)

    # Determine slice direction for longitude
    if lon[0] < lon[-1]:
        lon_slice = slice(lon_min, lon_max)
    else:
        lon_slice = slice(lon_max, lon_min)

    ndvi_sub = ndvi.sel({lat_name: lat_slice, lon_name: lon_slice})

    # To DataFrame
    df = ndvi_sub.to_dataframe(name="ndvi").reset_index()

    # Replace NOAA fill values (e.g. -0.9999 after scaling) with NaN
    # Anything below -0.9 is considered invalid NDVI here.
    fill_threshold = -0.9
    df["ndvi"] = df["ndvi"].where(df["ndvi"] > fill_threshold, np.nan)

    df = df.replace([np.inf, -np.inf], np.nan)

    # Add date column
    date_str = parse_date_from_attrs_or_name(ds, nc_path)
    df.insert(0, "date", date_str)

    # Rename coords to standard names if needed
    df = df.rename(columns={lon_name: "longitude", lat_name: "latitude"})

    df = df[["date", "longitude", "latitude", "ndvi"]]
    return df


# ============================================================
# 3. Main integrated pipeline + gap filling
# ============================================================

def run_ndvi_pipeline(settings_path: str):
    """
    Full automated NDVI data collection pipeline:

    1. Read date range, spatial extent, and output path from setting.json.
    2. For each year in the date range, query NOAA NCEI index for VIIRS v001 NDVI files.
    3. Download each daily NetCDF file (if not already present).
    4. For each file, clip NDVI to the bounding box and convert to a tidy table.
    5. Report missing (fill) values.
    6. Fill gaps using time interpolation + local spatial interpolation.
    7. Save the filled NDVI table to the ndvi output path (CSV or Parquet),
       with date column formatted as YYYY-MM-DD.
    """
    cfg = parse_settings(settings_path)
    start_date = cfg["start_date"]
    end_date = cfg["end_date"]
    lon_min = cfg["lon_min"]
    lon_max = cfg["lon_max"]
    lat_min = cfg["lat_min"]
    lat_max = cfg["lat_max"]
    ndvi_out_path = pathlib.Path(cfg["ndvi_out_path"])

    # Directory for raw NetCDF files (next to the final output)
    nc_dir = ndvi_out_path.parent / "viirs_nc"
    ensure_dir(nc_dir.as_posix())

    # 1) Find all VIIRS file URLs over all years in the range
    all_urls: List[str] = []
    for year in range(start_date.year, end_date.year + 1):
        year_start = max(start_date, date(year, 1, 1))
        year_end = min(end_date, date(year, 12, 31))

        base_url = (
            "https://www.ncei.noaa.gov/data/"
            "land-normalized-difference-vegetation-index/access/"
            f"{year}/"
        )

        urls_year = find_viirs_urls_for_year(
            base_url,
            start_date=year_start,
            end_date=year_end,
        )
        all_urls.extend(urls_year)

    if not all_urls:
        print("⚠️ No matching VIIRS v001 files found for the requested range.")
        return

    # 2) Download all files
    nc_paths = []
    for u in all_urls:
        p = download_file(u, str(nc_dir))
        nc_paths.append(p)

    # 3) Process each file → clip to bbox → tidy table
    all_days = []
    for p in sorted(nc_paths):
        name = pathlib.Path(p).name
        print(f"Processing NDVI: {name}")
        try:
            df_day = bbox_clip_and_table(
                p,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
            )
            all_days.append(df_day)
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")

    if not all_days:
        print("⚠️ No NDVI data extracted.")
        return

    # 4) Concatenate all days
    df_all = pd.concat(all_days, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"])

    # Count missing before interpolation
    n_total = len(df_all)
    n_missing_before = df_all["ndvi"].isna().sum()
    pct_missing_before = n_missing_before / n_total * 100
    print(
        f"NDVI missing (fill) BEFORE interpolation: "
        f"{n_missing_before} / {n_total} ({pct_missing_before:.2f}%)"
    )

    # 5) Convert to xarray for interpolation
    ds = df_all.set_index(["date", "latitude", "longitude"]).to_xarray()
    nd = ds["ndvi"]

    # Mask of pixels that ever have valid NDVI (to avoid filling oceans, etc.)
    valid_any = nd.notnull().any(dim="date")

    # (a) Time interpolation along date (per pixel)
    nd_time = nd.interpolate_na(dim="date", method="linear")

    # (b) Local spatial interpolation (nearest neighbor in latitude, then longitude)
    nd_space = nd_time.interpolate_na(dim="latitude", method="nearest")
    nd_space = nd_space.interpolate_na(dim="longitude", method="nearest")

    # (c) Reapply mask so pixels that are NEVER valid stay NaN
    nd_filled = nd_space.where(valid_any)

    # Back to DataFrame
    df_filled = nd_filled.to_dataframe().reset_index()

    # Force date format YYYY-MM-DD
    df_filled["date"] = pd.to_datetime(df_filled["date"]).dt.strftime("%Y-%m-%d")

    n_missing_after = df_filled["ndvi"].isna().sum()
    pct_missing_after = n_missing_after / len(df_filled) * 100
    print(
        f"NDVI missing AFTER interpolation: "
        f"{n_missing_after} / {len(df_filled)} ({pct_missing_after:.2f}%)"
    )

    # 6) Save
    ensure_dir(ndvi_out_path.parent.as_posix())
    if ndvi_out_path.suffix.lower() == ".parquet":
        df_filled.to_parquet(ndvi_out_path, index=False)
        print(f"✔ wrote {ndvi_out_path} (Parquet), rows={len(df_filled)}")
    else:
        df_filled.to_csv(ndvi_out_path, index=False, na_rep="NaN")
        print(f"✔ wrote {ndvi_out_path} (CSV), rows={len(df_filled)})")


if __name__ == "__main__":
    # Use setting.json in the SAME FOLDER as this script
    here = pathlib.Path(__file__).resolve().parent
    settings_path = here / "setting.json"
    run_ndvi_pipeline(str(settings_path))
