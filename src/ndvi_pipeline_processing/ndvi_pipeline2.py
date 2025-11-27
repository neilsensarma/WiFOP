import os
import re
import json
import time
import pathlib
from datetime import datetime
from typing import List, Dict

import requests
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# 0. Yearly directory NDVI URLs
# ============================================================

YEAR_DIRS: Dict[int, List[str]] = {
    2020: ["https://www.ncei.noaa.gov/data/land-normalized-difference-vegetation-index/access/2020/"],
    2021: ["https://www.ncei.noaa.gov/data/land-normalized-difference-vegetation-index/access/2021/"],
    2022: ["https://www.ncei.noaa.gov/data/land-normalized-difference-vegetation-index/access/2022/"],
    2023: ["https://www.ncei.noaa.gov/data/land-normalized-difference-vegetation-index/access/2023/"],
}

# Regex to extract date from VIIRS filenames (YYYYMMDD)
VIIRS_PATTERN = re.compile(
    r"VIIRS-Land_v001_.*_(\d{8})_.*\.nc$"
)


# ============================================================
# 1. Small utilities
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
      - spatial_extent (lon/lat bounding box)
      - input_files.ndvi (base path for NDVI outputs)

    We only use bbox + base path directory; time is handled by
    the directory URLs and June–September filtering per year.
    """
    with open(settings_path, "r") as f:
        cfg = json.load(f)

    extent = cfg["spatial_extent"]
    lon_min = extent["min_longitude"]
    lon_max = extent["max_longitude"]
    lat_min = extent["min_latitude"]
    lat_max = extent["max_latitude"]

    ndvi_out_path = cfg["input_files"]["ndvi"]

    return {
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "ndvi_out_path": ndvi_out_path,
    }


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
# 2. NetCDF helpers
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


def parse_date_from_attrs_or_name(ds: xr.Dataset, path_str: str) -> pd.Timestamp:
    """
    Get date from:
      1. 'time_coverage_start' attribute if present, or
      2. filename pattern '_YYYYMMDD_' using VIIRS_PATTERN.

    Returns pandas.Timestamp.
    """
    # Try attribute
    if "time_coverage_start" in ds.attrs:
        v = ds.attrs["time_coverage_start"]
        if isinstance(v, str) and len(v) >= 10:
            try:
                return pd.to_datetime(v[:10])
            except Exception:
                pass

    # Fallback to filename
    m = VIIRS_PATTERN.search(pathlib.Path(path_str).name)
    if m:
        yyyymmdd = m.group(1)
        return pd.to_datetime(yyyymmdd, format="%Y%m%d")

    # Last resort
    return pd.NaT


def scale_ndvi(da: xr.DataArray) -> xr.DataArray:
    """
    Scale raw NDVI integers to real NDVI using CF metadata.

    - Applies scale_factor and add_offset
    - Uses valid_range / valid_min / valid_max to keep only valid raw codes
    - Masks _FillValue / missing_value as NaN
    """
    attrs = da.attrs

    # scale & offset (e.g. scale_factor = 0.0001, add_offset = 0.0)
    scale = float(attrs.get("scale_factor", 1.0))
    offset = float(attrs.get("add_offset", 0.0))

    # work in float32 but keep raw codes for masking
    raw = da.astype("float32", copy=False)

    # start with all valid
    valid_mask = xr.ones_like(raw, dtype=bool)

    # 1) valid_range: typically a 2-element array [min_raw, max_raw]
    if "valid_range" in attrs:
        vr = np.asarray(attrs["valid_range"], dtype="float32")
        if vr.size == 2:
            vmin, vmax = float(vr[0]), float(vr[1])
            valid_mask = valid_mask & (raw >= vmin) & (raw <= vmax)

    # 2) valid_min / valid_max (sometimes provided instead of valid_range)
    if "valid_min" in attrs:
        vmin = float(attrs["valid_min"])
        valid_mask = valid_mask & (raw >= vmin)
    if "valid_max" in attrs:
        vmax = float(attrs["valid_max"])
        valid_mask = valid_mask & (raw <= vmax)

    # 3) explicit fill / missing codes (e.g. -9999)
    for key in ("_FillValue", "missing_value"):
        if key in attrs:
            fv = float(attrs[key])
            valid_mask = valid_mask & (raw != fv)

    # 4) apply scale/offset
    scaled = raw * np.float32(scale)
    if offset:
        scaled = scaled + np.float32(offset)

    # 5) mask invalids as NaN
    scaled = scaled.where(valid_mask, np.nan)

    # (Optional extra safety – uncomment if you want to drop anything outside [-1, 1])
    # scaled = scaled.where((scaled >= -1.0) & (scaled <= 1.0), np.nan)

    return scaled


def inspect_ndvi_metadata(nc_path: str):
    """
    Open one file and print NDVI variable metadata attributes
    (scale_factor, add_offset, valid_range, etc.).
    """
    ds = open_no_decode(nc_path)
    da_raw = get_var(ds, ["NDVI", "ndvi"])
    print("\n================ NDVI METADATA ================")
    print(f"File: {nc_path}")
    print("NDVI attributes:")
    for k, v in da_raw.attrs.items():
        print(f"  {k}: {v}")
    print("Dataset global attributes (a few):")
    for k, v in list(ds.attrs.items())[:10]:
        print(f"  {k}: {v}")
    print("==============================================\n")
    ds.close()


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

    - keeps all NDVI values as-is (no -0.9999 → NaN thresholding)
    - does NOT do any interpolation.
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

    # Replace infinities but keep real numeric values as-is
    df = df.replace([np.inf, -np.inf], np.nan)

    # Add date column
    date_ts = parse_date_from_attrs_or_name(ds, nc_path)
    df.insert(0, "date", date_ts)

    # Rename coords to standard names if needed
    df = df.rename(columns={lon_name: "longitude", lat_name: "latitude"})

    df = df[["date", "longitude", "latitude", "ndvi"]]

    ds.close()
    return df


# ============================================================
# 3. Discover file URLs for each year (from directory indexes)
# ============================================================

def discover_nc_urls_for_year(year: int, dir_urls: List[str]) -> List[str]:
    """
    From one or more directory index URLs, collect all VIIRS v001 NOAA-20
    NDVI .nc files for that year, and filter to June–September.
    """
    all_links: List[str] = []

    for base_url in dir_urls:
        print(f"Discovering .nc files for {year} in: {base_url}")
        html = robust_get(base_url, stream=False).text
        links = collect_nc_links(html, base_url)
        all_links.extend(links)

    urls: List[str] = []
    for u in all_links:
        m = VIIRS_PATTERN.search(u)
        if not m:
            continue

        yyyymmdd = m.group(1)
        dt = datetime.strptime(yyyymmdd, "%Y%m%d")
        if dt.year != year:
            continue
        if 6 <= dt.month <= 9:  # June–September
            urls.append(u)

    urls_sorted = sorted(set(urls))
    print(f"  Found {len(urls_sorted)} VIIRS NDVI files for {year} (June–September).")
    return urls_sorted


# ============================================================
# 4. Per-year processing + summaries
# ============================================================

def process_one_year(
    year: int,
    dir_urls: List[str],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    out_dir: pathlib.Path,
    nc_root: pathlib.Path,
):
    """
    For a single year:
      - discover .nc files from directory indexes
      - download them
      - extract bbox NDVI
      - filter to June–September of that year
      - save as ndvi_{year}_6_9.csv
      - print summary stats for that year
    """
    if not dir_urls:
        print(f"⚠️ No directory URLs provided for year {year}, skipping.")
        return

    print(f"\n================ PROCESSING YEAR {year} ================\n")

    # Discover all VIIRS NDVI .nc file URLs for June–September of this year
    urls = discover_nc_urls_for_year(year, dir_urls)
    if not urls:
        print(f"⚠️ No VIIRS NDVI files found for {year} June–September, skipping.")
        return

    # Subdirectory for this year's NetCDF files
    nc_dir = nc_root / f"{year}"
    ensure_dir(nc_dir.as_posix())

    # Download
    nc_paths = []
    for u in urls:
        p = download_file(u, str(nc_dir))
        nc_paths.append(p)

    # Inspect metadata from the first file for this year
    inspect_ndvi_metadata(nc_paths[0])

    # Process each file → bbox → tidy
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
        print(f"⚠️ No NDVI data extracted for year {year}.")
        return

    df_year = pd.concat(all_days, ignore_index=True)

    # Ensure datetime
    df_year["date"] = pd.to_datetime(df_year["date"])

    # Filter to this specific year and June–September (safety)
    mask = (
        (df_year["date"].dt.year == year)
        & (df_year["date"].dt.month >= 6)
        & (df_year["date"].dt.month <= 9)
    )
    df_year = df_year.loc[mask].copy()

    if df_year.empty:
        print(f"⚠️ NDVI table is empty after June–September filter for {year}.")
        return

    # Print descriptive stats
    print(f"\n------ NDVI SUMMARY FOR {year} (June–September) ------")
    print(df_year["ndvi"].describe())
    n_total = len(df_year)
    n_nan = df_year["ndvi"].isna().sum()
    print(f"\nTotal rows: {n_total}")
    print(f"NaN count in ndvi: {n_nan}")
    print("------------------------------------------------------\n")

    # Save CSV: ndvi_{year}_6_9.csv in out_dir
    df_year["date"] = df_year["date"].dt.strftime("%Y-%m-%d")
    ensure_dir(out_dir.as_posix())
    out_path = out_dir / f"ndvi_{year}_6_9.csv"
    df_year.to_csv(out_path, index=False, na_rep="NaN")
    print(f"✔ wrote {out_path} (CSV), rows={len(df_year)}")


def summarize_yearly_csvs(out_dir: pathlib.Path, years: List[int]):
    """
    After all per-year CSVs are written, load each one and print
    a compact summary: min, max, NaN count for ndvi.
    """
    print("\n================ FINAL YEARLY NDVI CHECK ================")
    for year in years:
        path = out_dir / f"ndvi_{year}_6_9.csv"
        if not path.exists():
            print(f"Year {year}: file {path.name} not found, skipping.")
            continue

        df = pd.read_csv(path)
        nd = df["ndvi"]
        print(f"\nYear {year}: {path.name}")
        print(f"  count: {nd.count()}")
        print(f"  mean : {nd.mean()}")
        print(f"  std  : {nd.std()}")
        print(f"  min  : {nd.min()}")
        print(f"  max  : {nd.max()}")
        print(f"  NaNs : {nd.isna().sum()}")
    print("=========================================================\n")


# ============================================================
# 5. Main entry point
# ============================================================

def run_ndvi_pipeline(settings_path: str):
    """
    Main pipeline:

    - Load bbox + output base path from setting.json.
    - For each year in YEAR_DIRS:
        * discover .nc URLs from NOAA directory indexes
        * download files
        * extract bbox NDVI for June–September only
        * write ndvi_{year}_6_9.csv
        * print per-year stats
    - Then reload those CSVs and print a compact summary for each year.
    """
    cfg = parse_settings(settings_path)
    lon_min = cfg["lon_min"]
    lon_max = cfg["lon_max"]
    lat_min = cfg["lat_min"]
    lat_max = cfg["lat_max"]
    ndvi_out_path = pathlib.Path(cfg["ndvi_out_path"])

    # Use the parent directory of ndvi_out_path as output directory
    out_dir = ndvi_out_path.parent
    ensure_dir(out_dir.as_posix())

    # Directory for NetCDFs
    nc_root = out_dir / "viirs_nc"
    ensure_dir(nc_root.as_posix())

    years = sorted(YEAR_DIRS.keys())

    for year in years:
        dir_urls = YEAR_DIRS[year]
        process_one_year(
            year=year,
            dir_urls=dir_urls,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            out_dir=out_dir,
            nc_root=nc_root,
        )

    # Final check across all years
    summarize_yearly_csvs(out_dir, years)


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    settings_path = here / "setting.json"
    run_ndvi_pipeline(str(settings_path))