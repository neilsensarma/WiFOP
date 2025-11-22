from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import geopandas as gpd

def _resolve_repo_root(script_file: Path) -> Path:
    here = script_file.resolve()
    for p in [here.parents[i] for i in range(1, min(6, len(here.parents)))]:
        names = {c.name.lower() for c in p.iterdir()} if p.exists() else set()
        if "scripts" in names and "data" in names:
            return p
    for p in [here.parents[i] for i in range(1, min(6, len(here.parents)))]:
        names = {c.name.lower() for c in p.iterdir()} if p.exists() else set()
        if "scripts" in names:
            return p
    return here.parents[1]

def _pick_data_root(repo_root: Path) -> Path:
    primary = repo_root / "data"
    legacy  = repo_root / "fire label data" / "data"
    if primary.exists():
        return primary
    if legacy.exists():
        print(f"[warn] using legacy data root: {legacy}", file=sys.stderr)
        return legacy
    primary.mkdir(parents=True, exist_ok=True)
    return primary

def _pick(cols, cand):
    for c in cand:
        if c in cols:
            return c
    return None

def build_labels():
    script_path = Path(__file__).resolve()
    repo_root   = _resolve_repo_root(script_path)
    DATA        = _pick_data_root(repo_root)
    RAW   = DATA / "raw"
    INTER = DATA / "interim"
    FINAL = DATA / "final"

    GRID_GEOJSON = INTER / "grid" / "CA_5km.geojson"
    IRWIN_CLEAN  = INTER / "fires" / "cleaned_irwin_CA_2023-07.csv"

    assert GRID_GEOJSON.exists(), f"grid not found: {GRID_GEOJSON}"
    assert IRWIN_CLEAN.exists(),  f"cleaned IRWIN not found: {IRWIN_CLEAN}"

    print("[path] grid   =", GRID_GEOJSON)
    print("[path] irwin  =", IRWIN_CLEAN)

    grid = gpd.read_file(GRID_GEOJSON)
    if grid.crs is None:
        grid.set_crs(4326, inplace=True)
    grid = grid[["grid_id", "geometry"]].copy()

    df = pd.read_csv(IRWIN_CLEAN, low_memory=False)

    lat_col  = _pick(df.columns, ["lat", "Latitude", "latitude", "InitialLatitude"])
    lon_col  = _pick(df.columns, ["lon", "Longitude", "longitude", "InitialLongitude"])
    date_col = _pick(df.columns, ["date", "local_date", "discovery_date"])
    time_col = _pick(df.columns, ["fire_time", "FireDiscoveryDateTime", "fire_time_local"])

    if lat_col is None or lon_col is None:
        raise KeyError(f"lat/lon columns not found in columns={list(df.columns)}")

    df = df.dropna(subset=[lat_col, lon_col]).copy()

    if date_col is not None:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        df = df[dt.notna()].copy()
        df["date"] = dt.dt.strftime("%Y-%m-%d")
    else:
        if time_col is None:
            raise KeyError(
                "Neither 'date' nor time column found. "
                "Expected one of date/local_date/discovery_date "
                "or fire_time/FireDiscoveryDateTime/fire_time_local."
            )
        dt = pd.to_datetime(df[time_col], errors="coerce")
        df = df[dt.notna()].copy()
        df["date"] = dt.dt.strftime("%Y-%m-%d")

    if "incident_id" in df.columns:
        df = df.drop_duplicates(subset=["incident_id", "date"])
    else:
        df = df.drop_duplicates(subset=[lat_col, lon_col, "date"])

    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col])

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=4326
    )

    events_joined = gpd.sjoin(
        gdf[["date", "geometry"]],
        grid,
        how="inner",
        predicate="intersects"
    )[["date", "grid_id"]].reset_index(drop=True)

    events_out = INTER / "labels" / "events_on_grid_CA_2023-07.csv"
    events_out.parent.mkdir(parents=True, exist_ok=True)
    events_joined.to_csv(events_out, index=False, encoding="utf-8")

    labels = (
        events_joined
        .assign(fire_today=1)
        .groupby(["date", "grid_id"], as_index=False)["fire_today"].max()
    )

    all_days = pd.date_range("2023-07-01", "2023-07-31", freq="D").strftime("%Y-%m-%d")
    all_ids  = grid["grid_id"].tolist()
    full = (pd.MultiIndex.from_product([all_days, all_ids], names=["date", "grid_id"])
            .to_frame(index=False))
    labels_full = full.merge(labels, on=["date", "grid_id"], how="left").fillna({"fire_today": 0})
    labels_full["fire_today"] = labels_full["fire_today"].astype(int)

    labels_out = FINAL / "labels_CA_2023-07.csv"
    labels_full.to_csv(labels_out, index=False, encoding="utf-8")

    print(f"[events] rows={len(events_joined)} -> {events_out}")
    print(f"[labels] rows={len(labels_full)} -> {labels_out}")

if __name__ == "__main__":
    build_labels()