from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import geopandas as gpd
from shapely.geometry import box

#constants
GRID_SIZE_M = 5_000          # 5 km
CRS_M       = 3310           # California Albers (meter)
WGS84       = 4326


def _resolve_repo_root(script_file: Path) -> Path:
    here = script_file.resolve()
    for p in [here.parents[i] for i in range(1, min(6, len(here.parents)))]:
        entries = {c.name.lower() for c in p.iterdir()} if p.exists() else set()
        if "scripts" in entries and "data" in entries:
            return p
    for p in [here.parents[i] for i in range(1, min(6, len(here.parents)))]:
        entries = {c.name.lower() for c in p.iterdir()} if p.exists() else set()
        if "scripts" in entries:
            return p
    return here.parents[1]


def _pick_data_root(repo_root: Path) -> Path:
    primary = repo_root / "data"
    legacy  = repo_root / "fire label data" / "data"
    if primary.exists():
        return primary
    if legacy.exists():
        print(f"[warn] data/ not found at repo root. Using legacy folder: {legacy}", file=sys.stderr)
        return legacy
    primary.mkdir(parents=True, exist_ok=True)
    return primary


def _locate_ca_boundary(raw_root: Path) -> Path:
    canonical = raw_root / "boundary" / "california.geojson"
    if canonical.exists():
        return canonical

    repo_root = raw_root.parents[2] if len(raw_root.parents) >= 3 else raw_root.parents[-1]
    found = list(repo_root.rglob("california.geojson"))
    if found:
        print(f"[warn] canonical boundary not found. Using discovered file: {found[0]}", file=sys.stderr)
        return found[0]

    raise FileNotFoundError(
        f"California boundary not found.\n"
        f"Expected at: {canonical}\n"
        f"Tip: run 'scripts/download_ca_boundary.py' first to create it."
    )


def build_grid():
    script_path = Path(__file__).resolve()
    repo_root   = _resolve_repo_root(script_path)
    data_root   = _pick_data_root(repo_root)
    raw_root    = data_root / "raw"
    inter_root  = data_root / "interim"
    final_root  = data_root / "final"

    ca_border   = _locate_ca_boundary(raw_root)
    grid_out    = inter_root / "grid" / "CA_5km.geojson"
    lookup_out  = final_root / "grid_lookup_CA_5km.csv"

    print("[path] repo_root  =", repo_root)
    print("[path] data_root  =", data_root)
    print("[path] CA_BORDER  =", ca_border)

    ca = gpd.read_file(ca_border)
    if ca.crs is None:
        ca.set_crs(WGS84, inplace=True)
    ca_m     = ca.to_crs(CRS_M)
    ca_union = ca_m.geometry.unary_union

    minx, miny, maxx, maxy = ca_union.bounds
    xs = np.arange(minx, maxx + GRID_SIZE_M, GRID_SIZE_M)
    ys = np.arange(miny, maxy + GRID_SIZE_M, GRID_SIZE_M)

    cells = [box(x, y, x + GRID_SIZE_M, y + GRID_SIZE_M)
             for x in xs[:-1] for y in ys[:-1]]
    grid = gpd.GeoDataFrame({"geometry": cells}, crs=CRS_M)

    grid = gpd.overlay(
        grid,
        gpd.GeoDataFrame(geometry=[ca_union], crs=CRS_M),
        how="intersection",
        keep_geom_type=True
    )
    grid = grid[~grid.is_empty].reset_index(drop=True)

    grid["grid_id"] = [f"CA5K_{i:05d}" for i in range(len(grid))]
    cent_ll = gpd.GeoSeries(grid.geometry.centroid, crs=CRS_M).to_crs(WGS84)
    grid["lon"] = cent_ll.x
    grid["lat"] = cent_ll.y

    grid_ll = grid.to_crs(WGS84)[["grid_id", "lon", "lat", "geometry"]]

    grid_out.parent.mkdir(parents=True, exist_ok=True)
    grid_ll.to_file(grid_out, driver="GeoJSON")

    lookup_out.parent.mkdir(parents=True, exist_ok=True)
    grid_ll[["grid_id", "lon", "lat"]].to_csv(lookup_out, index=False, encoding="utf-8")

    print(f"[grid] cells={len(grid_ll)}")
    print(f"[save] grid   -> {grid_out}")
    print(f"[save] lookup -> {lookup_out}")


if __name__ == "__main__":
    build_grid()
