import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box

IRWIN_CLEAN = "irwin_CA_2023-07.csv"
FIRMS_CLEAN = "firms_CA_2023-07_viirs.csv"
CA_BORDER   = "ca.geojson"
OUT_LABELS  = "labels_CA_2023-07.csv"
OUT_EXAMPLES= "examples_CA_2023-07.csv"

CELL_SIZE_M = 5000                    # 5 km
CRS_WGS84   = 4326
CRS_CA      = 3310                    # California Albers

DATE_START  = "2023-07-01"
DATE_END    = "2023-07-31"

BBOX_WGS84 = (-124.48, 32.53, -114.13, 42.01)  # (minx, miny, maxx, maxy)


#functions
def _read_points_csv(path, lat_col="lat", lon_col="lon", date_col="date"):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    lat_col = lat_col if lat_col in df.columns else ("latitude" if "latitude" in df.columns else lat_col)
    lon_col = lon_col if lon_col in df.columns else ("longitude" if "longitude" in df.columns else lon_col)
    date_col = date_col if date_col in df.columns else ("acq_date" if "acq_date" in df.columns else date_col)

    df = df[pd.to_numeric(df[lat_col], errors="coerce").notna() &
            pd.to_numeric(df[lon_col], errors="coerce").notna()].copy()
    df[lat_col] = df[lat_col].astype(float)
    df[lon_col] = df[lon_col].astype(float)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df[df[date_col].notna()]

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=CRS_WGS84
    )
    return gdf.rename(columns={lat_col: "lat", lon_col: "lon", date_col: "date"})


def _make_grid(poly_gdf, cell_size_m=CELL_SIZE_M):
    minx, miny, maxx, maxy = poly_gdf.total_bounds
    xs = list(range(int(minx), int(maxx)+cell_size_m, cell_size_m))
    ys = list(range(int(miny), int(maxy)+cell_size_m, cell_size_m))

    cells, ids = [], []
    gid = 0
    for x in xs[:-1]:
        for y in ys[:-1]:
            cells.append(box(x, y, x+cell_size_m, y+cell_size_m))
            ids.append(gid)
            gid += 1

    grid = gpd.GeoDataFrame({"grid_id": ids, "geometry": cells}, crs=poly_gdf.crs)
    grid = gpd.overlay(grid, poly_gdf[["geometry"]], how="intersection", keep_geom_type=True)

    # centroid lat&lon
    centers_proj = grid.geometry.centroid
    centers_ll = gpd.GeoSeries(centers_proj, crs=poly_gdf.crs).to_crs(CRS_WGS84)
    grid["centroid_lat"] = centers_ll.y
    grid["centroid_lon"] = centers_ll.x
    return grid


def _aggregate_to_cellday(points_gdf, grid, add_cols=None, date_col="date"):
    pts = points_gdf.to_crs(grid.crs)
    joined = gpd.sjoin(pts, grid[["grid_id", "geometry"]], how="inner", predicate="intersects")
    base = joined[[date_col, "grid_id"]].copy()
    base["count"] = 1
    agg = base.groupby([date_col, "grid_id"], as_index=False).agg({"count": "sum"})

    if add_cols:
        for col, how in add_cols.items():
            tmp = joined.groupby([date_col, "grid_id"], as_index=False)[col].agg(how)
            agg = agg.merge(tmp, on=[date_col, "grid_id"], how="left")
    return agg


def _normalize_confidence_to_012(series):
    s = pd.Series(0, index=series.index)  # 默认 0
    if series.dtype == "object":
        low = series.str.lower()
        s = s.where(True, other=0)  # 占位
        s[low.isin(["l", "low"])] = 0
        s[low.isin(["n", "nominal"])] = 1
        s[low.isin(["h", "high"])] = 2
        num = pd.to_numeric(series, errors="coerce")
    else:
        num = pd.to_numeric(series, errors="coerce")
    s = s.where(num.isna(), other=pd.cut(num, bins=[-1, 49.999, 79.999, 1e9], labels=[0, 1, 2]).astype(int))
    return s.astype(int)

def main():
    try:
        ca = gpd.read_file(CA_BORDER).to_crs(CRS_CA)
        print(f"Using boundary file: {CA_BORDER}")
    except Exception:
        print(f"[WARN] cannot find {CA_BORDER}，use bbox")
        ca = gpd.GeoDataFrame(geometry=[box(*BBOX_WGS84)], crs=CRS_WGS84).to_crs(CRS_CA)
    #grid
    grid = _make_grid(ca, CELL_SIZE_M)
    irwin = _read_points_csv(IRWIN_CLEAN)
    irwin_agg = _aggregate_to_cellday(irwin, grid)
    irwin_agg = irwin_agg.rename(columns={"count": "irwin_incidents"})
    irwin_agg["fire_today"] = (irwin_agg["irwin_incidents"] > 0).astype(int)

    firms = _read_points_csv(FIRMS_CLEAN)
    if "confidence" in firms.columns:
        firms["conf_num"] = _normalize_confidence_to_012(firms["confidence"])
        add = {"conf_num": "max"}
    else:
        add = None
    firms_agg = _aggregate_to_cellday(firms, grid, add_cols=add)
    firms_agg = firms_agg.rename(columns={"count": "firms_detections", "conf_num": "firms_conf_max"})

    dates = pd.date_range(DATE_START, DATE_END, freq="D").strftime("%Y-%m-%d")
    full = pd.MultiIndex.from_product([dates, grid["grid_id"].tolist()], names=["date", "grid_id"]).to_frame(index=False)

    df = (full
          .merge(irwin_agg[["date","grid_id","fire_today","irwin_incidents"]], on=["date","grid_id"], how="left")
          .merge(firms_agg[["date","grid_id","firms_detections","firms_conf_max"]], on=["date","grid_id"], how="left")
          .merge(grid[["grid_id","centroid_lat","centroid_lon"]], on="grid_id", how="left")
          )

    for col in ["fire_today", "irwin_incidents", "firms_detections", "firms_conf_max"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    labels = df[["date","grid_id","fire_today"]].copy()
    labels.to_csv(OUT_LABELS, index=False, encoding="utf-8")
    df.to_csv(OUT_EXAMPLES, index=False, encoding="utf-8")

    pos = int(df["fire_today"].sum())
    total = len(df)
    print(f"Saved: {OUT_LABELS}, {OUT_EXAMPLES}")
    print(f"Rows: {total:,} | Positives: {pos:,} | Pos rate: {pos/total:.4%}")


if __name__ == "__main__":
    main()
