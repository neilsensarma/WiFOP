import os
import pandas as pd
import geopandas as gpd

INP = "fire_archive_SV-C2_683401.csv"
CA_GEO = "ca.geojson"
OUT = "firms_CA_2023-07_viirs.csv"

BBOX = (-124.48, 32.53, -114.13, 42.01)  # lon_min, lat_min, lon_max, lat_max

print("cwd =", os.getcwd())

df = pd.read_csv(INP)
df = df.rename(columns={"latitude": "lat", "longitude": "lon", "acq_date": "date"})

df["confidence"] = df["confidence"].astype(str).str.lower()
df = df[df["confidence"].isin(["n", "h"])]

df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
df = df.dropna(subset=["date", "lat", "lon"])

xmin, ymin, xmax, ymax = BBOX
df = df[(df["lon"].between(xmin, xmax)) & (df["lat"].between(ymin, ymax))]

if os.path.exists(CA_GEO):
    ca = gpd.read_file(CA_GEO).to_crs(4326)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326)
    gdf = gpd.sjoin(gdf, ca, predicate="within")
    df = pd.DataFrame(gdf.drop(columns=["geometry"]))

df["lat_r"] = df["lat"].round(6)
df["lon_r"] = df["lon"].round(6)
df = df.drop_duplicates(subset=["date", "lat_r", "lon_r"])

map_conf = {"l": 33, "n": 66, "h": 90}
df["confidence"] = df["confidence"].map(map_conf)

out = df[["date", "lat", "lon", "confidence"]].copy()
out["lat"] = out["lat"].astype(float).round(6)
out["lon"] = out["lon"].astype(float).round(6)

out.to_csv(OUT, index=False)
print("saved ->", os.path.abspath(OUT), "| rows =", len(out))
