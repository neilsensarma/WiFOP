import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

INP = "irwin_2023-07.csv"          
CA  = "ca.geojson"       
OUT = "data/interim/fires/irwin_CA_2023-07.csv"

START, END = "2023-07-01", "2023-07-31"
CA_BBOX = (-124.48, 32.53, -114.13, 42.01)  # lon_min, lat_min, lon_max, lat_max

df = pd.read_csv(INP, low_memory=False)

df["date"] = pd.to_datetime(df["FireDiscoveryDateTime"], errors="coerce").dt.strftime("%Y-%m-%d")

lat = pd.to_numeric(df.get("InitialLatitude"), errors="coerce")
lon = pd.to_numeric(df.get("InitialLongitude"), errors="coerce")
lat = lat.where(lat.notna() & (lat != 0), df["y"])   # 回填 y → lat
lon = lon.where(lon.notna() & (lon != 0), df["x"])   # 回填 x → lon

df["lat"] = pd.to_numeric(lat, errors="coerce")
df["lon"] = pd.to_numeric(lon, errors="coerce")

df = df.dropna(subset=["date", "lat", "lon"])
df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
df = df[(df["date"] >= START) & (df["date"] <= END)]

if "POOState" in df.columns:
    df = df[df["POOState"] == "US-CA"]

if os.path.exists(CA):
    ca = gpd.read_file(CA).to_crs(4326)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326)
    gdf = gpd.sjoin(gdf, ca, predicate="within")
    df = pd.DataFrame(gdf.drop(columns=["geometry"]))
else:
    xmin, ymin, xmax, ymax = CA_BBOX
    df = df[(df["lon"].between(xmin, xmax)) & (df["lat"].between(ymin, ymax))]

if "IrwinID" in df.columns:
    df = df.drop_duplicates(subset=["IrwinID"])

out = df[["date", "lat", "lon"] + (["IrwinID"] if "IrwinID" in df.columns else [])].copy()
out.rename(columns={"IrwinID": "incident_id"}, inplace=True)
out["lat"] = out["lat"].astype(float).round(6)
out["lon"] = out["lon"].astype(float).round(6)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
out.to_csv(OUT, index=False)
print(f"IRWIN cleaned -> {OUT} | rows={len(out)} | cols={list(out.columns)}")
