"""
Download California boundary (GeoJSON) from U.S. Census TIGER/CB (2023),
and save to data/boundary/california.geojson (EPSG:4326).

Source:
https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip
"""

import io
import zipfile
import requests
import geopandas as gpd
from pathlib import Path

CB_ZIP_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip"
ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT /"fire label data"/"data" / "raw" / "boundary" / "california.geojson"

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[download] {CB_ZIP_URL}")
    r = requests.get(CB_ZIP_URL, timeout=120)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))

    shp_name = [n for n in z.namelist() if n.lower().endswith(".shp")]
    if not shp_name:
        raise RuntimeError("No SHP found in the ZIP.")
    shp_name = shp_name[0]
    stem = shp_name[:-4]
    members = [n for n in z.namelist() if n.startswith(stem)]

    tmp_dir = OUT_PATH.parent / "_tmp_cb_2023_us_state_500k"
    if tmp_dir.exists():
        for p in tmp_dir.iterdir():
            p.unlink()
        tmp_dir.rmdir()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for m in members:
        with z.open(m) as src, open(tmp_dir / Path(m).name, "wb") as dst:
            dst.write(src.read())

    shp_file = next(tmp_dir.glob("*.shp"))
    gdf = gpd.read_file(shp_file)

    if "STUSPS" in gdf.columns:
        ca = gdf[gdf["STUSPS"] == "CA"].copy()
    else:
        ca = gdf[gdf["STATEFP"] == "06"].copy()

    if ca.empty:
        raise RuntimeError("California polygon not found in the shapefile.")

    ca = ca.to_crs(4326)
    ca = ca.dissolve()
    ca = ca.explode(index_parts=False).dissolve()

    # save to GeoJSON
    ca.to_file(OUT_PATH, driver="GeoJSON")
    print(f"[save] -> {OUT_PATH.resolve()}")

    for p in tmp_dir.iterdir():
        p.unlink()
    tmp_dir.rmdir()

if __name__ == "__main__":
    main()
