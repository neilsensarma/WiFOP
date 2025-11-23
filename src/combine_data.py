import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

# ============================================================
# 0. Load 5 km grid
# ============================================================

print("\n[1/10] Loading 5km grid...")
grid = pd.read_csv("../data/california_grid_5km.csv")

grid = grid.rename(columns={"centroid_lat": "lat", "centroid_lon": "lon"})
assert "lat" in grid and "lon" in grid, "Grid rename failed!"

# Build BallTree
print("[2/10] Building BallTree...")
tree = BallTree(np.radians(grid[["lat","lon"]].to_numpy()), metric='haversine')

def assign_grid_id(df, lat_col, lon_col):
    assert lat_col in df.columns, f"Missing {lat_col} in dataframe!"
    assert lon_col in df.columns, f"Missing {lon_col} in dataframe!"

    coords = np.radians(df[[lat_col, lon_col]].to_numpy())
    dist, idx = tree.query(coords, k=1)
    df["grid_id"] = grid.iloc[idx.flatten()].grid_id.values
    return df

# ============================================================
# 1. NASA
# ============================================================

print("\n[3/10] Loading NASA...")
nasa = pd.read_csv("../data/interim/nasa.csv")

# FORCE rename
nasa.rename(columns={"LAT": "latitude", "LON": "longitude", "DATE": "date"}, inplace=True)
assert "latitude" in nasa and "longitude" in nasa, "NASA rename failed!"

nasa["date"] = pd.to_datetime(nasa["date"])

print("Assigning NASA grid IDs...")
nasa = assign_grid_id(nasa, "latitude", "longitude")
assert "grid_id" in nasa, "NASA grid_id assignment FAILED!"

nasa = nasa.drop(columns=["latitude","longitude"])
nasa_daily = nasa.groupby(["grid_id","date"], as_index=False).mean()

# ============================================================
# 2. NDVI
# ============================================================

print("\n[4/10] Loading NDVI...")
ndvi = pd.read_csv("../data/interim/ndvi.csv")
assert "latitude" in ndvi and "longitude" in ndvi, "NDVI rename failed!"

ndvi["date"] = pd.to_datetime(ndvi["date"], format="%m/%d/%Y")

print("Assigning NDVI grid IDs...")
ndvi = assign_grid_id(ndvi, "latitude", "longitude")
ndvi = ndvi.drop(columns=["latitude","longitude"])
ndvi_daily = ndvi.groupby(["grid_id","date"], as_index=False).mean()

# ============================================================
# 3. IRWIN
# ============================================================

print("\n[5/10] Loading IRWIN...")
irwin = pd.read_csv("../data/interim/irwin.csv")
assert "lat" in irwin and "lon" in irwin, "IRWIN columns missing!"

irwin["date"] = pd.to_datetime(irwin["date"])

print("Assigning IRWIN grid IDs...")
irwin = assign_grid_id(irwin, "lat", "lon")
assert "grid_id" in irwin, "IRWIN grid_id assignment FAILED!"

irwin = irwin.drop(columns=["lat","lon","state","type","incident_id"])

fire_labels = (
    irwin.groupby(["grid_id","date"]).size().reset_index(name="fire_count")
)
fire_labels["fire_today"] = (fire_labels["fire_count"] > 0).astype(int)

# ============================================================
# 4. NLCD
# ============================================================

print("\n[6/10] Loading NLCD...")
nlcd = pd.read_csv("../data/interim/nlcd.csv")
assert "latitude" in nlcd and "longitude" in nlcd, "NLCD columns missing!"

print("Assigning NLCD grid IDs...")
nlcd = assign_grid_id(nlcd, "latitude", "longitude")
nlcd = nlcd.drop(columns=["latitude","longitude"])

nlcd_grid = nlcd.groupby("grid_id", as_index=False).mean()

# ============================================================
# 5. Build daily panel
# ============================================================

all_dates = nasa_daily["date"].sort_values().unique()
panel = pd.MultiIndex.from_product(
    [grid["grid_id"], all_dates],
    names=["grid_id","date"]
).to_frame(index=False)

# ============================================================
# 6. Merge
# ============================================================

print("\nMerging datasets...")
panel = panel.merge(nasa_daily, on=["grid_id","date"], how="left")
panel = panel.merge(ndvi_daily, on=["grid_id","date"], how="left")
panel = panel.merge(fire_labels[["grid_id","date","fire_today"]], on=["grid_id","date"], how="left")
panel = panel.merge(nlcd_grid, on="grid_id", how="left")

panel["fire_today"] = panel["fire_today"].fillna(0).astype(int)

# ============================================================
# 7. Save
# ============================================================

print("\nSaving...")
panel.to_csv("../data/processed/california_5km_daily_panel_2023-07.csv", index=False)

print("\n🎉 DONE — merged panel built successfully!")
print(panel.head())
