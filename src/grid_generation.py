import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point

# === 1. Define approximate California bounding box in lat/lon ===
MIN_LAT, MAX_LAT = 32.0, 42.5
MIN_LON, MAX_LON = -125.0, -113.0

# Create a bounding box polygon in WGS84 (lon/lat)
bbox = box(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT)
gdf_bbox = gpd.GeoDataFrame(
    {"id": [1]},
    geometry=[bbox],
    crs="EPSG:4326"  # WGS84
)

# === 2. Reproject to a projected CRS (California Albers, meters) ===
gdf_bbox_3310 = gdf_bbox.to_crs(epsg=3310)
min_x, min_y, max_x, max_y = gdf_bbox_3310.total_bounds

# === 3. Define grid resolution: 5 km x 5 km (meters) ===
cell_size = 5000  # meters

# Centroids at the middle of each cell
xs = np.arange(min_x + cell_size / 2, max_x, cell_size)
ys = np.arange(min_y + cell_size / 2, max_y, cell_size)

# === 4. Build grid centroids in projected CRS ===
points = []
grid_id = 1
for y in ys:
    for x in xs:
        points.append(Point(x, y))
        grid_id += 1

gdf_points_3310 = gpd.GeoDataFrame(
    geometry=points,
    crs="EPSG:3310"
)

# === 5. Transform centroids back to WGS84 (lat/lon) ===
gdf_points_wgs84 = gdf_points_3310.to_crs(epsg=4326)

# Extract lon/lat from geometry
gdf_points_wgs84["centroid_lon"] = gdf_points_wgs84.geometry.x
gdf_points_wgs84["centroid_lat"] = gdf_points_wgs84.geometry.y

# Add grid_id
gdf_points_wgs84["grid_id"] = range(1, len(gdf_points_wgs84) + 1)

# === 6. Save to CSV ===
df = gdf_points_wgs84[["grid_id", "centroid_lat", "centroid_lon"]].copy()
df.to_csv("california_grid_5km.csv", index=False)

print(f"Saved {len(df)} grid cells to california_grid_5km.csv")
