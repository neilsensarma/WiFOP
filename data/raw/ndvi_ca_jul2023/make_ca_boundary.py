import geopandas as gpd

# Read the U.S. state boundaries shapefile you just unzipped
gdf = gpd.read_file("cb_2023_us_state_20m.shp")

# Filter only California
ca = gdf[gdf["NAME"] == "California"]

# Save California as its own shapefile
ca.to_file("/Users/pangdiyang/Desktop/WiFOP/CA_boundary.shp")

print("Saved California boundary shapefile to /Users/pangdiyang/Desktop/WiFOP/CA_boundary.shp")
