# requirements: pip install rasterio pyproj pandas numpy
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_origin, xy
from rasterio.windows import from_bounds as window_from_bounds
from pyproj import Transformer
import math

# ==== user inputs ====
INPUT_TIF  = "Annual_NLCD_LndCov_2023_CU_C1V1.tif"
OUTPUT_CSV = "nlcd_1km_CA_composition_2023.csv"
RENORMALIZE_ROW_SUM_TO_100 = False

# California bounding box in lon/lat (WGS84)
# (min_lon, min_lat, max_lon, max_lat)
CA_BBOX_LONLAT = (-124.48, 32.53, -114.13, 42.01)

# Equal-area CRS for CONUS; set local CRS if you’re doing AK/HI/PR
DST_CRS = "EPSG:5070"
DST_RES_METERS = 1000  # 1 km

# Making sure the code is running
print('dududu')

# Optional readable names (trim/extend for your NLCD vintage)
NLCD_CLASSES = {
    11: "Open Water", 12: "Perennial Ice/Snow",
    21: "Developed, Open Space", 22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity", 24: "Developed, High Intensity",
    31: "Barren Land (Rock/Sand/Clay)",
    41: "Deciduous Forest", 42: "Evergreen Forest", 43: "Mixed Forest",
    52: "Shrub/Scrub", 71: "Grassland/Herbaceous",
    81: "Pasture/Hay", 82: "Cultivated Crops",
    90: "Woody Wetlands", 95: "Emergent Herbaceous Wetlands",
}

with rasterio.open(INPUT_TIF) as src:
    src_crs = src.crs
    src_nodata = src.nodata

    # 1) Clip source to California bbox (convert lon/lat bbox to source CRS)
    bbox_src = transform_bounds("EPSG:4326", src_crs,
                                *CA_BBOX_LONLAT, densify_pts=21)
    win = window_from_bounds(*bbox_src, transform=src.transform)
    src_arr = src.read(1, window=win)
    src_transform = src.window_transform(win)

    # Identify classes present (excluding nodata if defined)
    unique_vals = np.unique(src_arr)
    if src_nodata is not None:
        unique_vals = unique_vals[unique_vals != src_nodata]
    unique_vals = unique_vals.astype(int)

# 2) Build the 1 km target grid directly from the CA bbox projected to DST_CRS
bbox_proj = transform_bounds("EPSG:4326", DST_CRS,
                             *CA_BBOX_LONLAT, densify_pts=21)
xmin, ymin, xmax, ymax = bbox_proj

dst_width = int(math.ceil((xmax - xmin) / DST_RES_METERS))
dst_height = int(math.ceil((ymax - ymin) / DST_RES_METERS))
dst_transform = from_origin(xmin, ymax, DST_RES_METERS, DST_RES_METERS)

# 3) For each class, resample a 0/1 mask with average -> fraction per 1 km cell
class_to_fraction = {}
for code in unique_vals.tolist():
    bin_src = (src_arr == code).astype("float32")
    dst_frac = np.zeros((dst_height, dst_width), dtype="float32")

    reproject(
        source=bin_src,
        destination=dst_frac,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=DST_CRS,
        resampling=Resampling.average,
        dst_nodata=0.0
    )
    class_to_fraction[code] = dst_frac

codes = sorted(class_to_fraction.keys())
stack = np.stack([class_to_fraction[c] for c in codes], axis=0)  # (K, H, W)

# Sum across classes (to identify valid cells and for optional renorm)
sum_frac = stack.sum(axis=0)  # (H, W)

if RENORMALIZE_ROW_SUM_TO_100:
    safe_sum = np.where(sum_frac > 0, sum_frac, 1.0)
    stack = stack / safe_sum

stack_pct = 100.0 * stack
valid_mask = sum_frac > 0  # keep cells that intersect valid source data

# 4) Cell-center lon/lat for each 1 km pixel
rows, cols = np.indices((dst_height, dst_width))
xs, ys = xy(dst_transform, rows, cols, offset="center")
xs = np.array(xs).flatten()
ys = np.array(ys).flatten()

to_lonlat = Transformer.from_crs(DST_CRS, "EPSG:4326", always_xy=True)
lon, lat = to_lonlat.transform(xs, ys)

# 5) Build the output table
df = pd.DataFrame({"longitude": lon, "latitude": lat})
for i, code in enumerate(codes):
    df[f"pct_{int(code)}"] = stack_pct[i].flatten()

# keep only cells that touch valid data
df = df[valid_mask.flatten()].reset_index(drop=True)

# (Optional) add readable-name columns
# for code in codes:
#     name = NLCD_CLASSES.get(int(code), f"class_{int(code)}")
#     safe = name.replace(" ", "_").replace("/", "_").replace(",", "")
#     df[f"pct_{safe}"] = df[f"pct_{int(code)}"]

df.to_csv(OUTPUT_CSV, index=False)
print(f"Done. Wrote {len(df):,} rows (California bbox) to {OUTPUT_CSV} with {len(codes)} class columns.")