import pandas as pd

# Load the parquet
df = pd.read_parquet("/Users/pangdiyang/Desktop/WiFOP/viirs_daily/JP113C1_CApoly_202307_all_days_lonlat_ndvi.parquet")

# Save as CSV
df.to_csv("/Users/pangdiyang/Desktop/WiFOP/JP113C1_CApoly_202307_all_days_lonlat_ndvi.csv", index=False)

print("Converted to CSV successfully!")
