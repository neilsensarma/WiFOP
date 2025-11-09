 Wildfire — California 5 km, July 2023 (Anchor + QC Examples)

Scope: California, 2023-07-01 … 2023-07-31  
Grid: 5 km squares (built in EPSG:3310); centroids in WGS84 (EPSG:4326)  
Key (all tables): `date (YYYY-MM-DD)` + `grid_id (int)`

- labels_CA_2023-07.csv — *Anchor (ground truth)*  
  Columns: `date, grid_id, fire_today` (0/1 new ignition from IRWIN).  
  Full cell×day panel (includes zero rows).

- grid_lookup_CA_5km.csv — *Grid lookup*  
  Mapping `grid_id → centroid_lat, centroid_lon`.  
  Single source of truth for coordinates (do not recompute elsewhere).

- fire/examples_CA_2023-07.csv — *QC/visualization only*  
  Panel table with same-day IRWIN/FIRMS summaries (e.g., `irwin_incidents`, `firms_detections`, `firms_conf_max`) plus `fire_today`.  
  ⚠ **Do not use for training** (contains same-day information → leakage). Use it to sanity-check counts, dates, and spatial joins.

## Quick QC snippets

```python
import pandas as pd

labels = pd.read_csv("data/final datasets/labels_CA_2023-07.csv")
lookup = pd.read_csv("data/final datasets/grid_lookup_CA_5km.csv")
examples = pd.read_csv("data/final datasets/fire/examples_CA_2023-07.csv")

# Label sparsity / date coverage
print("Rows:", len(labels), "Positives:", int(labels["fire_today"].sum()))
print(labels.groupby("date")["fire_today"].sum().head())

# Join coords if needed
labels_map = labels.merge(lookup, on="grid_id", how="left")
print(labels_map.head(3))

# Compare IRWIN/FIRMS daily totals (QC only)
cols = [c for c in ["irwin_incidents","firms_detections","firms_conf_max"] if c in examples.columns]
print(examples[cols].sum())
