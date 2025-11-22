## 1. Data sources

### 1.1 IRWIN / WFIGS incident locations

- Portal: National Interagency Fire Center – Wildland Fire Incident Locations  
  (IRWIN / WFIGS FeatureServer)
- Spatial coverage: United States
- Temporal filter used here:
  - `FireDiscoveryDateTime` between **2023-07-01 00:00:00** and **2023-07-31 23:59:59**
  - Local time zone: **US/Pacific**
- Attribute filters:
  - `POOState = 'US-CA'` (California)
  - `IncidentTypeCategory IN ('WF', 'RX', 'CX')` (wildfire / prescribed / complex)

The script `scripts/IRWIN_dataset_download.py` reproduces the same query
programmatically and saves a raw CSV for July 2023.

---

### 1.2 California boundary

- Source: U.S. Census TIGER/Line state boundaries  
  (TIGER/CB 2023, `cb_2023_us_state_500k.zip`)
- We download the full U.S. states shapefile and then clip to California.
- Coordinate system in the downloaded shapefile: EPSG:4269 (NAD83)
- We re-project to:
  - EPSG:3310 (California Albers, meters) for building the 5 km grid
  - EPSG:4326 (WGS84, degrees) for output lat/lon

The script `scripts/download_ca_boundary.py` downloads the ZIP, extracts the
shapefile and converts it to a GeoJSON file for California only.

---

## 2. Folder layout

All paths below are **relative to the WiFOP repo root**.

```text
data/
  fire_label_data/
    data/
      raw/
        boundary/
          california.geojson        # CA boundary (GeoJSON, WGS84)
        ca_2023-07/
          irwin_US-CA_2023-07_PT.csv  # IRWIN incidents, July 2023, Pacific time

      interim/
        fires/
          cleaned_irwin_CA_2023-07.csv  # cleaned, filtered IRWIN incidents
        grid/
          CA_5km.geojson                # 5 km grid polygons over CA

      final/
        grid_lookup_CA_5km.csv          # grid_id ↔ lat, lon (centroid)
        labels_CA_2023-07.csv           # daily fire label per grid cell
        qa/
          labels_pos_rate_by_day.csv    # daily positive ratio for QA
          qa_summary_2023-07.txt        # text summary from check script

    scripts/
      IRWIN_dataset_download.py   # download raw IRWIN CSV for a date range
      clean_irwin_portal.py       # clean & filter IRWIN CSV to July 2023 CA wildfire set
      download_ca_boundary.py     # download & build california.geojson
      make_grid_5km_ca.py         # build 5 km grid & grid lookup
      make_fire_labels.py         # assign daily labels to grid cells
      check_label.py              # QA checks on labels (coverage & imbalance)
## 3.how to reproduce
# 1. Download boundary
python data/fire_label_data/scripts/download_ca_boundary.py

# 2. Download IRWIN incidents for July 2023
python data/fire_label_data/scripts/IRWIN_dataset_download.py

# 3. Clean IRWIN incidents
python data/fire_label_data/scripts/clean_irwin_portal.py

# 4. Build 5 km grid + lookup table
python data/fire_label_data/scripts/make_grid_5km_ca.py

# 5. Build fire labels
python data/fire_label_data/scripts/make_fire_labels.py

# 6. Run QA checks (optional but recommended)
python data/fire_label_data/scripts/check_label.py
