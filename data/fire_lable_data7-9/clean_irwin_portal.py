import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IN_CSV  = ROOT / "fire label data" / "data" / "raw" / "ca_2023-07" /"irwin_US-CA_2023-7_9_PT.csv"

OUT_CSV = ROOT / "fire label data" / "data" / "interim" / "fires" / "cleaned_irwin_CA_2023-7_9.csv"


def main():
    df = pd.read_csv(IN_CSV, low_memory=False)
    df = df.rename(columns={
        "IrwinID": "incident_id",
        "IncidentName": "incident_name",
        "InitialLatitude": "lat",
        "InitialLongitude": "lon",
        "POOState": "state",
        "IncidentTypeCategory": "type",
        "FireDiscoveryDateTime": "fire_time",
    })

    df = df[df["state"].astype(str).eq("US-CA")]
    df = df[df["type"].astype(str).eq("WF")]

    df["date"] = pd.to_datetime(df["fire_time"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["lat", "lon", "date"])

    df = (df.sort_values(["incident_id", "date"])
            .drop_duplicates(subset=["incident_id"], keep="first"))

    out = df[["incident_id", "date", "lat", "lon", "state", "type"]].reset_index(drop=True)

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[clean] input rows={len(pd.read_csv(IN_CSV))}")
    print(f"[clean] kept rows={len(out)} | cols={len(out.columns)}")
    print(f"[clean] saved -> {OUT_CSV}")

if __name__ == "__main__":
    main()
