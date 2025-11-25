import argparse, time
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime, timedelta
import pytz, requests, pandas as pd

LAYER_URL = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/ArcGIS/rest/services/"
             "WFIGS_Incident_Locations/FeatureServer/0/query")

DATE_FIELDS = [
    "FireDiscoveryDateTime","ContainmentDateTime","ControlDateTime",
    "CompletionDateTime","CreatedOnDateTime_dt","ModifiedOnDateTime_dt",
    "InitialResponseDateTime","ICS209ReportDateTime","FinalFireReportApprovedDate"
]

def local_window_to_utc(start_local_date: str, end_local_date: str, tz_name="US/Pacific"):
    tz = pytz.timezone(tz_name)
    start_local = tz.localize(datetime.strptime(start_local_date + " 00:00:00", "%Y-%m-%d %H:%M:%S"))
    end_local   = tz.localize(datetime.strptime(end_local_date   + " 23:59:59", "%Y-%m-%d %H:%M:%S"))
    start_utc = start_local.astimezone(pytz.utc)
    end_utc   = end_local.astimezone(pytz.utc)
    start_s = start_utc.strftime("%Y-%m-%d %H:%M:%S")
    end_s   = end_utc.strftime("%Y-%m-%d %H:%M:%S")
    return start_s, end_s

def _expand_state(tok: str) -> List[str]:
    tok = tok.strip()
    if not tok: return []
    if tok.startswith("US-"): return [tok]
    return [f"US-{tok}", tok]

def build_where_pt(start_local: str, end_local: str,
                   states: List[str], types: List[str],
                   adsp_default: bool, valid_only: bool, non_quarantined_only: bool):
    start_utc, end_utc = local_window_to_utc(start_local, end_local, "US/Pacific")
    time_clause = (f"FireDiscoveryDateTime >= TIMESTAMP '{start_utc}' AND "
                   f"FireDiscoveryDateTime <= TIMESTAMP '{end_utc}'")
    parts = [time_clause]

    if types:
        parts.append("IncidentTypeCategory IN (" + ",".join([f"'{t}'" for t in types]) + ")")

    state_terms=[]
    for s in states:

        ex=_expand_state(s)
        if len(ex)==1: state_terms.append(f"POOState = '{ex[0]}'")
        else:          state_terms.append(f"(POOState = '{ex[0]}' OR POOState = '{ex[1]}')")
    if state_terms:
        parts.append("(" + " OR ".join(state_terms) + ")")

    if adsp_default:
        parts.append("ADSPermissionState = 'DEFAULT'")
    if valid_only:
        parts.append("IsValid = 1")
    if non_quarantined_only:
        parts.append("(IsQuarantined = 0 OR IsQuarantinedInIRWIN = 0 OR IsQuarantined IS NULL)")

    return " AND ".join(parts)

def count_only(where: str):
    r = requests.get(LAYER_URL, params={"f":"json","where":where,"returnCountOnly":"true"}, timeout=60)
    r.raise_for_status()
    return int(r.json().get("count", 0))

def fetch_all(where: str, page=2000, sleep=0.2) -> Dict[str, Any]:
    feats, offset = [], 0
    while True:
        params = {"f":"geojson","where":where,"outFields":"*","returnGeometry":"false",
                  "outSR":4326,"resultRecordCount":page,"resultOffset":offset}
        r = requests.get(LAYER_URL, params=params, timeout=90)
        r.raise_for_status()
        gj = r.json()
        batch = gj.get("features", [])
        if not batch: break
        feats.extend(batch); offset += len(batch)
        if len(batch) < page: break
        time.sleep(sleep)
    return {"features":feats}

def epoch_ms_to_pt_str(x):
    if pd.isna(x): return x
    try:
        dt = datetime.utcfromtimestamp(float(x)/1000.0).replace(tzinfo=pytz.utc).astimezone(pytz.timezone("US/Pacific"))
        h12 = dt.hour % 12 or 12
        ampm = "AM" if dt.hour < 12 else "PM"
        return f"{dt.month}/{dt.day}/{dt.year} {h12}:{dt.minute:02d}:{dt.second:02d} {ampm}"
    except Exception:
        return x

def format_dates(df: pd.DataFrame):
    for c in DATE_FIELDS:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].map(epoch_ms_to_pt_str)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end",   required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--types", default="WF,RX,CX")
    ap.add_argument("--adsp-default", action="store_true")
    ap.add_argument("--valid-only", action="store_true")
    ap.add_argument("--non-quarantined-only", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-epoch", action="store_true")
    args = ap.parse_args()

    states = [s.strip() for s in args.states.split(",") if s.strip()]
    types  = [t.strip() for t in args.types.split(",")  if t.strip()]
    where = build_where_pt(args.start, args.end, states, types,
                           args.adsp_default, args.valid_only, args.non_quarantined_only)
    print("[WHERE]", where)
    print("[COUNT]", count_only(where))

    gj = fetch_all(where)
    rows = [f.get("properties", {}) for f in gj["features"]]
    df = pd.DataFrame(rows)
    if not args.keep_epoch and not df.empty:
        format_dates(df)

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False, encoding="utf-8")
    print(f"[SAVE] -> {outp} | rows={len(df)} cols={len(df.columns)}")
    print("[ABS]", outp.resolve())
if __name__ == "__main__":
    main()
