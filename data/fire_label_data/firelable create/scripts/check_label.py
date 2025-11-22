from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

def _resolve_repo_root(script_file: Path) -> Path:
    here = script_file.resolve()
    for p in [here.parents[i] for i in range(1, min(6, len(here.parents)))]:
        try:
            names = {c.name.lower() for c in p.iterdir()}
        except Exception:
            names = set()
        if "scripts" in names and "data" in names:
            return p
    for p in [here.parents[i] for i in range(1, min(6, len(here.parents)))]:
        try:
            names = {c.name.lower() for c in p.iterdir()}
        except Exception:
            names = set()
        if "scripts" in names:
            return p
    return here.parents[1]

def _pick_data_root(repo_root: Path) -> Path:
    primary = repo_root / "data"
    legacy  = repo_root / "fire label data" / "data"
    if primary.exists():
        return primary
    if legacy.exists():
        print(f"[warn] using legacy data root: {legacy}", file=sys.stderr)
        return legacy
    primary.mkdir(parents=True, exist_ok=True)
    return primary

def _fmt_ratio(n, d):
    return f"{n} / {d} = {n/d:.6%}" if d else "N/A"

def main():
    script_path = Path(__file__).resolve()
    repo_root   = _resolve_repo_root(script_path)
    DATA        = _pick_data_root(repo_root)
    RAW   = DATA / "raw"
    INTER = DATA / "interim"
    FINAL = DATA / "final"

    f_labels  = FINAL / "labels_CA_2023-07.csv"
    f_lookup  = FINAL / "grid_lookup_CA_5km.csv"
    f_irwin   = INTER / "fires" / "cleaned_irwin_CA_2023-07.csv"
    f_events  = INTER / "labels" / "events_on_grid_CA_2023-07.csv"   # optional

    assert f_labels.exists(), f"labels not found: {f_labels}"
    assert f_lookup.exists(), f"grid lookup not found: {f_lookup}"
    assert f_irwin.exists(),  f"cleaned IRWIN not found: {f_irwin}"

    out_dir = INTER / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    f_summary = out_dir / "qa_summary_2023-07.txt"

    labels = pd.read_csv(f_labels, dtype={"grid_id": str})
    lookup = pd.read_csv(f_lookup, dtype={"grid_id": str})
    irwin  = pd.read_csv(f_irwin)

    days = pd.date_range("2023-07-01", "2023-07-31", freq="D").strftime("%Y-%m-%d")
    n_days   = len(days)
    n_grids  = lookup["grid_id"].nunique()
    n_expect = n_days * n_grids

    with f_summary.open("w", encoding="utf-8") as fw:
        fw.write(f"[paths]\n")
        fw.write(f"labels : {f_labels}\n")
        fw.write(f"lookup : {f_lookup}\n")
        fw.write(f"irwin  : {f_irwin}\n")
        fw.write(f"events : {f_events if f_events.exists() else '(not provided)'}\n\n")

        fw.write("[shape]\n")
        fw.write(f"lookup grids unique : {n_grids}\n")
        fw.write(f"labels rows         : {len(labels)}\n")
        fw.write(f"expected rows       : {n_expect} (= {n_days} days × {n_grids} grids)\n\n")

        # 列检查
        expected_cols = {"date","grid_id","fire_today"}
        fw.write(f"[columns]\nlabels columns = {list(labels.columns)}\n")
        miss_cols = expected_cols - set(labels.columns)
        fw.write(f"labels missing columns: {miss_cols}\n\n")

    # 1) DATE
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce")
    bad_date = labels[~labels["date"].between("2023-07-01", "2023-07-31")]
    # 2) fire_today VALUE
    bad_val = labels[~labels["fire_today"].isin([0,1])]

    # 3) grid_id
    labels["grid_id"] = labels["grid_id"].astype(str)
    lookup["grid_id"] = lookup["grid_id"].astype(str)
    not_in_lookup = labels.merge(lookup[["grid_id"]].drop_duplicates(), on="grid_id", how="left", indicator=True)
    not_in_lookup = not_in_lookup[not_in_lookup["_merge"]=="left_only"][["date","grid_id","fire_today"]]

    n_pos = int((labels["fire_today"]==1).sum())
    pos_rate = n_pos / len(labels) if len(labels) else 0

    # positive rate
    pos_by_day = (
        labels
        .groupby(labels["date"].dt.strftime("%Y-%m-%d"))["fire_today"]
        .mean()
        .rename("pos_rate")
        .reset_index()
        .sort_values("date")
    )
    pos_by_day.to_csv(out_dir / "labels_pos_rate_by_day.csv", index=False, encoding="utf-8")

    ir_cols = set(irwin.columns)
    lat_col = next((c for c in ["lat","Latitude","latitude","InitialLatitude"] if c in ir_cols), None)
    lon_col = next((c for c in ["lon","Longitude","longitude","InitialLongitude"] if c in ir_cols), None)
    date_col= "date" if "date" in ir_cols else None
    if date_col is None:
        time_col = next((c for c in ["fire_time","FireDiscoveryDateTime","fire_time_local"] if c in ir_cols), None)
        if time_col:
            dt = pd.to_datetime(irwin[time_col], errors="coerce")
            irwin["_tmp_date"] = dt.dt.strftime("%Y-%m-%d")
            date_col = "_tmp_date"
    ir_summary = {
        "rows": len(irwin),
        "date_col": date_col,
        "lat_col": lat_col,
        "lon_col": lon_col,
        "date_min": pd.to_datetime(irwin[date_col], errors="coerce").min() if date_col else None,
        "date_max": pd.to_datetime(irwin[date_col], errors="coerce").max() if date_col else None,
        "type_counts": irwin["type"].value_counts(dropna=False).to_dict() if "type" in ir_cols else {},
        "state_counts": irwin["state"].value_counts(dropna=False).to_dict() if "state" in ir_cols else {},
    }

    events_subset_violations = pd.DataFrame()
    if f_events.exists():
        ev = pd.read_csv(f_events, dtype={"grid_id": str})
        ev["date"] = pd.to_datetime(ev["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        lab_pos = labels[labels["fire_today"]==1].copy()
        lab_pos["date"] = lab_pos["date"].dt.strftime("%Y-%m-%d")

        m = lab_pos.merge(ev.drop_duplicates(), on=["date","grid_id"], how="left", indicator=True)
        violations = m[m["_merge"]=="left_only"][["date","grid_id"]].drop_duplicates()
        if len(violations) > 0:
            events_subset_violations = violations.copy()
            events_subset_violations.to_csv(out_dir / "labels_not_in_events.csv", index=False, encoding="utf-8")

    #summary
    with f_summary.open("a", encoding="utf-8") as fw:
        fw.write("[basic QA]\n")
        fw.write(f"bad_date rows: {len(bad_date)}\n")
        fw.write(f"bad fire_today (not in {{0,1}}): {len(bad_val)}\n")
        fw.write(f"grid_id not in lookup: {len(not_in_lookup)}\n\n")

        if len(not_in_lookup):
            not_in_lookup.to_csv(out_dir / "missing_grid_ids.csv", index=False, encoding="utf-8")

        fw.write("[positives]\n")
        fw.write(f"positives: {n_pos}\n")
        fw.write(f"pos rate : {_fmt_ratio(n_pos, len(labels))}\n")
        fw.write(f"daily pos rate csv: {out_dir / 'labels_pos_rate_by_day.csv'}\n\n")

        fw.write("[IRWIN overview]\n")
        for k,v in ir_summary.items():
            fw.write(f"{k}: {v}\n")
        fw.write("\n")

        if f_events.exists():
            fw.write("[events_on_grid cross-check]\n")
            fw.write(f"labels==1 not found in events_on_grid: {len(events_subset_violations)}\n")
            if len(events_subset_violations):
                fw.write(f"saved -> {out_dir / 'labels_not_in_events.csv'}\n")
            fw.write("\n")

    print(f"[done] QA summary -> {f_summary}")
    if f_events.exists():
        print("[hint] Also opened events_on_grid cross-check section since file exists.")
    else:
        print("[hint] events_on_grid not provided; subset cross-check skipped.")

if __name__ == "__main__":
    main()
