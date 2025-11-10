# download_viirs_jp113c1_july2023.py
import os
import re
import sys
import time
import math
import errno
import shutil
import pathlib
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

INDEX_URL = "https://www.ncei.noaa.gov/data/land-normalized-difference-vegetation-index/access/2023/"
OUT_DIR   = "/Users/pangdiyang/Desktop/WiFOP/viirs_daily"   # <= change to you username

# Choose version: "v001"
PREFER_VERSION = "v001"  

# Only NOAA-20 JP113C1 for July 2023 (YYYYMM)
MONTH_PREFIX = "202307"

# -------- utils --------
def ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def filename_from_url(u: str) -> str:
    return u.rstrip("/").split("/")[-1]

def human(n):
    units = ["", "K", "M", "G"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units)-1:
        f /= 1024.0
        i += 1
    return f"{f:.1f}{units[i]}B"

def robust_get(url, stream=True, timeout=60):
    # simple robust GET with a retry or two
    tries = 3
    for i in range(tries):
        try:
            r = requests.get(url, stream=stream, timeout=timeout)
            if r.status_code == 200:
                return r
            else:
                print(f"[warn] HTTP {r.status_code} for {url}")
        except Exception as e:
            print(f"[warn] {e} ({i+1}/{tries})")
        time.sleep(2)
    raise RuntimeError(f"Failed to GET after {tries} tries: {url}")

# -------- scrape listing & filter --------
def collect_links(index_html: str):
    soup = BeautifulSoup(index_html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.lower().endswith(".nc"):
            continue
        # build absolute URL
        if href.startswith("http"):
            url = href
        else:
            url = INDEX_URL.rstrip("/") + "/" + href.lstrip("/")
        links.append(url)
    return links

def pick_month_noaa20_jp113c1(links, month_prefix="202307", prefer_version="v002"):
    """
    Return URLs for all JP113C1 NOAA-20 files for given month (YYYYMM),
    preferring v002 when both v001 & v002 exist for the same date.
    """
    # Filter down to NOAA-20 + JP113C1 + given month
    pat = re.compile(r"VIIRS-Land_(v00[12])_JP113C1_NOAA-20_(\d{8})_.*\.nc$")
    candidates = {}
    for u in links:
        m = pat.search(u)
        if not m:
            continue
        version = m.group(1)   # v001 or v002
        yyyymmdd = m.group(2)
        if not yyyymmdd.startswith(month_prefix):
            continue
        # keep best version for a given date
        prev = candidates.get(yyyymmdd)
        if prev is None:
            candidates[yyyymmdd] = (version, u)
        else:
            # prefer chosen version
            if prefer_version == "v002":
                # overwrite if new is v002
                if version == "v002":
                    candidates[yyyymmdd] = (version, u)
            else:
                # prefer v001
                if version == "v001":
                    candidates[yyyymmdd] = (version, u)
    # return URLs sorted by date
    return [candidates[k][1] for k in sorted(candidates.keys())]

# -------- download --------
def download_file(url: str, out_dir: str):
    ensure_dir(out_dir)
    fname = filename_from_url(url)
    out_path = os.path.join(out_dir, fname)
    tmp_path = out_path + ".part"

    # skip if exists
    if os.path.exists(out_path):
        print(f"✔ exists: {fname}")
        return out_path

    r = robust_get(url, stream=True)
    total = int(r.headers.get("Content-Length", "0"))
    desc = f"↓ {fname}"
    with open(tmp_path, "wb") as f, tqdm(total=total if total>0 else None, unit="B", unit_scale=True, desc=desc) as pbar:
        for chunk in r.iter_content(chunk_size=1024*256):
            if chunk:
                f.write(chunk)
                if total > 0:
                    pbar.update(len(chunk))

    os.replace(tmp_path, out_path)
    print(f"✔ saved {fname} ({human(total) if total else 'unknown size'})")
    return out_path

def main():
    ensure_dir(OUT_DIR)
    print("Fetching index…")
    r = robust_get(INDEX_URL, stream=False)
    links = collect_links(r.text)
    month_urls = pick_month_noaa20_jp113c1(links, MONTH_PREFIX, PREFER_VERSION)

    if not month_urls:
        print("No matching files found. Check month/version filters and the index URL.")
        sys.exit(1)

    print(f"Found {len(month_urls)} files for NOAA-20 JP113C1 {MONTH_PREFIX} ({PREFER_VERSION}).")
    for url in month_urls:
        download_file(url, OUT_DIR)

if __name__ == "__main__":
    main()
