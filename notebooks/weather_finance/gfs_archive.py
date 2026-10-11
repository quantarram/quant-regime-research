"""
Archive of GFS 00z 2 m temperature forecasts at 34 large US metros, 2021-03-23 onward (NOAA's open-data bucket on AWS).

Each cycle: for 6-hourly steps f000..f240 (10 days), read the GRIB index, fetch only the 2 m temperature message by byte
range, sample it at the metro points. Nothing raw is kept; one float32 array [cycle, step, metro] is saved with resume support.
Used to build population-weighted degree-day forecasts and their day-to-day revisions for the natural-gas study.

  python gfs_archive.py                      # extend the archive up to yesterday
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import numpy as np
import requests

SIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "climate-energy-dashboard")
sys.path.insert(0, os.path.abspath(SIB))
from weatherdash import grib  # noqa: E402

BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gfs_t2m_archive.npz")
STEPS = list(range(0, 241, 6))
START = date(2021, 3, 23)

# name, lat, lon, tz, metro population (thousands, rounded; approximate)
METROS = [
    ("New York", 40.71, -74.01, "America/New_York", 20100), ("Los Angeles", 34.05, -118.24, "America/Los_Angeles", 13200),
    ("Chicago", 41.88, -87.63, "America/Chicago", 9500), ("Dallas", 32.78, -96.80, "America/Chicago", 7600),
    ("Houston", 29.76, -95.37, "America/Chicago", 7100), ("Washington", 38.90, -77.04, "America/New_York", 6300),
    ("Philadelphia", 39.95, -75.17, "America/New_York", 6200), ("Miami", 25.76, -80.19, "America/New_York", 6100),
    ("Atlanta", 33.75, -84.39, "America/New_York", 6100), ("Boston", 42.36, -71.06, "America/New_York", 4900),
    ("Phoenix", 33.45, -112.07, "America/Phoenix", 4800), ("San Francisco", 37.77, -122.42, "America/Los_Angeles", 4700),
    ("Detroit", 42.33, -83.05, "America/Detroit", 4300), ("Seattle", 47.61, -122.33, "America/Los_Angeles", 4000),
    ("Minneapolis", 44.98, -93.27, "America/Chicago", 3700), ("San Diego", 32.72, -117.16, "America/Los_Angeles", 3300),
    ("Tampa", 27.95, -82.46, "America/New_York", 3200), ("Denver", 39.74, -104.99, "America/Denver", 3000),
    ("Baltimore", 39.29, -76.61, "America/New_York", 2800), ("St. Louis", 38.63, -90.20, "America/Chicago", 2800),
    ("Orlando", 28.54, -81.38, "America/New_York", 2700), ("Charlotte", 35.23, -80.84, "America/New_York", 2700),
    ("San Antonio", 29.42, -98.49, "America/Chicago", 2600), ("Portland", 45.52, -122.68, "America/Los_Angeles", 2500),
    ("Sacramento", 38.58, -121.49, "America/Los_Angeles", 2400), ("Pittsburgh", 40.44, -79.99, "America/New_York", 2400),
    ("Austin", 30.27, -97.74, "America/Chicago", 2300), ("Las Vegas", 36.17, -115.14, "America/Los_Angeles", 2300),
    ("Cincinnati", 39.10, -84.51, "America/New_York", 2300), ("Kansas City", 39.10, -94.58, "America/Chicago", 2200),
    ("Columbus", 39.96, -83.00, "America/New_York", 2100), ("Indianapolis", 39.77, -86.16, "America/Indiana/Indianapolis", 2100),
    ("Cleveland", 41.50, -81.69, "America/New_York", 2100), ("Nashville", 36.16, -86.78, "America/Chicago", 2000),
]
_s = requests.Session()


def _get(url, headers=None, tries=7):
    last = None
    for a in range(tries):
        try:
            r = _s.get(url, headers=headers, timeout=60)
            if r.status_code in (200, 206):
                return r.content
            last = r.status_code
        except requests.RequestException as e:
            last = type(e).__name__
        time.sleep(min(2 ** a, 20))
    raise RuntimeError(f"GET failed ({last}): {url}")


def t2m_step(day, step):
    """2 m temperature (deg C) of the 00z cycle of `day` at lead `step`, at the metro points."""
    d = day.strftime("%Y%m%d")
    f = f"gfs.{d}/00/atmos/gfs.t00z.pgrb2.0p25.f{step:03d}"
    lines = _get(f"{BASE}/{f}.idx").decode().splitlines()
    for i, l in enumerate(lines):
        if ":TMP:2 m above ground:" in l:
            start = int(l.split(":")[1])
            end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else None
            break
    else:
        raise RuntimeError(f"no 2 m TMP in index {f}")
    hdr = {"Range": f"bytes={start}-{end}" if end else f"bytes={start}-"}
    buf = _get(f"{BASE}/{f}", headers=hdr)
    if len(buf) >= 16 and buf[:4] == b"GRIB":
        total = int.from_bytes(buf[8:16], "big")
        if len(buf) < total:  # the index's next offset was not this message's end (index order differs on a few days)
            buf = _get(f"{BASE}/{f}", headers={"Range": f"bytes={start}-{start + total - 1}"})
    for h in grib.messages(buf):
        v, _ = grib.sample(h, [m[1] for m in METROS], [m[2] for m in METROS])
        return v - 273.15
    raise RuntimeError("no message")


def cycle(day, workers=16):
    with ThreadPoolExecutor(workers) as ex:
        return np.array(list(ex.map(lambda s: t2m_step(day, s), STEPS)), dtype=np.float32)  # [step, metro]


def load():
    if os.path.exists(OUT):
        z = np.load(OUT, allow_pickle=True)
        return list(z["days"]), list(z["data"])
    return [], []


def save(days, data):
    np.savez(OUT, days=np.array(days), data=np.array(data, dtype=np.float32), steps=np.array(STEPS),
             metros=np.array([m[0] for m in METROS]))


def main():
    days, data = load()
    have = set(days)
    today = datetime.now(timezone.utc).date()
    todo = [START + timedelta(n) for n in range((today - timedelta(days=1) - START).days + 1)]
    todo = [d for d in todo if d.isoformat() not in have]
    print(f"{len(have)} cycles archived, {len(todo)} to fetch", flush=True)
    t0 = time.time()
    for i, d in enumerate(todo):
        try:
            data.append(cycle(d))
            days.append(d.isoformat())
        except Exception as e:  # noqa: BLE001
            print("skip", d, str(e)[:80], flush=True)
            continue
        if (i + 1) % 20 == 0:
            save(days, data)
            print(f"  {d} ({i + 1}/{len(todo)}) {int(time.time() - t0)}s", flush=True)
    save(days, data)
    print("done", len(days), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        t = time.time()
        a = cycle(date(2022, 1, 15))
        print(a.shape, round(time.time() - t, 1), "s", a[0, :5].round(1), a[:, 0].round(1)[:8])
    else:
        main()
