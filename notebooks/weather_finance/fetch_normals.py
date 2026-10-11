"""ERA5 daily mean temperature 1996-2025 at the 34 metros (Open-Meteo daily endpoint) -> population-weighted national
HDD/CDD climatology by leap-aware day of year (15-day circular smoothing), for the anomaly signal."""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "climate-energy-dashboard")))
from gfs_archive import METROS  # noqa: E402
from weatherdash import truth  # noqa: E402
from weatherdash.calendar_utils import doy366  # noqa: E402

CACHE = os.path.join(HERE, "data", "metro_hist")
os.makedirs(CACHE, exist_ok=True)
w = np.array([m[4] for m in METROS], float)
w /= w.sum()
hdd_tot = None
for j, (name, lat, lon, tz, pop) in enumerate(METROS):
    cp = os.path.join(CACHE, f"{name.replace(' ', '_').replace('.', '')}.csv")
    if os.path.exists(cp):
        t = pd.read_csv(cp, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        t = truth.city_daily_history(lat, lon, tz, "1996-01-01", "2025-12-31").rename("t")
        t.to_csv(cp)
        time.sleep(1.0)
    f = t * 9 / 5 + 32
    hdd = np.maximum(65 - f, 0) * w[j]
    cdd = np.maximum(f - 65, 0) * w[j]
    part = pd.DataFrame({"hdd": hdd, "cdd": cdd})
    hdd_tot = part if hdd_tot is None else hdd_tot.add(part, fill_value=0)
    print(j + 1, name, len(t), flush=True)
d = hdd_tot.groupby(doy366(hdd_tot.index)).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
sm = pd.concat([d.iloc[-7:], d, d.iloc[:7]]).rolling(15, center=True).mean().iloc[7:-7]
sm.index = range(1, 367)
sm.to_csv(os.path.join(HERE, "data", "era5_degree_day_normals.csv"))
print(sm.iloc[[0, 59, 180, 355]].round(2))
