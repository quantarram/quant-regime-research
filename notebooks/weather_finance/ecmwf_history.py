"""Historical IFS 00z forecasts (2024-06 onward, steps limited to 240 h, which is all the 2024 archive has) turned into regional demand, wind and
solar forecasts with the dashboard's own models. Writes data/ecmwf_hist_<tag>.csv (resumable); never touches the live ledger.

  python ecmwf_history.py START END TAG     # e.g. 20240601 20241231 a
"""
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DASH = os.path.abspath(os.path.join(HERE, "..", "..", "..", "climate-energy-dashboard"))
sys.path.insert(0, DASH)
from weatherdash import forecast  # noqa: E402

forecast.STEPS = [s for s in forecast.STEPS if s <= 240]


def _renew_at_points(date, cycle, step, lats, lons):
    """As the dashboard's, but tolerant of archive cycles whose 100 m wind messages decode as shortName u / v instead of 100u / 100v."""
    buf = forecast.fetch.ifs_fetch(date, cycle, step, forecast.RENEW_FIELDS)
    out = {}
    for h in forecast.grib.messages(buf):
        v, m = forecast.grib.sample(h, lats, lons)
        out[{"u": "100u", "v": "100v"}.get(m["shortName"], m["shortName"])] = v
    return out


forecast._renew_at_points = _renew_at_points
start, end, tag = sys.argv[1], sys.argv[2], sys.argv[3]
stride = int(sys.argv[4]) if len(sys.argv) > 4 else 1
OUT = os.path.join(HERE, "data", f"ecmwf_hist_{tag}.csv")
done = set()
if os.path.exists(OUT):
    done = set(pd.read_csv(OUT, usecols=["issue_date"])["issue_date"].astype(str))
d = datetime.strptime(start, "%Y%m%d")
last = datetime.strptime(end, "%Y%m%d")
while d <= last:
    iso = d.strftime("%Y-%m-%d")
    if iso not in done:
        try:
            a = forecast.run(d.strftime("%Y%m%d"), 0, workers=24)
            b = forecast.run_renewables(d.strftime("%Y%m%d"), 0, workers=24)
            a = a[a.lead_days <= 9]
            b = b[b.lead_days <= 9]
            w = b.pivot_table(index=["region", "valid_date", "lead_days"], columns="resource", values=["gen_fc_mwh", "gen_clim_mwh"]).reset_index()
            w.columns = ["_".join([c for c in col if c]) for col in w.columns]
            m = a[["region", "valid_date", "lead_days", "t_fc_c", "t_clim_c", "demand_fc_mwh", "demand_clim_mwh"]].merge(w, on=["region", "valid_date", "lead_days"], how="left")
            m.insert(0, "issue_date", iso)
            m.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(iso, len(m), flush=True)
        except Exception as e:  # noqa: BLE001
            print("FAILED", iso, type(e).__name__, str(e)[:120], flush=True)
    d += timedelta(days=stride)
print("done", tag)
