"""Recompute every CPE row of monitor_history.csv from the current (internally consistent) price history.

Run from notebooks/:  python legacy_adjustment_vintage/restate_cpe_monitor.py
Each date's CPE metrics are recomputed with only events entered on or before that date (monitor_history.cpe_metrics),
and only the rows with dashboard == 'cpe' are replaced; other dashboards' rows are untouched.
"""
import os, sys
sys.path.insert(0, os.getcwd())
import pandas as pd
import monitor_history as mh

hist = mh.load_history()
dates = sorted(hist.loc[hist["dashboard"] == "cpe", "date"].unique())
new = []
for d in dates:
    new += [(d, a, b, round(v, 6), int(n)) for a, b, v, n in mh.cpe_metrics(pd.Timestamp(d))]
keep = hist[hist["dashboard"] != "cpe"]
out = pd.concat([keep, pd.DataFrame(new, columns=mh.COLS)], ignore_index=True).sort_values(["date", "dashboard", "metric"])
out.to_csv(mh.HISTORY_PATH, index=False)
print(f"restated CPE rows for {len(dates)} dates ({len(new)} rows)")
