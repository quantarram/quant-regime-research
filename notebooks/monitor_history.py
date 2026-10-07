#!/usr/bin/env python3
"""
============================================================
  PERFORMANCE MONITOR -- SCORE HISTORY
  Dr. Arun Ramanathan
============================================================
The monitor page shows each dashboard's score as of today. This keeps the
score as of every earlier day too, in monitor_history.csv, so the page can show
how a score has moved as outcomes arrive.

Long format, one row per (date, dashboard, metric):
    date, dashboard, metric, value, n

`date` means "using only outcomes known by the close of that date". Every
metric is recomputed from the dashboards' own ledgers with outcomes filtered to
outcome_date <= date, so earlier days can be reconstructed exactly and there is
no difference between a backfilled row and one written live. Re-running a date
replaces its rows.

Usage:
  python monitor_history.py                 # daily: write yesterday's (latest complete day's) snapshot
  python monitor_history.py --backfill      # one-off: rebuild every day since each ledger's first outcome
============================================================
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import predictor_ledger as pled  # noqa: E402
import cpe_signal_ledger as cled  # noqa: E402

HISTORY_PATH = os.path.join(HERE, "monitor_history.csv")
COLS = ["date", "dashboard", "metric", "value", "n"]


def _csv(name):
    return pd.read_csv(os.path.join(HERE, name))


# ---------------------------------------------------------------------------
# metric builders: each returns a list of (dashboard, metric, value, n) as of `d`
# ---------------------------------------------------------------------------
def _direction_metrics(label, df, d):
    df = df[(df["status"] == "RESOLVED") & (pd.to_datetime(df["outcome_date"]) <= d)].copy()
    if df.empty:
        return []
    df["prediction_correct"] = df["prediction_correct"].astype(float)
    df["rose"] = df["actual_return_pct"].astype(float) > 0
    bull = df[df["direction"] == "BULLISH"]
    out = [(label, "price_rose_all", float(df["rose"].mean()), len(df))]
    if len(bull):
        out.append((label, "bullish_hit", float(bull["prediction_correct"].mean()), len(bull)))
    return out


def gold_metrics(d):
    return _direction_metrics("gold", _csv("gold_predictions.csv"), d)


def metals_metrics(d):
    m = _csv("metals_predictions.csv")
    return _direction_metrics("metals", m, d)


def portfolio_metrics(d):
    p = _csv("portfolio_predictions.csv")
    p = p[(p["status"] == "RESOLVED") & (pd.to_datetime(p["outcome_date"]) <= d)]
    if p.empty:
        return []
    t, n = p["tilt_pnl"].astype(float), p["neutral_pnl"].astype(float)
    return [("portfolio", "tilt_pnl_avg", float(t.mean()), len(p)),
            ("portfolio", "neutral_pnl_avg", float(n.mean()), len(p))]


def football_metrics(d):
    q = _csv(os.path.join("football_betting", "output", "qualifying_log.csv"))
    q = q[(q["status"] == "RESOLVED") & (pd.to_datetime(q["start_time"]).dt.tz_localize(None) <= d + timedelta(days=1))]
    if q.empty:
        return []
    placed = q[q["bet_placed"].astype(str).str.lower() == "true"]
    out = [("football", "win_rate", float(q["won"].astype(float).mean()), len(q)),
           ("football", "implied_rate", float((1 / q["odds"].astype(float)).mean()), len(q))]
    if len(placed):
        out.append(("football", "cum_profit_sgd", float(placed["actual_pnl"].astype(float).sum()), len(placed)))
    return out


def predictor_metrics(d, ledger=None):
    df = pled.load_ledger() if ledger is None else ledger
    df = df.copy()
    keep = (df["status"] == "RESOLVED") & (pd.to_datetime(df["outcome_date"]) <= d)
    df = df[keep]
    if df.empty:
        return []
    res = pled._resolved_numeric(df)
    out = []
    for h, g in res.groupby("horizon_days"):
        out += [("predictor", f"mape_{int(h)}d", float(g["abs_pct_error_q50"].mean()), len(g)),
                ("predictor", f"no_change_mape_{int(h)}d", float(g["naive_abs_pct_error"].mean()), len(g))]
    return out


def cpe_metrics(d):
    ev = cled.load_events()
    if pd.Timestamp(d) <= cled.TABLE_END:
        return []
    s = cled.live_stats(ev=ev, asof=d)
    out = []
    for r in s["by_horizon"]:
        if r["n"] >= 1 and r["horizon"] in (1, 21, 63):
            h = r["horizon"]
            out += [("cpe", f"claimed_{h}d", r["claimed"], r["n"]), ("cpe", f"realised_{h}d", r["hit"], r["n"]),
                    ("cpe", f"live_base_{h}d", r["live_base"], r["n"])]
    return out


def snapshot(d, include_cpe=True):
    d = pd.Timestamp(d)
    rows = gold_metrics(d) + metals_metrics(d) + portfolio_metrics(d) + football_metrics(d) + predictor_metrics(d)
    if include_cpe:
        rows += cpe_metrics(d)
    return [(d.strftime("%Y-%m-%d"), a, b, round(v, 6), int(n)) for a, b, v, n in rows]


# ---------------------------------------------------------------------------
def load_history():
    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(HISTORY_PATH)


def write_rows(new_rows):
    hist = load_history()
    new = pd.DataFrame(new_rows, columns=COLS)
    if new.empty:
        return hist
    dates = set(new["date"])
    hist = hist[~hist["date"].isin(dates)]  # re-running a date replaces its rows
    hist = pd.concat([hist, new], ignore_index=True).sort_values(["date", "dashboard", "metric"])
    hist.to_csv(HISTORY_PATH, index=False)
    return hist


def snapshot_date_today():
    """Latest complete day: yesterday (UTC). Skips weekends, when no new bars or outcomes can have arrived."""
    d = datetime.now(timezone.utc).date() - timedelta(days=1)
    return pd.Timestamp(d)


def append_today():
    d = snapshot_date_today()
    if d.weekday() >= 5:
        print(f"Monitor history: {d.date()} is a weekend, no snapshot written")
        return load_history()
    rows = snapshot(d)
    hist = write_rows(rows)
    print(f"Monitor history: {len(rows)} metric(s) written for {d.date()}; {hist['date'].nunique()} snapshot dates in file")
    return hist


def backfill():
    first = min(pd.to_datetime(_csv("gold_predictions.csv").query("status=='RESOLVED'")["outcome_date"]).min(),
                pd.Timestamp("2026-06-09"))
    last = snapshot_date_today()
    days = [d for d in pd.bdate_range(first, last)]
    ledger = pled.load_ledger()
    fridays = {d for d in days if d.weekday() == 4} | {days[-1]}
    all_rows = []
    for i, d in enumerate(days):
        rows = gold_metrics(d) + metals_metrics(d) + portfolio_metrics(d) + football_metrics(d) + predictor_metrics(d, ledger)
        if d in fridays:  # CPE scoring is the slow part; weekly through the past, daily from here on
            rows += cpe_metrics(d)
        all_rows += [(d.strftime("%Y-%m-%d"), a, b, round(v, 6), int(n)) for a, b, v, n in rows]
        if i % 10 == 0:
            print(f"  {d.date()}  ({i + 1}/{len(days)})")
    hist = write_rows(all_rows)
    print(f"Backfilled {len(days)} days -> {len(hist)} rows, {hist['date'].nunique()} dates")
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    args = ap.parse_args()
    backfill() if args.backfill else append_today()


if __name__ == "__main__":
    sys.exit(main())
