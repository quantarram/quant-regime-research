#!/usr/bin/env python3
"""
============================================================
  PREDICTOR_V1 LIVE FORECAST LEDGER
  Dr. Arun Ramanathan
============================================================
build_predictor_dashboard.py overwrites predictor_dashboard.html every
day, so until this ledger existed there was no persistent record of the
22-instrument price forecasts and nothing scoring them against what the
price later did (the gold / portfolio / metals dashboards already have
that via log_predictions.py).

This script keeps one row per (ticker, horizon, as_of_date) forecast in
predictor_forecasts.csv:
  - the forecast as shown on the dashboard (price at forecast, q10..q90)
  - once enough trading days have passed, the realized outcome and the
    scores: absolute % error of the median forecast, the no-change
    benchmark's error, whether the realized price landed inside the
    q10-q90 and q25-q75 intervals, and whether the median forecast
    called the direction of the move.

Horizons are in TRADING DAYS (the same definition the models are
trained and backtested on): the outcome is the close `horizon` rows
after the as-of row of that instrument's own price series, not the
calendar-day target_date_est shown on the dashboard (an estimate only).
A forecast resolves once that close exists and its date is before
today (UTC), so a still-forming bar is never used.

A forecast key (ticker, horizon, as-of date) can be rebuilt more than once
(weekend rebuilds, a still-forming last bar on 24h instruments such as
EURUSD/GLD, model retrains). The ledger keeps the FIRST version published,
so a forecast is always scored as it stood before its outcome was known.

Usage:
  python predictor_ledger.py              # daily: append today's forecasts, resolve what has matured
  python predictor_ledger.py --backfill   # one-off: rebuild the ledger from predictor_dashboard.html's git history
  python predictor_ledger.py --summary    # print resolved-so-far scores by horizon and instrument
============================================================
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(HERE, "predictor_dashboard.html")
LEDGER_PATH = os.path.join(HERE, "predictor_forecasts.csv")
PRICES_PATH = os.path.join(HERE, "multiasset_prices_live_history.parquet")
PROXY_PATH = os.path.join(HERE, "predictor_v1", "sector_proxy_cache.parquet")
PROXY_TICKERS = ("IYR", "VOX")

KEY = ["ticker", "horizon_days", "forecast_date"]
FORECAST_COLS = ["forecast_date", "ticker", "model_type", "horizon_days", "post_processed",
                 "price_at_forecast", "target_date_est",
                 "q10", "q25", "q50", "q75", "q90"]
OUTCOME_COLS = ["status", "outcome_date", "price_at_outcome", "actual_return_pct",
                "forecast_return_pct", "abs_pct_error_q50", "naive_abs_pct_error",
                "in_80_interval", "in_50_interval", "direction_correct"]
COLS = FORECAST_COLS + OUTCOME_COLS + ["source"]


def _empty():
    return pd.DataFrame(columns=COLS)


def load_ledger():
    if not os.path.exists(LEDGER_PATH):
        return _empty()
    df = pd.read_csv(LEDGER_PATH, dtype={"forecast_date": str, "target_date_est": str, "outcome_date": str})
    return df.reindex(columns=COLS)


def save_ledger(df):
    df = df.sort_values(["forecast_date", "ticker", "horizon_days"]).reset_index(drop=True)
    df.to_csv(LEDGER_PATH, index=False)


def parse_dashboard(html, source):
    """Extract the embedded data bundle (`const D = {...};`) into forecast rows."""
    m = re.search(r"const D = (\{.*?\});\s*\n", html, re.S)
    if not m:
        raise ValueError("no data bundle found in dashboard HTML")
    D = json.loads(m.group(1))
    rows = []
    for i in D["instruments"]:
        rows.append({
            "forecast_date": str(i["as_of_date"])[:10],
            "ticker": i["ticker"],
            "model_type": i["model_type"],
            "horizon_days": int(i["horizon"]),
            "post_processed": bool(i["post_processed"]),
            "price_at_forecast": i["price_now"],
            "target_date_est": str(i["target_date_est"])[:10],
            "q10": i["q10"], "q25": i["q25"], "q50": i["q50"], "q75": i["q75"], "q90": i["q90"],
            "status": "PENDING", "source": source,
        })
    return rows


def load_price_series():
    prices = pd.read_parquet(PRICES_PATH)
    proxy = pd.read_parquet(PROXY_PATH)
    series = {}
    for t in prices.columns:
        series[t] = prices[t].dropna()
    for t in PROXY_TICKERS:
        series[t] = proxy[t].dropna()
    return series


def resolve(df, series, today_utc):
    """Fill outcomes for PENDING rows whose horizon-th trading-day close exists."""
    n_resolved = 0
    for idx in df.index[df["status"] == "PENDING"]:
        r = df.loc[idx]
        s = series.get(r["ticker"])
        if s is None:
            continue
        asof = pd.Timestamp(r["forecast_date"])
        pos = s.index.searchsorted(asof)
        if pos >= len(s) or s.index[pos] != asof:
            continue  # as-of date not in this series (should not happen) -- leave pending
        tpos = pos + int(r["horizon_days"])
        if tpos >= len(s):
            continue
        outcome_date = s.index[tpos]
        if outcome_date.date() >= today_utc:
            continue  # bar may still be forming
        actual = float(s.iloc[tpos])
        p0 = float(r["price_at_forecast"])
        q = {k: float(r[k]) for k in ("q10", "q25", "q50", "q75", "q90")}
        df.loc[idx, "status"] = "RESOLVED"
        df.loc[idx, "outcome_date"] = str(outcome_date.date())
        df.loc[idx, "price_at_outcome"] = round(actual, 4)
        df.loc[idx, "actual_return_pct"] = round((actual / p0 - 1) * 100, 3)
        df.loc[idx, "forecast_return_pct"] = round((q["q50"] / p0 - 1) * 100, 3)
        df.loc[idx, "abs_pct_error_q50"] = round(abs(q["q50"] - actual) / actual * 100, 3)
        df.loc[idx, "naive_abs_pct_error"] = round(abs(p0 - actual) / actual * 100, 3)
        df.loc[idx, "in_80_interval"] = bool(q["q10"] <= actual <= q["q90"])
        df.loc[idx, "in_50_interval"] = bool(q["q25"] <= actual <= q["q75"])
        df.loc[idx, "direction_correct"] = bool(np.sign(q["q50"] - p0) == np.sign(actual - p0))
        n_resolved += 1
    return n_resolved


def append_new(df, new_rows):
    """Add forecasts whose key is not yet in the ledger (first-logged version is kept)."""
    if not new_rows:
        return df, 0
    new = pd.DataFrame(new_rows).reindex(columns=COLS)
    have = set(map(tuple, df[KEY].astype(str).values)) if len(df) else set()
    keep = [tuple(map(str, k)) not in have for k in new[KEY].values]
    new = new[keep].drop_duplicates(KEY, keep="first")
    if new.empty:
        return df, 0
    out = new if df.empty else pd.concat([df, new], ignore_index=True)
    return out, len(new)


def git_history_rows():
    """Every historical version of predictor_dashboard.html, oldest first."""
    top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True).strip()
    repo_rel = os.path.relpath(HTML_PATH, top)
    commits = subprocess.check_output(
        ["git", "log", "--reverse", "--format=%h", "--follow", "--", repo_rel], cwd=top, text=True).split()
    rows, n_ok = [], 0
    for h in commits:
        try:
            html = subprocess.check_output(["git", "show", f"{h}:{repo_rel}"], cwd=top, text=True,
                                           stderr=subprocess.DEVNULL)
            rows.extend(parse_dashboard(html, f"git:{h}"))
            n_ok += 1
        except Exception as e:
            print(f"  skipped {h}: {e}")
    print(f"  parsed {n_ok}/{len(commits)} historical dashboard versions")
    return rows


def _resolved_numeric(df):
    res = df[df["status"] == "RESOLVED"].copy()
    for c in ("abs_pct_error_q50", "naive_abs_pct_error", "actual_return_pct", "forecast_return_pct"):
        res[c] = res[c].astype(float)
    for c in ("in_80_interval", "in_50_interval", "direction_correct"):
        res[c] = res[c].astype(str).str.lower() == "true"
    return res


def _score_block(g):
    return {
        "n": int(len(g)),
        "mape": round(float(g["abs_pct_error_q50"].mean()), 3),
        "naive_mape": round(float(g["naive_abs_pct_error"].mean()), 3),
        "cov80": round(float(g["in_80_interval"].mean()), 4),
        "cov50": round(float(g["in_50_interval"].mean()), 4),
        "direction": round(float(g["direction_correct"].mean()), 4),
    }


def live_stats(df):
    """JSON-safe summary of the ledger for dashboards (predictor panel, performance monitor)."""
    res = _resolved_numeric(df)
    pend = df[df["status"] == "PENDING"]
    out = {
        "first_forecast": str(df["forecast_date"].min()) if len(df) else None,
        "last_forecast": str(df["forecast_date"].max()) if len(df) else None,
        "last_outcome": str(res["outcome_date"].max()) if len(res) else None,
        "n_logged": int(len(df)), "n_resolved": int(len(res)), "n_pending": int(len(pend)),
        "n_instruments": int(df["ticker"].nunique()) if len(df) else 0,
        "n_instruments_resolved": int(res["ticker"].nunique()) if len(res) else 0,
        "by_horizon": [], "by_ticker": {},
    }
    for h in sorted(df["horizon_days"].unique()) if len(df) else []:
        gh = res[res["horizon_days"] == h]
        ph = pend[pend["horizon_days"] == h]
        row = {"horizon": int(h), "tickers": sorted(df.loc[df["horizon_days"] == h, "ticker"].unique().tolist()),
               "n_pending": int(len(ph)),
               "first_pending_target": str(ph["target_date_est"].min()) if len(ph) else None,
               "n": 0}
        if len(gh):
            row.update(_score_block(gh))
            row["n_instruments"] = int(gh["ticker"].nunique())
        out["by_horizon"].append(row)
    for t, g in res.groupby("ticker"):
        out["by_ticker"][t] = _score_block(g)
    return out


def summary(df):
    res = df[df["status"] == "RESOLVED"].copy()
    print(f"\nLedger: {len(df)} forecasts logged, {len(res)} resolved, {len(df) - len(res)} pending")
    if res.empty:
        return
    for c in ("abs_pct_error_q50", "naive_abs_pct_error", "actual_return_pct"):
        res[c] = res[c].astype(float)
    for c in ("in_80_interval", "in_50_interval", "direction_correct"):
        res[c] = res[c].astype(str).str.lower() == "true"
    g = res.groupby("horizon_days").agg(
        n=("ticker", "size"), instruments=("ticker", "nunique"),
        mape=("abs_pct_error_q50", "mean"), naive_mape=("naive_abs_pct_error", "mean"),
        cov80=("in_80_interval", "mean"), cov50=("in_50_interval", "mean"),
        direction=("direction_correct", "mean")).round(3)
    print(g.to_string())
    pend = df[df["status"] == "PENDING"].groupby("horizon_days")["target_date_est"].min()
    print("\nEarliest estimated resolution still pending, by horizon:")
    print(pend.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    df = load_ledger()
    today_utc = datetime.now(timezone.utc).date()

    if args.summary:
        summary(df)
        return

    if args.backfill:
        print("Backfilling from git history of predictor_dashboard.html ...")
        df, n_new = append_new(df, git_history_rows())
        print(f"  added {n_new} unique forecasts")
    else:
        with open(HTML_PATH, encoding="utf-8") as f:
            rows = parse_dashboard(f.read(), "live")
        df, n_new = append_new(df, rows)
        print(f"Predictor ledger: {n_new} new forecast(s) logged")

    df[OUTCOME_COLS] = df[OUTCOME_COLS].astype(object)  # mixed str/float/bool columns
    n_res = resolve(df, load_price_series(), today_utc)
    print(f"Predictor ledger: {n_res} forecast(s) resolved")
    save_ledger(df)
    summary(df)


if __name__ == "__main__":
    sys.exit(main())
