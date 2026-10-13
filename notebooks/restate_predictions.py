#!/usr/bin/env python3
"""
One-off restatement of gold_predictions.csv, metals_predictions.csv and portfolio_predictions.csv
under the v2 scoring rules in log_predictions.py (2026-10-13):

  * horizons in TRADING days (the unit CPE, the dashboards, predictor_ledger.py and cpe_signal_ledger.py use),
  * entry = last settled close before the prediction date, outcome = the close h trading bars later,
  * one instrument at both ends (gold/silver/platinum on GLD/SLV/PPLT; portfolio on its five proxies).

It does NOT use any separate scoring code: it clears every outcome and runs the same resolve_gold /
resolve_metals / resolve_portfolio functions the daily pipeline runs, so a restated row is exactly what the
daily run would have produced. Rows whose trading-day horizon has not elapsed go back to PENDING.

The original files are copied to legacy_calendar_horizon/ first and never overwritten.
monitor_history.csv rows for the gold / metals / portfolio dashboards are then rebuilt from the restated ledgers
(rows for every other dashboard are left untouched).

Usage:
  python restate_predictions.py                 # restate the real files in this folder
  python restate_predictions.py --dir /tmp/x    # restate copies in another folder (tests); skips history
"""
import argparse
import contextlib
import importlib.util
import io
import os
import shutil
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["gold_predictions.csv", "metals_predictions.csv", "portfolio_predictions.csv"]

OUTCOME_RESET = {
    "gold_predictions.csv": ["outcome_date", "gcf_price_at_outcome", "actual_return_pct", "prediction_correct",
                             "scoring_ticker", "scored_entry_date", "scored_entry_px", "scored_outcome_px"],
    "metals_predictions.csv": ["outcome_date", "price_at_outcome", "actual_return_pct", "prediction_correct",
                               "scoring_ticker", "scored_entry_date", "scored_entry_px", "scored_outcome_px"],
    "portfolio_predictions.csv": ["outcome_date", "equities_price_out", "gold_price_out", "bonds_price_out",
                                  "crypto_price_out", "fx_price_out", "equities_return", "gold_return",
                                  "bonds_return", "crypto_return", "fx_return", "tilt_pnl", "neutral_pnl",
                                  "tilt_beat_neutral", "scored_entry_date", "equities_entry_px", "gold_entry_px",
                                  "bonds_entry_px", "crypto_entry_px", "fx_entry_px"],
}


def load_logger(directory):
    spec = importlib.util.spec_from_file_location("lp", os.path.join(HERE, "log_predictions.py"))
    lp = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(lp)
    lp.GOLD_CSV = os.path.join(directory, "gold_predictions.csv")
    lp.METALS_CSV = os.path.join(directory, "metals_predictions.csv")
    lp.PORT_CSV = os.path.join(directory, "portfolio_predictions.csv")
    return lp


def backup(directory):
    legacy = os.path.join(directory, "legacy_calendar_horizon")
    os.makedirs(legacy, exist_ok=True)
    for f in FILES:
        dst = os.path.join(legacy, f)
        if os.path.exists(dst):
            print(f"  backup exists, kept: {dst}")
        else:
            shutil.copy2(os.path.join(directory, f), dst)
            print(f"  backed up -> {dst}")


def reset_outcomes(directory):
    for f in FILES:
        path = os.path.join(directory, f)
        df = pd.read_csv(path)
        for c in OUTCOME_RESET[f]:
            if c not in df.columns:
                df[c] = ""
            df[c] = ""
        df["status"] = "PENDING"
        df.to_csv(path, index=False)


def rebuild_history(directory):
    sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location("mh", os.path.join(HERE, "monitor_history.py"))
    mh = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mh)
    hist = mh.load_history()
    first = pd.to_datetime(hist["date"]).min()
    last = pd.to_datetime(hist["date"]).max()
    days = list(pd.bdate_range(first, last))
    rows, names = [], set()
    for d in days:
        for a, b, v, n in mh.gold_metrics(d) + mh.metals_metrics(d) + mh.portfolio_metrics(d):
            rows.append((d.strftime("%Y-%m-%d"), a, b, round(v, 6), int(n)))
            names.add(a)
    kept = hist[~hist["dashboard"].isin(names)]
    new = pd.DataFrame(rows, columns=mh.COLS)
    out = pd.concat([kept, new], ignore_index=True).sort_values(["date", "dashboard", "metric"])
    out.to_csv(mh.HISTORY_PATH, index=False)
    print(f"  monitor_history.csv: replaced dashboards {sorted(names)}; {len(new)} rows over {len(days)} days; "
          f"other dashboards untouched ({len(kept)} rows kept)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    args = ap.parse_args()
    directory = os.path.abspath(args.dir)
    real = directory == HERE

    print("1) backing up the original ledgers")
    backup(directory)
    print("2) clearing every outcome")
    reset_outcomes(directory)
    print("3) re-scoring every row with the daily run's own resolvers")
    lp = load_logger(directory)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lp.resolve_gold()
        lp.resolve_metals()
        lp.resolve_portfolio()
    for line in buf.getvalue().splitlines():
        if "prediction(s) resolved" in line:
            print("  " + line.strip())
    if real:
        print("4) rebuilding the monitor's score history for gold / metals / portfolio")
        rebuild_history(directory)
    else:
        print("4) (test folder: monitor history not rebuilt)")


if __name__ == "__main__":
    main()
