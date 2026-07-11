#!/usr/bin/env python3
"""
============================================================
  IBKR PAPER TRADING LEDGER (simulated, no broker connection)
  Dr. Arun Ramanathan
============================================================
IBKR's own paper-trading API is unavailable on this account
(Lite tier ineligible for Pro without USD 20k net worth), so
this tracks a virtual portfolio locally instead, at the same
250,000 SGD notional size, using only BULLISH tilts (the
bearish side of the signal underperforms in backtest and is
held at neutral weight here rather than shorted/underweighted).

Run this AFTER build_portfolio_dashboard.py each day. It reads
today's suggested weights straight from portfolio_dashboard.html
(same source of truth as log_predictions.py), marks the ledger
to market, rebalances to today's bullish-only target weights,
and appends one row to ibkr_paper_ledger.csv.

A parallel "neutral" (no tilt, constant weights) track runs
alongside it as the benchmark, same as portfolio_predictions.csv
already does for resolved predictions.

Usage:
  python ibkr_paper_ledger.py
============================================================
"""

import os
import re
import json
import warnings
from datetime import date

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_HTML = os.path.join(BASE_DIR, "portfolio_dashboard.html")
LEDGER_CSV     = os.path.join(BASE_DIR, "ibkr_paper_ledger.csv")

STARTING_CAPITAL_SGD = 250_000.0  # matches intended IBKR paper account size
# All instruments (SPY, GC=F, TLT, IBIT, UUP) trade in USD, but the account
# is SGD — NAV/returns are converted to SGD daily using USDSGD=X, so the
# ledger reflects real SGD currency risk, not just the USD asset returns.

TICKERS = {"equities": "SPY", "gold": "GC=F", "bonds": "TLT",
           "crypto": "IBIT", "fx": "UUP"}
CLS_MAP = {"equities": "Equities", "gold": "Gold", "bonds": "Bonds",
           "crypto": "Crypto", "fx": "FX"}

TODAY = date.today()

LEDGER_COLS = [
    "date",
    "usdsgd_rate",
    "equities_tilt", "gold_tilt", "bonds_tilt", "crypto_tilt", "fx_tilt",
    "equities_weight", "gold_weight", "bonds_weight", "crypto_weight", "fx_weight",
    "equities_price", "gold_price", "bonds_price", "crypto_price", "fx_price",
    "equities_shares", "gold_shares", "bonds_shares", "crypto_shares", "fx_shares",
    "equities_shares_neutral", "gold_shares_neutral", "bonds_shares_neutral",
    "crypto_shares_neutral", "fx_shares_neutral",
    "nav_tilt_sgd", "nav_neutral_sgd",
    "daily_return_tilt", "daily_return_neutral",
    "cum_return_tilt", "cum_return_neutral",
    "notes",
]


def fetch_usdsgd_rate():
    hist = yf.download("USDSGD=X", period="5d", auto_adjust=True, progress=False)
    return float(hist["Close"].squeeze().dropna().iloc[-1])


def fetch_dashboard_state():
    """Read today's tilts, weights and prices from portfolio_dashboard.html."""
    if not os.path.exists(DASHBOARD_HTML):
        raise FileNotFoundError(
            "portfolio_dashboard.html not found. Run build_portfolio_dashboard.py first."
        )

    with open(DASHBOARD_HTML, "r", encoding="utf-8") as f:
        html = f.read()

    match = re.search(r'const D = (\{.*?\});\n', html, re.DOTALL)
    if not match:
        raise ValueError("Could not find 'const D = {...}' in portfolio_dashboard.html")
    D = json.loads(match.group(1))

    generated_date = D.get("gen", "")[:10]
    if generated_date != str(TODAY):
        raise RuntimeError(
            f"portfolio_dashboard.html was generated on {generated_date}, not today ({TODAY}).\n"
            f"Run build_portfolio_dashboard.py first, then re-run this script."
        )

    ts   = D["tilt_summaries"]
    snap = D["snapshot"]

    state = {}
    for cls, label in CLS_MAP.items():
        state[cls] = {
            "tilt":       ts[label]["overall_label"],
            "neutral_w":  float(ts[label]["neutral_w"]) / 100.0,
            "suggested_w": float(ts[label]["suggested_w"]) / 100.0,
            "price":      float(snap[label]["price"]),
        }
    return state


def bullish_only_weights(state):
    """
    Apply only the bullish side of the tilt. Bearish labels (TILT DOWN /
    UNDERWEIGHT) fall back to neutral weight instead of being shorted —
    the backtest showed bearish calls are not reliable yet.
    Renormalises so weights sum to 1.
    """
    raw = {}
    for cls, s in state.items():
        if s["tilt"] in ("TILT DOWN", "UNDERWEIGHT"):
            raw[cls] = s["neutral_w"]
        else:  # OVERWEIGHT, TILT UP, NEUTRAL
            raw[cls] = s["suggested_w"]

    total = sum(raw.values())
    return {cls: w / total for cls, w in raw.items()}


def neutral_weights(state):
    raw = {cls: s["neutral_w"] for cls, s in state.items()}
    total = sum(raw.values())
    return {cls: w / total for cls, w in raw.items()}


def main():
    print("=" * 60)
    print("  IBKR PAPER LEDGER (simulated)")
    print(f"  {TODAY}")
    print("=" * 60)

    state = fetch_dashboard_state()
    target_tilt    = bullish_only_weights(state)
    target_neutral = neutral_weights(state)
    usdsgd = fetch_usdsgd_rate()

    if not os.path.exists(LEDGER_CSV):
        pd.DataFrame(columns=LEDGER_COLS).to_csv(LEDGER_CSV, index=False)
        print(f"  Created: {LEDGER_CSV}")

    df = pd.read_csv(LEDGER_CSV)

    if str(TODAY) in df["date"].astype(str).values:
        print(f"  Already logged for {TODAY} — skipping.")
        return

    prices = {cls: state[cls]["price"] for cls in TICKERS}

    if df.empty:
        capital_usd = STARTING_CAPITAL_SGD / usdsgd
        nav_tilt_usd = nav_neutral_usd = capital_usd
        nav_tilt_sgd = nav_neutral_sgd = STARTING_CAPITAL_SGD
        daily_return_tilt = daily_return_neutral = 0.0
        cum_return_tilt = cum_return_neutral = 0.0
        notes = "Ledger opened."
    else:
        prev = df.iloc[-1]
        shares_tilt_prev    = {cls: float(prev[f"{cls}_shares"]) for cls in TICKERS}
        shares_neutral_prev = {cls: float(prev[f"{cls}_shares_neutral"]) for cls in TICKERS}
        nav_tilt_sgd_prev    = float(prev["nav_tilt_sgd"])
        nav_neutral_sgd_prev = float(prev["nav_neutral_sgd"])

        nav_tilt_usd    = sum(shares_tilt_prev[cls] * prices[cls] for cls in TICKERS)
        nav_neutral_usd = sum(shares_neutral_prev[cls] * prices[cls] for cls in TICKERS)
        nav_tilt_sgd    = nav_tilt_usd * usdsgd
        nav_neutral_sgd = nav_neutral_usd * usdsgd

        daily_return_tilt    = nav_tilt_sgd / nav_tilt_sgd_prev - 1
        daily_return_neutral = nav_neutral_sgd / nav_neutral_sgd_prev - 1
        cum_return_tilt    = nav_tilt_sgd / STARTING_CAPITAL_SGD - 1
        cum_return_neutral = nav_neutral_sgd / STARTING_CAPITAL_SGD - 1
        notes = ""

    shares_tilt    = {cls: target_tilt[cls] * nav_tilt_usd / prices[cls] for cls in TICKERS}
    shares_neutral = {cls: target_neutral[cls] * nav_neutral_usd / prices[cls] for cls in TICKERS}

    row = {"date": str(TODAY), "usdsgd_rate": round(usdsgd, 5)}
    for cls in TICKERS:
        row[f"{cls}_tilt"]            = state[cls]["tilt"]
        row[f"{cls}_weight"]          = round(target_tilt[cls] * 100, 2)
        row[f"{cls}_price"]           = round(prices[cls], 4)
        row[f"{cls}_shares"]          = round(shares_tilt[cls], 6)
        row[f"{cls}_shares_neutral"]  = round(shares_neutral[cls], 6)
    row["nav_tilt_sgd"]          = round(nav_tilt_sgd, 2)
    row["nav_neutral_sgd"]       = round(nav_neutral_sgd, 2)
    row["daily_return_tilt"]     = round(daily_return_tilt * 100, 4)
    row["daily_return_neutral"]  = round(daily_return_neutral * 100, 4)
    row["cum_return_tilt"]       = round(cum_return_tilt * 100, 4)
    row["cum_return_neutral"]    = round(cum_return_neutral * 100, 4)
    row["notes"]                 = notes

    df = pd.concat([df, pd.DataFrame([row], columns=LEDGER_COLS)], ignore_index=True)
    df.to_csv(LEDGER_CSV, index=False)

    print(f"  USDSGD        : {usdsgd:.4f}")
    print(f"  NAV (tilt)    : SGD {nav_tilt_sgd:,.2f}  ({cum_return_tilt*100:+.2f}% cum, "
          f"{daily_return_tilt*100:+.2f}% today)")
    print(f"  NAV (neutral) : SGD {nav_neutral_sgd:,.2f}  ({cum_return_neutral*100:+.2f}% cum, "
          f"{daily_return_neutral*100:+.2f}% today)")
    print(f"  Weights (bullish-only): " +
          "  ".join(f"{cls}={target_tilt[cls]*100:.1f}%" for cls in TICKERS))
    print(f"  Row appended → {LEDGER_CSV}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
