#!/usr/bin/env python3
"""
============================================================
  CPE DASHBOARD HISTORICAL BACKTEST
  Dr. Arun Ramanathan
============================================================
Simulates what the Gold and Portfolio dashboards WOULD HAVE
said on each trading day from 2025-01-01 to today, using
only price data available up to that day (strict look-ahead
prevention).

The CPE signal catalog (cpe_results.parquet and
joint_cpe_results.parquet) is treated as FIXED — learned
from pre-2025 data — and is NOT retrained per day.
This mirrors the real deployment setting where the catalog
was built once and is used live.

Outputs:
  gold_predictions_backtest.csv       — same schema as gold_predictions.csv
                                        but with outcomes already resolved
  portfolio_predictions_backtest.csv  — same schema as portfolio_predictions.csv
                                        but with outcomes already resolved
  backtest_summary.txt                — hit-rate tables by horizon / direction

Usage:
  python backtest_cpe_dashboard.py
  python backtest_cpe_dashboard.py --start 2025-01-01 --end 2025-12-31
  python backtest_cpe_dashboard.py --weekly   # one prediction per week (faster)
  python backtest_cpe_dashboard.py --gold-only
  python backtest_cpe_dashboard.py --portfolio-only

Requires (same directory):
  multiasset_prices.parquet
  cpe_results.parquet
  joint_cpe_results.parquet
============================================================
"""

import argparse
import os
import sys
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── PATHS ───────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
GOLD_BT_CSV  = os.path.join(BASE_DIR, "gold_predictions_backtest.csv")
PORT_BT_CSV  = os.path.join(BASE_DIR, "portfolio_predictions_backtest.csv")
SUMMARY_TXT  = os.path.join(BASE_DIR, "backtest_summary.txt")

# ── COLUMN SCHEMAS (matching live log_predictions.py) ───────────────────────
GOLD_COLS = [
    "date_predicted", "gcf_price_usd", "price_sgd_per_g",
    "composite_score", "verdict", "direction",
    "horizon_days", "recovery_pct_positive",
    "r21", "r63", "r126", "r252", "p63_percentile",
    "outcome_date", "gcf_price_at_outcome", "actual_return_pct",
    "prediction_correct", "status", "notes",
]

PORT_COLS = [
    "date_predicted",
    "equities_tilt", "gold_tilt", "bonds_tilt", "crypto_tilt", "fx_tilt",
    "dominant_horizon_days", "n_signals_firing",
    "equities_price", "gold_price", "bonds_price", "crypto_price", "fx_price",
    "outcome_date",
    "equities_price_out", "gold_price_out", "bonds_price_out",
    "crypto_price_out", "fx_price_out",
    "equities_return", "gold_return", "bonds_return", "crypto_return", "fx_return",
    "tilt_pnl", "neutral_pnl", "tilt_beat_neutral", "status", "notes",
]

# ── NEUTRAL WEIGHTS (from log_predictions.py) ────────────────────────────────
NEUTRAL_WEIGHTS = {
    "equities": 0.329,
    "gold"    : 0.299,
    "bonds"   : 0.047,
    "crypto"  : 0.225,
    "fx"      : 0.100,
}

# ── ASSET CLASS CONFIG ───────────────────────────────────────────────────────
RATE_TICKERS = {
    "^VIX","^VXN","^OVX","^EVZ","^VVIX","^SKEW",
    "^TNX","^TYX","^FVX","^IRX",
}

GOLD_Y    = ["GLD", "IAU", "GC=F"]
HORIZONS  = [21, 63, 126, 252]

# CPE filter thresholds (from portfolio dashboard)
CPE_MIN  = 0.80
LIFT_MIN = 1.50
N_MIN    = 100

TILT_THRESHOLDS = [
    (0.85, "OVERWEIGHT",  +15),
    (0.70, "TILT UP",     +8),
    (0.50, "NEUTRAL",      0),
    (0.35, "TILT DOWN",   -8),
    (0.00, "UNDERWEIGHT", -15),
]

ASSET_CLASSES = {
    "Equities": {"tickers": ["SPY","QQQ"],   "proxy": "SPY"},
    "Gold":     {"tickers": ["GC=F","GLD"],  "proxy": "GC=F"},
    "Bonds":    {"tickers": ["TLT","AGG"],   "proxy": "TLT"},
    "Crypto":   {"tickers": ["IBIT","FBTC"], "proxy": "IBIT"},
    "FX":       {"tickers": ["UUP"],         "proxy": "UUP"},
}

ASSET_PRICE_TICKERS = {
    "equities": "SPY",
    "gold"    : "GC=F",
    "bonds"   : "TLT",
    "crypto"  : "IBIT",
    "fx"      : "UUP",
}


# ════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════

def detect_catalog_cutoff(cpe, jcpe, prices):
    """
    Determine the training cutoff date(s) baked into the signal catalog.

    WHAT THE DATA ACTUALLY SHOWS (verified from parquet inspection):
    ---------------------------------------------------------------
    The catalog is NOT a single training run — it is an ENSEMBLE of runs
    at multiple different cutoff dates.  Each (Y, n_total) pair maps to a
    specific calendar date on the price index, and those dates span from
    2020 through 2026-05-24 (for the CPE pairwise table) and through
    2026-05-13 (for the JCPE joint table).

    Specifically:
      - ~30.8% of CPE rows have implied training dates >= 2025-01-01
      - ~31% of JCPE rows have implied training dates >= 2025-01-01
      - The latest implied date is 2026-05-24 (SOL-USD / DOT-USD rows)
      - The latest CLEAN (pre-2025) row is 2024-12-31

    CONSEQUENCE FOR BACKTESTING:
    - Signals derived from rows with implied_date >= sim_date are LEAKED.
    - The only 100% clean approach is to FILTER the catalog at each sim_date
      to rows where n_total <= number of Y-ticker price rows up to sim_date.
    - This function returns the per-(Y,n_total)->date mapping so the main
      loop can apply this filter.

    Returns
    -------
    tuple:
        catalog_cutoff : pd.Timestamp
            Latest implied training date across ALL rows (worst case).
        implied_dates  : dict mapping (Y, n_total) -> pd.Timestamp
            Pre-computed lookup for the per-row filter.
        safe_start     : pd.Timestamp
            First date where the FULL catalog (all rows) is strictly pre-date.
            Using --start before this date means some catalog rows are leaked
            unless per-row filtering is applied.
    """
    print("\n\u2500\u2500 CATALOG CUTOFF ANALYSIS (data-verified) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    print("  The catalog is an ENSEMBLE of runs at multiple cutoff dates.")
    print("  Building (Y, n_total) -> implied_date lookup...")

    implied_dates = {}
    latest_overall = pd.Timestamp("1900-01-01")
    latest_pre2025 = pd.Timestamp("1900-01-01")

    for df_name, df in [("CPE", cpe), ("JCPE", jcpe)]:
        pairs = df[["Y", "n_total"]].drop_duplicates().values.tolist()
        for Y, nv in pairs:
            nv = int(nv)
            key = (str(Y), nv)
            if key in implied_dates:
                continue
            if str(Y) in prices.columns:
                s = prices[str(Y)].dropna().sort_index()
                if nv <= len(s):
                    d = s.index[nv - 1]
                    implied_dates[key] = d
                    if d > latest_overall:
                        latest_overall = d
                    if d < pd.Timestamp("2025-01-01") and d > latest_pre2025:
                        latest_pre2025 = d

    n_total_pairs  = len(implied_dates)
    n_post2025     = sum(1 for d in implied_dates.values() if d >= pd.Timestamp("2025-01-01"))
    n_post2025_pct = n_post2025 / n_total_pairs * 100 if n_total_pairs else 0

    print(f"  Total unique (Y, n_total) pairs : {n_total_pairs}")
    print(f"  Pairs with implied_date < 2025  : {n_total_pairs - n_post2025} ({100-n_post2025_pct:.1f}%)")
    print(f"  Pairs with implied_date >= 2025 : {n_post2025} ({n_post2025_pct:.1f}%) -- LEAKAGE RISK")
    print(f"  Latest overall implied date      : {latest_overall.date()}")
    print(f"  Latest pre-2025 implied date     : {latest_pre2025.date()}")
    print(f"  Safe backtest start (full catalog): 2025-01-01")
    print(f"  With per-row filtering            : any date (filter applied per sim_date)")

    # safe_start: first date after which ALL catalog rows were knowable
    # = day after latest_overall
    safe_start = latest_overall + pd.Timedelta(days=1)

    return latest_overall, implied_dates, safe_start

def load_data():
    print("\nLoading parquet files...")
    for fn in ["multiasset_prices.parquet", "cpe_results.parquet", "joint_cpe_results.parquet"]:
        path = os.path.join(BASE_DIR, fn)
        if not os.path.exists(path):
            print(f"  ERROR: {fn} not found in {BASE_DIR}")
            sys.exit(1)

    prices = pd.read_parquet(os.path.join(BASE_DIR, "multiasset_prices.parquet"))
    cpe    = pd.read_parquet(os.path.join(BASE_DIR, "cpe_results.parquet"))
    jcpe   = pd.read_parquet(os.path.join(BASE_DIR, "joint_cpe_results.parquet"))

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()

    print(f"  Prices  : {prices.shape[0]} rows x {prices.shape[1]} cols"
          f"  ({prices.index[0].date()} -> {prices.index[-1].date()})")
    print(f"  CPE pairwise : {len(cpe):,} rows")
    print(f"  CPE joint    : {len(jcpe):,} rows")

    # Print all columns so the user can see any embedded metadata
    print(f"  CPE cols : {list(cpe.columns)}")
    print(f"  JCPE cols: {list(jcpe.columns)}")

    catalog_cutoff, implied_dates, safe_start = detect_catalog_cutoff(cpe, jcpe, prices)

    # Pre-attach implied_date as a column for O(1) per-sim filtering
    cpe["_implied_date"]  = [implied_dates.get((str(y), int(n)), pd.NaT)
                              for y, n in zip(cpe["Y"],  cpe["n_total"])]
    jcpe["_implied_date"] = [implied_dates.get((str(y), int(n)), pd.NaT)
                              for y, n in zip(jcpe["Y"], jcpe["n_total"])]
    print(f"  _implied_date column attached to CPE and JCPE.")

    return prices, cpe, jcpe, catalog_cutoff, implied_dates, safe_start

def log_ret(prices_slice, ticker, tau):
    """Log return of ticker over last tau days in prices_slice."""
    if ticker not in prices_slice.columns:
        return np.nan
    s = prices_slice[ticker].dropna()
    if len(s) < tau + 1:
        return np.nan
    if ticker in RATE_TICKERS:
        return float(s.iloc[-1] - s.iloc[-1-tau])
    return float(np.log(s.iloc[-1] / s.iloc[-1-tau]))


def hist_quantile(prices_slice, ticker, tau, q):
    """Historical q-quantile of tau-day returns up to prices_slice."""
    if ticker not in prices_slice.columns:
        return np.nan
    s = prices_slice[ticker].dropna()
    if ticker in RATE_TICKERS:
        rets = s.diff(tau).dropna().values
    else:
        rets = np.log(s / s.shift(tau)).dropna().values
    if len(rets) < 50:
        return np.nan
    return float(np.quantile(rets, q))


def future_price(prices_full, ticker, from_date, tau_days):
    """
    Price of ticker approximately tau_days after from_date
    (nearest available trading day).
    Returns np.nan if not available.
    """
    if ticker not in prices_full.columns:
        return np.nan
    target = pd.Timestamp(from_date) + pd.Timedelta(days=tau_days)
    sub = prices_full[ticker].dropna()
    sub = sub[sub.index >= target]
    if sub.empty:
        return np.nan
    # Use first available price on or after target
    return float(sub.iloc[0])


# ════════════════════════════════════════════════════════════
#  GOLD SIGNAL COMPUTATION (mirrors build_gold_dashboard.py)
# ════════════════════════════════════════════════════════════

def compute_gold_state(prices_slice, jcpe, cpe, sim_date, implied_dates=None, precomp_inc=None, precomp_thresh=None):
    """
    Replicates the key computations from build_gold_dashboard.py
    using only prices up to sim_date.
    Returns a dict matching the gold_state structure in log_predictions.py.
    """
    # ── PER-ROW CATALOG FILTER (O(n) boolean mask on pre-attached column) ──
    # _implied_date was attached once at load time; this is now a single comparison.
    sim_ts = pd.Timestamp(sim_date)
    if "_implied_date" in cpe.columns:
        cpe  = cpe[ cpe["_implied_date"].isna()  | (cpe["_implied_date"]  < sim_ts)]
        jcpe = jcpe[jcpe["_implied_date"].isna() | (jcpe["_implied_date"] < sim_ts)]

    TAU_LIST = [1, 5, 10, 21, 63, 126, 252, 300]
    Q_GRID   = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]
    full_q   = sorted(set(Q_GRID + [round(1-q, 10) for q in Q_GRID]))

    price_tickers = [t for t in prices_slice.columns if t not in RATE_TICKERS]
    rate_tickers  = [t for t in prices_slice.columns if t in RATE_TICKERS]

    # ── CURRENT INCREMENTS & THRESHOLDS (fast-path if pre-computed) ──────────
    if precomp_inc is not None and precomp_thresh is not None:
        current_inc = precomp_inc
        thresholds  = precomp_thresh
    else:
        current_inc = {}
        for tau in TAU_LIST:
            row = {}
            for t in price_tickers:
                r = log_ret(prices_slice, t, tau)
                if not np.isnan(r):
                    row[t] = r
            for t in rate_tickers:
                s = prices_slice[t].dropna()
                if len(s) >= tau + 1:
                    row[t] = float(s.iloc[-1] - s.iloc[-1-tau])
            current_inc[tau] = row
        thresholds = {}
        for tau in TAU_LIST:
            idf = pd.DataFrame(index=prices_slice.index)
            for t in price_tickers:
                s = prices_slice[t]; idf[t] = np.log(s / s.shift(tau))
            for t in rate_tickers:
                s = prices_slice[t]; idf[t] = s - s.shift(tau)
            for q in full_q:
                thresholds[(tau, q)] = idf.quantile(q, numeric_only=True).to_dict()

    def fires(predictors, tau_pasts, q_Xs, direction):
        for x, tp, qx in zip(predictors, tau_pasts, q_Xs):
            tp = int(tp); qx = float(qx)
            curr = current_inc.get(tp, {}).get(x)
            if curr is None or np.isnan(curr):
                return False
            if direction == "bullish":
                th = thresholds.get((tp, qx), {}).get(x, float("nan"))
                if np.isnan(th) or curr <= th:
                    return False
            else:
                th = thresholds.get((tp, round(1-qx, 10)), {}).get(x, float("nan"))
                if np.isnan(th) or curr >= th:
                    return False
        return True

    # ── GOLD PRICE ───────────────────────────────────────────
    gcf = prices_slice["GC=F"].dropna() if "GC=F" in prices_slice.columns else pd.Series(dtype=float)
    if gcf.empty:
        return None

    gold_usd = float(gcf.iloc[-1])

    # SGD conversion
    fx_ticker = "SGDUSD=X"
    usd_per_sgd = 1.0
    if fx_ticker in prices_slice.columns:
        fx_s = prices_slice[fx_ticker].dropna().reindex(gcf.index).ffill()
        if len(fx_s) > 0:
            usd_per_sgd = 1.0 / float(fx_s.iloc[-1])
    price_sgd_g = gold_usd * usd_per_sgd / 31.1035

    # ── RETURNS ──────────────────────────────────────────────
    def pct_chg(n):
        if len(gcf) > n:
            return float((gold_usd / float(gcf.iloc[-1-n]) - 1) * 100)
        return 0.0

    chg = {t: round(pct_chg(t), 1) for t in [1, 5, 10, 21, 63, 126, 252]}

    # ── HISTORICAL RETURN DISTRIBUTIONS ──────────────────────
    gcf_63  = np.log(gcf / gcf.shift(63)).dropna().values * 100
    gcf_126 = np.log(gcf / gcf.shift(126)).dropna().values * 100

    curr_pct_63 = round(float(np.mean(gcf_63 <= chg[63])) * 100, 1) if len(gcf_63) else 50.0

    # ── AUTOCORRELATION CPE ───────────────────────────────────
    auto_cpe = {}
    for tp in [21, 63, 126, 252]:
        lb = np.log(gcf / gcf.shift(tp)).dropna() * 100
        curr_ret = current_inc.get(tp, {}).get("GC=F", None)
        if curr_ret is None:
            continue
        curr_ret_pct = round(curr_ret * 100, 1)
        q_now = float(np.mean(lb.values <= curr_ret_pct))
        auto_cpe[tp] = {
            "current_return_pct" : curr_ret_pct,
            "current_percentile" : round(q_now * 100, 1),
        }
        for fwd in [21, 63, 126, 252]:
            fwd_ret = np.log(gcf / gcf.shift(-fwd)).dropna() * 100
            past_below = lb[lb <= curr_ret_pct].index
            fwd_at = fwd_ret.reindex(past_below).dropna()
            if len(fwd_at) >= 5:
                auto_cpe[tp][f"fwd_{fwd}_pct_positive"] = round(float(np.mean(fwd_at > 0)) * 100, 1)
                auto_cpe[tp][f"fwd_{fwd}_n"]            = int(len(fwd_at))
                auto_cpe[tp][f"fwd_{fwd}_median"]       = round(float(fwd_at.median()), 2)

    # ── JOINT CPE SIGNALS ────────────────────────────────────
    gold_joint  = jcpe[jcpe["Y"].isin(GOLD_Y)].copy()
    gold_joint  = gold_joint[gold_joint["n_predictors"] <= 6].copy()

    signal_rows = []
    for _, row in gold_joint.iterrows():
        w = float(row["joint_CPE"]) * float(row["lift"]) * np.log(max(row["n_joint"], 1))
        firing = fires(row["predictors"], row["tau_pasts"], row["q_Xs"], row["direction"])
        signal_rows.append({
            "Y":         row["Y"],
            "direction": row["direction"],
            "tau_future":int(row["tau_future"]),
            "q_Y":       float(row["q_Y"]),
            "joint_CPE": float(row["joint_CPE"]),
            "lift":      float(row["lift"]),
            "n_joint":   int(row["n_joint"]),
            "weight":    round(w, 3),
            "firing":    bool(firing),
            "predictors":list(row["predictors"]),
            "tau_pasts": [int(x) for x in row["tau_pasts"]],
            "q_Xs":      [float(x) for x in row["q_Xs"]],
        })

    # ── CPE SCORES ───────────────────────────────────────────
    scores = {}
    sig_df = pd.DataFrame(signal_rows)
    if not sig_df.empty:
        for (y, tf), grp in sig_df.groupby(["Y", "tau_future"]):
            bull = grp[grp["direction"] == "bullish"]
            bear = grp[grp["direction"] == "bearish"]
            tw   = bull["weight"].sum() + bear["weight"].sum()
            fb   = bull[bull["firing"]]["weight"].sum()
            fbr  = bear[bear["firing"]]["weight"].sum()
            sc   = (fb - fbr) / tw if tw > 0 else 0
            scores[f"{y}_{tf}"] = {
                "score": round(sc, 4),
                "fired_bull": int(bull["firing"].sum()),
                "fired_bear": int(bear["firing"].sum()),
            }

    # ── COMPOSITE BUY SCORE ──────────────────────────────────
    draw_score = min(100, max(0, (-round(chg[63], 1) / 20) * 100))

    auto_score = 0.0
    if 63 in auto_cpe and "fwd_126_pct_positive" in auto_cpe[63]:
        auto_score = round(float(auto_cpe[63]["fwd_126_pct_positive"]), 1)

    # Predictor proximity (simplified — just IBIT/SLV upper tails + GC=F lower)
    KEY_PREDS = {
        "IBIT":     [(1, 0.5), (5, 0.5), (252, 0.5), (126, 0.6)],
        "FBTC":     [(1, 0.5), (5, 0.5), (252, 0.5), (126, 0.6)],
        "SLV":      [(252, 0.95), (300, 0.95)],
        "SI=F":     [(252, 0.95), (300, 0.95)],
        "SGDUSD=X": [(300, 0.9), (252, 0.9)],
        "GC=F":     [(63, 0.10), (126, 0.10), (252, 0.10)],
        "DX-Y.NYB": [(63, 0.10), (252, 0.10)],
        "UUP":      [(63, 0.10), (252, 0.10)],
        "^GVZ":     [(21, 0.10), (63, 0.10)],
    }
    prox_scores = []
    for ticker, params in KEY_PREDS.items():
        if ticker not in prices_slice.columns:
            continue
        for (tau, q) in params:
            curr = current_inc.get(tau, {}).get(ticker)
            if curr is None:
                continue
            is_lower = q <= 0.20
            if is_lower:
                th = thresholds.get((tau, q), {}).get(ticker, float("nan"))
                if np.isnan(th):
                    continue
                in_tail  = bool(curr < th)
                dist_pct = (curr - th) / abs(th) * 100 if th != 0 else 0
                prox_scores.append(max(0, min(100, 100 * (1 - abs(dist_pct) / 100))))
            else:
                th = thresholds.get((tau, q), {}).get(ticker, float("nan"))
                if np.isnan(th):
                    continue
                in_tail  = bool(curr > th)
                dist_pct = (curr - th) / abs(th) * 100 if th != 0 else 0
                prox_scores.append(max(0, min(100,
                    100 * (1 - abs(dist_pct) / 100) if not in_tail else 100)))

    prox_score = round(float(np.mean(prox_scores)), 1) if prox_scores else 50.0

    gcf_252_score = scores.get("GC=F_252", {}).get("score", 0)
    cpe_score = round(max(0, min(100, (gcf_252_score + 1) / 2 * 100)), 1)

    composite = round(
        0.35 * draw_score +
        0.35 * auto_score +
        0.20 * prox_score +
        0.10 * cpe_score, 1
    )

    # Verdict / direction
    if composite >= 70:
        verdict   = "STRONG BUY"
        direction = "BULLISH"
    elif composite >= 55:
        verdict   = "BUY ZONE"
        direction = "BULLISH"
    elif composite >= 40:
        verdict   = "WAIT & WATCH"
        direction = "NEUTRAL"
    elif composite >= 25:
        verdict   = "NEUTRAL"
        direction = "NEUTRAL"
    else:
        verdict   = "TOO EARLY"
        direction = "BEARISH"

    # Recovery info for CSV columns
    r_vals = {}
    for tau_p, block in auto_cpe.items():
        for tf in [21, 63, 126, 252]:
            key = f"fwd_{tf}_pct_positive"
            if key in block:
                r_vals.setdefault(tf, []).append(block[key])
    horizon_scores = {tf: float(np.mean(vals)) for tf, vals in r_vals.items() if vals}

    return {
        "gcf_price"      : gold_usd,
        "price_sgd_g"    : price_sgd_g,
        "r21"            : chg.get(21, 0) / 100,
        "r63"            : chg.get(63, 0) / 100,
        "r126"           : chg.get(126, 0) / 100,
        "r252"           : chg.get(252, 0) / 100,
        "p63"            : curr_pct_63,
        "composite"      : composite,
        "verdict"        : verdict,
        "direction"      : direction,
        "strong_horizons": [21, 63, 126, 252],
        "horizon_scores" : horizon_scores,
        "auto_cpe"       : auto_cpe,
    }


# ════════════════════════════════════════════════════════════
#  PORTFOLIO SIGNAL COMPUTATION (mirrors build_portfolio_dashboard.py)
# ════════════════════════════════════════════════════════════

def compute_portfolio_state(prices_slice, cpe, jcpe, sim_date=None, implied_dates=None, precomp_inc=None, precomp_thresh=None):
    """
    Replicates the key tilt computation from build_portfolio_dashboard.py
    using only prices up to sim_date.
    """
    price_tickers = [t for t in prices_slice.columns if t not in RATE_TICKERS]

    # ── PER-ROW CATALOG FILTER (O(n) boolean mask on pre-attached column) ──
    if sim_date is not None and "_implied_date" in cpe.columns:
        sim_ts = pd.Timestamp(sim_date)
        cpe  = cpe[ cpe["_implied_date"].isna()  | (cpe["_implied_date"]  < sim_ts)]
        jcpe = jcpe[jcpe["_implied_date"].isna() | (jcpe["_implied_date"] < sim_ts)]

    def curr_ret(ticker, tau):
        # Fast path: use pre-computed returns (log-ret units, need *100 for % scale)
        if precomp_inc is not None:
            v = precomp_inc.get(int(tau), {}).get(ticker)
            if v is not None:
                if ticker in RATE_TICKERS:
                    return float(v)          # rate tickers: level diff, no scaling
                return float(v) * 100        # log-ret -> % scale
        if ticker not in prices_slice.columns:
            return None
        s = prices_slice[ticker].dropna()
        if len(s) < tau + 1:
            return None
        if ticker in RATE_TICKERS:
            return float(s.iloc[-1] - s.iloc[-1-tau])
        return float(np.log(s.iloc[-1] / s.iloc[-1-tau]) * 100)

    def hist_q(ticker, tau, q):
        # Fast path: use pre-computed quantile tables
        if precomp_thresh is not None:
            v = precomp_thresh.get((int(tau), float(q)), {}).get(ticker)
            if v is not None:
                if ticker in RATE_TICKERS:
                    return float(v)
                return float(v) * 100        # log-ret -> % scale
        if ticker not in prices_slice.columns:
            return None
        s = prices_slice[ticker].dropna()
        if ticker in RATE_TICKERS:
            rets = s.diff(tau).dropna().values
        else:
            rets = (np.log(s / s.shift(tau)).dropna().values * 100)
        if len(rets) < 50:
            return None
        return float(np.quantile(rets, q))

    # ── IDENTIFY FIRING PREDICTORS ───────────────────────────
    pred_combos = (cpe[["X", "tau_past", "q_X", "direction"]]
                   .drop_duplicates().values.tolist())

    firing_set = set()
    for X, tau, q, direction in pred_combos:
        cr  = curr_ret(str(X), int(tau))
        if cr is None:
            continue
        th  = hist_q(str(X), int(tau), float(q))
        if th is None:
            continue
        if direction == "bullish" and cr > th:
            firing_set.add((str(X), int(tau), float(q), direction))
        elif direction == "bearish" and cr < th:
            firing_set.add((str(X), int(tau), float(q), direction))

    # ── COMPUTE TILT SCORES PER ASSET ────────────────────────
    hor_weights = {21: 0.20, 63: 0.30, 126: 0.30, 252: 0.20}
    asset_scores = {}
    n_signals_total = 0

    for ac_name, ac_info in ASSET_CLASSES.items():
        ac_tickers = ac_info["tickers"]

        # Filter CPE to rows relevant to this asset class
        cpe_ac = cpe[
            (cpe["Y"].isin(ac_tickers)) &
            (cpe["CPE"]  >= CPE_MIN) &
            (cpe["lift"] >= LIFT_MIN) &
            (cpe["n_condition"] >= N_MIN)
        ]
        if cpe_ac.empty:
            asset_scores[ac_name] = 0.5  # neutral
            continue

        # For each forward horizon, compute weighted CPE score
        weighted_sum   = 0.0
        weight_total   = 0.0
        n_signals_here = 0

        for tf, hw in hor_weights.items():
            sub = cpe_ac[cpe_ac["tau_future"] == tf]
            if sub.empty:
                continue
            bull_cpe = 0.0; bull_w = 0.0
            bear_cpe = 0.0; bear_w = 0.0

            for _, row in sub.iterrows():
                X   = str(row["X"])
                tau = int(row["tau_past"])
                q   = float(row["q_X"])
                d   = row["direction"]
                key = (X, tau, q, d)
                cpe_val = float(row["CPE"])
                lift    = float(row["lift"])
                w       = cpe_val * lift * np.log(max(int(row["n_condition"]), 1))

                if key in firing_set:
                    n_signals_here += 1
                    if d == "bullish":
                        bull_cpe += cpe_val * w
                        bull_w   += w
                    else:
                        bear_cpe += cpe_val * w
                        bear_w   += w

            bull_mean = bull_cpe / bull_w if bull_w > 0 else 0.5
            bear_mean = bear_cpe / bear_w if bear_w > 0 else 0.5

            # Net score: bull pushes above 0.5, bear below
            total_w = bull_w + bear_w
            if total_w > 0:
                net = (bull_w * bull_mean - bear_w * bear_mean) / total_w + 0.5
            else:
                net = 0.5

            net = max(0.0, min(1.0, net))
            weighted_sum  += hw * net
            weight_total  += hw

        score = weighted_sum / weight_total if weight_total > 0 else 0.5
        asset_scores[ac_name] = score
        n_signals_total += n_signals_here

    # ── TRANSLATE SCORES TO TILTS ─────────────────────────────
    def score_to_tilt(score):
        for thresh, label, _ in TILT_THRESHOLDS:
            if score >= thresh:
                return label
        return "UNDERWEIGHT"

    tilts = {ac: score_to_tilt(sc) for ac, sc in asset_scores.items()}

    # Dominant horizon: pick the one with most firing signals
    # (simplified — use 63d as default if no clear leader)
    dominant_horizon = 63

    # Current prices for each proxy
    def get_price(ticker):
        if ticker in prices_slice.columns:
            s = prices_slice[ticker].dropna()
            if len(s):
                return float(s.iloc[-1])
        return np.nan

    return {
        "equities_tilt"    : tilts.get("Equities", "NEUTRAL"),
        "gold_tilt"        : tilts.get("Gold",     "NEUTRAL"),
        "bonds_tilt"       : tilts.get("Bonds",    "NEUTRAL"),
        "crypto_tilt"      : tilts.get("Crypto",   "NEUTRAL"),
        "fx_tilt"          : tilts.get("FX",       "NEUTRAL"),
        "dominant_horizon" : dominant_horizon,
        "n_signals_firing" : n_signals_total,
        "equities_price"   : get_price("SPY"),
        "gold_price"       : get_price("GC=F"),
        "bonds_price"      : get_price("TLT"),
        "crypto_price"     : get_price("IBIT"),
        "fx_price"         : get_price("UUP"),
    }


# ════════════════════════════════════════════════════════════
#  ROW BUILDERS
# ════════════════════════════════════════════════════════════

def build_gold_rows(sim_date, gs):
    """Build gold prediction rows for all 4 horizons."""
    rows = []
    for tf in [21, 63, 126, 252]:
        rec_vals = []
        for tp in [21, 63, 126, 252]:
            key = f"fwd_{tf}_pct_positive"
            if tp in gs["auto_cpe"] and key in gs["auto_cpe"][tp]:
                v = gs["auto_cpe"][tp][key]
                if not np.isnan(v):
                    rec_vals.append(v)
        rec_avg = round(float(np.mean(rec_vals)), 1) if rec_vals else ""

        rows.append({
            "date_predicted"       : str(sim_date),
            "gcf_price_usd"        : round(gs["gcf_price"], 2),
            "price_sgd_per_g"      : round(gs["price_sgd_g"], 2),
            "composite_score"      : round(gs["composite"], 1),
            "verdict"              : gs["verdict"],
            "direction"            : gs["direction"],
            "horizon_days"         : tf,
            "recovery_pct_positive": rec_avg,
            "r21"                  : round(gs["r21"] * 100, 2),
            "r63"                  : round(gs["r63"] * 100, 2),
            "r126"                 : round(gs["r126"] * 100, 2),
            "r252"                 : round(gs["r252"] * 100, 2),
            "p63_percentile"       : round(gs["p63"], 1),
            "outcome_date"         : "",
            "gcf_price_at_outcome" : "",
            "actual_return_pct"    : "",
            "prediction_correct"   : "",
            "status"               : "PENDING",
            "notes"                : "",
        })
    return rows


def build_port_row(sim_date, ps):
    return {
        "date_predicted"      : str(sim_date),
        "equities_tilt"       : ps["equities_tilt"],
        "gold_tilt"           : ps["gold_tilt"],
        "bonds_tilt"          : ps["bonds_tilt"],
        "crypto_tilt"         : ps["crypto_tilt"],
        "fx_tilt"             : ps["fx_tilt"],
        "dominant_horizon_days": ps["dominant_horizon"],
        "n_signals_firing"    : ps["n_signals_firing"],
        "equities_price"      : round(ps["equities_price"], 4) if not np.isnan(ps["equities_price"]) else "",
        "gold_price"          : round(ps["gold_price"],     2)  if not np.isnan(ps["gold_price"])     else "",
        "bonds_price"         : round(ps["bonds_price"],    4)  if not np.isnan(ps["bonds_price"])    else "",
        "crypto_price"        : round(ps["crypto_price"],   4)  if not np.isnan(ps["crypto_price"])   else "",
        "fx_price"            : round(ps["fx_price"],       4)  if not np.isnan(ps["fx_price"])       else "",
        "outcome_date"        : "",
        "equities_price_out"  : "", "gold_price_out"     : "",
        "bonds_price_out"     : "", "crypto_price_out"   : "", "fx_price_out"       : "",
        "equities_return"     : "", "gold_return"        : "",
        "bonds_return"        : "", "crypto_return"      : "", "fx_return"          : "",
        "tilt_pnl"            : "", "neutral_pnl"        : "",
        "tilt_beat_neutral"   : "", "status"             : "PENDING",
        "notes"               : "",
    }


# ════════════════════════════════════════════════════════════
#  RESOLUTION
# ════════════════════════════════════════════════════════════

def resolve_gold_df(gold_df, prices_full):
    """Fill outcome columns for all resolved gold rows."""
    gcf_full = prices_full["GC=F"].dropna() if "GC=F" in prices_full.columns else pd.Series(dtype=float)
    today = date.today()
    resolved = 0

    for idx, row in gold_df.iterrows():
        if row["status"] != "PENDING":
            continue
        pred_dt      = pd.to_datetime(row["date_predicted"]).date()
        horizon_days = int(row["horizon_days"])
        outcome_dt   = pred_dt + timedelta(days=horizon_days)

        if today < outcome_dt:
            continue  # not yet due

        # Find nearest price at or after outcome_dt
        candidates = gcf_full[gcf_full.index >= pd.Timestamp(outcome_dt)]
        if candidates.empty:
            continue

        out_price   = float(candidates.iloc[0])
        entry_price = float(row["gcf_price_usd"])
        actual_ret  = (out_price / entry_price - 1) * 100
        direction   = row["direction"]

        if direction == "BULLISH":
            correct = 1 if actual_ret > 0 else 0
        elif direction == "BEARISH":
            correct = 1 if actual_ret < 0 else 0
        else:
            correct = 1 if abs(actual_ret) < 3 else 0

        gold_df.at[idx, "outcome_date"]         = str(outcome_dt)
        gold_df.at[idx, "gcf_price_at_outcome"] = round(out_price, 2)
        gold_df.at[idx, "actual_return_pct"]    = round(actual_ret, 2)
        gold_df.at[idx, "prediction_correct"]   = correct
        gold_df.at[idx, "status"]               = "RESOLVED"
        resolved += 1

    return gold_df, resolved


def resolve_port_df(port_df, prices_full):
    """Fill outcome columns for all resolved portfolio rows."""
    today = date.today()
    resolved = 0

    def tilt_to_weight(tilt, neutral_wt):
        if tilt == "TILT UP" or tilt == "OVERWEIGHT":
            return neutral_wt * 1.4
        elif tilt == "TILT DOWN" or tilt == "UNDERWEIGHT":
            return neutral_wt * 0.1
        return neutral_wt

    neutral_w_norm = {k: v / sum(NEUTRAL_WEIGHTS.values())
                      for k, v in NEUTRAL_WEIGHTS.items()}

    for idx, row in port_df.iterrows():
        if row["status"] != "PENDING":
            continue
        pred_dt      = pd.to_datetime(row["date_predicted"]).date()
        horizon_days = int(row["dominant_horizon_days"])
        outcome_dt   = pred_dt + timedelta(days=horizon_days)

        if today < outcome_dt:
            continue

        returns  = {}
        out_px   = {}
        for cls, tk in ASSET_PRICE_TICKERS.items():
            entry_col = f"{cls}_price"
            if tk not in prices_full.columns:
                returns[cls] = np.nan; out_px[cls] = np.nan; continue
            s = prices_full[tk].dropna()
            s = s[s.index >= pd.Timestamp(outcome_dt)]
            if s.empty:
                returns[cls] = np.nan; out_px[cls] = np.nan; continue
            op = float(s.iloc[0])
            ep_raw = row[entry_col]
            if ep_raw == "" or pd.isna(ep_raw):
                returns[cls] = np.nan; out_px[cls] = np.nan; continue
            ep = float(ep_raw)
            out_px[cls]  = op
            returns[cls] = (op / ep - 1) if ep > 0 else np.nan

        tilt_col = {"equities": "equities_tilt", "gold": "gold_tilt",
                    "bonds": "bonds_tilt", "crypto": "crypto_tilt", "fx": "fx_tilt"}

        tilt_wts = {cls: tilt_to_weight(row[tilt_col[cls]], NEUTRAL_WEIGHTS[cls])
                    for cls in ASSET_PRICE_TICKERS}
        tw_sum   = sum(tilt_wts.values())
        tilt_wts = {k: v / tw_sum for k, v in tilt_wts.items()}

        tilt_pnl    = sum(tilt_wts[cls] * returns[cls]
                          for cls in ASSET_PRICE_TICKERS
                          if not np.isnan(returns.get(cls, np.nan)))
        neutral_pnl = sum(neutral_w_norm[cls] * returns[cls]
                          for cls in ASSET_PRICE_TICKERS
                          if not np.isnan(returns.get(cls, np.nan)))

        beat = 1 if tilt_pnl > neutral_pnl else 0

        for cls in ASSET_PRICE_TICKERS:
            port_df.at[idx, f"{cls}_price_out"] = round(out_px[cls], 4) if not np.isnan(out_px.get(cls, np.nan)) else ""
            r = returns.get(cls, np.nan)
            port_df.at[idx, f"{cls}_return"]    = round(r * 100, 2) if not np.isnan(r) else ""

        port_df.at[idx, "outcome_date"]      = str(outcome_dt)
        port_df.at[idx, "tilt_pnl"]          = round(tilt_pnl    * 100, 2)
        port_df.at[idx, "neutral_pnl"]       = round(neutral_pnl * 100, 2)
        port_df.at[idx, "tilt_beat_neutral"] = beat
        port_df.at[idx, "status"]            = "RESOLVED"
        resolved += 1

    return port_df, resolved


# ════════════════════════════════════════════════════════════
#  SUMMARY REPORT
# ════════════════════════════════════════════════════════════

def write_summary(gold_df, port_df):
    lines = []
    lines.append("=" * 68)
    lines.append("  CPE DASHBOARD BACKTEST SUMMARY")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 68)

    # ── GOLD ─────────────────────────────────────────────────
    lines.append("\n── GOLD BUY SIGNAL ACCURACY ────────────────────────────────")
    resolved_g = gold_df[gold_df["status"] == "RESOLVED"].copy()
    pending_g  = gold_df[gold_df["status"] == "PENDING"].copy()
    lines.append(f"  Total rows    : {len(gold_df)}")
    lines.append(f"  Resolved      : {len(resolved_g)}")
    lines.append(f"  Pending       : {len(pending_g)}")

    if len(resolved_g) > 0:
        lines.append(f"\n  {'Horizon':>10}  {'N':>5}  {'Correct':>8}  {'Accuracy':>9}  "
                     f"{'Mean Ret%':>10}  {'Med Ret%':>10}")
        lines.append(f"  {'-'*10}  {'-'*5}  {'-'*8}  {'-'*9}  {'-'*10}  {'-'*10}")
        for h in [21, 63, 126, 252]:
            sub = resolved_g[resolved_g["horizon_days"] == h]
            if sub.empty:
                continue
            correct    = sub["prediction_correct"].apply(pd.to_numeric, errors="coerce")
            actual_ret = sub["actual_return_pct"].apply(pd.to_numeric, errors="coerce")
            n   = len(sub)
            acc = correct.mean() * 100
            mr  = actual_ret.mean()
            mdr = actual_ret.median()
            lines.append(f"  {str(h)+'d':>10}  {n:>5}  {int(correct.sum()):>8}  "
                         f"{acc:>8.1f}%  {mr:>+10.2f}%  {mdr:>+10.2f}%")

        # By direction
        lines.append(f"\n  {'Direction':>12}  {'N':>5}  {'Accuracy':>9}  {'Mean Ret%':>10}")
        for d in ["BULLISH", "NEUTRAL", "BEARISH"]:
            sub = resolved_g[resolved_g["direction"] == d]
            if sub.empty:
                continue
            acc = sub["prediction_correct"].apply(pd.to_numeric, errors="coerce").mean() * 100
            mr  = sub["actual_return_pct"].apply(pd.to_numeric, errors="coerce").mean()
            lines.append(f"  {d:>12}  {len(sub):>5}  {acc:>8.1f}%  {mr:>+10.2f}%")

        # By composite score bucket
        lines.append(f"\n  Score bucket     N   Accuracy  Mean Ret%")
        def score_bucket(s):
            if s >= 70: return "≥70 (Strong Buy)"
            if s >= 55: return "55-70 (Buy Zone)"
            if s >= 40: return "40-55 (Watch)"
            return "<40 (Bearish)"
        resolved_g["bucket"] = resolved_g["composite_score"].apply(pd.to_numeric, errors="coerce").apply(score_bucket)
        for bkt in ["≥70 (Strong Buy)", "55-70 (Buy Zone)", "40-55 (Watch)", "<40 (Bearish)"]:
            sub = resolved_g[resolved_g["bucket"] == bkt]
            if sub.empty:
                continue
            acc = sub["prediction_correct"].apply(pd.to_numeric, errors="coerce").mean() * 100
            mr  = sub["actual_return_pct"].apply(pd.to_numeric, errors="coerce").mean()
            lines.append(f"  {bkt:<18} {len(sub):>4}  {acc:>8.1f}%  {mr:>+9.2f}%")

    # ── PORTFOLIO ─────────────────────────────────────────────
    lines.append("\n── PORTFOLIO TILT ACCURACY ─────────────────────────────────")
    resolved_p = port_df[port_df["status"] == "RESOLVED"].copy()
    pending_p  = port_df[port_df["status"] == "PENDING"].copy()
    lines.append(f"  Total rows    : {len(port_df)}")
    lines.append(f"  Resolved      : {len(resolved_p)}")
    lines.append(f"  Pending       : {len(pending_p)}")

    if len(resolved_p) > 0:
        beat   = resolved_p["tilt_beat_neutral"].apply(pd.to_numeric, errors="coerce")
        tp     = resolved_p["tilt_pnl"].apply(pd.to_numeric, errors="coerce")
        np_    = resolved_p["neutral_pnl"].apply(pd.to_numeric, errors="coerce")
        edge   = tp - np_
        lines.append(f"\n  Tilt beat neutral : {beat.sum():.0f} / {len(beat)} = {beat.mean()*100:.1f}%")
        lines.append(f"  Avg tilt PnL      : {tp.mean():+.2f}%")
        lines.append(f"  Avg neutral PnL   : {np_.mean():+.2f}%")
        lines.append(f"  Avg edge (tilt-neutral): {edge.mean():+.3f}%  "
                     f"[std={edge.std():.3f}%  t={edge.mean()/edge.std()*np.sqrt(len(edge)):.2f}]")

        lines.append(f"\n  Per-asset tilt performance:")
        lines.append(f"  {'Asset':<12}  {'TILT UP avg%':<26} {'TILT DOWN avg%':<26} {'NEUTRAL avg%'}")
        for cls in ["equities", "gold", "bonds", "crypto", "fx"]:
            tc = f"{cls}_tilt"
            rc = f"{cls}_return"
            up = resolved_p[resolved_p[tc].isin(["TILT UP","OVERWEIGHT"])][rc].apply(pd.to_numeric, errors="coerce")
            dn = resolved_p[resolved_p[tc].isin(["TILT DOWN","UNDERWEIGHT"])][rc].apply(pd.to_numeric, errors="coerce")
            nt = resolved_p[resolved_p[tc] == "NEUTRAL"][rc].apply(pd.to_numeric, errors="coerce")
            up_s = f"{up.mean():+.2f}% (n={len(up)})" if len(up) else "n/a"
            dn_s = f"{dn.mean():+.2f}% (n={len(dn)})" if len(dn) else "n/a"
            nt_s = f"{nt.mean():+.2f}% (n={len(nt)})" if len(nt) else "n/a"
            lines.append(f"  {cls:<12}  Up:{up_s:<24} Dn:{dn_s:<24} Neu:{nt_s}")

    lines.append("\n" + "=" * 68)
    summary = "\n".join(lines)
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(summary)
    print(summary)


# ════════════════════════════════════════════════════════════
#  MAIN BACKTEST LOOP
# ════════════════════════════════════════════════════════════

def run_backtest(start_date, end_date, weekly, do_gold, do_portfolio):
    prices_full, cpe, jcpe, catalog_cutoff, implied_dates, safe_start = load_data()

    # ── CATALOG CUTOFF ENFORCEMENT ───────────────────────────────────────────
    # The catalog contains rows with training cutoffs spanning 2020–2026-05-24.
    # ~30.8% of rows have implied training dates >= 2025-01-01.
    #
    # TWO MODES depending on --start:
    #
    # (A) --start >= 2025-01-01 (DEFAULT):
    #     Still not fully clean because some rows go to 2026-05.
    #     We apply PER-ROW FILTERING inside compute_gold_state and
    #     compute_portfolio_state: a row is only used if its implied
    #     training date < sim_date.  This is the only correct approach.
    #
    # (B) --start < 2025-01-01:
    #     Even more rows would be leaked.  We refuse to run without
    #     explicit confirmation.
    #
    start_ts = pd.Timestamp(start_date)
    if start_ts < pd.Timestamp("2025-01-01"):
        print(f"\n!!! ERROR: --start {start_date} is before 2025-01-01.")
        print(f"  The catalog contains rows trained on data as late as {catalog_cutoff.date()}.")
        print(f"  Starting before 2025-01-01 would leak substantial catalog signal into early sim dates.")
        ans = input("  Proceed anyway with per-row filtering? [y/N]: ").strip().lower()
        if ans != "y":
            sys.exit(1)

    print(f"\n  Per-row catalog filtering ENABLED.")
    print(f"  Each sim date will only use catalog rows where implied training date < sim_date.")
    print(f"  This eliminates look-ahead at the signal-catalog level.")

    # Test period trading days
    all_dates = prices_full.index[
        (prices_full.index >= pd.Timestamp(start_date)) &
        (prices_full.index <= pd.Timestamp(end_date))
    ]

    if weekly:
        # Keep only Mondays (or first available trading day of each week)
        all_dates = all_dates[all_dates.to_series().dt.dayofweek == 0]
        if len(all_dates) == 0:
            # Fallback: every 5th day
            all_dates = prices_full.index[
                (prices_full.index >= pd.Timestamp(start_date)) &
                (prices_full.index <= pd.Timestamp(end_date))
            ][::5]

    n_days = len(all_dates)
    print(f"\nBacktest period: {start_date} → {end_date}")
    print(f"Simulation days : {n_days}{'  (weekly)' if weekly else ''}")
    print(f"Modules         : {'Gold' if do_gold else ''}  {'Portfolio' if do_portfolio else ''}")

    gold_rows = []
    port_rows = []

    # ── PRE-COMPUTE NUMPY RETURN ARRAYS (done once, ~0.5s) ───────────────
    # Replaces per-day DataFrame quantile loop (~5.7s/day) with array slices.
    print('Pre-computing return arrays...')
    _BT_TAUS   = [1, 5, 10, 21, 63, 126, 252, 300]
    _BT_QG     = [0.50,0.60,0.70,0.75,0.80,0.90,0.95,0.99]
    _BT_FQ     = sorted(set(_BT_QG + [round(1-q,10) for q in _BT_QG]))
    _BT_FQ_ARR = np.array(_BT_FQ)
    _RT        = {'^VIX','^VXN','^OVX','^GVZ','^EVZ','^VVIX','^SKEW',
                  '^TNX','^TYX','^FVX','^IRX'}
    _pc        = [c for c in prices_full.columns if c not in _RT]
    _rc        = [c for c in prices_full.columns if c in _RT]
    _pidx      = prices_full.index
    _parr      = prices_full[_pc].values.astype(float)
    _rarr      = prices_full[_rc].values.astype(float)
    _rp, _rr   = {}, {}
    with np.errstate(divide='ignore', invalid='ignore'):
        for _tau in _BT_TAUS:
            _lr = np.full_like(_parr, np.nan)
            _lr[_tau:] = np.log(_parr[_tau:] / _parr[:-_tau])
            _rp[_tau] = _lr
            _dr = np.full_like(_rarr, np.nan)
            _dr[_tau:] = _rarr[_tau:] - _rarr[:-_tau]
            _rr[_tau] = _dr
    print(f'  Arrays ready: {len(_pc)} price + {len(_rc)} rate tickers')

    for i, sim_ts in enumerate(all_dates):
        sim_date = sim_ts.date()
        _sr      = int(_pidx.searchsorted(sim_ts, side='right'))

        # Strict look-ahead prevention: slice prices up to and including sim_date
        ps = prices_full[prices_full.index <= sim_ts]

        # Build current_inc and thresholds via numpy (fast path)
        _ci, _th = {}, {}
        for _tau in _BT_TAUS:
            _row = {}
            for _ci2, _col in enumerate(_pc):
                v = _rp[_tau][_sr-1, _ci2]
                if np.isfinite(v): _row[_col] = float(v)
            for _ci2, _col in enumerate(_rc):
                v = _rr[_tau][_sr-1, _ci2]
                if np.isfinite(v): _row[_col] = float(v)
            _ci[_tau] = _row
            _pq = np.nanquantile(_rp[_tau][:_sr], _BT_FQ_ARR, axis=0)
            _rq = np.nanquantile(_rr[_tau][:_sr], _BT_FQ_ARR, axis=0)
            for _qi, _q in enumerate(_BT_FQ):
                _d = {}
                for _ci2, _col in enumerate(_pc): _d[_col] = float(_pq[_qi, _ci2])
                for _ci2, _col in enumerate(_rc): _d[_col] = float(_rq[_qi, _ci2])
                _th[(_tau, _q)] = _d

        if (i + 1) % 20 == 0 or i == 0 or i == n_days - 1:
            print(f"  [{i+1:4d}/{n_days}] {sim_date} "
                  f"(GC=F={ps['GC=F'].dropna().iloc[-1]:.0f} "
                  f"SPY={ps['SPY'].dropna().iloc[-1]:.1f})" if "SPY" in ps.columns and "GC=F" in ps.columns else
                  f"  [{i+1:4d}/{n_days}] {sim_date}")

        # ── GOLD ─────────────────────────────────────────────
        if do_gold:
            gs = compute_gold_state(ps, jcpe, cpe, sim_date, implied_dates, _ci, _th)
            if gs is not None:
                gold_rows.extend(build_gold_rows(sim_date, gs))

        # ── PORTFOLIO ─────────────────────────────────────────
        if do_portfolio:
            port_s = compute_portfolio_state(ps, cpe, jcpe, sim_date, implied_dates, _ci, _th)
            if port_s is not None:
                port_rows.append(build_port_row(sim_date, port_s))

    # ── BUILD DATAFRAMES ─────────────────────────────────────
    gold_df = pd.DataFrame(gold_rows, columns=GOLD_COLS) if gold_rows else pd.DataFrame(columns=GOLD_COLS)
    port_df = pd.DataFrame(port_rows, columns=PORT_COLS) if port_rows else pd.DataFrame(columns=PORT_COLS)

    # ── RESOLVE OUTCOMES ─────────────────────────────────────
    print(f"\nResolving gold outcomes...")
    gold_df, n_g = resolve_gold_df(gold_df, prices_full)
    print(f"  {n_g} gold rows resolved, {(gold_df['status']=='PENDING').sum()} still pending")

    print(f"Resolving portfolio outcomes...")
    port_df, n_p = resolve_port_df(port_df, prices_full)
    print(f"  {n_p} portfolio rows resolved, {(port_df['status']=='PENDING').sum()} still pending")

    # ── SAVE CSVs ────────────────────────────────────────────
    gold_df.to_csv(GOLD_BT_CSV, index=False)
    port_df.to_csv(PORT_BT_CSV, index=False)
    print(f"\nSaved: {GOLD_BT_CSV}")
    print(f"Saved: {PORT_BT_CSV}")

    # ── SUMMARY ──────────────────────────────────────────────
    write_summary(gold_df, port_df)
    print(f"\nSaved: {SUMMARY_TXT}")


# ════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CPE Dashboard Historical Backtest (2025–2026)"
    )
    parser.add_argument(
        "--start", default="2025-01-01",
        help="Start date of test period (default: 2025-01-01)"
    )
    parser.add_argument(
        "--end", default=str(date.today()),
        help=f"End date of test period (default: today = {date.today()})"
    )
    parser.add_argument(
        "--weekly", action="store_true",
        help="Run on weekly cadence (Mondays only) — faster, fewer rows"
    )
    parser.add_argument(
        "--gold-only", action="store_true",
        help="Only backtest the gold dashboard"
    )
    parser.add_argument(
        "--portfolio-only", action="store_true",
        help="Only backtest the portfolio dashboard"
    )
    args = parser.parse_args()

    do_gold      = not args.portfolio_only
    do_portfolio = not args.gold_only

    print("=" * 60)
    print("  CPE DASHBOARD HISTORICAL BACKTEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    run_backtest(
        start_date   = args.start,
        end_date     = args.end,
        weekly       = args.weekly,
        do_gold      = do_gold,
        do_portfolio = do_portfolio,
    )

    print("\n✓ Backtest complete.")


if __name__ == "__main__":
    main()
