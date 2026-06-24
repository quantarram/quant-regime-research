#!/usr/bin/env python3
"""
============================================================
  CPE PROPER BACKTEST
  Dr. Arun Ramanathan
============================================================
Tests the actual claim of the CPE framework:

  "When predictor X was in its tail over tau_past days,
   target Y exceeded its q_Y threshold over tau_future days
   with probability CPE — significantly above the
   unconditional probability uncond_prob."

For each trading day in the test period (2025-01-01 → today):
  1. Check which CPE/JCPE signal rows FIRED (predictor condition met)
  2. For each fired signal, record the outcome tau_future days later
     (did Y actually exceed its threshold?)
  3. Compare realised hit rate vs stated CPE vs uncond_prob

This is the correct out-of-sample calibration test.

SEPARATE LAYER: Dashboard usefulness
  Also replays the real dashboard tilt logic day-by-day
  and measures whether following it would have added value.

Outputs:
  cpe_signal_backtest.csv       — one row per (signal_date, CPE_row)
  jcpe_signal_backtest.csv      — one row per (signal_date, JCPE_row)
  dashboard_gold_replay.csv     — daily gold dashboard state + outcome
  dashboard_portfolio_replay.csv— daily portfolio tilt + outcome
  cpe_backtest_report.txt       — summary report

Usage:
  python cpe_proper_backtest.py
  python cpe_proper_backtest.py --start 2025-01-01 --weekly
  python cpe_proper_backtest.py --gold-only
  python cpe_proper_backtest.py --portfolio-only
============================================================
"""

import argparse, os, sys, warnings
from datetime import date, timedelta
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RATE_TICKERS = {"^VIX","^VXN","^OVX","^GVZ","^EVZ","^VVIX","^SKEW",
                "^TNX","^TYX","^FVX","^IRX"}

GOLD_Y  = ["GC=F","GLD","IAU"]
PORT_AC = {
    "Equities": {"tickers": ["SPY","QQQ"],         "proxy": "SPY"},
    "Gold":     {"tickers": ["GC=F","GLD","IAU"],   "proxy": "GC=F"},
    "Bonds":    {"tickers": ["TLT","AGG","SHY"],    "proxy": "TLT"},
    "Crypto":   {"tickers": ["IBIT","FBTC","BTC-USD"],"proxy":"IBIT"},
    "FX":       {"tickers": ["UUP","SGDUSD=X"],     "proxy": "UUP"},
}
PORT_PROXY = {ac: info["proxy"] for ac, info in PORT_AC.items()}
PORT_ALL_Y = [t for info in PORT_AC.values() for t in info["tickers"]]

CPE_MIN  = 0.80
LIFT_MIN = 1.50
N_MIN    = 100
HOR_WEIGHTS = {21: 0.20, 63: 0.30, 126: 0.30, 252: 0.20}
NEUTRAL_W   = {"Equities": 0.329, "Gold": 0.299,
               "Bonds": 0.047, "Crypto": 0.225, "FX": 0.100}


# ═══════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════

def load():
    print("\nLoading data...")
    for fn in ["multiasset_prices.parquet","cpe_results.parquet",
               "joint_cpe_results.parquet"]:
        if not os.path.exists(os.path.join(BASE_DIR, fn)):
            print(f"  ERROR: {fn} not found"); sys.exit(1)

    prices = pd.read_parquet(os.path.join(BASE_DIR,"multiasset_prices.parquet"))
    cpe    = pd.read_parquet(os.path.join(BASE_DIR,"cpe_results.parquet"))
    jcpe   = pd.read_parquet(os.path.join(BASE_DIR,"joint_cpe_results.parquet"))
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()

    print(f"  Prices : {prices.shape[0]} rows × {prices.shape[1]} cols "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")
    print(f"  CPE    : {len(cpe):,} rows")
    print(f"  JCPE   : {len(jcpe):,} rows")

    # Attach implied_date for catalog filtering
    print("  Building catalog implied_date lookup...")
    implied = {}
    for df in [cpe, jcpe]:
        for Y, nv in df[["Y","n_total"]].drop_duplicates().values:
            key = (str(Y), int(nv))
            if key in implied: continue
            if str(Y) in prices.columns:
                s = prices[str(Y)].dropna().sort_index()
                nv_i = int(nv)
                if nv_i <= len(s):
                    implied[key] = s.index[nv_i - 1]

    cpe["_implied"]  = [implied.get((str(y),int(n)), pd.NaT)
                        for y,n in zip(cpe["Y"],cpe["n_total"])]
    jcpe["_implied"] = [implied.get((str(y),int(n)), pd.NaT)
                        for y,n in zip(jcpe["Y"],jcpe["n_total"])]
    print(f"  Implied dates attached.")
    return prices, cpe, jcpe


# ═══════════════════════════════════════════════════════════
#  PRE-COMPUTE NUMPY RETURN ARRAYS
# ═══════════════════════════════════════════════════════════

def precompute_returns(prices):
    """Return (ret_price, ret_rate, price_cols, rate_cols) numpy arrays."""
    price_cols = [c for c in prices.columns if c not in RATE_TICKERS]
    rate_cols  = [c for c in prices.columns if c in RATE_TICKERS]
    col_idx    = {c: i for i,c in enumerate(price_cols)}
    rcol_idx   = {c: i for i,c in enumerate(rate_cols)}

    TAUS = [1,5,10,21,63,126,252,300]
    parr = prices[price_cols].values.astype(float)
    rarr = prices[rate_cols].values.astype(float)
    N    = len(prices)

    ret_p, ret_r = {}, {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for tau in TAUS:
            lr = np.full((N, len(price_cols)), np.nan)
            lr[tau:] = np.log(parr[tau:] / parr[:-tau])
            ret_p[tau] = lr
            dr = np.full((N, len(rate_cols)), np.nan)
            dr[tau:] = rarr[tau:] - rarr[:-tau]
            ret_r[tau] = dr

    return ret_p, ret_r, price_cols, rate_cols, col_idx, rcol_idx, TAUS


# ═══════════════════════════════════════════════════════════
#  RETURN HELPERS (per sim-day slice)
# ═══════════════════════════════════════════════════════════

def get_current_return(ticker, tau, row_idx,
                       ret_p, ret_r, col_idx, rcol_idx):
    """Log-return (or level diff) of ticker over tau days ending at row_idx."""
    if ticker in col_idx:
        v = ret_p[tau][row_idx, col_idx[ticker]]
        return float(v) if np.isfinite(v) else np.nan
    if ticker in rcol_idx:
        v = ret_r[tau][row_idx, rcol_idx[ticker]]
        return float(v) if np.isfinite(v) else np.nan
    return np.nan


def get_quantile(ticker, tau, q, row_idx,
                 ret_p, ret_r, col_idx, rcol_idx):
    """Historical q-th quantile of ticker's tau-day return up to row_idx."""
    if ticker in col_idx:
        vals = ret_p[tau][:row_idx, col_idx[ticker]]
    elif ticker in rcol_idx:
        vals = ret_r[tau][:row_idx, rcol_idx[ticker]]
    else:
        return np.nan
    finite = vals[np.isfinite(vals)]
    if len(finite) < 30:
        return np.nan
    return float(np.quantile(finite, q))


def condition_fired(X, tau_past, q_X, direction, row_idx,
                    ret_p, ret_r, col_idx, rcol_idx):
    """
    Returns True if the predictor condition is met at row_idx.
    bullish: X's return > q_X quantile  (X in upper tail)
    bearish: X's return < (1-q_X) quantile  (X in lower tail)
    """
    curr = get_current_return(X, tau_past, row_idx,
                               ret_p, ret_r, col_idx, rcol_idx)
    if np.isnan(curr):
        return False
    if direction == "bullish":
        th = get_quantile(X, tau_past, q_X, row_idx,
                          ret_p, ret_r, col_idx, rcol_idx)
        return (not np.isnan(th)) and curr > th
    else:  # bearish: X in lower tail
        th = get_quantile(X, tau_past, 1.0 - q_X, row_idx,
                          ret_p, ret_r, col_idx, rcol_idx)
        return (not np.isnan(th)) and curr < th


def outcome_hit(Y, tau_future, q_Y, direction, signal_row_idx,
                prices_idx, prices,
                ret_p, ret_r, col_idx, rcol_idx):
    """
    Returns (hit: bool|None, actual_return: float).
    hit=None if outcome date not yet in price history.

    bullish outcome: Y's tau_future return > q_Y quantile at signal date
    bearish outcome: Y's tau_future return < (1-q_Y) quantile at signal date
    """
    signal_ts   = prices_idx[signal_row_idx]
    target_ts   = signal_ts + pd.Timedelta(days=tau_future)
    future_rows = prices_idx[prices_idx >= target_ts]
    if len(future_rows) == 0:
        return None, np.nan

    outcome_row = prices_idx.searchsorted(future_rows[0])
    # forward return at the nearest available date
    if Y in col_idx:
        p_now = prices[Y].iloc[signal_row_idx]
        p_fut = prices[Y].iloc[outcome_row]
        if p_now <= 0 or np.isnan(p_now) or np.isnan(p_fut):
            return None, np.nan
        actual_ret = float(np.log(p_fut / p_now))
    elif Y in ret_r[1].shape:   # rate ticker — use level diff
        return None, np.nan     # skip rate tickers as Y for now
    else:
        if Y not in prices.columns:
            return None, np.nan
        p_now = prices[Y].iloc[signal_row_idx]
        p_fut = prices[Y].iloc[outcome_row]
        if np.isnan(p_now) or np.isnan(p_fut):
            return None, np.nan
        actual_ret = float(np.log(p_fut / p_now))

    # threshold: historical q_Y-th quantile at signal date (same tau_future window)
    th = get_quantile(Y, tau_future, q_Y, signal_row_idx,
                      ret_p, ret_r, col_idx, rcol_idx)
    if np.isnan(th):
        return None, actual_ret

    if direction == "bullish":
        hit = bool(actual_ret > th)
    else:  # bearish: Y drops below (1-q_Y)-th pct
        hit = bool(actual_ret < th)

    return hit, actual_ret


# ═══════════════════════════════════════════════════════════
#  LAYER 1: PAIRWISE CPE SIGNAL BACKTEST
# ═══════════════════════════════════════════════════════════

def run_pairwise_backtest(prices, cpe, sim_dates, prices_idx,
                          ret_p, ret_r, col_idx, rcol_idx,
                          focus_Y=None):
    """
    For each sim_date × each CPE row whose predictor fires:
      record (signal_date, Y, X, tau_past, tau_future, q_X, q_Y,
              direction, CPE, uncond_prob, lift, n_condition,
              condition_fired=True, outcome_hit, actual_return_log)
    """
    print(f"\n{'='*60}")
    print(f"  LAYER 1: PAIRWISE CPE SIGNAL BACKTEST")
    print(f"  {'Gold signals only' if focus_Y else 'All dashboard Y tickers'}")
    print(f"{'='*60}")

    if focus_Y:
        cpe_sub = cpe[cpe["Y"].isin(focus_Y)].copy()
    else:
        cpe_sub = cpe[cpe["Y"].isin(PORT_ALL_Y + GOLD_Y)].copy()

    print(f"  CPE rows in scope: {len(cpe_sub):,}")

    records = []
    n_days  = len(sim_dates)

    for i, sim_ts in enumerate(sim_dates):
        if (i+1) % 50 == 0 or i == 0 or i == n_days-1:
            print(f"  [{i+1:4d}/{n_days}] {sim_ts.date()}")

        row_idx = int(prices_idx.searchsorted(sim_ts, side="right")) - 1
        if row_idx < 252:   # need enough history for quantiles
            continue

        sim_ts_pd = pd.Timestamp(sim_ts)

        # Filter catalog to rows trained before this date
        cpe_day = cpe_sub[
            cpe_sub["_implied"].isna() | (cpe_sub["_implied"] < sim_ts_pd)
        ]

        # Group by predictor condition to avoid rechecking the same (X,tau,q,dir) twice
        pred_fired_cache = {}

        for _, row in cpe_day.iterrows():
            X         = str(row["X"])
            tau_past  = int(row["tau_past"])
            q_X       = float(row["q_X"])
            direction = str(row["direction"])
            Y         = str(row["Y"])
            tau_future= int(row["tau_future"])
            q_Y       = float(row["q_Y"])

            # Check condition (cached)
            pred_key = (X, tau_past, q_X, direction)
            if pred_key not in pred_fired_cache:
                pred_fired_cache[pred_key] = condition_fired(
                    X, tau_past, q_X, direction, row_idx,
                    ret_p, ret_r, col_idx, rcol_idx)
            if not pred_fired_cache[pred_key]:
                continue

            # Signal fired — check outcome
            hit, actual_ret = outcome_hit(
                Y, tau_future, q_Y, direction,
                row_idx, prices_idx, prices,
                ret_p, ret_r, col_idx, rcol_idx)

            records.append({
                "signal_date"   : str(sim_ts.date()),
                "Y"             : Y,
                "X"             : X,
                "tau_past"      : tau_past,
                "tau_future"    : tau_future,
                "q_X"           : q_X,
                "q_Y"           : q_Y,
                "direction"     : direction,
                "CPE"           : round(float(row["CPE"]), 4),
                "uncond_prob"   : round(float(row["uncond_prob"]), 4),
                "lift"          : round(float(row["lift"]), 3),
                "n_condition"   : int(row["n_condition"]),
                "outcome_hit"   : int(hit) if hit is not None else "",
                "actual_ret_log": round(actual_ret, 5) if np.isfinite(actual_ret) else "",
                "status"        : "RESOLVED" if hit is not None else "PENDING",
            })

    df = pd.DataFrame(records)
    print(f"\n  Total signal-instances recorded: {len(df):,}")
    if len(df):
        res = df[df["status"]=="RESOLVED"]
        print(f"  Resolved: {len(res):,}  Pending: {(df['status']=='PENDING').sum():,}")
        if len(res):
            hit_rate = res["outcome_hit"].astype(float).mean()
            avg_cpe  = res["CPE"].mean()
            avg_unc  = res["uncond_prob"].mean()
            print(f"  Overall hit rate    : {hit_rate:.1%}")
            print(f"  Avg stated CPE      : {avg_cpe:.1%}")
            print(f"  Avg uncond_prob     : {avg_unc:.1%}")
            print(f"  Lift preserved      : {hit_rate/avg_unc:.2f}x (vs stated {avg_cpe/avg_unc:.2f}x)")
    return df


# ═══════════════════════════════════════════════════════════
#  LAYER 2: JOINT CPE SIGNAL BACKTEST
# ═══════════════════════════════════════════════════════════

def run_joint_backtest(prices, jcpe, sim_dates, prices_idx,
                       ret_p, ret_r, col_idx, rcol_idx,
                       focus_Y=None):
    print(f"\n{'='*60}")
    print(f"  LAYER 2: JOINT CPE SIGNAL BACKTEST")
    print(f"{'='*60}")

    if focus_Y:
        jcpe_sub = jcpe[jcpe["Y"].isin(focus_Y)].copy()
    else:
        jcpe_sub = jcpe[jcpe["Y"].isin(PORT_ALL_Y + GOLD_Y)].copy()

    print(f"  JCPE rows in scope: {len(jcpe_sub):,}")

    records = []
    n_days  = len(sim_dates)

    for i, sim_ts in enumerate(sim_dates):
        if (i+1) % 50 == 0 or i == 0 or i == n_days-1:
            print(f"  [{i+1:4d}/{n_days}] {sim_ts.date()}")

        row_idx = int(prices_idx.searchsorted(sim_ts, side="right")) - 1
        if row_idx < 252:
            continue

        sim_ts_pd = pd.Timestamp(sim_ts)

        jcpe_day = jcpe_sub[
            jcpe_sub["_implied"].isna() | (jcpe_sub["_implied"] < sim_ts_pd)
        ]

        for _, row in jcpe_day.iterrows():
            predictors = list(row["predictors"])
            tau_pasts  = [int(t) for t in row["tau_pasts"]]
            q_Xs       = [float(q) for q in row["q_Xs"]]
            direction  = str(row["direction"])
            Y          = str(row["Y"])
            tau_future = int(row["tau_future"])
            q_Y        = float(row["q_Y"])

            # ALL predictors must fire simultaneously
            all_fired = all(
                condition_fired(X, tp, qx, direction, row_idx,
                                ret_p, ret_r, col_idx, rcol_idx)
                for X, tp, qx in zip(predictors, tau_pasts, q_Xs)
            )
            if not all_fired:
                continue

            hit, actual_ret = outcome_hit(
                Y, tau_future, q_Y, direction,
                row_idx, prices_idx, prices,
                ret_p, ret_r, col_idx, rcol_idx)

            records.append({
                "signal_date"   : str(sim_ts.date()),
                "Y"             : Y,
                "direction"     : direction,
                "tau_future"    : tau_future,
                "q_Y"           : q_Y,
                "n_predictors"  : int(row["n_predictors"]),
                "predictors"    : "|".join(predictors),
                "joint_CPE"     : round(float(row["joint_CPE"]), 4),
                "uncond_prob"   : round(float(row["uncond_prob"]), 4),
                "lift"          : round(float(row["lift"]), 3),
                "n_joint"       : int(row["n_joint"]),
                "outcome_hit"   : int(hit) if hit is not None else "",
                "actual_ret_log": round(actual_ret, 5) if np.isfinite(actual_ret) else "",
                "status"        : "RESOLVED" if hit is not None else "PENDING",
            })

    df = pd.DataFrame(records)
    print(f"\n  Total joint signal-instances: {len(df):,}")
    if len(df):
        res = df[df["status"]=="RESOLVED"]
        if len(res):
            hit_rate = res["outcome_hit"].astype(float).mean()
            print(f"  Overall hit rate : {hit_rate:.1%}")
            print(f"  Avg joint_CPE    : {res['joint_CPE'].mean():.1%}")
            print(f"  Avg uncond_prob  : {res['uncond_prob'].mean():.1%}")
    return df


# ═══════════════════════════════════════════════════════════
#  LAYER 3: DASHBOARD REPLAY — GOLD
# ═══════════════════════════════════════════════════════════

def run_gold_dashboard_replay(prices, cpe, jcpe, sim_dates, prices_idx,
                               ret_p, ret_r, col_idx, rcol_idx):
    """
    Replays what the gold dashboard CPE signals would have said each day.
    Records: which signals fired, net bull/bear balance, and whether
    gold actually rose/fell at each forward horizon.
    This tests dashboard-level usefulness, not individual signal calibration.
    """
    print(f"\n{'='*60}")
    print(f"  LAYER 3: GOLD DASHBOARD REPLAY")
    print(f"{'='*60}")

    gold_cpe  = cpe[cpe["Y"].isin(GOLD_Y)].copy()
    gold_jcpe = jcpe[jcpe["Y"].isin(GOLD_Y)].copy()

    records = []

    for i, sim_ts in enumerate(sim_dates):
        if (i+1) % 50 == 0 or i == 0:
            print(f"  [{i+1:4d}/{len(sim_dates)}] {sim_ts.date()}")

        row_idx   = int(prices_idx.searchsorted(sim_ts, side="right")) - 1
        if row_idx < 252: continue
        sim_ts_pd = pd.Timestamp(sim_ts)

        # Catalog filter
        gc_day = gold_cpe[gold_cpe["_implied"].isna() | (gold_cpe["_implied"] < sim_ts_pd)]
        gj_day = gold_jcpe[gold_jcpe["_implied"].isna() | (gold_jcpe["_implied"] < sim_ts_pd)]

        # Check each pairwise signal
        pred_cache = {}
        bull_fired = []
        bear_fired = []

        for _, row in gc_day.iterrows():
            X, tp, qx, d = str(row["X"]), int(row["tau_past"]), float(row["q_X"]), str(row["direction"])
            key = (X, tp, qx, d)
            if key not in pred_cache:
                pred_cache[key] = condition_fired(X, tp, qx, d, row_idx, ret_p, ret_r, col_idx, rcol_idx)
            if not pred_cache[key]: continue
            w = float(row["CPE"]) * float(row["lift"]) * np.log(max(int(row["n_condition"]),1))
            if d == "bullish": bull_fired.append(w)
            else:              bear_fired.append(w)

        # Joint signals
        joint_bull = []
        joint_bear = []
        for _, row in gj_day.iterrows():
            preds = list(row["predictors"]); tps = [int(t) for t in row["tau_pasts"]]
            qxs = [float(q) for q in row["q_Xs"]]; d = str(row["direction"])
            if all(condition_fired(X, tp, qx, d, row_idx, ret_p, ret_r, col_idx, rcol_idx)
                   for X, tp, qx in zip(preds, tps, qxs)):
                w = float(row["joint_CPE"]) * float(row["lift"]) * np.log(max(int(row["n_joint"]),1))
                if d == "bullish": joint_bull.append(w)
                else:              joint_bear.append(w)

        n_bull    = len(bull_fired)
        n_bear    = len(bear_fired)
        sum_bull  = sum(bull_fired)
        sum_bear  = sum(bear_fired)
        net_score = (sum_bull - sum_bear) / (sum_bull + sum_bear) if (sum_bull + sum_bear) > 0 else 0.0
        signal    = "BULLISH" if net_score > 0.1 else ("BEARISH" if net_score < -0.1 else "NEUTRAL")

        # Current GC=F price
        gcf_price = np.nan
        if "GC=F" in prices.columns:
            gcf_price = float(prices["GC=F"].iloc[row_idx]) if np.isfinite(prices["GC=F"].iloc[row_idx]) else np.nan

        rec = {
            "signal_date"  : str(sim_ts.date()),
            "gcf_price"    : round(gcf_price, 2) if np.isfinite(gcf_price) else "",
            "n_bull_firing": n_bull,
            "n_bear_firing": n_bear,
            "sum_bull_wt"  : round(sum_bull, 3),
            "sum_bear_wt"  : round(sum_bear, 3),
            "n_joint_bull" : len(joint_bull),
            "n_joint_bear" : len(joint_bear),
            "net_score"    : round(net_score, 4),
            "signal"       : signal,
        }

        # Forward outcomes at each horizon
        for tau_f in [21, 63, 126, 252]:
            target_ts = sim_ts + pd.Timedelta(days=tau_f)
            future    = prices_idx[prices_idx >= target_ts]
            if len(future) == 0 or "GC=F" not in prices.columns:
                rec[f"gcf_ret_{tau_f}d"]     = ""
                rec[f"gcf_positive_{tau_f}d"] = ""
                rec[f"status_{tau_f}d"]       = "PENDING"
                continue
            out_row = int(prices_idx.searchsorted(future[0]))
            p_now   = prices["GC=F"].iloc[row_idx]
            p_fut   = prices["GC=F"].iloc[out_row]
            if np.isnan(p_now) or np.isnan(p_fut) or p_now <= 0:
                rec[f"gcf_ret_{tau_f}d"]     = ""
                rec[f"gcf_positive_{tau_f}d"] = ""
                rec[f"status_{tau_f}d"]       = "PENDING"
            else:
                ret = float(np.log(p_fut / p_now)) * 100
                rec[f"gcf_ret_{tau_f}d"]      = round(ret, 3)
                rec[f"gcf_positive_{tau_f}d"] = int(ret > 0)
                rec[f"status_{tau_f}d"]        = "RESOLVED"

        records.append(rec)

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════
#  LAYER 4: DASHBOARD REPLAY — PORTFOLIO
# ═══════════════════════════════════════════════════════════

def run_portfolio_dashboard_replay(prices, cpe, sim_dates, prices_idx,
                                    ret_p, ret_r, col_idx, rcol_idx):
    """
    Replays the EXACT portfolio dashboard logic:
      get_firing_cpe_signals → compute_regime_score → net_score → tilt_label
    Then measures whether the tilt beat the neutral weight.
    """
    print(f"\n{'='*60}")
    print(f"  LAYER 4: PORTFOLIO DASHBOARD REPLAY")
    print(f"{'='*60}")

    TILT_MAP = [(0.85,"OVERWEIGHT"),(0.70,"TILT UP"),(0.50,"NEUTRAL"),
                (0.35,"TILT DOWN"),(0.00,"UNDERWEIGHT")]

    def score_to_tilt(score):
        if score is None: return "NO SIGNAL"
        for th, label in TILT_MAP:
            if score >= th: return label
        return "UNDERWEIGHT"

    def tilt_to_weight_mult(label):
        return 1.4 if label in ("TILT UP","OVERWEIGHT") else \
               0.1 if label in ("TILT DOWN","UNDERWEIGHT") else 1.0

    records = []

    for i, sim_ts in enumerate(sim_dates):
        if (i+1) % 50 == 0 or i == 0:
            print(f"  [{i+1:4d}/{len(sim_dates)}] {sim_ts.date()}")

        row_idx   = int(prices_idx.searchsorted(sim_ts, side="right")) - 1
        if row_idx < 252: continue
        sim_ts_pd = pd.Timestamp(sim_ts)

        cpe_day = cpe[
            cpe["Y"].isin(PORT_ALL_Y) &
            (cpe["CPE"]          >= CPE_MIN) &
            (cpe["lift"]         >= LIFT_MIN) &
            (cpe["n_condition"]  >= N_MIN) &
            (cpe["_implied"].isna() | (cpe["_implied"] < sim_ts_pd))
        ]

        # Check firing for each unique predictor condition
        pred_cache = {}
        def fired(X, tp, qx, d):
            key = (X, int(tp), float(qx), d)
            if key not in pred_cache:
                pred_cache[key] = condition_fired(X, int(tp), float(qx), d,
                                                  row_idx, ret_p, ret_r, col_idx, rcol_idx)
            return pred_cache[key]

        rec = {"signal_date": str(sim_ts.date())}
        tilt_weights = {}

        for ac, info in PORT_AC.items():
            bull_tickers = info["tickers"]
            bear_tickers = info["tickers"]
            ac_tilts = {}

            for hor in [21, 63, 126, 252]:
                bull_sigs = cpe_day[
                    cpe_day["Y"].isin(bull_tickers) &
                    (cpe_day["tau_future"] == hor) &
                    (cpe_day["direction"] == "bullish") &
                    cpe_day.apply(lambda r: fired(r["X"], r["tau_past"], r["q_X"], "bullish"), axis=1)
                ]
                bear_sigs = cpe_day[
                    cpe_day["Y"].isin(bear_tickers) &
                    (cpe_day["tau_future"] == hor) &
                    (cpe_day["direction"] == "bearish") &
                    cpe_day.apply(lambda r: fired(r["X"], r["tau_past"], r["q_X"], "bearish"), axis=1)
                ]

                def regime_score(sigs):
                    if len(sigs) == 0: return None
                    dedup = (sigs.sort_values("CPE", ascending=False)
                                 .drop_duplicates(subset=["X","tau_past","q_X","direction"]))
                    wts = dedup["n_condition"].values.astype(float)
                    return float(np.average(dedup["CPE"].values, weights=wts))

                bs = regime_score(bull_sigs)
                rs = regime_score(bear_sigs)
                if bs is not None and rs is not None:
                    net = bs * 0.6 + (1 - rs) * 0.4
                elif bs is not None:
                    net = bs
                elif rs is not None:
                    net = 1 - rs
                else:
                    net = None
                ac_tilts[hor] = net

            # Weighted average across horizons
            # Convert net_score to delta: score=0.7 → +4, score=0.3 → -4
            # None means no signals fired → delta=0
            total_d = sum(
                HOR_WEIGHTS[h] * (0.0 if s is None else (s - 0.5) * 20)
                for h, s in ac_tilts.items()
            )
            # Reconstruct net_score from weighted delta average
            all_none = all(s is None for s in ac_tilts.values())
            if all_none:
                overall_tilt = "NEUTRAL"
            else:
                net_score_overall = 0.5 + total_d / 20
                net_score_overall = max(0.0, min(1.0, net_score_overall))
                overall_tilt = score_to_tilt(net_score_overall)

            proxy = info["proxy"]
            p_now = float(prices[proxy].iloc[row_idx]) if proxy in prices.columns else np.nan
            rec[f"{ac.lower()}_tilt"]  = overall_tilt
            rec[f"{ac.lower()}_price"] = round(p_now, 4) if np.isfinite(p_now) else ""
            tilt_weights[ac] = tilt_to_weight_mult(overall_tilt) * NEUTRAL_W[ac]

        # Normalise tilt weights
        tw_sum = sum(tilt_weights.values())
        nw_sum = sum(NEUTRAL_W.values())
        tilt_norm    = {k: v/tw_sum for k,v in tilt_weights.items()}
        neutral_norm = {k: v/nw_sum for k,v in NEUTRAL_W.items()}

        # Forward returns at dominant horizon (63d)
        for hor in [21, 63, 126]:
            target_ts = sim_ts + pd.Timedelta(days=hor)
            future    = prices_idx[prices_idx >= target_ts]
            if len(future) == 0:
                rec[f"tilt_pnl_{hor}d"]    = ""
                rec[f"neutral_pnl_{hor}d"] = ""
                rec[f"tilt_beat_{hor}d"]   = ""
                rec[f"status_{hor}d"]      = "PENDING"
                continue
            out_row = int(prices_idx.searchsorted(future[0]))
            tilt_pnl    = 0.0
            neutral_pnl = 0.0
            for ac, info in PORT_AC.items():
                proxy = info["proxy"]
                if proxy not in prices.columns: continue
                p_now = prices[proxy].iloc[row_idx]
                p_fut = prices[proxy].iloc[out_row]
                if np.isnan(p_now) or np.isnan(p_fut) or p_now <= 0: continue
                ret = float(np.log(p_fut / p_now))
                tilt_pnl    += tilt_norm[ac]    * ret
                neutral_pnl += neutral_norm[ac] * ret
            rec[f"tilt_pnl_{hor}d"]    = round(tilt_pnl * 100, 3)
            rec[f"neutral_pnl_{hor}d"] = round(neutral_pnl * 100, 3)
            rec[f"tilt_beat_{hor}d"]   = int(tilt_pnl > neutral_pnl)
            rec[f"status_{hor}d"]      = "RESOLVED"

        records.append(rec)

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════
#  SUMMARY REPORT
# ═══════════════════════════════════════════════════════════

def write_report(pair_df, joint_df, gold_df, port_df, path):
    lines = []
    sep   = "=" * 68
    lines += [sep, "  CPE PROPER BACKTEST REPORT", f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", sep]

    # ── PAIRWISE CALIBRATION ─────────────────────────────────
    lines += ["", "── PAIRWISE CPE CALIBRATION ─────────────────────────────────────"]
    lines += ["  Core question: do the stated CPE values hold out-of-sample?",
              "  Metric: realised hit rate vs stated CPE vs unconditional baseline",
              "  A well-calibrated signal should have: hit_rate ≈ CPE > uncond_prob"]

    if pair_df is not None and len(pair_df):
        res = pair_df[pair_df["status"]=="RESOLVED"].copy()
        res["outcome_hit"] = pd.to_numeric(res["outcome_hit"], errors="coerce")
        lines.append(f"\n  Total signal-instances fired  : {len(pair_df):,}")
        lines.append(f"  Resolved                      : {len(res):,}")
        lines.append(f"  Pending                       : {(pair_df['status']=='PENDING').sum():,}")

        if len(res) > 0:
            overall_hit  = res["outcome_hit"].mean()
            overall_cpe  = res["CPE"].mean()
            overall_unc  = res["uncond_prob"].mean()
            lines.append(f"\n  Overall realised hit rate : {overall_hit:.1%}")
            lines.append(f"  Avg stated CPE            : {overall_cpe:.1%}  (target)")
            lines.append(f"  Avg uncond_prob (baseline): {overall_unc:.1%}")
            lines.append(f"  Lift preserved            : {overall_hit/overall_unc:.2f}x "
                         f"(stated: {overall_cpe/overall_unc:.2f}x)")

            lines.append(f"\n  By horizon:")
            lines.append(f"  {'Horizon':>8}  {'N':>6}  {'Hit%':>7}  {'CPE':>7}  "
                         f"{'Uncond':>7}  {'Lift':>6}  {'Calibration'}")
            lines.append(f"  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*15}")
            for tf in sorted(res["tau_future"].unique()):
                sub = res[res["tau_future"]==tf]
                hr  = sub["outcome_hit"].mean()
                cp  = sub["CPE"].mean()
                un  = sub["uncond_prob"].mean()
                li  = hr/un if un>0 else 0
                calib = ("GOOD" if abs(hr-cp) < 0.05
                         else "OVER-stated" if hr < cp - 0.05
                         else "UNDER-stated")
                lines.append(f"  {str(tf)+'d':>8}  {len(sub):>6,}  {hr:>7.1%}  {cp:>7.1%}  "
                             f"{un:>7.1%}  {li:>6.2f}x  {calib}")

            lines.append(f"\n  By direction:")
            for d in ["bullish","bearish"]:
                sub = res[res["direction"]==d]
                if not len(sub): continue
                hr  = sub["outcome_hit"].mean()
                cp  = sub["CPE"].mean()
                un  = sub["uncond_prob"].mean()
                lines.append(f"  {d:>10}: hit={hr:.1%}  CPE={cp:.1%}  uncond={un:.1%}  "
                             f"lift={hr/un:.2f}x  n={len(sub):,}")

            lines.append(f"\n  Gold signals specifically:")
            gold_res = res[res["Y"].isin(GOLD_Y)]
            if len(gold_res):
                hr = gold_res["outcome_hit"].mean()
                cp = gold_res["CPE"].mean()
                un = gold_res["uncond_prob"].mean()
                lines.append(f"  GC=F/GLD/IAU: hit={hr:.1%}  CPE={cp:.1%}  "
                             f"uncond={un:.1%}  lift={hr/un:.2f}x  n={len(gold_res):,}")
                for tf in sorted(gold_res["tau_future"].unique()):
                    sub = gold_res[gold_res["tau_future"]==tf]
                    hr2 = sub["outcome_hit"].mean()
                    lines.append(f"    tau_future={tf:3d}d: hit={hr2:.1%}  "
                                 f"CPE={sub['CPE'].mean():.1%}  n={len(sub):,}")

    # ── JOINT CALIBRATION ────────────────────────────────────
    lines += ["", "── JOINT CPE CALIBRATION ────────────────────────────────────────"]
    if joint_df is not None and len(joint_df):
        res = joint_df[joint_df["status"]=="RESOLVED"].copy()
        res["outcome_hit"] = pd.to_numeric(res["outcome_hit"], errors="coerce")
        lines.append(f"  Joint signal-instances fired: {len(joint_df):,}")
        lines.append(f"  Resolved: {len(res):,}")
        if len(res):
            hr = res["outcome_hit"].mean()
            cp = res["joint_CPE"].mean()
            un = res["uncond_prob"].mean()
            lines.append(f"  Realised hit rate : {hr:.1%}")
            lines.append(f"  Avg joint_CPE     : {cp:.1%}  (target)")
            lines.append(f"  Avg uncond_prob   : {un:.1%}")
            lines.append(f"  Lift preserved    : {hr/un:.2f}x  (stated {cp/un:.2f}x)")
            for tf in sorted(res["tau_future"].unique()):
                sub = res[res["tau_future"]==tf]
                hr2 = sub["outcome_hit"].mean()
                lines.append(f"  tau_future={tf:3d}d: hit={hr2:.1%}  "
                             f"joint_CPE={sub['joint_CPE'].mean():.1%}  n={len(sub):,}")

    # ── GOLD DASHBOARD REPLAY ────────────────────────────────
    lines += ["", "── GOLD DASHBOARD CPE SIGNAL REPLAY ────────────────────────────"]
    lines += ["  Does following the CPE bull/bear signal balance predict gold direction?",
              "  Signal: net_score > 0.1 = BULLISH, < -0.1 = BEARISH, else NEUTRAL"]
    if gold_df is not None and len(gold_df):
        for hor in [21, 63, 126, 252]:
            col_ret  = f"gcf_ret_{hor}d"
            col_pos  = f"gcf_positive_{hor}d"
            col_stat = f"status_{hor}d"
            if col_stat not in gold_df.columns: continue
            res = gold_df[gold_df[col_stat]=="RESOLVED"].copy()
            if not len(res): continue
            res[col_pos] = pd.to_numeric(res[col_pos], errors="coerce")
            res[col_ret] = pd.to_numeric(res[col_ret], errors="coerce")

            lines.append(f"\n  Horizon {hor}d  (n={len(res)}):")
            for sig in ["BULLISH","NEUTRAL","BEARISH"]:
                sub = res[res["signal"]==sig]
                if not len(sub): continue
                pct_pos  = sub[col_pos].mean()
                mean_ret = sub[col_ret].mean()
                lines.append(f"    {sig:>8}: n={len(sub):3d}  "
                             f"gold_positive={pct_pos:.1%}  mean_ret={mean_ret:+.2f}%")

            # Is BULLISH signal associated with higher returns than BEARISH?
            bull_sub = res[res["signal"]=="BULLISH"][col_ret].dropna()
            bear_sub = res[res["signal"]=="BEARISH"][col_ret].dropna()
            neut_sub = res[res["signal"]=="NEUTRAL"][col_ret].dropna()
            if len(bull_sub) and len(bear_sub):
                diff = bull_sub.mean() - bear_sub.mean()
                lines.append(f"    BULLISH vs BEARISH mean return spread: {diff:+.2f}%")

    # ── PORTFOLIO DASHBOARD REPLAY ───────────────────────────
    lines += ["", "── PORTFOLIO DASHBOARD REPLAY ───────────────────────────────────"]
    lines += ["  Does the CPE tilt signal beat a neutral equal-weight portfolio?"]
    if port_df is not None and len(port_df):
        for hor in [21, 63, 126]:
            col_tb   = f"tilt_beat_{hor}d"
            col_tp   = f"tilt_pnl_{hor}d"
            col_np   = f"neutral_pnl_{hor}d"
            col_stat = f"status_{hor}d"
            if col_stat not in port_df.columns: continue
            res = port_df[port_df[col_stat]=="RESOLVED"].copy()
            if not len(res): continue
            for c in [col_tb, col_tp, col_np]:
                res[c] = pd.to_numeric(res[c], errors="coerce")
            beat      = res[col_tb].mean()
            tp        = res[col_tp].mean()
            np_       = res[col_np].mean()
            edge      = res[col_tp] - res[col_np]
            t_stat    = edge.mean() / edge.std() * np.sqrt(len(edge)) if edge.std() > 0 else 0
            lines.append(f"\n  Horizon {hor}d  (n={len(res)}):")
            lines.append(f"    Tilt beat neutral : {beat:.1%}  ({int(res[col_tb].sum())}/{len(res)})")
            lines.append(f"    Avg tilt PnL      : {tp:+.3f}%")
            lines.append(f"    Avg neutral PnL   : {np_:+.3f}%")
            lines.append(f"    Avg edge          : {edge.mean():+.3f}%  (t={t_stat:.2f})")

    lines.append(f"\n{sep}")
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CPE Proper Backtest")
    parser.add_argument("--start",  default="2025-01-01")
    parser.add_argument("--end",    default=str(date.today()))
    parser.add_argument("--weekly", action="store_true",
                        help="Weekly cadence (faster, fewer observations)")
    parser.add_argument("--gold-only",      action="store_true")
    parser.add_argument("--portfolio-only", action="store_true")
    args = parser.parse_args()

    do_gold = not args.portfolio_only
    do_port = not args.gold_only

    print("=" * 60)
    print("  CPE PROPER BACKTEST")
    print(f"  {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    prices, cpe, jcpe = load()
    prices_idx        = prices.index

    print("\nPre-computing return arrays...")
    ret_p, ret_r, price_cols, rate_cols, col_idx, rcol_idx, _ = precompute_returns(prices)
    print(f"  Done: {len(price_cols)} price + {len(rate_cols)} rate tickers")

    # Sim dates
    all_dates = prices_idx[
        (prices_idx >= pd.Timestamp(args.start)) &
        (prices_idx <= pd.Timestamp(args.end))
    ]
    if args.weekly:
        all_dates = all_dates[all_dates.to_series().dt.dayofweek == 0]
        if len(all_dates) == 0:
            all_dates = prices_idx[
                (prices_idx >= pd.Timestamp(args.start)) &
                (prices_idx <= pd.Timestamp(args.end))
            ][::5]

    print(f"\nTest period : {args.start} → {args.end}")
    print(f"Sim days    : {len(all_dates)}{'  (weekly)' if args.weekly else ''}")

    pair_df = joint_df = gold_df = port_df = None

    if do_gold:
        pair_df  = run_pairwise_backtest(prices, cpe, all_dates, prices_idx,
                                         ret_p, ret_r, col_idx, rcol_idx,
                                         focus_Y=GOLD_Y)
        joint_df = run_joint_backtest(prices, jcpe, all_dates, prices_idx,
                                      ret_p, ret_r, col_idx, rcol_idx,
                                      focus_Y=GOLD_Y)
        gold_df  = run_gold_dashboard_replay(prices, cpe, jcpe, all_dates, prices_idx,
                                              ret_p, ret_r, col_idx, rcol_idx)

    if do_port:
        port_df  = run_portfolio_dashboard_replay(prices, cpe, all_dates, prices_idx,
                                                   ret_p, ret_r, col_idx, rcol_idx)

    # Save CSVs
    for df, name in [
        (pair_df,  "cpe_signal_backtest.csv"),
        (joint_df, "jcpe_signal_backtest.csv"),
        (gold_df,  "dashboard_gold_replay.csv"),
        (port_df,  "dashboard_portfolio_replay.csv"),
    ]:
        if df is not None and len(df):
            path = os.path.join(BASE_DIR, name)
            df.to_csv(path, index=False)
            print(f"\nSaved: {path}")

    # Report
    report_path = os.path.join(BASE_DIR, "cpe_backtest_report.txt")
    write_report(pair_df, joint_df, gold_df, port_df, report_path)
    print(f"\nSaved: {report_path}")
    print("\n✓ Proper backtest complete.")


if __name__ == "__main__":
    main()
