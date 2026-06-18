#!/usr/bin/env python3
"""
============================================================
  CPE PREDICTION LOGGER
  Dr. Arun Ramanathan
============================================================
Run this AFTER build_gold_dashboard.py and build_portfolio_dashboard.py
each morning. It reads current dashboard state and appends
timestamped prediction rows to two CSVs:

  gold_predictions.csv       — one row per signal horizon
  portfolio_predictions.csv  — one row per day (all tilts)

After each horizon expires, run with --resolve to fill
outcome columns for all PENDING rows that are now due.

Usage:
  python log_predictions.py              # log today's predictions
  python log_predictions.py --resolve    # resolve expired predictions
  python log_predictions.py --both       # log + resolve in one step
============================================================
"""

import sys
import os
import warnings
import argparse
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── PATHS ────────────────────────────────────────────────────
# Adjust BASE_DIR to your notebooks folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GOLD_CSV = os.path.join(BASE_DIR, "gold_predictions.csv")
PORT_CSV = os.path.join(BASE_DIR, "portfolio_predictions.csv")

TODAY = date.today()
NOW   = datetime.now().strftime("%Y-%m-%d %H:%M")

print("=" * 60)
print("  CPE PREDICTION LOGGER")
print(f"  {NOW}")
print("=" * 60)


# ════════════════════════════════════════════════════════════
#  SECTION 1 — GOLD STATE READER
#  Replicates the key computations from build_gold_dashboard.py
#  so we can extract today's signal state without re-running
#  the full dashboard.
# ════════════════════════════════════════════════════════════

def fetch_gold_state():
    """
    Fetch current gold dashboard state.
    Returns a dict with all fields needed for logging.
    """
    print("\n[GOLD] Fetching current state...")

    # Price data
    gld  = yf.download("GLD",      period="2y", auto_adjust=True, progress=False)
    gcf  = yf.download("GC=F",     period="2y", auto_adjust=True, progress=False)
    ibit = yf.download("IBIT",     period="2y", auto_adjust=True, progress=False)
    fbtc = yf.download("FBTC",     period="2y", auto_adjust=True, progress=False)
    slv  = yf.download("SLV",      period="2y", auto_adjust=True, progress=False)
    sif  = yf.download("SI=F",     period="2y", auto_adjust=True, progress=False)
    uup  = yf.download("UUP",      period="2y", auto_adjust=True, progress=False)
    sgd  = yf.download("SGDUSD=X", period="2y", auto_adjust=True, progress=False)

    def last_close(df):
        return float(df["Close"].dropna().iloc[-1])

    def ret(df, days):
        c = df["Close"].dropna()
        if len(c) < days + 1:
            return np.nan
        return float((c.iloc[-1] / c.iloc[-days-1]) - 1)

    def pctile(df, days):
        c  = df["Close"].dropna()
        r  = c.pct_change(days).dropna()
        if len(r) < 2:
            return np.nan
        curr = r.iloc[-1]
        return float((r < curr).mean() * 100)

    # GLD ratio calibration (oz per share)
    GLD_RATIO = 0.09182

    gld_price  = last_close(gld)
    gcf_price  = last_close(gcf)

    # Returns
    r21  = ret(gcf, 21)
    r63  = ret(gcf, 63)
    r126 = ret(gcf, 126)
    r252 = ret(gcf, 252)

    # Percentile of 63d return
    p63 = pctile(gcf, 63)

    # SGD/USD
    sgd_rate = last_close(sgd)
    price_sgd_g = (gcf_price / 32.1507) * (1 / sgd_rate)  # USD/oz → SGD/g

    # ── AUTOCORRELATION CPE (simplified replication) ──────────
    # For each tau_past, compute % positive at each tau_future
    gcf_close = gcf["Close"].dropna()

    def autocpe(tau_past, tau_future, q_threshold=0.5):
        """% of dates where past return <= q_threshold quantile
           AND future return > 0."""
        past_ret   = gcf_close.pct_change(tau_past).dropna()
        future_ret = gcf_close.pct_change(tau_future).shift(-tau_future).dropna()
        idx        = past_ret.index.intersection(future_ret.index)
        past_ret   = past_ret.loc[idx]
        future_ret = future_ret.loc[idx]
        # current past return percentile
        curr_past  = past_ret.iloc[-1]
        q_val      = past_ret.quantile(q_threshold)
        cond       = past_ret <= q_val
        if cond.sum() < 10:
            return np.nan, cond.sum()
        pct_pos = float(future_ret[cond].gt(0).mean() * 100)
        return pct_pos, int(cond.sum())

    # Horizons to test
    horizons = [21, 63, 126, 252]
    tau_pasts = [21, 63, 126, 252]

    # Build recovery table
    recovery = {}
    for tp in tau_pasts:
        recovery[tp] = {}
        for tf in horizons:
            pct, n = autocpe(tp, tf, q_threshold=0.10)  # lowest 10%
            recovery[tp][tf] = {"pct_positive": pct, "n": n}

    # Dominant horizon: highest pct_positive across all tau_pasts at each tf
    horizon_scores = {}
    for tf in horizons:
        vals = [recovery[tp][tf]["pct_positive"] for tp in tau_pasts
                if not np.isnan(recovery[tp][tf]["pct_positive"])]
        horizon_scores[tf] = np.mean(vals) if vals else np.nan

    # ── COMPOSITE SCORE (simplified) ─────────────────────────
    # Drawdown depth (35%)
    drawdown_score = min(100, max(0, (abs(r63) / 0.15) * 60)) if not np.isnan(r63) else 50

    # Historical recovery % (35%) — use 126d horizon
    rec_126 = np.nanmean([recovery[tp][126]["pct_positive"] for tp in tau_pasts])
    recovery_score = min(100, max(0, rec_126)) if not np.isnan(rec_126) else 50

    # Predictor proximity (20%) — simplified
    ibit_5d = ret(ibit, 5)
    slv_252 = pctile(slv, 252)
    prox_score = 50  # default
    firing_count = 0
    if ibit_5d is not None and not np.isnan(ibit_5d) and ibit_5d > 0:
        firing_count += 1
    if slv_252 is not None and not np.isnan(slv_252) and slv_252 > 95:
        firing_count += 1
    prox_score = 30 + firing_count * 20

    # CPE signal score (10%)
    cpe_score = 50

    composite = (
        0.35 * drawdown_score +
        0.35 * recovery_score +
        0.20 * prox_score +
        0.10 * cpe_score
    )

    # ── VERDICT ──────────────────────────────────────────────
    if composite >= 60:
        verdict   = "BUY"
        direction = "BULLISH"
    elif composite >= 50:
        verdict   = "WAIT & WATCH"
        direction = "NEUTRAL"
    else:
        verdict   = "TOO EARLY"
        direction = "BEARISH"

    # ── STRONG HORIZONS ───────────────────────────────────────
    # Log all horizons where avg recovery pct > 45% (meaningful signal)
    # or all if none cross threshold
    strong_horizons = [tf for tf, sc in horizon_scores.items()
                       if not np.isnan(sc) and sc > 40]
    if not strong_horizons:
        strong_horizons = [max(horizon_scores, key=lambda k: horizon_scores[k]
                               if not np.isnan(horizon_scores[k]) else -1)]

    print(f"  GC=F price : ${gcf_price:,.2f}")
    print(f"  Score      : {composite:.1f}/100  →  {verdict}")
    print(f"  Direction  : {direction}")
    print(f"  63d return : {r63*100:.1f}%  (P{p63:.1f}ile)")
    print(f"  Strong horizons: {strong_horizons}")

    return {
        "gcf_price"      : gcf_price,
        "gld_price"      : gld_price,
        "price_sgd_g"    : price_sgd_g,
        "r21"            : r21,
        "r63"            : r63,
        "r126"           : r126,
        "r252"           : r252,
        "p63"            : p63,
        "composite"      : composite,
        "verdict"        : verdict,
        "direction"      : direction,
        "strong_horizons": strong_horizons,
        "horizon_scores" : horizon_scores,
        "recovery"       : recovery,
    }


def fetch_portfolio_state():
    """
    Fetch current portfolio dashboard state.
    Returns tilt signals and prices for all asset classes.
    """
    print("\n[PORTFOLIO] Fetching current state...")

    tickers = {
        "equities": "SPY",
        "gold"    : "GC=F",
        "bonds"   : "TLT",
        "crypto"  : "IBIT",
        "fx"      : "UUP",
    }

    prices = {}
    for cls, tk in tickers.items():
        df = yf.download(tk, period="5d", auto_adjust=True, progress=False)
        prices[cls] = float(df["Close"].dropna().iloc[-1])
        print(f"  {cls:10s}: {tk:10s} = {prices[cls]:.4f}")

    # ── READ TILT STATE FROM PORTFOLIO DASHBOARD OUTPUT ───────
    # The portfolio dashboard computes tilts from cpe_results.parquet.
    # Here we read the parquet directly to get today's tilt state.
    # If parquet not found, fall back to manual entry prompt.

    parquet_path = os.path.join(BASE_DIR, "cpe_results.parquet")
    tilt_state = {
        "equities_tilt": "NEUTRAL",
        "gold_tilt"    : "NEUTRAL",
        "bonds_tilt"   : "NEUTRAL",
        "crypto_tilt"  : "NEUTRAL",
        "fx_tilt"      : "NEUTRAL",
        "dominant_horizon": 63,
        "n_signals_firing": 0,
    }

    if os.path.exists(parquet_path):
        # Replicate tilt logic from build_portfolio_dashboard.py
        try:
            cpe = pd.read_parquet(parquet_path)
            # Download all relevant tickers for regime check
            all_tickers = list(set(cpe["X"].unique().tolist()[:50]))  # sample
            # Use simplified tilt: check if bearish signals dominate per asset class
            asset_map = {
                "equities": ["SPY", "QQQ"],
                "gold"    : ["GC=F", "GLD", "IAU"],
                "bonds"   : ["TLT", "AGG", "SHY"],
                "crypto"  : ["IBIT", "FBTC", "BTC-USD"],
                "fx"      : ["UUP", "SGDUSD=X"],
            }
            for cls, syms in asset_map.items():
                cls_rows = cpe[cpe["Y"].isin(syms)]
                bull = cls_rows[cls_rows["direction"] == "bullish"]["lift"].mean()
                bear = cls_rows[cls_rows["direction"] == "bearish"]["lift"].mean()
                if pd.isna(bull): bull = 0
                if pd.isna(bear): bear = 0
                if bull > 2.0 and bull > bear * 1.2:
                    tilt_state[f"{cls}_tilt"] = "TILT UP"
                elif bear > 2.0 and bear > bull * 1.2:
                    tilt_state[f"{cls}_tilt"] = "TILT DOWN"
                else:
                    tilt_state[f"{cls}_tilt"] = "NEUTRAL"
        except Exception as e:
            print(f"  [WARN] Could not read parquet: {e}")
            print("  [INFO] Using manually specified tilt state below.")
            # ── MANUAL OVERRIDE — update these daily if parquet unavailable ──
            tilt_state["equities_tilt"]    = "NEUTRAL"
            tilt_state["gold_tilt"]        = "NEUTRAL"
            tilt_state["bonds_tilt"]       = "TILT DOWN"
            tilt_state["crypto_tilt"]      = "NEUTRAL"
            tilt_state["fx_tilt"]          = "TILT UP"
            tilt_state["dominant_horizon"] = 63
            tilt_state["n_signals_firing"] = 79
    else:
        print("  [INFO] parquet not found — using manual tilt state.")
        tilt_state["equities_tilt"]    = "NEUTRAL"
        tilt_state["gold_tilt"]        = "NEUTRAL"
        tilt_state["bonds_tilt"]       = "TILT DOWN"
        tilt_state["crypto_tilt"]      = "NEUTRAL"
        tilt_state["fx_tilt"]          = "TILT UP"
        tilt_state["dominant_horizon"] = 63
        tilt_state["n_signals_firing"] = 79

    return {**tilt_state, **{f"{cls}_price": prices[cls] for cls in tickers}}


# ════════════════════════════════════════════════════════════
#  SECTION 2 — CSV INITIALISATION
# ════════════════════════════════════════════════════════════

GOLD_COLS = [
    # Prediction side
    "date_predicted",
    "gcf_price_usd",
    "price_sgd_per_g",
    "composite_score",
    "verdict",
    "direction",
    "horizon_days",
    "recovery_pct_positive",  # % of historical episodes that were positive at this horizon
    "r21", "r63", "r126", "r252",
    "p63_percentile",
    # Outcome side (filled on resolution)
    "outcome_date",
    "gcf_price_at_outcome",
    "actual_return_pct",
    "prediction_correct",     # 1=correct, 0=wrong, blank=pending
    "status",                 # PENDING / RESOLVED
    "notes",
]

PORT_COLS = [
    # Prediction side
    "date_predicted",
    "equities_tilt", "gold_tilt", "bonds_tilt", "crypto_tilt", "fx_tilt",
    "dominant_horizon_days",
    "n_signals_firing",
    "equities_price", "gold_price", "bonds_price", "crypto_price", "fx_price",
    # Outcome side (filled on resolution)
    "outcome_date",
    "equities_price_out", "gold_price_out", "bonds_price_out",
    "crypto_price_out", "fx_price_out",
    "equities_return", "gold_return", "bonds_return", "crypto_return", "fx_return",
    "tilt_pnl",          # return of tilt-weighted portfolio
    "neutral_pnl",       # return of equal-weight portfolio
    "tilt_beat_neutral", # 1=yes, 0=no, blank=pending
    "status",
    "notes",
]


def init_csv(path, cols):
    if not os.path.exists(path):
        pd.DataFrame(columns=cols).to_csv(path, index=False)
        print(f"  Created: {path}")


# ════════════════════════════════════════════════════════════
#  SECTION 3 — LOG TODAY'S PREDICTIONS
# ════════════════════════════════════════════════════════════

def log_gold(gold_state):
    init_csv(GOLD_CSV, GOLD_COLS)
    df = pd.read_csv(GOLD_CSV)

    # Check if already logged today
    if str(TODAY) in df["date_predicted"].astype(str).values:
        print(f"\n[GOLD] Already logged for {TODAY} — skipping.")
        return

    gs = gold_state
    rows = []

    for tf in gs["strong_horizons"]:
        # Average recovery pct across all tau_pasts for this horizon
        rec_vals = [gs["recovery"][tp][tf]["pct_positive"]
                    for tp in [21, 63, 126, 252]
                    if not np.isnan(gs["recovery"][tp][tf]["pct_positive"])]
        rec_avg = np.mean(rec_vals) if rec_vals else np.nan

        row = {
            "date_predicted"      : str(TODAY),
            "gcf_price_usd"       : round(gs["gcf_price"], 2),
            "price_sgd_per_g"     : round(gs["price_sgd_g"], 2),
            "composite_score"     : round(gs["composite"], 1),
            "verdict"             : gs["verdict"],
            "direction"           : gs["direction"],
            "horizon_days"        : tf,
            "recovery_pct_positive": round(rec_avg, 1) if not np.isnan(rec_avg) else "",
            "r21"                 : round(gs["r21"] * 100, 2) if not np.isnan(gs["r21"]) else "",
            "r63"                 : round(gs["r63"] * 100, 2) if not np.isnan(gs["r63"]) else "",
            "r126"                : round(gs["r126"] * 100, 2) if not np.isnan(gs["r126"]) else "",
            "r252"                : round(gs["r252"] * 100, 2) if not np.isnan(gs["r252"]) else "",
            "p63_percentile"      : round(gs["p63"], 1) if not np.isnan(gs["p63"]) else "",
            # Outcome — blank until resolved
            "outcome_date"        : "",
            "gcf_price_at_outcome": "",
            "actual_return_pct"   : "",
            "prediction_correct"  : "",
            "status"              : "PENDING",
            "notes"               : "",
        }
        rows.append(row)
        print(f"  [GOLD] Logged horizon {tf}d  |  direction={gs['direction']}  |  score={gs['composite']:.1f}")

    new_rows = pd.DataFrame(rows, columns=GOLD_COLS)
    df = pd.concat([df, new_rows], ignore_index=True)
    df.to_csv(GOLD_CSV, index=False)
    print(f"  [GOLD] {len(rows)} row(s) appended → {GOLD_CSV}")


def log_portfolio(port_state):
    init_csv(PORT_CSV, PORT_COLS)
    df = pd.read_csv(PORT_CSV)

    if str(TODAY) in df["date_predicted"].astype(str).values:
        print(f"\n[PORTFOLIO] Already logged for {TODAY} — skipping.")
        return

    ps = port_state
    row = {
        "date_predicted"      : str(TODAY),
        "equities_tilt"       : ps["equities_tilt"],
        "gold_tilt"           : ps["gold_tilt"],
        "bonds_tilt"          : ps["bonds_tilt"],
        "crypto_tilt"         : ps["crypto_tilt"],
        "fx_tilt"             : ps["fx_tilt"],
        "dominant_horizon_days": ps["dominant_horizon"],
        "n_signals_firing"    : ps["n_signals_firing"],
        "equities_price"      : round(ps["equities_price"], 4),
        "gold_price"          : round(ps["gold_price"], 2),
        "bonds_price"         : round(ps["bonds_price"], 4),
        "crypto_price"        : round(ps["crypto_price"], 4),
        "fx_price"            : round(ps["fx_price"], 4),
        # Outcome blank
        "outcome_date"        : "",
        "equities_price_out"  : "",
        "gold_price_out"      : "",
        "bonds_price_out"     : "",
        "crypto_price_out"    : "",
        "fx_price_out"        : "",
        "equities_return"     : "",
        "gold_return"         : "",
        "bonds_return"        : "",
        "crypto_return"       : "",
        "fx_return"           : "",
        "tilt_pnl"            : "",
        "neutral_pnl"         : "",
        "tilt_beat_neutral"   : "",
        "status"              : "PENDING",
        "notes"               : "",
    }

    df = pd.concat([df, pd.DataFrame([row], columns=PORT_COLS)], ignore_index=True)
    df.to_csv(PORT_CSV, index=False)
    print(f"  [PORTFOLIO] Row appended → {PORT_CSV}")
    print(f"  Tilts: EQ={ps['equities_tilt']}  AU={ps['gold_tilt']}  "
          f"BD={ps['bonds_tilt']}  CR={ps['crypto_tilt']}  FX={ps['fx_tilt']}")


# ════════════════════════════════════════════════════════════
#  SECTION 4 — RESOLVE EXPIRED PREDICTIONS
# ════════════════════════════════════════════════════════════

def tilt_to_weight(tilt, neutral_wt):
    """Convert tilt label to suggested portfolio weight."""
    if tilt == "TILT UP":
        return neutral_wt * 1.4
    elif tilt == "TILT DOWN":
        return neutral_wt * 0.1
    else:
        return neutral_wt


NEUTRAL_WEIGHTS = {
    "equities": 0.329,
    "gold"    : 0.299,
    "bonds"   : 0.047,
    "crypto"  : 0.225,
    "fx"      : 0.100,
}


def resolve_gold():
    if not os.path.exists(GOLD_CSV):
        print("[GOLD] No CSV found to resolve.")
        return

    df = pd.read_csv(GOLD_CSV)
    pending = df[df["status"] == "PENDING"].copy()
    if pending.empty:
        print("[GOLD] No pending predictions to resolve.")
        return

    # Fetch current GC=F price
    gcf = yf.download("GC=F", period="5d", auto_adjust=True, progress=False)
    current_price = float(gcf["Close"].dropna().iloc[-1])
    print(f"\n[GOLD RESOLVE] Current GC=F price: ${current_price:,.2f}")

    resolved = 0
    for idx, row in pending.iterrows():
        pred_date    = pd.to_datetime(row["date_predicted"]).date()
        horizon_days = int(row["horizon_days"])
        outcome_date = pred_date + timedelta(days=horizon_days)

        if TODAY >= outcome_date:
            # Fetch price at outcome date (nearest trading day)
            start = outcome_date - timedelta(days=5)
            end   = outcome_date + timedelta(days=5)
            hist  = yf.download("GC=F", start=str(start), end=str(end),
                                auto_adjust=True, progress=False)
            if hist.empty:
                print(f"  [SKIP] No price data for outcome date {outcome_date}")
                continue

            out_price   = float(hist["Close"].dropna().iloc[0])
            entry_price = float(row["gcf_price_usd"])
            actual_ret  = (out_price / entry_price - 1) * 100
            direction   = row["direction"]

            # Correct if direction matches actual move
            if direction == "BULLISH":
                correct = 1 if actual_ret > 0 else 0
            elif direction == "BEARISH":
                correct = 1 if actual_ret < 0 else 0
            else:  # NEUTRAL — correct if within ±3%
                correct = 1 if abs(actual_ret) < 3 else 0

            df.at[idx, "outcome_date"]         = str(outcome_date)
            df.at[idx, "gcf_price_at_outcome"] = round(out_price, 2)
            df.at[idx, "actual_return_pct"]    = round(actual_ret, 2)
            df.at[idx, "prediction_correct"]   = correct
            df.at[idx, "status"]               = "RESOLVED"

            result_str = "✓ CORRECT" if correct else "✗ WRONG"
            print(f"  [{pred_date} → {outcome_date}] {direction} {horizon_days}d | "
                  f"entry=${entry_price:.0f} → out=${out_price:.0f} | "
                  f"ret={actual_ret:+.1f}% | {result_str}")
            resolved += 1

    df.to_csv(GOLD_CSV, index=False)
    print(f"  [GOLD] {resolved} prediction(s) resolved.")


def resolve_portfolio():
    if not os.path.exists(PORT_CSV):
        print("[PORTFOLIO] No CSV found to resolve.")
        return

    df = pd.read_csv(PORT_CSV)
    pending = df[df["status"] == "PENDING"].copy()
    if pending.empty:
        print("[PORTFOLIO] No pending predictions to resolve.")
        return

    # Fetch current prices
    tickers = {"equities": "SPY", "gold": "GC=F",
               "bonds": "TLT", "crypto": "IBIT", "fx": "UUP"}
    current = {}
    for cls, tk in tickers.items():
        hist = yf.download(tk, period="5d", auto_adjust=True, progress=False)
        current[cls] = float(hist["Close"].dropna().iloc[-1])

    print(f"\n[PORTFOLIO RESOLVE] Current prices fetched.")
    resolved = 0

    for idx, row in pending.iterrows():
        pred_date    = pd.to_datetime(row["date_predicted"]).date()
        horizon_days = int(row["dominant_horizon_days"])
        outcome_date = pred_date + timedelta(days=horizon_days)

        if TODAY >= outcome_date:
            # Fetch prices at outcome date for each asset
            out_prices = {}
            for cls, tk in tickers.items():
                start = outcome_date - timedelta(days=5)
                end   = outcome_date + timedelta(days=5)
                hist  = yf.download(tk, start=str(start), end=str(end),
                                   auto_adjust=True, progress=False)
                if not hist.empty:
                    out_prices[cls] = float(hist["Close"].dropna().iloc[0])
                else:
                    out_prices[cls] = np.nan

            # Compute returns
            returns = {}
            for cls in tickers:
                entry = float(row[f"{cls}_price"])
                out   = out_prices.get(cls, np.nan)
                returns[cls] = (out / entry - 1) if (not np.isnan(out) and entry > 0) else np.nan

            # Tilt PnL — weighted by tilt-adjusted weights (normalised)
            tilt_weights = {}
            for cls in tickers:
                tilt = row[f"{cls}_tilt"]
                tilt_weights[cls] = tilt_to_weight(tilt, NEUTRAL_WEIGHTS[cls])
            tw_sum = sum(tilt_weights.values())
            tilt_weights = {k: v / tw_sum for k, v in tilt_weights.items()}

            neutral_weights = {k: v / sum(NEUTRAL_WEIGHTS.values())
                               for k, v in NEUTRAL_WEIGHTS.items()}

            tilt_pnl    = sum(tilt_weights[cls] * returns[cls]
                              for cls in tickers if not np.isnan(returns[cls]))
            neutral_pnl = sum(neutral_weights[cls] * returns[cls]
                              for cls in tickers if not np.isnan(returns[cls]))

            beat = 1 if tilt_pnl > neutral_pnl else 0

            # Fill outcome columns
            for cls in tickers:
                df.at[idx, f"{cls}_price_out"] = round(out_prices.get(cls, np.nan), 4)
                df.at[idx, f"{cls}_return"]    = round(returns[cls] * 100, 2) \
                    if not np.isnan(returns[cls]) else ""

            df.at[idx, "outcome_date"]      = str(outcome_date)
            df.at[idx, "tilt_pnl"]          = round(tilt_pnl * 100, 2)
            df.at[idx, "neutral_pnl"]       = round(neutral_pnl * 100, 2)
            df.at[idx, "tilt_beat_neutral"] = beat
            df.at[idx, "status"]            = "RESOLVED"

            result_str = "✓ TILT WON" if beat else "✗ NEUTRAL WON"
            print(f"  [{pred_date} → {outcome_date}] "
                  f"tilt={tilt_pnl*100:+.2f}% vs neutral={neutral_pnl*100:+.2f}% | {result_str}")
            resolved += 1

    df.to_csv(PORT_CSV, index=False)
    print(f"  [PORTFOLIO] {resolved} prediction(s) resolved.")


# ════════════════════════════════════════════════════════════
#  SECTION 5 — SUMMARY PRINTOUT
# ════════════════════════════════════════════════════════════

def print_summary():
    print("\n" + "=" * 60)
    print("  PREDICTION LOG SUMMARY")
    print("=" * 60)

    for label, path in [("GOLD", GOLD_CSV), ("PORTFOLIO", PORT_CSV)]:
        if not os.path.exists(path):
            print(f"  {label}: no CSV yet")
            continue
        df = pd.read_csv(path)
        n_total    = len(df)
        n_pending  = (df["status"] == "PENDING").sum()
        n_resolved = (df["status"] == "RESOLVED").sum()

        if label == "GOLD" and n_resolved > 0:
            res = df[df["status"] == "RESOLVED"]
            acc = res["prediction_correct"].astype(float).mean() * 100
            print(f"  GOLD  : {n_total} rows | {n_pending} pending | "
                  f"{n_resolved} resolved | accuracy={acc:.0f}%")
        elif label == "PORTFOLIO" and n_resolved > 0:
            res = df[df["status"] == "RESOLVED"]
            win = res["tilt_beat_neutral"].astype(float).mean() * 100
            avg_edge = (res["tilt_pnl"].astype(float) -
                        res["neutral_pnl"].astype(float)).mean()
            print(f"  PORT  : {n_total} rows | {n_pending} pending | "
                  f"{n_resolved} resolved | tilt won={win:.0f}% | "
                  f"avg edge={avg_edge:+.2f}%")
        else:
            print(f"  {label:5s}: {n_total} rows | {n_pending} pending | "
                  f"{n_resolved} resolved")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CPE Prediction Logger")
    parser.add_argument("--resolve", action="store_true",
                        help="Resolve expired predictions only")
    parser.add_argument("--both", action="store_true",
                        help="Log today + resolve expired")
    args = parser.parse_args()

    do_log     = not args.resolve          # log unless --resolve only
    do_resolve = args.resolve or args.both

    if do_log:
        print("\n── LOGGING TODAY'S PREDICTIONS ─────────────────────────")
        gold_state = fetch_gold_state()
        port_state = fetch_portfolio_state()
        log_gold(gold_state)
        log_portfolio(port_state)

    if do_resolve:
        print("\n── RESOLVING EXPIRED PREDICTIONS ───────────────────────")
        resolve_gold()
        resolve_portfolio()

    print_summary()
    print("\nDone.\n")


if __name__ == "__main__":
    main()
