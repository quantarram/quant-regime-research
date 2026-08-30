#!/usr/bin/env python3
"""
============================================================
  CPE PREDICTION LOGGER
  Dr. Arun Ramanathan
============================================================
Run this AFTER build_gold_dashboard.py, build_portfolio_dashboard.py,
and build_metals_dashboard.py each morning. It reads current dashboard
state and appends timestamped prediction rows to three CSVs:

  gold_predictions.csv       — one row per signal horizon
  portfolio_predictions.csv  — one row per day (all tilts)
  metals_predictions.csv     — one row per (Silver/Platinum) x horizon
                                 (Gold itself is already covered by
                                 gold_predictions.csv above)

After each horizon expires, run with --resolve to fill
outcome columns for all PENDING rows that are now due.

Usage:
  python log_predictions.py              # log today's predictions
  python log_predictions.py --resolve    # resolve expired predictions
  python log_predictions.py --both       # log + resolve in one step
============================================================
"""

import sys
import re
import os
import warnings
import argparse
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ── PATHS ────────────────────────────────────────────────────
# Adjust BASE_DIR to your notebooks folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GOLD_CSV   = os.path.join(BASE_DIR, "gold_predictions.csv")
PORT_CSV   = os.path.join(BASE_DIR, "portfolio_predictions.csv")
METALS_CSV = os.path.join(BASE_DIR, "metals_predictions.csv")

# Metal name -> ticker used for outcome-price resolution. Must be oz-equivalent
# spot-scale, matching the "spot_price_usd_oz" entry price logged by log_metals()
# (build_metals_dashboard.py's calibrated D["metals"][name]["spot_usd_oz"]).
# Silver: SI=F futures already trade at spot scale. Platinum: PPLT is a raw ETF
# share price (~$15, NOT oz-equivalent) — must use PL=F futures instead, which
# is what build_metals_dashboard.py's calibrate_ratio() actually calibrates
# spot_usd_oz against. Using PPLT here previously produced a bogus ~-99% "return"
# by diffing two different price scales (fixed 2026-08-03).
METALS_TICKERS = {"Silver": "SI=F", "Platinum": "PL=F"}

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

def composite_to_verdict(composite):
    """
    Derive (verdict, direction) from a composite buy score. Thresholds must
    match build_gold_dashboard.py's / build_metals_dashboard.py's composite
    scale exactly (both use the same >=70/55/40/25 ladder).
    """
    if composite >= 70:
        return "STRONG BUY", "BULLISH"
    elif composite >= 55:
        return "BUY ZONE", "BULLISH"
    elif composite >= 40:
        return "WAIT & WATCH", "NEUTRAL"
    elif composite >= 25:
        return "NEUTRAL", "NEUTRAL"
    else:
        return "TOO EARLY", "BEARISH"


def fetch_gold_state():
    """
    Parse today's state directly from the generated gold_dashboard.html.
    This is the single source of truth — no recomputation, no divergence.
    """
    print("\n[GOLD] Reading state from gold_dashboard.html...")

    html_path = os.path.join(BASE_DIR, "gold_dashboard.html")
    if not os.path.exists(html_path):
        raise FileNotFoundError(
            "gold_dashboard.html not found. Run build_gold_dashboard.py first."
        )

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract the JS data bundle: const D = {...};
    match = re.search(r'const D = (\{.*?\});\n', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find 'const D = {...}' in gold_dashboard.html")

    import json
    D = json.loads(match.group(1))

    # ── STALENESS GUARD ───────────────────────────────────────
    # Refuse to log if dashboard was not built today.
    # This prevents logging stale values if the script is run
    # before build_gold_dashboard.py has been executed today.
    generated_date = D.get("generated", "")[:10]  # "YYYY-MM-DD"
    today_str = str(TODAY)
    if generated_date != today_str:
        raise RuntimeError(
            f"gold_dashboard.html was generated on {generated_date}, not today ({today_str}).\n"
            f"Run build_gold_dashboard.py first, then re-run this script."
        )

    # ── Core prices ───────────────────────────────────────────
    gcf_price   = float(D["gold_usd"])
    price_sgd_g = float(D["gold_sgd_g"])

    # ── Returns (stored as % in chg dict) ────────────────────
    chg  = D.get("chg", {})
    r21  = float(chg.get("21", 0)) / 100
    r63  = float(chg.get("63", 0)) / 100
    r126 = float(chg.get("126", 0)) / 100
    r252 = float(chg.get("252", 0)) / 100

    # ── Composite score & verdict ─────────────────────────────
    comp       = D["components"]
    composite  = float(comp["composite"])
    label      = comp["label"]          # e.g. "WATCH — APPROACHING BUY"

    verdict, direction = composite_to_verdict(composite)

    # ── Percentile of 63d return ──────────────────────────────
    # Stored in auto_cpe block
    auto_cpe = D.get("auto_cpe", {})
    p63 = float(auto_cpe.get("63", {}).get("current_percentile", 0))

    # ── Recovery % at each horizon (from auto_cpe) ───────────
    # auto_cpe keys are tau_past; each has fwd_X_pct_positive
    horizon_scores = {21: [], 63: [], 126: [], 252: []}
    recovery = {}
    for tau_str, block in auto_cpe.items():
        try:
            tp = int(tau_str)
        except ValueError:
            continue
        recovery[tp] = {}
        for tf in [21, 63, 126, 252]:
            key = f"fwd_{tf}_pct_positive"
            pct = block.get(key, np.nan)
            n   = block.get(f"fwd_{tf}_n", 0)
            recovery[tp][tf] = {"pct_positive": float(pct) if pct is not None else np.nan,
                                 "n": int(n)}
            if not np.isnan(float(pct if pct is not None else np.nan)):
                horizon_scores[tf].append(float(pct))

    horizon_scores = {tf: np.mean(vals) if vals else np.nan
                      for tf, vals in horizon_scores.items()}

    # ── Strong horizons ───────────────────────────────────────
    # Log all horizons that have data. If composite >= 50, log all.
    # If composite < 50 (bearish), log only those with pct_positive < 40
    # (confirming the bearish call). Always log at least one.
    strong_horizons = [tf for tf, sc in horizon_scores.items()
                       if not np.isnan(sc)]
    if not strong_horizons:
        strong_horizons = [126]  # fallback

    print(f"  GC=F price : ${gcf_price:,.2f}")
    print(f"  SGD/g      : S${price_sgd_g:.2f}")
    print(f"  Score      : {composite:.1f}/100  →  {verdict}")
    print(f"  Direction  : {direction}")
    print(f"  63d return : {r63*100:.1f}%  (P{p63:.1f}ile)")
    print(f"  Strong horizons: {strong_horizons}")

    return {
        "gcf_price"      : gcf_price,
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
    Parse today's state directly from portfolio_dashboard.html.
    Reads const D bundle — exact same data the dashboard displays.
    """
    print("\n[PORTFOLIO] Reading state from portfolio_dashboard.html...")

    html_path = os.path.join(BASE_DIR, "portfolio_dashboard.html")
    if not os.path.exists(html_path):
        raise FileNotFoundError(
            "portfolio_dashboard.html not found. Run build_portfolio_dashboard.py first."
        )

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    import json
    match = re.search(r'const D = (\{.*?\});\n', html, re.DOTALL)
    if not match:
        raise ValueError("Could not find 'const D = {...}' in portfolio_dashboard.html")
    D = json.loads(match.group(1))

    # ── STALENESS GUARD ───────────────────────────────────────
    generated_date = D.get("gen", "")[:10]  # "YYYY-MM-DD"
    today_str = str(TODAY)
    if generated_date != today_str:
        raise RuntimeError(
            f"portfolio_dashboard.html was generated on {generated_date}, not today ({today_str}).\n"
            f"Run build_portfolio_dashboard.py first, then re-run this script."
        )

    # ── Tilts from tilt_summaries ─────────────────────────────
    # Keys: Equities, Gold, Bonds, Crypto, FX
    ts = D["tilt_summaries"]
    cls_map = {
        "equities": "Equities",
        "gold"    : "Gold",
        "bonds"   : "Bonds",
        "crypto"  : "Crypto",
        "fx"      : "FX",
    }
    tilt_map = {cls: ts[label]["overall_label"]
                for cls, label in cls_map.items()}

    # ── Dominant horizon: horizon with most non-NEUTRAL/NO SIGNAL tilts ──
    dominant_hz = 63  # default
    best_count = 0
    for hz in D.get("horizons", [21, 63, 126, 252]):
        hz_str = str(hz)
        count = sum(1 for label in cls_map.values()
                    if ts[label]["horizon_labels"].get(hz_str, "NO SIGNAL")
                    not in ("NO SIGNAL", "NEUTRAL"))
        if count > best_count:
            best_count = count
            dominant_hz = hz

    # ── Prices from snapshot ──────────────────────────────────
    snap = D["snapshot"]
    prices = {cls: float(snap[label]["price"])
              for cls, label in cls_map.items()}

    n_firing = int(D.get("firing_pred_count", 0))

    for cls in cls_map:
        print(f"  {cls:10s}: tilt={tilt_map[cls]:12s}  price={prices[cls]:.4f}")
    print(f"  n_firing={n_firing}  dominant_hz={dominant_hz}d")

    return {
        "equities_tilt"   : tilt_map["equities"],
        "gold_tilt"       : tilt_map["gold"],
        "bonds_tilt"      : tilt_map["bonds"],
        "crypto_tilt"     : tilt_map["crypto"],
        "fx_tilt"         : tilt_map["fx"],
        "dominant_horizon": dominant_hz,
        "n_signals_firing": n_firing,
        "equities_price"  : prices["equities"],
        "gold_price"      : prices["gold"],
        "bonds_price"     : prices["bonds"],
        "crypto_price"    : prices["crypto"],
        "fx_price"        : prices["fx"],
    }


def fetch_metals_state():
    """
    Parse today's state directly from precious_metals_dashboard.html for
    Silver and Platinum (Gold is already tracked via fetch_gold_state()).
    """
    print("\n[METALS] Reading state from precious_metals_dashboard.html...")

    html_path = os.path.join(BASE_DIR, "precious_metals_dashboard.html")
    if not os.path.exists(html_path):
        raise FileNotFoundError(
            "precious_metals_dashboard.html not found. Run build_metals_dashboard.py first."
        )

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    import json
    match = re.search(r'const D = (\{.*?\});\n', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find 'const D = {...}' in precious_metals_dashboard.html")
    D = json.loads(match.group(1))

    # ── STALENESS GUARD ───────────────────────────────────────
    generated_date = D.get("gen", "")[:10]  # "YYYY-MM-DD"
    today_str = str(TODAY)
    if generated_date != today_str:
        raise RuntimeError(
            f"precious_metals_dashboard.html was generated on {generated_date}, not today ({today_str}).\n"
            f"Run build_metals_dashboard.py first, then re-run this script."
        )

    states = {}
    for name in METALS_TICKERS:
        d = D["metals"][name]
        composite = float(d["composite"])
        verdict, direction = composite_to_verdict(composite)

        recovery = {int(tf): block for tf, block in d["recovery_63"].items()}
        strong_horizons = sorted(recovery.keys())
        if not strong_horizons:
            strong_horizons = [126]

        states[name] = {
            "spot_usd_oz"    : float(d["spot_usd_oz"]),
            "composite"      : composite,
            "verdict"        : verdict,
            "direction"      : direction,
            "chg"            : {int(k): float(v) for k, v in d["chg"].items()},
            "curr_pct_63"    : float(d["curr_pct_63"]),
            "strong_horizons": strong_horizons,
            "recovery"       : recovery,
        }
        print(f"  {name:9s}: spot=${float(d['spot_usd_oz']):,.2f}/oz  score={composite:.1f}/100  "
              f"-> {verdict} ({direction})  strong_horizons={strong_horizons}")

    return states


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


METALS_COLS = [
    # Prediction side
    "date_predicted",
    "metal",                  # Silver / Platinum
    "spot_price_usd_oz",
    "composite_score",
    "verdict",
    "direction",
    "horizon_days",
    "recovery_pct_positive",  # % of historical episodes that were positive at this horizon
    "chg_21", "chg_63", "chg_126", "chg_252",
    "curr_pct_63",            # current 63d-return percentile within own history
    # Outcome side (filled on resolution)
    "outcome_date",
    "price_at_outcome",
    "actual_return_pct",
    "prediction_correct",     # 1=correct, 0=wrong, blank=pending
    "status",                 # PENDING / RESOLVED
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


def log_metals(metals_state):
    init_csv(METALS_CSV, METALS_COLS)
    df = pd.read_csv(METALS_CSV)

    already = df[(df["date_predicted"].astype(str) == str(TODAY))]
    rows = []
    for name, ms in metals_state.items():
        if not already[already["metal"] == name].empty:
            print(f"  [METALS] {name} already logged for {TODAY} — skipping.")
            continue

        for tf in ms["strong_horizons"]:
            rec = ms["recovery"].get(tf, {})
            row = {
                "date_predicted"       : str(TODAY),
                "metal"                : name,
                "spot_price_usd_oz"    : round(ms["spot_usd_oz"], 2),
                "composite_score"      : round(ms["composite"], 1),
                "verdict"              : ms["verdict"],
                "direction"            : ms["direction"],
                "horizon_days"         : tf,
                "recovery_pct_positive": rec.get("pct_positive", ""),
                "chg_21"               : ms["chg"].get(21, ""),
                "chg_63"               : ms["chg"].get(63, ""),
                "chg_126"              : ms["chg"].get(126, ""),
                "chg_252"              : ms["chg"].get(252, ""),
                "curr_pct_63"          : round(ms["curr_pct_63"], 1),
                # Outcome — blank until resolved
                "outcome_date"         : "",
                "price_at_outcome"     : "",
                "actual_return_pct"    : "",
                "prediction_correct"   : "",
                "status"               : "PENDING",
                "notes"                : "",
            }
            rows.append(row)
            print(f"  [METALS] Logged {name} horizon {tf}d  |  direction={ms['direction']}  |  score={ms['composite']:.1f}")

    if not rows:
        return

    new_rows = pd.DataFrame(rows, columns=METALS_COLS)
    df = pd.concat([df, new_rows], ignore_index=True)
    df.to_csv(METALS_CSV, index=False)
    print(f"  [METALS] {len(rows)} row(s) appended → {METALS_CSV}")


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

    import yfinance as yf
    # Fetch current GC=F price
    gcf = yf.download("GC=F", period="5d", auto_adjust=True, progress=False)
    current_price = float(gcf["Close"].squeeze().dropna().iloc[-1])
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

            out_price   = float(hist["Close"].squeeze().dropna().iloc[0])
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


def resolve_metals():
    if not os.path.exists(METALS_CSV):
        print("[METALS] No CSV found to resolve.")
        return

    df = pd.read_csv(METALS_CSV)
    pending = df[df["status"] == "PENDING"].copy()
    if pending.empty:
        print("[METALS] No pending predictions to resolve.")
        return

    import yfinance as yf
    resolved = 0
    for name, ticker in METALS_TICKERS.items():
        rows = pending[pending["metal"] == name]
        if rows.empty:
            continue

        hist_cache = {}
        for idx, row in rows.iterrows():
            pred_date    = pd.to_datetime(row["date_predicted"]).date()
            horizon_days = int(row["horizon_days"])
            outcome_date = pred_date + timedelta(days=horizon_days)

            if TODAY >= outcome_date:
                if outcome_date not in hist_cache:
                    start = outcome_date - timedelta(days=5)
                    end   = outcome_date + timedelta(days=5)
                    hist_cache[outcome_date] = yf.download(
                        ticker, start=str(start), end=str(end),
                        auto_adjust=True, progress=False)
                hist = hist_cache[outcome_date]
                if hist.empty:
                    print(f"  [SKIP] No price data for {name} outcome date {outcome_date}")
                    continue

                out_price   = float(hist["Close"].squeeze().dropna().iloc[0])
                entry_price = float(row["spot_price_usd_oz"])
                actual_ret  = (out_price / entry_price - 1) * 100
                direction   = row["direction"]

                if direction == "BULLISH":
                    correct = 1 if actual_ret > 0 else 0
                elif direction == "BEARISH":
                    correct = 1 if actual_ret < 0 else 0
                else:  # NEUTRAL — correct if within ±3%
                    correct = 1 if abs(actual_ret) < 3 else 0

                df.at[idx, "outcome_date"]      = str(outcome_date)
                df.at[idx, "price_at_outcome"]  = round(out_price, 2)
                df.at[idx, "actual_return_pct"] = round(actual_ret, 2)
                df.at[idx, "prediction_correct"]= correct
                df.at[idx, "status"]            = "RESOLVED"

                result_str = "✓ CORRECT" if correct else "✗ WRONG"
                print(f"  [{name} {pred_date} → {outcome_date}] {direction} {horizon_days}d | "
                      f"entry=${entry_price:.0f} → out=${out_price:.0f} | "
                      f"ret={actual_ret:+.1f}% | {result_str}")
                resolved += 1

    df.to_csv(METALS_CSV, index=False)
    print(f"  [METALS] {resolved} prediction(s) resolved.")


def resolve_portfolio():
    if not os.path.exists(PORT_CSV):
        print("[PORTFOLIO] No CSV found to resolve.")
        return

    df = pd.read_csv(PORT_CSV)
    pending = df[df["status"] == "PENDING"].copy()
    if pending.empty:
        print("[PORTFOLIO] No pending predictions to resolve.")
        return

    import yfinance as yf
    # Fetch current prices
    tickers = {"equities": "SPY", "gold": "GC=F",
               "bonds": "TLT", "crypto": "IBIT", "fx": "UUP"}
    current = {}
    for cls, tk in tickers.items():
        hist = yf.download(tk, period="5d", auto_adjust=True, progress=False)
        current[cls] = float(hist["Close"].squeeze().dropna().iloc[-1])

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
                    out_prices[cls] = float(hist["Close"].squeeze().dropna().iloc[0])
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

    for label, path in [("GOLD", GOLD_CSV), ("METALS", METALS_CSV), ("PORTFOLIO", PORT_CSV)]:
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
        elif label == "METALS" and n_resolved > 0:
            res = df[df["status"] == "RESOLVED"]
            acc = res["prediction_correct"].astype(float).mean() * 100
            print(f"  METALS: {n_total} rows | {n_pending} pending | "
                  f"{n_resolved} resolved | accuracy={acc:.0f}%")
            for name in METALS_TICKERS:
                mres = res[res["metal"] == name]
                if len(mres) > 0:
                    macc = mres["prediction_correct"].astype(float).mean() * 100
                    print(f"    {name:9s}: {len(mres)} resolved | accuracy={macc:.0f}%")
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

    # Combined three-metal accuracy (Gold + Silver + Platinum). GOLD_CSV and METALS_CSV are
    # separate logs with their own accuracy figures above (METALS there is Silver+Platinum
    # only, Gold is tracked on its own) -- this blends all three into one number. Note: Gold's
    # resolved sample mixes 21d and 63d horizons while Silver/Platinum are still 21d-only, so
    # this isn't a perfectly apples-to-apples blend, just a quick combined read.
    combined_parts = []
    if os.path.exists(GOLD_CSV):
        g = pd.read_csv(GOLD_CSV)
        combined_parts.append(g[g["status"] == "RESOLVED"]["prediction_correct"])
    if os.path.exists(METALS_CSV):
        m = pd.read_csv(METALS_CSV)
        combined_parts.append(m[m["status"] == "RESOLVED"]["prediction_correct"])
    if combined_parts:
        combined = pd.concat(combined_parts).astype(float)
        if len(combined):
            print(f"  COMBINED (Gold+Silver+Platinum): {len(combined)} resolved | "
                  f"accuracy={combined.mean()*100:.0f}%")


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
        gold_state   = fetch_gold_state()
        port_state   = fetch_portfolio_state()
        metals_state = fetch_metals_state()
        log_gold(gold_state)
        log_portfolio(port_state)
        log_metals(metals_state)

    if do_resolve:
        print("\n── RESOLVING EXPIRED PREDICTIONS ───────────────────────")
        resolve_gold()
        resolve_portfolio()
        resolve_metals()

    print_summary()
    print("\nDone.\n")


if __name__ == "__main__":
    main()
