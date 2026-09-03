#!/usr/bin/env python3
"""
============================================================
  MULTI-ASSET PORTFOLIO TILT DASHBOARD BUILDER
  Based on CPE Multi-Asset Atlas
  Dr. Arun Ramanathan
============================================================
Reads:
  - multiasset_prices.parquet   (historical prices)
  - cpe_results.parquet         (pairwise CPE signals)
  - joint_cpe_results.parquet   (joint CPE signals)
Outputs:
  - portfolio_dashboard.html    (self-contained dashboard)
============================================================
"""

import json, math, os, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

print("=" * 60)
print("  PORTFOLIO TILT DASHBOARD BUILDER")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ── ASSET CLASS DEFINITIONS ──────────────────────────────────
ASSET_CLASSES = {
    "Equities": {
        "tickers":    ["SPY", "QQQ"],
        "label":      "Equities",
        "color":      "#4A90D9",
        "icon":       "📈",
        "proxy":      "SPY",
        "desc":       "US large-cap equities (SPY/QQQ)"
    },
    "Gold": {
        "tickers":    ["GC=F", "GLD", "IAU"],
        "label":      "Gold",
        "color":      "#C9882A",
        "icon":       "🥇",
        "proxy":      "GC=F",
        "desc":       "Gold futures and ETFs"
    },
    "Bonds": {
        "tickers":    ["TLT", "AGG"],      # Long duration only — SHY excluded
        "tickers_bull": ["TLT","AGG","SHY"], # For bullish signal: include SHY
        "label":      "Bonds",
        "color":      "#4B9E6F",
        "icon":       "📊",
        "proxy":      "TLT",
        "desc":       "US Treasury bonds (TLT/AGG long duration)"
    },
    "Crypto": {
        "tickers":    ["IBIT", "FBTC", "BTC-USD"],
        "label":      "Crypto",
        "color":      "#F5A623",
        "icon":       "₿",
        "proxy":      "IBIT",
        "desc":       "Bitcoin ETFs and BTC-USD"
    },
    "FX": {
        "tickers":    ["UUP", "SGDUSD=X"],
        "label":      "FX / USD",
        "color":      "#9B59B6",
        "icon":       "💱",
        "proxy":      "UUP",
        "desc":       "USD strength (UUP ETF, SGDUSD=X)"
    }
}

# Forward horizons and their weights for overall tilt
HORIZONS     = [21, 63, 126, 252]
HOR_WEIGHTS  = {21: 0.20, 63: 0.30, 126: 0.30, 252: 0.20}
HOR_LABELS   = {21: "1 month", 63: "3 months", 126: "6 months", 252: "1 year"}

# CPE filter thresholds (same as atlas)
CPE_MIN  = 0.80
LIFT_MIN = 1.50
N_MIN    = 100

# Tilt translation thresholds
TILT_THRESHOLDS = [
    (0.85, "OVERWEIGHT",   +15),
    (0.70, "TILT UP",      +8),
    (0.50, "NEUTRAL",       0),
    (0.35, "TILT DOWN",    -8),
    (0.00, "UNDERWEIGHT", -15),
]

# ── LOAD DATA ────────────────────────────────────────────────
print("\nLoading local data...")
prices = pd.read_parquet("multiasset_prices.parquet")
cpe    = pd.read_parquet("cpe_results.parquet")
jcpe   = pd.read_parquet("joint_cpe_results.parquet")

print(f"  Prices: {prices.shape[0]} rows × {prices.shape[1]} tickers")
print(f"  CPE pairwise: {cpe.shape[0]:,} rows")
print(f"  CPE joint: {jcpe.shape[0]:,} rows")

# ── FETCH FRESH PRICES ───────────────────────────────────────
print("\nFetching latest prices from Yahoo Finance...")
try:
    import yfinance as yf
    # Fetch the FULL CPE predictor universe (every X ticker the firing-predictor
    # scan below checks), not just the 5 asset-class proxies -- confirmed
    # 2026-09-03 that only ~15-22 of 158 X tickers were ever in this list across
    # all three dashboard builders, so ~90% of "which predictors are firing"
    # was silently running on multiasset_prices.parquet's frozen 2026-06-16
    # snapshot every single day, not genuinely live data. See ibkr_paper_ledger
    # commit history / project memory for the full investigation.
    fetch_list = list(set(
        ["SPY","QQQ","GC=F","GLD","IAU","TLT","AGG",
         "IBIT","FBTC","BTC-USD","UUP","SGDUSD=X",
         "SLV","^VIX","^GVZ","DX-Y.NYB"]
        # Note: XAUUSD=X removed — delisted on Yahoo Finance
        + cpe["X"].unique().tolist()
    ))
    raw = yf.download(fetch_list, period="400d",
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    # ── STRIP ANY PARTIAL/LIVE "TODAY" SESSION ROW, FOR EVERY TICKER ──
    # yfinance's most recent daily bar can be an intraday snapshot that keeps
    # changing as the trading day progresses, which was letting tilt scores
    # drift between multiple runs on the same calendar day. Drop any row dated
    # today-or-later before it reaches `prices`, so every run on a given day
    # sees identical, fully-settled data (see build_gold_dashboard.py for the
    # symptom this was first diagnosed from).
    import datetime as _dt_check
    if len(raw.index) and raw.index.max().date() >= _dt_check.date.today():
        raw = raw[raw.index.date < _dt_check.date.today()]

    for col in raw.columns:
        if col in prices.columns:
            new = raw[[col]].loc[raw.index > prices.index.max()]
            if not new.empty:
                prices = pd.concat([prices, new])
            fresh_col = raw[[col]].reindex(prices.index)
            prices[col] = prices[col].fillna(fresh_col[col])
        else:
            prices = prices.join(raw[[col]], how="left", rsuffix="_new")
            if col + "_new" in prices.columns:
                prices[col] = prices.get(col, float("nan"))
                prices[col] = prices[col].fillna(prices[col + "_new"])
                prices = prices.drop(columns=[col + "_new"], errors="ignore")
            else:
                prices[col] = raw[col].reindex(prices.index)

    prices = prices.sort_index().loc[~prices.index.duplicated(keep="last")]
    # ── CRITICAL: Force-override last 30 days from fresh Yahoo data ──
    # Prevents stale parquet values from being used for current price calculations
    for _key in ["SPY","QQQ","GC=F","GLD","IAU","TLT","AGG","SHY",
                 "IBIT","FBTC","BTC-USD","UUP","SGDUSD=X","^GVZ","SLV"]:
        if _key in raw.columns:
            _fresh = raw[_key].dropna()
            if len(_fresh) > 5:
                prices[_key] = prices[_key].fillna(_fresh.reindex(prices.index))
                _last30 = _fresh.index[-30:]
                prices.loc[prices.index.isin(_last30), _key] = _fresh.reindex(
                    prices.index[prices.index.isin(_last30)]
                )
    prices = prices.sort_index().loc[~prices.index.duplicated(keep="last")]
    print(f"  Latest date: {prices.index.max().date()}")
    # Verify key tickers are current
    for _chk in ["SPY","GC=F","TLT","IBIT"]:
        if _chk in prices.columns:
            _lv = prices[_chk].dropna()
            if len(_lv):
                print(f"  {_chk} last valid: {_lv.index[-1].date()} = ${_lv.iloc[-1]:.2f}")

    from datetime import date as _date
    _today = _date.today()
    _latest = prices.index.max().date()
    _age = (_today - _latest).days
    if _age > 4:
        print(f"  WARNING: data is {_age} days old — Yahoo may be lagging.")
    for _t in ["SPY", "GC=F", "TLT", "IBIT"]:
        if _t in prices.columns:
            _lv = prices[_t].dropna()
            if len(_lv):
                _tage = (_latest - _lv.index[-1].date()).days
                if _tage > 1:
                    print(f"  WARNING: {_t} last valid {_lv.index[-1].date()} ({_tage}d stale)")

    # ── Persist a live, ever-growing price history ────────────────────
    # NEVER overwrites multiasset_prices.parquet itself -- that file is a
    # frozen 2026-06-16 snapshot referenced by already-published papers, and
    # changing its update behavior would be a real reproducibility risk, not
    # a routine fix. This is a separate file that all three dashboard
    # builders (gold/portfolio/metals) merge their freshly-fetched prices
    # into, so a real, non-frozen daily price record exists on disk going
    # forward for anything that later needs to reconstruct a past date's
    # regime state (confirmed 2026-09-03 that no such record existed before
    # this, which is why a hold-to-horizon paper-ledger track couldn't be
    # backfilled).
    LIVE_HISTORY_PARQUET = "multiasset_prices_live_history.parquet"
    try:
        if os.path.exists(LIVE_HISTORY_PARQUET):
            existing_hist = pd.read_parquet(LIVE_HISTORY_PARQUET)
            merged_hist = existing_hist.copy()
            for col in prices.columns:
                merged_hist[col] = prices[col].combine_first(existing_hist.get(col))
        else:
            merged_hist = prices.copy()
        merged_hist = merged_hist.sort_index().loc[~merged_hist.index.duplicated(keep="last")]
        merged_hist.to_parquet(LIVE_HISTORY_PARQUET)
        print(f"  Live price history saved -> {LIVE_HISTORY_PARQUET} "
              f"({merged_hist.shape[0]} rows x {merged_hist.shape[1]} tickers, "
              f"through {merged_hist.index.max().date()})")
    except Exception as _hist_err:
        print(f"  WARNING: could not persist live price history: {_hist_err}")

except Exception as e:
    print(f"  yfinance error: {e}")
    print("  FATAL: Cannot build dashboard without fresh prices. Exiting.")
    import sys; sys.exit(1)

# ── FX RATE ──────────────────────────────────────────────────
usd_per_sgd = 1.0
if "SGDUSD=X" in prices.columns:
    fx = prices["SGDUSD=X"].dropna()
    if len(fx):
        usd_per_sgd = float(fx.iloc[-1])

# ── STEP 1: NEUTRAL WEIGHTS FROM SHARPE RATIOS ───────────────
print("\nComputing Sharpe-based neutral weights...")

RATE_TICKERS = {"^VIX","^VXN","^OVX","^EVZ","^VVIX","^SKEW",
                "^TNX","^TYX","^FVX","^IRX"}

def compute_sharpe(ticker, lookback_years=10):
    """Annualised Sharpe (rf=0) from log returns."""
    if ticker not in prices.columns:
        return None
    series = prices[ticker].dropna()
    # Use last N years
    cutoff = series.index.max() - pd.DateOffset(years=lookback_years)
    series = series[series.index >= cutoff]
    if len(series) < 252:
        return None
    if ticker in RATE_TICKERS:
        rets = series.diff().dropna()
    else:
        rets = np.log(series / series.shift(1)).dropna()
    ann_ret  = float(rets.mean() * 252)
    ann_vol  = float(rets.std() * np.sqrt(252))
    if ann_vol < 1e-8:
        return None
    return ann_ret / ann_vol

# Compute Sharpe for main proxy of each asset class
# FX gets fixed weight (near-zero Sharpe naturally)
sharpe_proxies = {
    "Equities": "SPY",
    "Gold":     "GC=F",
    "Bonds":    "TLT",
    "Crypto":   "BTC-USD",  # longest BTC history
}

sharpes = {}
for ac, proxy in sharpe_proxies.items():
    s = compute_sharpe(proxy)
    sharpes[ac] = s
    print(f"  {ac:10s} ({proxy}): Sharpe = {s:.3f}" if s else f"  {ac:10s} ({proxy}): insufficient data")

# Normalise positive Sharpes to weights (floor at 0)
FX_FIXED_WEIGHT = 10.0  # percent
remaining = 100.0 - FX_FIXED_WEIGHT

pos_sharpes = {k: max(v, 0) for k, v in sharpes.items() if v is not None}
total_sharpe = sum(pos_sharpes.values())

MIN_WEIGHT = 5.0  # minimum 5% per asset class in neutral portfolio

if total_sharpe > 0:
    raw_weights = {k: v / total_sharpe * remaining for k, v in pos_sharpes.items()}
    # Apply floor — redistribute from highest weights
    floored = {k: max(v, MIN_WEIGHT) for k, v in raw_weights.items()}
    # Also handle missing asset classes (Sharpe was None)
    for k in sharpe_proxies:
        if k not in floored:
            floored[k] = MIN_WEIGHT
    # Renormalise to remaining (90%)
    total_floored = sum(floored.values())
    neutral_weights = {k: round(v / total_floored * remaining, 1) for k, v in floored.items()}
else:
    # Fallback equal weight
    neutral_weights = {k: round(remaining / len(sharpe_proxies), 1) for k in sharpe_proxies}

neutral_weights["FX"] = FX_FIXED_WEIGHT

# Ensure sums to 100
_diff = 100.0 - sum(neutral_weights.values())
neutral_weights["Equities"] = round(neutral_weights["Equities"] + _diff, 1)

print(f"\n  Neutral weights:")
for ac, w in neutral_weights.items():
    print(f"    {ac:10s}: {w:.1f}%")

# ── STEP 2: COMPUTE CURRENT REGIME ───────────────────────────
print("\nComputing current regime (which predictors are in their tails)...")

def current_return(ticker, tau):
    """Log return of ticker over last tau days."""
    if ticker not in prices.columns:
        return None
    series = prices[ticker].dropna()
    if len(series) < tau + 1:
        return None
    if ticker in RATE_TICKERS:
        return float(series.iloc[-1] - series.iloc[-1-tau])
    return float(np.log(series.iloc[-1] / series.iloc[-1-tau]) * 100)

def historical_quantile(ticker, tau, q):
    """Historical q-th quantile of tau-day returns."""
    if ticker not in prices.columns:
        return None
    series = prices[ticker].dropna()
    if ticker in RATE_TICKERS:
        rets = series.diff(tau).dropna().values
    else:
        rets = (np.log(series / series.shift(tau)).dropna().values * 100)
    if len(rets) < 50:
        return None
    return float(np.quantile(rets, q))

# Get all unique (X, tau_past, q_X) combinations in the CPE table
# and check which ones are currently in their tail
print("  Checking predictor tail status...")

# Get unique predictors from CPE table
pred_combos = (cpe[["X","tau_past","q_X","direction"]]
               .drop_duplicates()
               .values.tolist())

firing_predictors = []   # list of (X, tau_past, q_X, direction, curr_ret, threshold)

checked = set()
for X, tau, q, direction in pred_combos:
    key = (X, int(tau), float(q), direction)
    if key in checked:
        continue
    checked.add(key)

    curr = current_return(X, int(tau))
    if curr is None:
        continue

    if direction == "bullish":
        # X must be in its UPPER tail (above q-th quantile)
        thresh = historical_quantile(X, int(tau), float(q))
        if thresh is not None and curr >= thresh:
            firing_predictors.append({
                "X": X, "tau": int(tau), "q": float(q),
                "direction": direction,
                "curr_ret": round(curr, 2),
                "threshold": round(thresh, 2)
            })
    else:  # bearish
        # X must be in its LOWER tail (below (1-q)-th quantile)
        thresh = historical_quantile(X, int(tau), 1.0 - float(q))
        if thresh is not None and curr <= thresh:
            firing_predictors.append({
                "X": X, "tau": int(tau), "q": float(q),
                "direction": direction,
                "curr_ret": round(curr, 2),
                "threshold": round(thresh, 2)
            })

print(f"  Firing predictors: {len(firing_predictors)}")

# Build a lookup set for fast matching
firing_set = set(
    (p["X"], p["tau"], p["q"], p["direction"])
    for p in firing_predictors
)

# ── STEP 3: CPE SIGNAL LOOKUP PER ASSET CLASS × HORIZON ─────
print("\nLooking up CPE signals per asset class...")

def get_firing_cpe_signals(target_tickers, horizon, direction_filter=None):
    """
    Return all CPE rows where:
      - Y in target_tickers
      - tau_future == horizon
      - (X, tau_past, q_X, direction) currently firing
      - CPE >= CPE_MIN, lift >= LIFT_MIN, n_condition >= N_MIN
    """
    mask = (
        cpe["Y"].isin(target_tickers) &
        (cpe["tau_future"] == horizon) &
        (cpe["CPE"] >= CPE_MIN) &
        (cpe["lift"] >= LIFT_MIN) &
        (cpe["n_condition"] >= N_MIN)
    )
    if direction_filter:
        mask &= (cpe["direction"] == direction_filter)

    subset = cpe[mask].copy()

    # Filter to currently firing predictors — vectorised for speed
    keys = list(zip(
        subset["X"].values,
        subset["tau_past"].astype(int).values,
        subset["q_X"].astype(float).values,
        subset["direction"].values
    ))
    mask = [k in firing_set for k in keys]
    return subset[mask]

def compute_regime_score(firing_signals):
    """
    Regime score = weighted mean of CPE for firing signals.
    Deduplicates by (X, tau_past, q_X) to avoid inflating counts
    when multiple Y tickers (GC=F, GLD, IAU) share the same predictor.
    Returns score in [0,1] and list of top signals (deduplicated).
    """
    if len(firing_signals) == 0:
        return None, []

    fs = firing_signals.copy()

    # Deduplicate: keep best CPE per unique (X, tau_past, q_X, direction)
    # This prevents GC=F, GLD, IAU all firing the same predictor counting 3x
    fs_dedup = (fs.sort_values("CPE", ascending=False)
                  .drop_duplicates(subset=["X","tau_past","q_X","direction"])
                  .reset_index(drop=True))

    # Weight by n_condition
    weights = fs_dedup["n_condition"].values.astype(float)
    scores  = fs_dedup["CPE"].values
    regime_score = float(np.average(scores, weights=weights))

    # Top signals for display — best CPE per unique predictor
    top = (fs_dedup
           .sort_values(["CPE","lift"], ascending=False)
           .head(5)
           [["Y","X","tau_past","tau_future","q_X","q_Y","CPE","lift","n_condition","direction"]]
           .to_dict("records"))
    return regime_score, top

def score_to_tilt(score):
    """Convert regime score to tilt label and weight delta."""
    if score is None:
        return "NO SIGNAL", 0
    for threshold, label, delta in TILT_THRESHOLDS:
        if score >= threshold:
            return label, delta
    return "UNDERWEIGHT", -15

# Compute for each asset class and horizon
print("  Computing tilt signals...")
results = {}
for ac, info in ASSET_CLASSES.items():
    results[ac] = {"info": info, "horizons": {}, "sharpe": sharpes.get(ac)}
    for hor in HORIZONS:
        # For bonds: bullish uses all durations, bearish uses long-duration only
        bull_tickers = info.get("tickers_bull", info["tickers"])
        bull_sigs = get_firing_cpe_signals(bull_tickers, hor, "bullish")
        bear_sigs = get_firing_cpe_signals(info["tickers"], hor, "bearish")

        bull_score, bull_top = compute_regime_score(bull_sigs)
        bear_score, bear_top = compute_regime_score(bear_sigs)

        # Net score: bullish pushes up, bearish pushes down
        if bull_score is not None and bear_score is not None:
            net_score = bull_score * 0.6 + (1 - bear_score) * 0.4
        elif bull_score is not None:
            net_score = bull_score
        elif bear_score is not None:
            net_score = 1 - bear_score
        else:
            net_score = None

        tilt_label, tilt_delta = score_to_tilt(net_score)

        results[ac]["horizons"][hor] = {
            "bull_score":  round(bull_score, 3) if bull_score else None,
            "bear_score":  round(bear_score, 3) if bear_score else None,
            "net_score":   round(net_score, 3) if net_score else None,
            "tilt_label":  tilt_label,
            "tilt_delta":  tilt_delta,
            "bull_n":      len(bull_sigs),
            "bear_n":      len(bear_sigs),
            "bull_top":    bull_top,
            "bear_top":    bear_top,
        }
        print(f"    {ac:10s} {hor:3d}d: {tilt_label:12s} (bull={len(bull_sigs)} sigs, bear={len(bear_sigs)} sigs)")

# ── STEP 4: COMPUTE OVERALL TILT AND SUGGESTED WEIGHTS ───────
print("\nComputing suggested portfolio weights...")

suggested_weights = {}
tilt_summaries   = {}

for ac in ASSET_CLASSES:
    # Weighted average delta across horizons
    total_w  = 0
    total_d  = 0
    labels   = []
    for hor in HORIZONS:
        h = results[ac]["horizons"][hor]
        w = HOR_WEIGHTS[hor]
        d = h["tilt_delta"]
        total_d += d * w
        total_w += w
        labels.append(h["tilt_label"])

    overall_delta = round(total_d / total_w, 1) if total_w > 0 else 0

    # Overall label from overall delta
    if overall_delta >= 10:
        overall_label = "OVERWEIGHT"
    elif overall_delta >= 4:
        overall_label = "TILT UP"
    elif overall_delta >= -4:
        overall_label = "NEUTRAL"
    elif overall_delta >= -10:
        overall_label = "TILT DOWN"
    else:
        overall_label = "UNDERWEIGHT"

    base      = neutral_weights.get(ac, 20.0)
    suggested = max(0, min(50, base + overall_delta))

    suggested_weights[ac] = round(suggested, 1)
    tilt_summaries[ac] = {
        "overall_label": overall_label,
        "overall_delta": overall_delta,
        "neutral_w":     base,
        "suggested_w":   round(suggested, 1),
        "horizon_labels": {hor: results[ac]["horizons"][hor]["tilt_label"] for hor in HORIZONS}
    }
    print(f"  {ac:10s}: neutral={base:.1f}% → suggested={suggested:.1f}% ({overall_label})")

# Renormalise to 100%
total_suggested = sum(suggested_weights.values())
if total_suggested > 0:
    suggested_weights = {k: round(v / total_suggested * 100, 1) for k, v in suggested_weights.items()}
    # Update summaries
    for ac in ASSET_CLASSES:
        tilt_summaries[ac]["suggested_w"] = suggested_weights[ac]

print(f"\n  Suggested weights (normalised): {suggested_weights}")

# ── STEP 5: CURRENT PRICE SNAPSHOT ───────────────────────────
snapshot = {}
for ac, info in ASSET_CLASSES.items():
    proxy = info["proxy"]
    if proxy in prices.columns:
        s = prices[proxy].dropna()
        if len(s) >= 2:
            last_price = float(s.iloc[-1])
            ret_1d     = float((s.iloc[-1]/s.iloc[-2]-1)*100) if len(s)>=2 else 0
            ret_5d     = float((s.iloc[-1]/s.iloc[-6]-1)*100) if len(s)>=6 else 0
            ret_21d    = float((s.iloc[-1]/s.iloc[-22]-1)*100) if len(s)>=22 else 0
            ret_63d    = float((s.iloc[-1]/s.iloc[-64]-1)*100) if len(s)>=64 else 0
            snapshot[ac] = {
                "proxy":      proxy,
                "price":      round(last_price, 2),
                "ret_1d":     round(ret_1d, 2),
                "ret_5d":     round(ret_5d, 2),
                "ret_21d":    round(ret_21d, 2),
                "ret_63d":    round(ret_63d, 2),
            }

# ── STEP 6: BUILD DATA BUNDLE ─────────────────────────────────
gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")

bundle = {
    "gen":               gen_time,
    "data_date":         str(prices.index.max().date()),
    "neutral_weights":   neutral_weights,
    "suggested_weights": suggested_weights,
    "tilt_summaries":    tilt_summaries,
    "results":           results,
    "snapshot":          snapshot,
    "sharpes":           {k: round(v, 3) if v else None for k, v in sharpes.items()},
    "firing_pred_count": len(firing_predictors),
    "asset_classes":     {k: v for k, v in ASSET_CLASSES.items()},
    "horizons":          HORIZONS,
    "hor_labels":        HOR_LABELS,
}

bundle_json = json.dumps(bundle, default=str)
print(f"\n  Data bundle: {len(bundle_json)/1024:.1f} KB")

# ── STEP 7: BUILD HTML DASHBOARD ─────────────────────────────
print("\nBuilding HTML dashboard...")

TILT_COLORS = {
    "OVERWEIGHT":   "#1a6b3a",
    "TILT UP":      "#4DB87A",
    "NEUTRAL":      "#8B7355",
    "TILT DOWN":    "#E8A020",
    "UNDERWEIGHT":  "#E05555",
    "NO SIGNAL":    "#aaaaaa",
}

TILT_BG = {
    "OVERWEIGHT":   "#f0faf4",
    "TILT UP":      "#f5fbf7",
    "NEUTRAL":      "#f9f7f4",
    "TILT DOWN":    "#fdf8f0",
    "UNDERWEIGHT":  "#fdf4f4",
    "NO SIGNAL":    "#f5f5f5",
}

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio Tilt Dashboard — CPE Framework</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=DM+Serif+Display:ital@0;1&display=swap');

  :root{{
    --bg:#0C0E0D; --card:#141614; --card2:#1B1E1B; --border:#2C302C; --border2:#3A3F3A;
    --text:#DDE8DD; --muted:#7A8F7A; --faint:#1B1E1B;
    --gold:#C9A84C; --green:#4DB87A; --red:#E05555; --warn:#E8A020;
    --mono:'IBM Plex Mono',monospace; --serif:'DM Serif Display',serif;
    --blue:#5B9BD5; --orange:#F5A623; --purple:#B07CC6;
  }}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);font-family:'Inter',sans-serif;color:var(--text);font-size:14px;}}
  a{{color:var(--gold);text-decoration:none;}}

  /* ── LAYOUT ── */
  .page{{max-width:1400px;margin:0 auto;padding:32px 24px 80px;}}

  /* ── HEADER ── */
  .header{{background:linear-gradient(160deg,#141614,#1B1E1B);border:1px solid var(--border2);
           border-radius:16px;padding:32px 40px;margin-bottom:28px;
           display:grid;grid-template-columns:1fr auto;align-items:center;gap:24px;}}
  .h-title{{font-family:var(--serif);font-size:28px;color:var(--text);margin-bottom:6px;}}
  .h-title em{{font-style:italic;color:var(--gold);}}
  .h-sub{{font-family:var(--mono);font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);margin-bottom:4px;}}
  .h-meta{{font-size:12px;color:var(--muted);}}
  .h-badge{{background:rgba(201,168,76,0.12);border:1px solid rgba(201,168,76,0.3);
            border-radius:12px;padding:12px 20px;text-align:center;}}
  .h-badge-num{{font-family:var(--mono);font-size:28px;font-weight:600;color:var(--gold);}}
  .h-badge-label{{font-size:10px;color:var(--muted);display:block;margin-top:2px;letter-spacing:.1em;text-transform:uppercase;}}

  /* ── SECTION ── */
  .section{{margin-bottom:28px;}}
  .section-title{{font-family:var(--mono);font-size:10px;letter-spacing:.18em;
                  text-transform:uppercase;color:var(--muted);margin-bottom:14px;}}

  /* ── CARDS ── */
  .card{{background:var(--card);border-radius:14px;border:1px solid var(--border);padding:20px 24px;}}
  .grid-5{{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;}}
  .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;}}
  .grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;}}

  /* ── PRICE CARDS ── */
  .price-card{{background:var(--card);border-radius:14px;border:1px solid var(--border);
               padding:16px 18px;}}
  .pc-ac{{font-size:10px;font-family:var(--mono);letter-spacing:.12em;
          text-transform:uppercase;color:var(--muted);margin-bottom:4px;}}
  .pc-price{{font-family:var(--mono);font-size:20px;font-weight:600;color:var(--text);}}
  .pc-ticker{{font-size:11px;color:var(--muted);margin-top:2px;}}
  .pc-rets{{display:flex;gap:10px;margin-top:10px;flex-wrap:wrap;}}
  .pc-ret{{font-family:var(--mono);font-size:10px;}}
  .up{{color:var(--green);}} .dn{{color:var(--red);}} .nt{{color:var(--muted);}}

  /* ── TILT TABLE ── */
  .tilt-table{{width:100%;border-collapse:collapse;}}
  .tilt-table th{{font-family:var(--mono);font-size:9px;letter-spacing:.12em;
                  text-transform:uppercase;color:var(--muted);padding:10px 14px;
                  border-bottom:2px solid var(--border);text-align:left;background:var(--faint);}}
  .tilt-table td{{padding:12px 14px;border-bottom:1px solid var(--border);vertical-align:middle;}}
  .tilt-table tr:last-child td{{border-bottom:none;}}
  .tilt-table tr:hover td{{background:var(--card2);}}

  .ac-cell{{display:flex;align-items:center;gap:10px;}}
  .ac-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0;}}
  .ac-name{{font-weight:600;font-size:13px;color:var(--text);}}
  .ac-desc{{font-size:10px;color:var(--muted);margin-top:2px;}}

  .tilt-pill{{display:inline-flex;align-items:center;padding:4px 10px;
              border-radius:100px;font-family:var(--mono);font-size:9px;
              font-weight:600;letter-spacing:.06em;white-space:nowrap;}}

  .weight-bar-bg{{background:var(--border);border-radius:4px;height:6px;width:80px;display:inline-block;vertical-align:middle;}}
  .weight-bar-fill{{height:100%;border-radius:4px;transition:width .3s;}}

  /* ── SIGNAL CARDS ── */
  .signal-section{{margin-top:28px;}}
  .signal-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px;}}
  .signal-card{{background:var(--card);border-radius:12px;border:1px solid var(--border);
                padding:16px 18px;}}
  .sig-header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}}
  .sig-ac{{font-size:12px;font-weight:600;color:var(--text);}}
  .sig-hor{{font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:.1em;}}
  .sig-row{{font-family:var(--mono);font-size:10px;color:var(--text);padding:5px 0;
             border-bottom:1px solid var(--border);display:flex;justify-content:space-between;gap:8px;}}
  .sig-row:last-child{{border-bottom:none;}}
  .sig-pred{{color:var(--muted);font-size:9px;}}
  .sig-cpe{{color:var(--green);font-weight:600;}}
  .sig-lift{{color:var(--gold);}}

  /* ── WEIGHT COMPARISON ── */
  .wt-row{{display:flex;align-items:center;gap:14px;padding:10px 0;border-bottom:1px solid var(--border);}}
  .wt-row:last-child{{border-bottom:none;}}
  .wt-ac{{width:90px;font-size:12px;font-weight:600;color:var(--text);}}
  .wt-bars{{flex:1;display:flex;flex-direction:column;gap:5px;}}
  .wt-bar-row{{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:10px;}}
  .wt-bar-label{{width:60px;color:var(--muted);}}
  .wt-bar-track{{flex:1;background:var(--border);border-radius:3px;height:8px;}}
  .wt-bar-fill{{height:100%;border-radius:3px;}}
  .wt-pct{{width:40px;text-align:right;}}

  /* ── DISCLAIMER ── */
  .disclaimer{{background:var(--card);border:1px solid var(--border);border-radius:10px;
               padding:16px 20px;font-size:11px;color:var(--muted);line-height:1.7;margin-top:28px;}}

  /* ── MISC ── */
  ::-webkit-scrollbar{{width:4px;height:4px;}}
  ::-webkit-scrollbar-track{{background:var(--bg);}}
  ::-webkit-scrollbar-thumb{{background:var(--border2);border-radius:2px;}}

  /* ── RESPONSIVE ── */
  @media(max-width:900px){{
    .grid-5{{grid-template-columns:repeat(2,1fr);}}
    .grid-3{{grid-template-columns:1fr 1fr;}}
    .header{{grid-template-columns:1fr;}}
  }}
  @media(max-width:600px){{
    .grid-5,.grid-2,.grid-3{{grid-template-columns:1fr;}}
    .page{{padding:16px 12px 60px;}}
  }}
</style>
</head>
<body>
<div class="page">

<!-- HEADER -->
<div class="header">
  <div>
    <div class="h-sub">CPE Multi-Asset Framework · Portfolio Tilt Dashboard</div>
    <div class="h-title"><em>Regime-Conditional</em> Portfolio Tilt</div>
    <div class="h-meta">
      Updated: <span id="hgen"></span> &nbsp;·&nbsp;
      Data: <span id="hdate"></span> &nbsp;·&nbsp;
      <span id="hpred"></span> predictors firing
    </div>
  </div>
  <div class="h-badge">
    <div class="h-badge-num" id="hfire">—</div>
    <span class="h-badge-label">CPE signals active</span>
  </div>
</div>

<!-- REGIME SUMMARY -->
<div class="section">
  <div class="card" id="regime-summary" style="background:linear-gradient(135deg,#1B1E1B 0%,#222522 100%);border:1px solid var(--border2);padding:24px 32px;">
    <div style="font-family:var(--mono);font-size:9px;letter-spacing:.18em;text-transform:uppercase;color:#9AAE9A;margin-bottom:10px;">Today's Regime — Plain English Summary</div>
    <div id="summary-text" style="font-size:16px;line-height:1.7;color:#eee;"></div>
    <div id="summary-signals" style="margin-top:12px;display:flex;flex-wrap:wrap;gap:8px;"></div>
  </div>
</div>

<!-- PRICE SNAPSHOT -->
<div class="section">
  <div class="section-title">Current Price Snapshot</div>
  <div class="grid-5" id="price-cards"></div>
</div>

<!-- TILT TABLE -->
<div class="section">
  <div class="section-title">CPE Regime Tilt — by Asset Class and Horizon</div>
  <div class="card" style="padding:0;overflow:hidden;">
    <table class="tilt-table" id="tilt-table">
      <thead>
        <tr>
          <th style="width:200px">Asset Class</th>
          <th>1 Month</th>
          <th>3 Months</th>
          <th>6 Months</th>
          <th>1 Year</th>
          <th>Overall Tilt</th>
          <th>Neutral Wt</th>
          <th>Suggested Wt</th>
          <th>Change</th>
        </tr>
      </thead>
      <tbody id="tilt-tbody"></tbody>
    </table>
  </div>
</div>

<!-- WEIGHT COMPARISON -->
<div class="section">
  <div class="section-title">Suggested vs Neutral Portfolio Weights</div>
  <div class="grid-2">
    <div class="card">
      <div id="chart-weights" style="height:320px;"></div>
    </div>
    <div class="card" style="padding:24px;">
      <div style="font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:16px;">Weight Comparison</div>
      <div id="weight-bars"></div>
    </div>
  </div>
</div>

<!-- FIRING CPE SIGNALS -->
<div class="section signal-section">
  <div class="section-title">Firing CPE Signals — Evidence Behind Each Tilt</div>
  <div class="signal-grid" id="signal-grid"></div>
</div>

<!-- DISCLAIMER -->
<div class="disclaimer">
  <strong>Research Disclaimer:</strong> All CPE values are within-sample empirical frequencies computed from historical data (1993–2026).
  The tilt labels, suggested weights, and regime scores are descriptive summaries of historical tail co-movement patterns — not forecasts or investment advice.
  Neutral weights are derived from long-run historical Sharpe ratios (rf=0, 10-year lookback) and are not optimal in any mean-variance sense.
  Past statistical structure does not guarantee future behaviour. This dashboard is for personal research only.
  Always consult a licensed financial adviser before making investment decisions.
  &nbsp;·&nbsp; <strong>Dr. Arun Ramanathan</strong>
  &nbsp;·&nbsp; zenodo.org/records/20606184
</div>

</div><!-- /page -->

<script>
const D = {bundle_json};

// ── TILT COLORS ───────────────────────────────────────────────
const TC = {{
  "OVERWEIGHT":  {{color:"#4DB87A", bg:"rgba(77,184,122,0.12)",  border:"rgba(77,184,122,0.35)"}},
  "TILT UP":     {{color:"#8FD4A0", bg:"rgba(143,212,160,0.10)", border:"rgba(143,212,160,0.30)"}},
  "NEUTRAL":     {{color:"#B0A387", bg:"rgba(176,163,135,0.10)", border:"rgba(176,163,135,0.28)"}},
  "TILT DOWN":   {{color:"#E8A020", bg:"rgba(232,160,32,0.10)",  border:"rgba(232,160,32,0.30)"}},
  "UNDERWEIGHT": {{color:"#E05555", bg:"rgba(224,85,85,0.12)",   border:"rgba(224,85,85,0.35)"}},
  "NO SIGNAL":   {{color:"#7A8F7A", bg:"rgba(122,143,122,0.10)", border:"rgba(122,143,122,0.28)"}},
}};

const AC_COLORS = {{
  "Equities": "#5B9BD5",
  "Gold":     "#C9A84C",
  "Bonds":    "#4DB87A",
  "Crypto":   "#F5A623",
  "FX":       "#B07CC6",
}};

function tiltPill(label){{
  const t = TC[label] || TC["NO SIGNAL"];
  return `<span class="tilt-pill" style="color:${{t.color}};background:${{t.bg}};border:1px solid ${{t.border}}">${{label}}</span>`;
}}

function fmt(v, decimals=2){{
  if(v===null||v===undefined) return '—';
  const s = v>=0?'+':'';
  return s + v.toFixed(decimals) + '%';
}}

function retClass(v){{
  if(v===null||v===undefined) return 'nt';
  return v>0?'up':v<0?'dn':'nt';
}}

// ── HEADER ───────────────────────────────────────────────────
document.getElementById('hgen').textContent  = D.gen;
document.getElementById('hdate').textContent = D.data_date;
document.getElementById('hpred').textContent = D.firing_pred_count;

// Count UNIQUE firing predictor-horizon combinations (deduplicated)
const seen = new Set();
let totalSigs = 0;
for(const ac of Object.keys(D.results)){{
  for(const hor of D.horizons){{
    const h = D.results[ac].horizons[hor];
    // Count deduplicated bull signals
    (h.bull_top||[]).forEach(s => {{
      const k = s.X + '_' + s.tau_past + '_' + s.q_X + '_bull';
      if(!seen.has(k)){{ seen.add(k); totalSigs++; }}
    }});
    // Count deduplicated bear signals
    (h.bear_top||[]).forEach(s => {{
      const k = s.X + '_' + s.tau_past + '_' + s.q_X + '_bear';
      if(!seen.has(k)){{ seen.add(k); totalSigs++; }}
    }});
  }}
}}
document.getElementById('hfire').textContent = totalSigs;

// ── REGIME SUMMARY ───────────────────────────────────────────
(function(){{
  const overweights  = [];
  const tiltups      = [];
  const tiltdowns    = [];
  const underweights = [];
  const ACS_order = ["Equities","Gold","Bonds","Crypto","FX"];

  for(const ac of ACS_order){{
    const ts = D.tilt_summaries[ac];
    if(!ts) continue;
    const info = D.asset_classes[ac];
    const label = ts.overall_label;
    const delta = ts.overall_delta;
    const sw    = ts.suggested_w;
    const nw    = ts.neutral_w;
    if(label==="OVERWEIGHT")  overweights.push({{ac, info, sw, nw}});
    else if(label==="TILT UP")     tiltups.push({{ac, info, sw, nw}});
    else if(label==="TILT DOWN")   tiltdowns.push({{ac, info, sw, nw}});
    else if(label==="UNDERWEIGHT") underweights.push({{ac, info, sw, nw}});
  }}

  // Build plain-English paragraph
  const parts = [];

  if(overweights.length || tiltups.length){{
    const favoured = [...overweights,...tiltups].map(x=>x.info.icon+' '+x.ac);
    parts.push('The current cross-asset regime <strong style="color:#4DB87A">favours '+favoured.join(' and ')+'</strong>'+
               ' — CPE signals show historically elevated conditional probabilities of large moves at the 1–3 month horizon.');
  }}

  if(underweights.length || tiltdowns.length){{
    const avoid = [...underweights,...tiltdowns].map(x=>x.info.icon+' '+x.ac);
    parts.push('The regime is <strong style="color:#E8A020">cautious on '+avoid.join(' and ')+'</strong>'+
               ' — bearish CPE signals dominate at the 3-month horizon, suggesting historical tail conditions that precede weakness.');
  }}

  if(!overweights.length && !tiltups.length && !underweights.length && !tiltdowns.length){{
    parts.push('No strong directional regime signals at 1–3 month horizons. CPE structure is broadly neutral — hold near-benchmark weights across all asset classes.');
  }}

  // Firing predictor highlights
  const highlights = [];
  if(D.firing_pred_count > 800)
    highlights.push({{text: D.firing_pred_count+' predictors in tail regimes', color:'#C9882A'}});

  // Check specific known signals
  for(const ac of ACS_order){{
    const h63 = D.results[ac]?.horizons[63];
    if(h63 && h63.tilt_label !== 'NO SIGNAL' && h63.tilt_label !== 'NEUTRAL'){{
      highlights.push({{text: D.asset_classes[ac].icon+' '+ac+': '+h63.tilt_label+' at 3mo', color: TC[h63.tilt_label]?.color||'#888'}});
    }}
  }}

  document.getElementById('summary-text').innerHTML = parts.join(' ');

  const sigEl = document.getElementById('summary-signals');
  highlights.forEach(h => {{
    const span = document.createElement('span');
    span.style.cssText = `background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);
      border-radius:100px;padding:4px 12px;font-family:var(--mono);font-size:10px;color:${{h.color}}`;
    span.textContent = h.text;
    sigEl.appendChild(span);
  }});
}})();

// ── PRICE CARDS ──────────────────────────────────────────────
const pcEl = document.getElementById('price-cards');
for(const [ac, snap] of Object.entries(D.snapshot)){{
  const info = D.asset_classes[ac];
  const el = document.createElement('div');
  el.className = 'price-card';
  el.innerHTML = `
    <div class="pc-ac" style="color:${{AC_COLORS[ac] || '#888'}}">${{info.icon}} ${{ac}}</div>
    <div class="pc-price">${{snap.price.toLocaleString('en-US',{{minimumFractionDigits:2,maximumFractionDigits:2}})}}</div>
    <div class="pc-ticker">${{snap.proxy}}</div>
    <div class="pc-rets">
      <span class="pc-ret ${{retClass(snap.ret_1d)}}">1d: ${{fmt(snap.ret_1d)}}</span>
      <span class="pc-ret ${{retClass(snap.ret_5d)}}">5d: ${{fmt(snap.ret_5d)}}</span>
      <span class="pc-ret ${{retClass(snap.ret_21d)}}">21d: ${{fmt(snap.ret_21d)}}</span>
      <span class="pc-ret ${{retClass(snap.ret_63d)}}">63d: ${{fmt(snap.ret_63d)}}</span>
    </div>`;
  pcEl.appendChild(el);
}}

// ── TILT TABLE ───────────────────────────────────────────────
const tbody = document.getElementById('tilt-tbody');
const ACS = ["Equities","Gold","Bonds","Crypto","FX"];
for(const ac of ACS){{
  const ts  = D.tilt_summaries[ac];
  const info = D.asset_classes[ac];
  if(!ts) continue;
  const delta = ts.overall_delta;
  const deltaStr = delta===0?'—':(delta>0?'+':'')+delta.toFixed(1)+'pp';
  const deltaColor = delta>0?'#4DB87A':delta<0?'#E05555':'#7A8F7A';

  const hl = ts.horizon_labels;
  const neutralW  = ts.neutral_w;
  const suggestedW = ts.suggested_w;
  const barW = Math.min(100, suggestedW / 50 * 100);
  const color = AC_COLORS[ac] || '#888';

  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>
      <div class="ac-cell">
        <div class="ac-dot" style="background:${{color}}"></div>
        <div>
          <div class="ac-name">${{info.icon}} ${{ac}}</div>
          <div class="ac-desc">${{info.desc}}</div>
        </div>
      </div>
    </td>
    <td>${{tiltPill(hl[21])}}</td>
    <td>${{tiltPill(hl[63])}}</td>
    <td>${{tiltPill(hl[126])}}</td>
    <td>${{tiltPill(hl[252])}}</td>
    <td>${{tiltPill(ts.overall_label)}}</td>
    <td style="font-family:var(--mono);font-size:12px;color:var(--muted)">${{neutralW.toFixed(1)}}%</td>
    <td>
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="font-family:var(--mono);font-size:13px;font-weight:600;color:${{color}}">${{suggestedW.toFixed(1)}}%</span>
        <div class="weight-bar-bg"><div class="weight-bar-fill" style="width:${{barW}}%;background:${{color}}"></div></div>
      </div>
    </td>
    <td style="font-family:var(--mono);font-size:12px;font-weight:600;color:${{deltaColor}}">${{deltaStr}}</td>`;
  tbody.appendChild(tr);
}}

// ── WEIGHT CHART (Donut) ──────────────────────────────────────
const acNames   = ACS;
const sugWts    = ACS.map(ac => D.suggested_weights[ac] || 0);
const neutralWts = ACS.map(ac => D.neutral_weights[ac]  || 0);
const colors    = ACS.map(ac => AC_COLORS[ac] || '#888');

Plotly.newPlot('chart-weights',[{{
  type:'pie', hole:0.52,
  values: sugWts,
  labels: ACS.map((ac,i) => ac + ' ' + sugWts[i].toFixed(1)+'%'),
  marker:{{colors, line:{{color:'#0C0E0D', width:2}}}},
  textinfo:'label',
  textfont:{{color:'#DDE8DD'}},
  outsidetextfont:{{color:'#DDE8DD'}},
  hovertemplate:'%{{label}}<br>Suggested: %{{value:.1f}}%<extra></extra>',
  sort:false,
}}],{{
  title:{{text:'Suggested Portfolio Weights',font:{{size:13,color:'#DDE8DD'}},x:0.5}},
  showlegend:false,
  margin:{{t:40,b:10,l:10,r:10}},
  paper_bgcolor:'rgba(0,0,0,0)',
  plot_bgcolor:'rgba(0,0,0,0)',
}});

// ── WEIGHT BARS ───────────────────────────────────────────────
const wbEl = document.getElementById('weight-bars');
for(const ac of ACS){{
  const nw = D.neutral_weights[ac]  || 0;
  const sw = D.suggested_weights[ac] || 0;
  const color = AC_COLORS[ac] || '#888';
  const maxW = 50;
  const div = document.createElement('div');
  div.className = 'wt-row';
  div.innerHTML = `
    <div class="wt-ac"><span style="color:${{color}}">${{D.asset_classes[ac].icon}}</span> ${{ac}}</div>
    <div class="wt-bars">
      <div class="wt-bar-row">
        <span class="wt-bar-label" style="color:var(--muted)">Neutral</span>
        <div class="wt-bar-track"><div class="wt-bar-fill" style="width:${{nw/maxW*100}}%;background:#ccc"></div></div>
        <span class="wt-pct" style="color:var(--muted)">${{nw.toFixed(1)}}%</span>
      </div>
      <div class="wt-bar-row">
        <span class="wt-bar-label" style="color:${{color}}">Suggested</span>
        <div class="wt-bar-track"><div class="wt-bar-fill" style="width:${{sw/maxW*100}}%;background:${{color}}"></div></div>
        <span class="wt-pct" style="font-weight:600;color:${{color}}">${{sw.toFixed(1)}}%</span>
      </div>
    </div>`;
  wbEl.appendChild(div);
}}

// ── FIRING SIGNAL CARDS ───────────────────────────────────────
const sgEl = document.getElementById('signal-grid');
for(const ac of ACS){{
  const info = D.asset_classes[ac];
  const color = AC_COLORS[ac] || '#888';
  for(const hor of D.horizons){{
    const h = D.results[ac].horizons[hor];
    if(!h) continue;
    const sigs = [...(h.bull_top||[]), ...(h.bear_top||[])];
    if(sigs.length === 0) continue;

    const card = document.createElement('div');
    card.className = 'signal-card';

    let rows = sigs.slice(0,5).map(s => `
      <div class="sig-row">
        <span>
          <span style="color:${{color}};font-weight:600">${{s.Y}}</span>
          <span class="sig-pred"> ← ${{s.X}} (τ=${{s.tau_past}}d, q=${{s.q_X}})</span>
        </span>
        <span>
          <span class="sig-cpe">CPE ${{s.CPE.toFixed(2)}}</span>
          <span style="color:var(--muted)"> · </span>
          <span class="sig-lift">${{s.lift.toFixed(2)}}×</span>
          <span style="color:var(--muted);font-size:9px"> n=${{s.n_condition}}</span>
        </span>
      </div>`).join('');

    const tilt = h.tilt_label;
    const t = TC[tilt] || TC["NO SIGNAL"];
    card.innerHTML = `
      <div class="sig-header">
        <span class="sig-ac" style="color:${{color}}">${{info.icon}} ${{ac}}</span>
        <span class="sig-hor">${{D.hor_labels[hor]}} (${{hor}}d)</span>
        ${{tiltPill(tilt)}}
      </div>
      ${{rows}}`;
    sgEl.appendChild(card);
  }}
}}
</script>
</body>
</html>"""

# ── WRITE OUTPUT ─────────────────────────────────────────────
outfile = "portfolio_dashboard.html"
with open(outfile, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nSaved: {outfile}  ({len(html)//1024} KB)")
print("Open portfolio_dashboard.html in Chrome/Firefox.")
print("Re-run this script to refresh.")
