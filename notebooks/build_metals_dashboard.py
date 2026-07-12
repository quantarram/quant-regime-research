#!/usr/bin/env python3
"""
============================================================
  PRECIOUS METALS BUY & TILT DASHBOARD
  Dr. Arun Ramanathan
============================================================
Combines:
  1. Gold-dashboard-style composite buy score, per metal
     (Gold / Silver / Platinum) — drawdown depth, mean-reversion
     recovery odds, and firing CPE regime score.
  2. Portfolio-tilt-style relative allocation across the three
     metals (which one(s) to overweight within a precious-metals
     sleeve right now).

Reads:
  - multiasset_prices.parquet   (historical prices)
  - cpe_results.parquet         (pairwise CPE signals)
  - joint_cpe_results.parquet   (joint CPE signals, unused directly
                                  but loaded for parity/future use)
Outputs:
  - precious_metals_dashboard.html
============================================================
"""

import json, os, re, warnings
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

print("=" * 60)
print("  PRECIOUS METALS DASHBOARD BUILDER")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── METAL DEFINITIONS ────────────────────────────────────────
METALS = {
    "Gold": {
        "y_tickers":  ["GC=F", "GLD", "IAU"],
        "hist":       "GC=F",   # full-history series used for drawdown/recovery stats
        "etf":        "GLD",    # ETF used for stable spot-price display
        "futures":    "GC=F",
        "color":      "#C9882A",
        "icon":       "\U0001F947",
        "desc":       "Gold futures and ETFs (GC=F/GLD/IAU)",
    },
    "Silver": {
        "y_tickers":  ["SI=F", "SLV"],
        "hist":       "SI=F",
        "etf":        "SLV",
        "futures":    "SI=F",
        "color":      "#9AA5AD",
        "icon":       "\U0001F948",
        "desc":       "Silver futures and ETF (SI=F/SLV)",
    },
    "Platinum": {
        "y_tickers":  ["PPLT"],
        "hist":       "PPLT",   # PL=F has no deep parquet history — use ETF for stats
        "etf":        "PPLT",
        "futures":    "PL=F",   # only used for fresh-price cross-check, not history
        "color":      "#7C8CA6",
        "icon":       "⬡",
        "desc":       "Platinum ETF (PPLT), futures cross-checked (PL=F)",
    },
}

FX_TICKER = "SGDUSD=X"
TROY_OZ_G = 31.1035

TAU_LIST = [21, 63, 126, 252]
HOR_WEIGHTS = {21: 0.20, 63: 0.30, 126: 0.30, 252: 0.20}
HOR_LABELS  = {21: "1 month", 63: "3 months", 126: "6 months", 252: "1 year"}

CPE_MIN, LIFT_MIN, N_MIN = 0.80, 1.50, 100

TILT_THRESHOLDS = [
    (0.85, "OVERWEIGHT",  +15),
    (0.70, "TILT UP",       +8),
    (0.50, "NEUTRAL",        0),
    (0.35, "TILT DOWN",     -8),
    (0.00, "UNDERWEIGHT",  -15),
]

RATE_TICKERS = {"^VIX","^VXN","^OVX","^EVZ","^VVIX","^SKEW",
                "^TNX","^TYX","^FVX","^IRX"}

# ── LOAD DATA ────────────────────────────────────────────────
print("\nLoading local data...")
prices = pd.read_parquet("multiasset_prices.parquet")
cpe    = pd.read_parquet("cpe_results.parquet")
jcpe   = pd.read_parquet("joint_cpe_results.parquet")
print(f"  Prices: {prices.shape[0]} rows x {prices.shape[1]} tickers")
print(f"  CPE pairwise: {cpe.shape[0]:,} rows")

PARQUET_MAX_DATE = prices.index.max()
print(f"  Parquet history frozen at: {PARQUET_MAX_DATE.date()}")

# ── FETCH FRESH PRICES ───────────────────────────────────────
print("\nFetching latest prices from Yahoo Finance...")
try:
    import yfinance as yf
    fetch_list = list(set(
        ["GC=F","GLD","IAU","SI=F","SLV","PL=F","PPLT",
         "SGDUSD=X","DX-Y.NYB","UUP","^GVZ"]
    ))
    raw = yf.download(fetch_list, period="400d", auto_adjust=True, progress=False)["Close"]
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    # ── STRIP ANY PARTIAL/LIVE "TODAY" SESSION ROW, FOR EVERY TICKER ──
    # yfinance's most recent daily bar can be an intraday snapshot that keeps
    # changing as the trading day progresses, which was letting composite buy
    # scores drift between multiple runs on the same calendar day. Drop any
    # row dated today-or-later before it reaches `prices`, so every run on a
    # given day sees identical, fully-settled data (see build_gold_dashboard.py
    # for the symptom this was first diagnosed from).
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
    for _key in ["GC=F","GLD","IAU","SI=F","SLV","PPLT","SGDUSD=X","DX-Y.NYB","UUP","^GVZ"]:
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

    _today = date.today()
    _latest = prices.index.max().date()
    _age = (_today - _latest).days
    if _age > 4:
        print(f"  WARNING: data is {_age} days old - Yahoo may be lagging.")
except Exception as e:
    print(f"  yfinance error: {e}")
    print("  FATAL: Cannot build dashboard without fresh prices. Exiting.")
    import sys; sys.exit(1)

usd_per_sgd = 1.0
if FX_TICKER in prices.columns:
    fx = prices[FX_TICKER].dropna()
    if len(fx):
        # SGDUSD=X quotes USD per 1 SGD (e.g. ~0.78) — invert to get SGD per USD
        # (matches build_gold_dashboard.py's usd_per_sgd convention).
        usd_per_sgd = 1.0 / float(fx.iloc[-1])

# ── HELPERS ──────────────────────────────────────────────────

def settled_price(ticker):
    """Last settled (non-partial-session) close from the fresh yfinance pull."""
    if ticker not in raw.columns:
        return None, None
    s = raw[ticker].dropna()
    if len(s) == 0:
        return None, None
    if len(s) >= 2 and s.index[-1].date() >= date.today():
        return float(s.iloc[-2]), s.index[-2].date()
    return float(s.iloc[-1]), s.index[-1].date()


def calibrate_ratio(etf_ticker, futures_ticker, cache_key):
    """Dynamically calibrate ETF-shares-per-oz vs its futures, cached weekly.
    Mirrors the GLD/GC=F calibration already used in build_gold_dashboard.py."""
    cache_path = os.path.join(BASE_DIR, f".{cache_key}_ratio_cache.json")
    today = datetime.now().date()
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
            cache_date = datetime.strptime(cache["date"], "%Y-%m-%d").date()
            if (today - cache_date).days < 7:
                return float(cache["ratio"])
        except Exception:
            pass
    if etf_ticker not in raw.columns or futures_ticker not in raw.columns:
        return None
    etf = raw[etf_ticker].dropna()
    fut = raw[futures_ticker].dropna()
    common = etf.index.intersection(fut.index)
    if len(common) < 10:
        return None
    ratio = float((etf.reindex(common) / fut.reindex(common)).tail(60).median())
    try:
        with open(cache_path, "w") as f:
            json.dump({"date": str(today), "ratio": ratio}, f)
    except Exception:
        pass
    return ratio


def hist_series(ticker):
    s = prices[ticker].dropna()
    return s[s.index <= PARQUET_MAX_DATE]


def current_return(ticker, tau):
    if ticker not in prices.columns:
        return None
    series = prices[ticker].dropna()
    if len(series) < tau + 1:
        return None
    if ticker in RATE_TICKERS:
        return float(series.iloc[-1] - series.iloc[-1 - tau])
    return float(np.log(series.iloc[-1] / series.iloc[-1 - tau]) * 100)


def historical_quantile(ticker, tau, q):
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


# ── FIRING PREDICTORS (generic, same pattern as portfolio dashboard) ──
print("\nComputing current regime (which predictors are in their tails)...")
pred_combos = (cpe[["X","tau_past","q_X","direction"]].drop_duplicates().values.tolist())
firing_predictors = []
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
        thresh = historical_quantile(X, int(tau), float(q))
        if thresh is not None and curr >= thresh:
            firing_predictors.append((X, int(tau), float(q), direction))
    else:
        thresh = historical_quantile(X, int(tau), 1.0 - float(q))
        if thresh is not None and curr <= thresh:
            firing_predictors.append((X, int(tau), float(q), direction))
firing_set = set(firing_predictors)
print(f"  Firing predictors: {len(firing_set)}")


def get_firing_cpe_signals(target_tickers, horizon, direction_filter):
    mask = (
        cpe["Y"].isin(target_tickers) &
        (cpe["tau_future"] == horizon) &
        (cpe["CPE"] >= CPE_MIN) &
        (cpe["lift"] >= LIFT_MIN) &
        (cpe["n_condition"] >= N_MIN) &
        (cpe["direction"] == direction_filter)
    )
    subset = cpe[mask].copy()
    keys = list(zip(subset["X"].values, subset["tau_past"].astype(int).values,
                     subset["q_X"].astype(float).values, subset["direction"].values))
    keep = [k in firing_set for k in keys]
    return subset[keep]


def compute_regime_score(firing_signals):
    if len(firing_signals) == 0:
        return None, []
    fs = firing_signals.copy()
    fs_dedup = (fs.sort_values("CPE", ascending=False)
                  .drop_duplicates(subset=["X","tau_past","q_X","direction"])
                  .reset_index(drop=True))
    weights = fs_dedup["n_condition"].values.astype(float)
    scores  = fs_dedup["CPE"].values
    regime_score = float(np.average(scores, weights=weights))
    top = (fs_dedup.sort_values(["CPE","lift"], ascending=False).head(5)
           [["Y","X","tau_past","tau_future","q_X","q_Y","CPE","lift","n_condition","direction"]]
           .to_dict("records"))
    return regime_score, top


def score_to_tilt(score):
    if score is None:
        return "NO SIGNAL", 0
    for threshold, label, delta in TILT_THRESHOLDS:
        if score >= threshold:
            return label, delta
    return "UNDERWEIGHT", -15


# ── PER-METAL COMPUTATION ────────────────────────────────────
print("\nComputing per-metal buy signals and regime tilts...")

metal_data = {}
for name, info in METALS.items():
    hist = hist_series(info["hist"])
    if len(hist) < 300:
        print(f"  WARNING: {name} history too short ({len(hist)} rows), skipping")
        continue

    # ── spot price (ETF-calibrated, settled-close, futures-checked) ──
    ratio = calibrate_ratio(info["etf"], info["futures"], name.lower())
    etf_price, etf_date = settled_price(info["etf"])
    fut_price, fut_date = settled_price(info["futures"])
    if ratio and etf_price:
        spot_usd_oz = etf_price / ratio
        spot_src = f"{info['etf']} ETF (calibrated to {info['futures']})"
        spot_date = etf_date
    elif fut_price:
        spot_usd_oz = fut_price
        spot_src = f"{info['futures']} futures"
        spot_date = fut_date
    else:
        spot_usd_oz = float(hist.iloc[-1])
        spot_src = f"{info['hist']} (fallback)"
        spot_date = hist.index[-1].date()

    spot_sgd_oz = spot_usd_oz * usd_per_sgd
    spot_sgd_g  = spot_sgd_oz / TROY_OZ_G

    # ── returns / drawdown vs current price series (parquet-clipped, own units) ──
    last_hist_price = float(hist.iloc[-1])
    def pct_chg(n):
        if len(hist) > n:
            return float((last_hist_price / float(hist.iloc[-1 - n]) - 1) * 100)
        return 0.0
    chg = {t: round(pct_chg(t), 2) for t in [1,5,21,63,126,252]}
    peak_252 = float(hist.iloc[-252:].max()) if len(hist) >= 252 else float(hist.max())
    dd_from_peak = round((last_hist_price / peak_252 - 1) * 100, 2)

    # ── mean-reversion / recovery odds (CORRECT forward-return sign) ──
    ret_63  = ((hist / hist.shift(63)) - 1).dropna() * 100
    curr_pct_63 = float(np.mean(ret_63.values <= chg[63]))   # percentile rank, 0-1
    draw_score = round(100 * (1 - curr_pct_63), 1)           # deeper drawdown -> higher score

    trigger_dates = ret_63[ret_63 <= chg[63]].index
    recovery = {}
    for fwd in [21, 63, 126, 252]:
        fwd_ret = ((hist.shift(-fwd) / hist) - 1).dropna() * 100   # true forward return
        vals = fwd_ret.reindex(trigger_dates).dropna().values
        if len(vals) >= 5:
            recovery[fwd] = {
                "n": int(len(vals)),
                "p10": round(float(np.percentile(vals,10)),2),
                "p50": round(float(np.percentile(vals,50)),2),
                "p90": round(float(np.percentile(vals,90)),2),
                "pct_positive": round(float(np.mean(vals>0))*100,1),
                "mean": round(float(np.mean(vals)),2),
            }
    auto_score = recovery.get(126, {}).get("pct_positive", 50.0)

    # ── firing CPE regime score, per horizon ──
    horizons = {}
    for hor in TAU_LIST:
        bull_sigs = get_firing_cpe_signals(info["y_tickers"], hor, "bullish")
        bear_sigs = get_firing_cpe_signals(info["y_tickers"], hor, "bearish")
        bull_score, bull_top = compute_regime_score(bull_sigs)
        bear_score, bear_top = compute_regime_score(bear_sigs)
        if bull_score is not None and bear_score is not None:
            net_score = bull_score * 0.6 + (1 - bear_score) * 0.4
        elif bull_score is not None:
            net_score = bull_score
        elif bear_score is not None:
            net_score = 1 - bear_score
        else:
            net_score = None
        tilt_label, tilt_delta = score_to_tilt(net_score)
        horizons[hor] = {
            "bull_score": round(bull_score,3) if bull_score else None,
            "bear_score": round(bear_score,3) if bear_score else None,
            "net_score":  round(net_score,3) if net_score else None,
            "tilt_label": tilt_label, "tilt_delta": tilt_delta,
            "bull_n": len(bull_sigs), "bear_n": len(bear_sigs),
            "bull_top": bull_top, "bear_top": bear_top,
        }

    cpe_126 = horizons[126]["net_score"]
    cpe_score = round((cpe_126 if cpe_126 is not None else 0.5) * 100, 1)

    composite = round(0.40*draw_score + 0.35*auto_score + 0.25*cpe_score, 1)
    buy_label = ("STRONG BUY ZONE" if composite >= 70 else
                 "BUY ZONE" if composite >= 55 else
                 "WATCH - APPROACHING BUY" if composite >= 40 else
                 "NEUTRAL - WAIT" if composite >= 25 else
                 "NOT YET")
    # verdict_tier drives the buy-card's ring/text color in the browser — kept
    # on the same 4-value scale (bull/bull_light/warn/bear) build_gold_dashboard.py
    # uses, so Gold's imported verdict (below) colors identically to Silver/Platinum.
    verdict_tier = ("bull" if composite >= 70 else
                     "bull_light" if composite >= 55 else
                     "warn" if composite >= 25 else
                     "bear")

    print(f"  {name:9s}: spot ${spot_usd_oz:,.2f}/oz | draw={draw_score:5.1f} "
          f"auto={auto_score:5.1f} cpe={cpe_score:5.1f} -> composite={composite:5.1f} ({buy_label})")

    chart = hist.iloc[-365:]
    metal_data[name] = {
        "info": info,
        "spot_usd_oz": round(spot_usd_oz, 2),
        "spot_sgd_oz": round(spot_sgd_oz, 2),
        "spot_sgd_g":  round(spot_sgd_g, 4),
        "spot_src": spot_src,
        "spot_date": str(spot_date) if spot_date else None,
        "chg": chg,
        "dd_from_peak": dd_from_peak,
        "peak_252": round(peak_252, 2),
        "draw_score": draw_score,
        "auto_score": round(auto_score, 1),
        "cpe_score": cpe_score,
        "composite": composite,
        "buy_label": buy_label,
        "verdict_tier": verdict_tier,
        "curr_pct_63": round(curr_pct_63*100, 1),
        "recovery_63": {str(k): v for k, v in recovery.items()},
        "horizons": horizons,
        "chart_dates":  [str(d.date()) for d in chart.index],
        "chart_prices": [round(float(p), 2) for p in chart.values],
    }

# ── IMPORT GOLD'S AUTHORITATIVE NUMBERS FROM gold_dashboard.html ────
# Gold's own dashboard uses a more elaborate composite formula (adds a 4th
# "predictor proximity" component hand-tuned just for gold, plus GLD-ETF
# settled-close price calibration) than the simplified 3-component formula
# above. Rather than reimplement that formula a second time here — which is
# exactly how the two dashboards drifted out of sync and showed conflicting
# verdicts for gold on the same day — read gold_dashboard.html's already-
# computed values directly (same technique log_predictions.py already uses)
# so gold's number/verdict/spot-price can never disagree between the two
# dashboards. Falls back to this script's own gold computation above if
# gold_dashboard.html is missing or stale (e.g. running this script standalone
# before build_gold_dashboard.py has ever run).
if "Gold" in metal_data:
    _gold_html = os.path.join(BASE_DIR, "gold_dashboard.html")
    try:
        if os.path.exists(_gold_html):
            with open(_gold_html, encoding="utf-8") as _f:
                _gold_content = _f.read()
            _m = re.search(r'const D = (\{.*?\});\n', _gold_content, re.DOTALL)
            if _m:
                _GD = json.loads(_m.group(1))
                _gd_date = datetime.strptime(_GD["generated"][:10], "%Y-%m-%d").date()
                if (datetime.now().date() - _gd_date).days <= 1:
                    _gc = _GD["components"]
                    g = metal_data["Gold"]
                    g["spot_usd_oz"]   = round(_GD["gold_usd"], 2)
                    g["spot_sgd_oz"]   = round(_GD["gold_sgd_oz"], 2)
                    g["spot_sgd_g"]    = round(_GD["gold_sgd_g"], 4)
                    g["spot_src"]      = "GLD ETF (from gold_dashboard.html)"
                    g["spot_date"]     = _GD["latest_date"]
                    # _GD["chg"] came through json.loads so its keys are strings ("1","63",...)
                    # while g["chg"]'s keys are still native ints at this point in the script.
                    g["chg"]           = {k: _GD["chg"].get(str(k), g["chg"].get(k, 0)) for k in g["chg"]}
                    g["dd_from_peak"]  = _GD["dd_from_peak"]
                    g["peak_252"]      = _GD["peak_252"]
                    g["draw_score"]    = _gc["draw_score"]
                    g["auto_score"]    = _gc["auto_score"]
                    g["cpe_score"]     = _gc["cpe_score"]
                    g["composite"]     = _gc["composite"]
                    g["buy_label"]     = _gc["label"]
                    g["verdict_tier"]  = _gc["verdict_tier"]
                    g["curr_pct_63"]   = _GD["curr_pct_63"]
                    print(f"  Gold     : overridden with gold_dashboard.html's authoritative "
                          f"composite={_gc['composite']} ({_gc['label']})")
                else:
                    print(f"  Gold     : gold_dashboard.html is {(datetime.now().date()-_gd_date).days}d "
                          f"stale, keeping this script's own gold computation")
    except Exception as _e:
        print(f"  Gold     : could not import gold_dashboard.html ({_e}), keeping own computation")

# ── SHARPE-BASED NEUTRAL WEIGHTS WITHIN THE METALS SLEEVE ────
print("\nComputing Sharpe-based neutral weights within precious-metals sleeve...")

def compute_sharpe(ticker, lookback_years=10):
    if ticker not in prices.columns:
        return None
    series = prices[ticker].dropna()
    cutoff = series.index.max() - pd.DateOffset(years=lookback_years)
    series = series[series.index >= cutoff]
    if len(series) < 252:
        return None
    rets = np.log(series / series.shift(1)).dropna()
    ann_ret = float(rets.mean() * 252)
    ann_vol = float(rets.std() * np.sqrt(252))
    if ann_vol < 1e-8:
        return None
    return ann_ret / ann_vol

sharpes = {name: compute_sharpe(info["hist"]) for name, info in METALS.items() if name in metal_data}
for name, s in sharpes.items():
    print(f"  {name:9s} ({METALS[name]['hist']}): Sharpe = {s:.3f}" if s is not None else f"  {name:9s}: insufficient data")

MIN_WEIGHT = 15.0
pos_sharpes = {k: max(v, 0.05) for k, v in sharpes.items() if v is not None}
total_sharpe = sum(pos_sharpes.values())
if total_sharpe > 0:
    raw_weights = {k: v/total_sharpe*100 for k, v in pos_sharpes.items()}
    floored = {k: max(v, MIN_WEIGHT) for k, v in raw_weights.items()}
    for k in metal_data:
        if k not in floored:
            floored[k] = MIN_WEIGHT
    total_floored = sum(floored.values())
    neutral_weights = {k: round(v/total_floored*100, 1) for k, v in floored.items()}
else:
    neutral_weights = {k: round(100/len(metal_data), 1) for k in metal_data}

print(f"  Neutral weights: {neutral_weights}")

# ── OVERALL TILT AND SUGGESTED WEIGHTS ───────────────────────
tilt_summaries = {}
suggested_weights = {}
for name in metal_data:
    total_w, total_d = 0, 0
    for hor in TAU_LIST:
        h = metal_data[name]["horizons"][hor]
        w = HOR_WEIGHTS[hor]
        total_d += h["tilt_delta"] * w
        total_w += w
    overall_delta = round(total_d/total_w, 1) if total_w > 0 else 0
    if overall_delta >= 10: overall_label = "OVERWEIGHT"
    elif overall_delta >= 4: overall_label = "TILT UP"
    elif overall_delta >= -4: overall_label = "NEUTRAL"
    elif overall_delta >= -10: overall_label = "TILT DOWN"
    else: overall_label = "UNDERWEIGHT"

    base = neutral_weights.get(name, 100/len(metal_data))
    suggested = max(0, min(70, base + overall_delta))
    suggested_weights[name] = round(suggested, 1)
    tilt_summaries[name] = {
        "overall_label": overall_label, "overall_delta": overall_delta,
        "neutral_w": base, "suggested_w": round(suggested, 1),
        "horizon_labels": {hor: metal_data[name]["horizons"][hor]["tilt_label"] for hor in TAU_LIST},
    }
    print(f"  {name:9s}: neutral={base:.1f}% -> suggested={suggested:.1f}% ({overall_label})")

total_suggested = sum(suggested_weights.values())
if total_suggested > 0:
    suggested_weights = {k: round(v/total_suggested*100, 1) for k, v in suggested_weights.items()}
    for name in metal_data:
        tilt_summaries[name]["suggested_w"] = suggested_weights[name]

print(f"  Suggested weights (normalised): {suggested_weights}")

# ── DATA BUNDLE ───────────────────────────────────────────────
gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")
bundle = {
    "gen": gen_time,
    "data_date": str(prices.index.max().date()),
    "usd_per_sgd": round(usd_per_sgd, 4),
    "metals": {name: {
        "spot_usd_oz": d["spot_usd_oz"], "spot_sgd_oz": d["spot_sgd_oz"],
        "spot_sgd_g": d["spot_sgd_g"], "spot_src": d["spot_src"], "spot_date": d["spot_date"],
        "chg": d["chg"], "dd_from_peak": d["dd_from_peak"], "peak_252": d["peak_252"],
        "draw_score": d["draw_score"], "auto_score": d["auto_score"], "cpe_score": d["cpe_score"],
        "composite": d["composite"], "buy_label": d["buy_label"], "verdict_tier": d["verdict_tier"],
        "curr_pct_63": d["curr_pct_63"],
        "recovery_63": d["recovery_63"],
        "chart_dates": d["chart_dates"], "chart_prices": d["chart_prices"],
        "color": d["info"]["color"], "icon": d["info"]["icon"], "desc": d["info"]["desc"],
    } for name, d in metal_data.items()},
    "horizons_by_metal": {name: metal_data[name]["horizons"] for name in metal_data},
    "neutral_weights": neutral_weights,
    "suggested_weights": suggested_weights,
    "tilt_summaries": tilt_summaries,
    "sharpes": {k: round(v,3) if v is not None else None for k, v in sharpes.items()},
    "firing_pred_count": len(firing_set),
    "horizons": TAU_LIST,
    "hor_labels": HOR_LABELS,
}
bundle_json = json.dumps(bundle, allow_nan=False)
print(f"\n  Data bundle: {len(bundle_json)/1024:.1f} KB")

# ── HTML ──────────────────────────────────────────────────────
print("\nBuilding HTML dashboard...")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Precious Metals Buy &amp; Tilt Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=DM+Serif+Display:ital@0;1&display=swap');
  :root{{
    --bg:#F5F0E8; --card:#fff; --border:#e5e5e5;
    --text:#1a1a1a; --muted:#666; --faint:#f5f5f5;
    --gold:#C9882A; --silver:#9AA5AD; --platinum:#7C8CA6;
    --green:#1a6b3a; --red:#E05555; --amber:#C87000;
    --mono:'IBM Plex Mono',monospace; --serif:'DM Serif Display',serif;
  }}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);font-family:'Inter',sans-serif;color:var(--text);font-size:14px;}}
  .page{{max-width:1400px;margin:0 auto;padding:32px 24px 80px;}}

  .header{{background:#1a1a1a;border-radius:16px;padding:32px 40px;margin-bottom:28px;
           display:grid;grid-template-columns:1fr auto;align-items:center;gap:24px;}}
  .h-title{{font-family:var(--serif);font-size:28px;color:#fff;margin-bottom:6px;}}
  .h-title em{{font-style:italic;color:var(--gold);}}
  .h-sub{{font-family:var(--mono);font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:#888;margin-bottom:4px;}}
  .h-meta{{font-size:12px;color:#aaa;}}
  .h-badge{{background:rgba(201,136,42,0.15);border:1px solid rgba(201,136,42,0.3);
            border-radius:12px;padding:12px 20px;text-align:center;}}
  .h-badge-num{{font-family:var(--mono);font-size:28px;font-weight:600;color:var(--gold);}}
  .h-badge-label{{font-size:10px;color:#888;display:block;margin-top:2px;letter-spacing:.1em;text-transform:uppercase;}}

  .section{{margin-bottom:28px;}}
  .section-title{{font-family:var(--mono);font-size:10px;letter-spacing:.18em;
                  text-transform:uppercase;color:var(--muted);margin-bottom:14px;}}
  .card{{background:var(--card);border-radius:14px;border:1px solid var(--border);padding:20px 24px;}}
  .grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;}}
  .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;}}

  /* ── BUY CARDS ── */
  .buy-card{{background:var(--card);border-radius:16px;border:1px solid var(--border);
             padding:22px;position:relative;overflow:hidden;}}
  .buy-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:4px;
                      background:var(--accent);}}
  .buy-head{{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;}}
  .buy-name{{font-family:var(--serif);font-size:22px;display:flex;align-items:center;gap:8px;}}
  .buy-price{{text-align:right;}}
  .buy-price-v{{font-family:var(--mono);font-size:16px;font-weight:600;}}
  .buy-price-s{{font-size:10px;color:var(--muted);margin-top:2px;}}
  .buy-ring-wrap{{display:flex;align-items:center;gap:16px;margin:14px 0;}}
  .buy-ring{{position:relative;width:100px;height:100px;flex-shrink:0;}}
  .buy-ring svg{{width:100%;height:100%;}}
  .buy-ring-inner{{position:absolute;inset:0;display:flex;flex-direction:column;
    align-items:center;justify-content:center;}}
  .buy-num{{font-family:var(--mono);font-size:24px;font-weight:600;}}
  .buy-verdict{{font-family:var(--mono);font-size:12px;font-weight:600;letter-spacing:.03em;}}
  .buy-verdict-sub{{font-size:11px;color:var(--muted);margin-top:2px;}}
  .comp-row{{display:flex;flex-direction:column;gap:6px;flex:1;}}
  .comp-item{{background:var(--faint);border-radius:6px;padding:6px 10px;}}
  .comp-label{{font-family:var(--mono);font-size:8px;color:var(--muted);
    text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px;display:flex;justify-content:space-between;}}
  .comp-bar-wrap{{height:4px;background:#e5e5e5;border-radius:2px;overflow:hidden;}}
  .comp-bar{{height:100%;border-radius:2px;}}
  .buy-stats{{display:flex;gap:14px;margin-top:12px;padding-top:12px;border-top:1px solid var(--border);
              font-family:var(--mono);font-size:11px;}}
  .buy-stat-l{{color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.06em;}}

  /* ── TILT TABLE ── */
  .tilt-table{{width:100%;border-collapse:collapse;}}
  .tilt-table th{{font-family:var(--mono);font-size:9px;letter-spacing:.12em;
                  text-transform:uppercase;color:var(--muted);padding:10px 14px;
                  border-bottom:2px solid var(--border);text-align:left;background:var(--faint);}}
  .tilt-table td{{padding:12px 14px;border-bottom:1px solid var(--border);vertical-align:middle;}}
  .tilt-table tr:last-child td{{border-bottom:none;}}
  .tilt-table tr:hover td{{background:#fafafa;}}
  .ac-cell{{display:flex;align-items:center;gap:10px;}}
  .ac-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0;}}
  .ac-name{{font-weight:600;font-size:13px;}}
  .ac-desc{{font-size:10px;color:var(--muted);margin-top:2px;}}
  .tilt-pill{{display:inline-flex;align-items:center;padding:4px 10px;
              border-radius:100px;font-family:var(--mono);font-size:9px;
              font-weight:600;letter-spacing:.06em;white-space:nowrap;}}
  .weight-bar-bg{{background:#eee;border-radius:4px;height:6px;width:80px;display:inline-block;vertical-align:middle;}}
  .weight-bar-fill{{height:100%;border-radius:4px;}}

  .wt-row{{display:flex;align-items:center;gap:14px;padding:10px 0;border-bottom:1px solid var(--border);}}
  .wt-row:last-child{{border-bottom:none;}}
  .wt-ac{{width:100px;font-size:12px;font-weight:600;}}
  .wt-bars{{flex:1;display:flex;flex-direction:column;gap:5px;}}
  .wt-bar-row{{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:10px;}}
  .wt-bar-label{{width:60px;color:var(--muted);}}
  .wt-bar-track{{flex:1;background:#f0f0f0;border-radius:3px;height:8px;}}
  .wt-bar-fill{{height:100%;border-radius:3px;}}
  .wt-pct{{width:40px;text-align:right;}}

  .signal-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px;}}
  .signal-card{{background:var(--card);border-radius:12px;border:1px solid var(--border);padding:16px 18px;}}
  .sig-header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}}
  .sig-ac{{font-size:12px;font-weight:600;}}
  .sig-hor{{font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:.1em;}}
  .sig-row{{font-family:var(--mono);font-size:10px;padding:5px 0;
             border-bottom:1px solid #f0f0f0;display:flex;justify-content:space-between;gap:8px;}}
  .sig-row:last-child{{border-bottom:none;}}
  .sig-pred{{color:var(--muted);font-size:9px;}}
  .sig-cpe{{color:var(--green);font-weight:600;}}
  .sig-lift{{color:var(--gold);}}

  .disclaimer{{background:#f9f7f4;border:1px solid #e5e0d5;border-radius:10px;
               padding:16px 20px;font-size:11px;color:var(--muted);line-height:1.7;margin-top:28px;}}

  @media(max-width:1000px){{
    .grid-3{{grid-template-columns:1fr;}}
    .header{{grid-template-columns:1fr;}}
  }}
  @media(max-width:600px){{
    .grid-2{{grid-template-columns:1fr;}}
    .page{{padding:16px 12px 60px;}}
    .buy-ring-wrap{{flex-direction:column;align-items:flex-start;}}
  }}
</style>
</head>
<body>
<div class="page">

<div class="header">
  <div>
    <div class="h-sub">CPE Multi-Asset Framework - Precious Metals Dashboard</div>
    <div class="h-title"><em>Gold, Silver &amp; Platinum</em> Buy Signal &amp; Sleeve Tilt</div>
    <div class="h-meta">
      Updated: <span id="hgen"></span> &nbsp;.&nbsp;
      Data: <span id="hdate"></span> &nbsp;.&nbsp;
      <span id="hpred"></span> predictors firing
    </div>
  </div>
  <div class="h-badge">
    <div class="h-badge-num" id="hfx">-</div>
    <span class="h-badge-label">USD / SGD</span>
  </div>
</div>

<!-- BUY SIGNAL CARDS -->
<div class="section">
  <div class="section-title">Which Metal To Buy Now — Composite Buy Score</div>
  <div class="grid-3" id="buy-cards"></div>
</div>

<!-- PRICE CHART -->
<div class="section">
  <div class="section-title">1-Year Price Performance (Indexed to 100)</div>
  <div class="card">
    <div style="font-size:11px;color:var(--muted);margin-bottom:10px;">
      Each line shows % return since 1 year ago, not dollar price — gold (~$4,000/oz), silver (~$60/oz)
      and platinum (~$1,600/oz) trade on very different scales, so this puts them on one comparable axis.
      A line at 150 means that metal is up 50% over the year; it does not mean it costs more per ounce.
      Hover a point to see both the native-unit price and the % change.
    </div>
    <div id="chart-prices" style="height:340px;"></div>
  </div>
</div>

<!-- TILT TABLE -->
<div class="section">
  <div class="section-title">Sleeve Tilt — How To Split Your Precious-Metals Allocation</div>
  <div class="card" style="padding:0;overflow:hidden;">
    <table class="tilt-table" id="tilt-table">
      <thead>
        <tr>
          <th style="width:220px">Metal</th>
          <th>1 Month</th><th>3 Months</th><th>6 Months</th><th>1 Year</th>
          <th>Overall Tilt</th><th>Neutral Wt</th><th>Suggested Wt</th><th>Change</th>
        </tr>
      </thead>
      <tbody id="tilt-tbody"></tbody>
    </table>
  </div>
</div>

<!-- WEIGHT COMPARISON -->
<div class="section">
  <div class="section-title">Suggested vs Neutral Sleeve Weights</div>
  <div class="grid-2">
    <div class="card"><div id="chart-weights" style="height:300px;"></div></div>
    <div class="card" style="padding:24px;">
      <div style="font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:16px;">Weight Comparison</div>
      <div id="weight-bars"></div>
    </div>
  </div>
</div>

<!-- FIRING SIGNALS -->
<div class="section">
  <div class="section-title">Firing CPE Signals — Evidence Behind Each Tilt</div>
  <div class="signal-grid" id="signal-grid"></div>
</div>

<div class="disclaimer">
  <strong>Research Disclaimer:</strong> All CPE values are within-sample empirical frequencies computed from historical data.
  Buy scores, tilt labels, and suggested weights are descriptive summaries of historical tail co-movement and mean-reversion
  patterns — not forecasts or investment advice. Spot prices are ETF-derived (calibrated weekly against the metal's futures
  contract) and may lag intraday moves. Past statistical structure does not guarantee future behaviour. This dashboard is
  for personal research only. Always consult a licensed financial adviser before making investment decisions.
  &nbsp;.&nbsp; <strong>Dr. Arun Ramanathan</strong>
</div>

</div>

<script>
const D = {bundle_json};
const METAL_ORDER = {list(metal_data.keys())!r};
const TC = {{
  "OVERWEIGHT":  {{color:"#1a6b3a", bg:"#f0faf4", border:"#c8e6d0"}},
  "TILT UP":     {{color:"#2A7A4B", bg:"#f5fbf7", border:"#d0ead8"}},
  "NEUTRAL":     {{color:"#8B7355", bg:"#f9f7f4", border:"#e5e0d5"}},
  "TILT DOWN":   {{color:"#C87000", bg:"#fdf8f0", border:"#f0ddb8"}},
  "UNDERWEIGHT": {{color:"#E05555", bg:"#fdf4f4", border:"#f0c8c8"}},
  "NO SIGNAL":   {{color:"#aaa",    bg:"#f5f5f5", border:"#e0e0e0"}},
}};
// Keyed by verdict_tier (not the label text) since Gold's label/tier come
// straight from gold_dashboard.html's own 4-value scheme (bull/bull_light/
// warn/bear) while Silver/Platinum use this script's 5-label scheme mapped
// onto the same 4 tiers — one color table works for all three cards.
const BUY_COLORS = {{
  "bull":       "#1a6b3a",
  "bull_light": "#4DB87A",
  "warn":       "#C87000",
  "bear":       "#E05555",
}};

function tiltPill(label){{
  const t = TC[label] || TC["NO SIGNAL"];
  return `<span class="tilt-pill" style="color:${{t.color}};background:${{t.bg}};border:1px solid ${{t.border}}">${{label}}</span>`;
}}
function fmt(v, decimals=2){{
  if(v===null||v===undefined) return '-';
  const s = v>=0?'+':'';
  return s + v.toFixed(decimals) + '%';
}}

document.getElementById('hgen').textContent  = D.gen;
document.getElementById('hdate').textContent = D.data_date;
document.getElementById('hpred').textContent = D.firing_pred_count;
document.getElementById('hfx').textContent   = D.usd_per_sgd.toFixed(4);

// ── BUY CARDS ─────────────────────────────────────────────────
const bcEl = document.getElementById('buy-cards');
for(const name of METAL_ORDER){{
  const m = D.metals[name];
  const color = m.color;
  const buyColor = BUY_COLORS[m.verdict_tier] || '#888';
  const circumference = 2*Math.PI*40;
  const offset = circumference * (1 - m.composite/100);
  const card = document.createElement('div');
  card.className = 'buy-card';
  card.style.setProperty('--accent', color);
  card.innerHTML = `
    <div class="buy-head">
      <div class="buy-name">${{m.icon}} ${{name}}</div>
      <div class="buy-price">
        <div class="buy-price-v">S$${{m.spot_sgd_g.toFixed(2)}}/g</div>
        <div class="buy-price-s">$${{m.spot_usd_oz.toLocaleString('en-US',{{minimumFractionDigits:2,maximumFractionDigits:2}})}}/oz</div>
      </div>
    </div>
    <div class="buy-ring-wrap">
      <div class="buy-ring">
        <svg viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="40" fill="none" stroke="#eee" stroke-width="8"/>
          <circle cx="50" cy="50" r="40" fill="none" stroke="${{buyColor}}" stroke-width="8"
                  stroke-linecap="round" stroke-dasharray="${{circumference}}" stroke-dashoffset="${{offset}}"
                  style="transform:rotate(-90deg);transform-origin:50px 50px;"/>
        </svg>
        <div class="buy-ring-inner">
          <div class="buy-num" style="color:${{buyColor}}">${{m.composite}}</div>
        </div>
      </div>
      <div style="flex:1;min-width:0;">
        <div class="buy-verdict" style="color:${{buyColor}}">${{m.buy_label}}</div>
        <div class="buy-verdict-sub">${{m.dd_from_peak}}% from 52w high &middot; ${{m.curr_pct_63}}th pct (63d)</div>
        <div class="comp-row" style="margin-top:8px;">
          <div class="comp-item">
            <div class="comp-label"><span>Drawdown depth</span><span>${{m.draw_score.toFixed(0)}}</span></div>
            <div class="comp-bar-wrap"><div class="comp-bar" style="width:${{m.draw_score}}%;background:${{color}}"></div></div>
          </div>
          <div class="comp-item">
            <div class="comp-label"><span>Recovery odds (fwd 6mo)</span><span>${{m.auto_score.toFixed(0)}}</span></div>
            <div class="comp-bar-wrap"><div class="comp-bar" style="width:${{m.auto_score}}%;background:${{color}}"></div></div>
          </div>
          <div class="comp-item">
            <div class="comp-label"><span>CPE regime score (6mo)</span><span>${{m.cpe_score.toFixed(0)}}</span></div>
            <div class="comp-bar-wrap"><div class="comp-bar" style="width:${{m.cpe_score}}%;background:${{color}}"></div></div>
          </div>
        </div>
      </div>
    </div>
    <div class="buy-stats">
      <div><div class="buy-stat-l">1d</div><span>${{fmt(m.chg['1'])}}</span></div>
      <div><div class="buy-stat-l">21d</div><span>${{fmt(m.chg['21'])}}</span></div>
      <div><div class="buy-stat-l">63d</div><span>${{fmt(m.chg['63'])}}</span></div>
      <div><div class="buy-stat-l">252d</div><span>${{fmt(m.chg['252'])}}</span></div>
    </div>
    <div style="font-size:9px;color:var(--muted);margin-top:10px;">Spot: ${{m.spot_src}} (${{m.spot_date}})</div>
  `;
  bcEl.appendChild(card);
}}

// ── PRICE CHART (indexed to 100 -- % return since 1yr ago, NOT dollar price;
//    gold/silver/platinum trade on very different dollar scales so raw prices
//    can't share one axis) ────────────────────────────────────────────────
const chartTraces = METAL_ORDER.map(name => {{
  const m = D.metals[name];
  const base = m.chart_prices[0];
  return {{
    x: m.chart_dates,
    y: m.chart_prices.map(p => p/base*100),
    customdata: m.chart_prices.map(p => [p, p/base*100 - 100]),
    type: 'scatter', mode: 'lines', name: name,
    line: {{color: m.color, width: 2}},
    hovertemplate: '<b>'+name+'</b> · %{{x}}<br>'+
                   '$%{{customdata[0]:,.2f}} (native units)<br>'+
                   '%{{customdata[1]:+.1f}}% since 1yr ago<extra></extra>',
  }};
}});
Plotly.newPlot('chart-prices', chartTraces, {{
  margin:{{t:10,b:40,l:50,r:20}},
  paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
  legend:{{orientation:'h', y:-0.15}},
  yaxis:{{title:'Cumulative return since 1yr ago (start = 100, i.e. 150 = +50%)', gridcolor:'#eee'}},
  xaxis:{{gridcolor:'#eee'}},
}});

// ── TILT TABLE ───────────────────────────────────────────────
const tbody = document.getElementById('tilt-tbody');
for(const name of METAL_ORDER){{
  const ts = D.tilt_summaries[name];
  const m = D.metals[name];
  const delta = ts.overall_delta;
  const deltaStr = delta===0?'-':(delta>0?'+':'')+delta.toFixed(1)+'pp';
  const deltaColor = delta>0?'#1a6b3a':delta<0?'#E05555':'#888';
  const hl = ts.horizon_labels;
  const barW = Math.min(100, ts.suggested_w/70*100);
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>
      <div class="ac-cell">
        <div class="ac-dot" style="background:${{m.color}}"></div>
        <div>
          <div class="ac-name">${{m.icon}} ${{name}}</div>
          <div class="ac-desc">${{m.desc}}</div>
        </div>
      </div>
    </td>
    <td>${{tiltPill(hl['21'])}}</td>
    <td>${{tiltPill(hl['63'])}}</td>
    <td>${{tiltPill(hl['126'])}}</td>
    <td>${{tiltPill(hl['252'])}}</td>
    <td>${{tiltPill(ts.overall_label)}}</td>
    <td style="font-family:var(--mono);font-size:12px;color:var(--muted)">${{ts.neutral_w.toFixed(1)}}%</td>
    <td>
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="font-family:var(--mono);font-size:13px;font-weight:600;color:${{m.color}}">${{ts.suggested_w.toFixed(1)}}%</span>
        <div class="weight-bar-bg"><div class="weight-bar-fill" style="width:${{barW}}%;background:${{m.color}}"></div></div>
      </div>
    </td>
    <td style="font-family:var(--mono);font-size:12px;font-weight:600;color:${{deltaColor}}">${{deltaStr}}</td>`;
  tbody.appendChild(tr);
}}

// ── WEIGHT DONUT ─────────────────────────────────────────────
Plotly.newPlot('chart-weights',[{{
  type:'pie', hole:0.52,
  values: METAL_ORDER.map(n => D.suggested_weights[n] || 0),
  labels: METAL_ORDER.map(n => n + ' ' + (D.suggested_weights[n]||0).toFixed(1)+'%'),
  marker:{{colors: METAL_ORDER.map(n => D.metals[n].color)}},
  textinfo:'label',
  hovertemplate:'%{{label}}<br>Suggested: %{{value:.1f}}%<extra></extra>',
  sort:false,
}}],{{
  title:{{text:'Suggested Sleeve Weights',font:{{size:13}},x:0.5}},
  showlegend:false, margin:{{t:40,b:10,l:10,r:10}},
  paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
}});

// ── WEIGHT BARS ───────────────────────────────────────────────
const wbEl = document.getElementById('weight-bars');
for(const name of METAL_ORDER){{
  const nw = D.neutral_weights[name] || 0;
  const sw = D.suggested_weights[name] || 0;
  const color = D.metals[name].color;
  const maxW = 70;
  const div = document.createElement('div');
  div.className = 'wt-row';
  div.innerHTML = `
    <div class="wt-ac"><span>${{D.metals[name].icon}}</span> ${{name}}</div>
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
for(const name of METAL_ORDER){{
  const color = D.metals[name].color;
  for(const hor of D.horizons){{
    const h = D.horizons_by_metal[name][hor];
    if(!h) continue;
    const sigs = [...(h.bull_top||[]), ...(h.bear_top||[])];
    if(sigs.length === 0) continue;
    const card = document.createElement('div');
    card.className = 'signal-card';
    let rows = sigs.slice(0,5).map(s => `
      <div class="sig-row">
        <span>
          <span style="color:${{color}};font-weight:600">${{s.Y}}</span>
          <span class="sig-pred"> &larr; ${{s.X}} (&tau;=${{s.tau_past}}d, q=${{s.q_X}})</span>
        </span>
        <span>
          <span class="sig-cpe">CPE ${{s.CPE.toFixed(2)}}</span>
          <span style="color:var(--muted)"> &middot; </span>
          <span class="sig-lift">${{s.lift.toFixed(2)}}x</span>
          <span style="color:var(--muted);font-size:9px"> n=${{s.n_condition}}</span>
        </span>
      </div>`).join('');
    const tilt = h.tilt_label;
    const t = TC[tilt] || TC["NO SIGNAL"];
    card.innerHTML = `
      <div class="sig-header">
        <span class="sig-ac" style="color:${{color}}">${{D.metals[name].icon}} ${{name}}</span>
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

outfile = os.path.join(BASE_DIR, "precious_metals_dashboard.html")
with open(outfile, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nSaved: {outfile}  ({len(html)//1024} KB)")
print("Open precious_metals_dashboard.html in Chrome/Firefox.")
print("Re-run this script to refresh.")
