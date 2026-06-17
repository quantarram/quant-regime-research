"""
Gold CPE Buy Signal Dashboard — Enhanced
=========================================
Comprehensive gold buy/sell signal dashboard using:
  1. Current drawdown context vs historical distribution
  2. Forward return distributions after similar drawdowns
  3. All CPE predictors for gold (pairwise + joint)
  4. Gold autocorrelation CPE (mean-reversion signal)
  5. Composite buy score
  6. Gold spot price cone (SGD per gram)

Run: python build_gold_dashboard.py
Requires: multiasset_prices.parquet, joint_cpe_results.parquet, cpe_results.parquet
"""

import pandas as pd
import numpy as np
import json, os, warnings
from datetime import datetime
warnings.filterwarnings("ignore")

print("="*60)
print("  GOLD BUY SIGNAL DASHBOARD BUILDER")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

# ── LOAD & REFRESH DATA ───────────────────────────────────────────────────────
print("\nLoading local data...")
prices = pd.read_parquet("multiasset_prices.parquet")
joint  = pd.read_parquet("joint_cpe_results.parquet")
pair   = pd.read_parquet("cpe_results.parquet")

GOLD_Y    = ["GLD","IAU","GC=F"]
FX_TICKER = "SGDUSD=X"
RATE_TICKERS = {"^VIX","^VXN","^OVX","^EVZ","^VVIX","^SKEW",
                "^TNX","^TYX","^FVX","^IRX"}
# Note: ^GVZ (Gold Vol) and DX-Y.NYB (DXY) use log returns like price series
# ^VIX family stays as level changes since they are pure vol levels
price_tickers = [t for t in prices.columns if t not in RATE_TICKERS]
rate_tickers  = [t for t in prices.columns if t in RATE_TICKERS]

print("Fetching latest prices from Yahoo Finance...")
try:
    import yfinance as yf
    fetch_list = list(set(GOLD_Y + [FX_TICKER,"SLV","SI=F","IBIT","FBTC","BITB",
                                    "SGDUSD=X","SOXX","XLK","QQQ","VUG","EWY","XAUUSD=X","GLD","IAU",
                                    "DX-Y.NYB","UUP","^GVZ"]))  # DXY index, USD ETF backup, Gold Vol Index
    raw = yf.download(fetch_list, period="400d", auto_adjust=True, progress=False)["Close"]
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    for col in raw.columns:
        if col in prices.columns:
            # For existing columns: append new rows after parquet max date
            new = raw[[col]].loc[raw.index > prices.index.max()]
            if not new.empty:
                prices = pd.concat([prices, new])
            # Also fill NaN values in existing column with fresh data
            fresh_col = raw[[col]].reindex(prices.index)
            prices[col] = prices[col].fillna(fresh_col[col])
        else:
            # For new tickers (e.g. GLD, UUP, ^GVZ not in original parquet):
            # add the full fresh series
            prices = prices.join(raw[[col]], how='left', rsuffix='_new')
            if col+'_new' in prices.columns:
                prices[col] = prices.get(col, float('nan'))
                prices[col] = prices[col].fillna(prices[col+'_new'])
                prices = prices.drop(columns=[col+'_new'], errors='ignore')
            else:
                prices[col] = raw[col].reindex(prices.index)
    prices = prices.sort_index().loc[~prices.index.duplicated(keep="last")]
    print(f"  Latest date: {prices.index.max().date()}")
    # ── STALENESS WARNINGS ──
    from datetime import date as _date
    _today = _date.today()
    _latest = prices.index.max().date()
    _age = (_today - _latest).days
    if _age > 4:
        print(f"  WARNING: latest data is {_age} days old ({_latest}). Yahoo may be lagging or markets closed.")
    for _t in ["GLD","GC=F","SGDUSD=X"]:
        if _t in prices.columns:
            _last_valid = prices[_t].dropna()
            if len(_last_valid):
                _tage = (_latest - _last_valid.index[-1].date()).days
                if _tage > 1:
                    print(f"  WARNING: {_t} last valid {_last_valid.index[-1].date()} ({_tage}d behind latest). Price may be stale.")
            else:
                print(f"  WARNING: {_t} has NO valid data after merge.")
except Exception as e:
    print(f"  yfinance error: {e}")

latest_date = prices.index.max()
TAU_LIST = [1,5,10,21,63,126,252,300]
Q_GRID   = [0.50,0.60,0.70,0.75,0.80,0.90,0.95,0.99]

# ── INCREMENTS & THRESHOLDS ───────────────────────────────────────────────────
print("Computing increments and thresholds...")
current_inc = {}
for tau in TAU_LIST:
    row = {}
    for t in price_tickers:
        s = prices[t].dropna()
        if len(s) >= tau+1:
            row[t] = float(np.log(s.iloc[-1]/s.iloc[-1-tau]))
    for t in rate_tickers:
        s = prices[t].dropna()
        if len(s) >= tau+1:
            row[t] = float(s.iloc[-1] - s.iloc[-1-tau])
    current_inc[tau] = row

full_q = sorted(set(Q_GRID + [round(1-q,10) for q in Q_GRID]))
thresholds = {}
inc_series_cache = {}
for tau in TAU_LIST:
    idf = pd.DataFrame(index=prices.index)
    for t in price_tickers:
        s = prices[t]; idf[t] = np.log(s/s.shift(tau))
    for t in rate_tickers:
        s = prices[t]; idf[t] = s - s.shift(tau)
    inc_series_cache[tau] = idf
    for q in full_q:
        thresholds[(tau,q)] = idf.quantile(q, numeric_only=True).to_dict()

def fires(predictors, tau_pasts, q_Xs, direction):
    for x,tp,qx in zip(predictors,tau_pasts,q_Xs):
        tp=int(tp); qx=float(qx)
        curr = current_inc.get(tp,{}).get(x)
        if curr is None or np.isnan(curr): return False
        if direction == "bullish":
            th = thresholds.get((tp,qx),{}).get(x,float("nan"))
            if np.isnan(th) or curr<=th: return False
        else:
            th = thresholds.get((tp,round(1-qx,10)),{}).get(x,float("nan"))
            if np.isnan(th) or curr>=th: return False
    return True

# ── GOLD PRICE STATS ──────────────────────────────────────────────────────────
print("Computing gold price stats...")
gcf = prices["GC=F"].dropna()
sgd_fx = prices[FX_TICKER].dropna()
usd_per_sgd = 1.0 / float(sgd_fx.reindex(gcf.index).ffill().iloc[-1])

# ── GOLD SPOT PRICE: use GLD ETF as primary (most reliable, never rolls)
# GLD tracks spot gold: 1 share = 0.0963 troy oz (NAV ratio)
# Convert GLD price to USD/oz by dividing by 0.0963
# Dynamically calibrate GLD-to-gold ratio using overlapping GC=F/GLD data
# This avoids hardcoding a ratio that drifts over time due to GLD management fees
if "GLD" in prices.columns:
    gld_series = prices["GLD"].dropna()
    if len(gld_series) > 30:
        # Find dates where both GC=F and GLD have data
        common_idx = gcf.index.intersection(gld_series.index)
        if len(common_idx) >= 30:
            # Compute ratio over last 30 overlapping days
            gcf_common = gcf.reindex(common_idx).dropna()
            gld_common = gld_series.reindex(common_idx).dropna()
            shared = gcf_common.index.intersection(gld_common.index)
            GLD_OZ_RATIO = float((gld_common.reindex(shared) / gcf_common.reindex(shared)).tail(30).median())
            print(f"  GLD/GCF calibrated ratio: {GLD_OZ_RATIO:.5f} oz/share (last 30 days)")
        else:
            GLD_OZ_RATIO = 0.0861  # fallback from June 2026 calibration
            print(f"  Using fallback GLD ratio: {GLD_OZ_RATIO:.5f}")

        gold_spot_usd = float(gld_series.iloc[-1]) / GLD_OZ_RATIO
        gold_usd = gold_spot_usd
        print(f"  Using GLD ETF: ${float(gld_series.iloc[-1]):.2f}/share -> ${gold_usd:.2f}/oz")
        # Extend gcf with GLD-derived prices for latest dates where GC=F is stale
        gld_as_gcf = gld_series / GLD_OZ_RATIO
        gcf = gcf.combine_first(gld_as_gcf).dropna().sort_index()
    else:
        GLD_OZ_RATIO = 0.0861
        gold_spot_usd = float(gcf.iloc[-1]) - 12.0
        gold_usd = float(gcf.iloc[-1])
        print(f"  GLD insufficient data, using GC=F: ${gold_usd:.2f}/oz")
elif "XAUUSD=X" in prices.columns:
    spot_series = prices["XAUUSD=X"].dropna()
    if len(spot_series) > 0:
        gold_spot_usd = float(spot_series.iloc[-1])
        gold_usd = gold_spot_usd
        print(f"  Using XAUUSD=X spot: ${gold_spot_usd:.2f}/oz")
    else:
        gold_spot_usd = float(gcf.iloc[-1]) - 12.0
        gold_usd = float(gcf.iloc[-1])
        print(f"  Fallback GC=F - $12: ${gold_usd:.2f}/oz")
else:
    gold_spot_usd = float(gcf.iloc[-1]) - 12.0
    gold_usd = float(gcf.iloc[-1])
    print(f"  Fallback GC=F - $12: ${gold_usd:.2f}/oz")
gold_sgd_oz = gold_usd * usd_per_sgd
gold_sgd_g  = gold_sgd_oz / 31.1035

# Spot price in SGD per gram — verifiable against published spot rates
gold_spot_sgd_oz = gold_spot_usd * usd_per_sgd
gold_spot_sgd_g  = gold_spot_sgd_oz / 31.1035
print(f"  Spot SGD/g: S${gold_spot_sgd_g:.2f} | SGD/oz: S${gold_spot_sgd_oz:.2f}")
# Always define these for the data bundle
# US market closes at 4 PM New York = 4 AM next day Singapore time
# So when running in Singapore morning, the last US close was YESTERDAY Singapore date
from datetime import timedelta
_sgt_hour = (datetime.utcnow().hour + 8) % 24
_us_close_label = (datetime.now() - timedelta(days=1)).strftime('%d %b %Y') if _sgt_hour < 4 else datetime.now().strftime('%d %b %Y')
# Before 4 AM SGT: US market still open, so close was 2 days ago SGT
if _sgt_hour < 4:
    _us_close_label = (datetime.now() - timedelta(days=2)).strftime('%d %b %Y')
elif _sgt_hour < 16:  # 4 AM to 4 PM SGT: last close was yesterday SGT
    _us_close_label = (datetime.now() - timedelta(days=1)).strftime('%d %b %Y')
else:  # after 4 PM SGT: today's US session still open, last close was yesterday
    _us_close_label = (datetime.now() - timedelta(days=1)).strftime('%d %b %Y')
bar_sub_text = (f"Gold spot price in Singapore dollars per gram · "
                f"Last US close: {_us_close_label} · "
                f"Derived from GLD ETF and USD/SGD rate.")
peak_252 = float(gcf.iloc[-252:].max()) if len(gcf) >= 252 else float(gcf.max())
dd_from_peak = round((gold_usd / peak_252 - 1) * 100, 2)
peak_spot_sgd_g = round(peak_252 * usd_per_sgd / 31.1035, 2)

def pct_chg(n):
    if len(gcf) > n:
        return float((gcf.iloc[-1]/gcf.iloc[-1-n]-1)*100)
    return 0.0

chg = {t: round(pct_chg(t),2) for t in [1,5,10,21,63,126,252]}

# Historical distribution of 63d and 126d returns for GC=F
gcf_63  = np.log(gcf/gcf.shift(63)).dropna().values * 100
gcf_126 = np.log(gcf/gcf.shift(126)).dropna().values * 100
gcf_252 = np.log(gcf/gcf.shift(252)).dropna().values * 100

curr_pct_63  = float(np.mean(gcf_63  <= chg[63]))  * 100
curr_pct_126 = float(np.mean(gcf_126 <= chg[126])) * 100

# ── RECOVERY ANALYSIS ─────────────────────────────────────────────────────────
print("Computing historical recovery analysis...")

def recovery_dist(lookback_tau, threshold_pct, forward_taus):
    """
    Find dates when GC=F had a lookback_tau-day return <= threshold_pct,
    then compute forward return distributions.
    """
    lb_returns = np.log(gcf/gcf.shift(lookback_tau)).dropna() * 100
    trigger_dates = lb_returns[lb_returns <= threshold_pct].index
    results = {}
    for fwd_tau in forward_taus:
        fwd_ret = np.log(gcf/gcf.shift(-fwd_tau)).dropna() * 100
        vals = fwd_ret.reindex(trigger_dates).dropna().values
        if len(vals) >= 5:
            results[fwd_tau] = {
                "n": int(len(vals)),
                "p10": round(float(np.percentile(vals,10)),2),
                "p25": round(float(np.percentile(vals,25)),2),
                "p50": round(float(np.percentile(vals,50)),2),
                "p75": round(float(np.percentile(vals,75)),2),
                "p90": round(float(np.percentile(vals,90)),2),
                "pct_positive": round(float(np.mean(vals>0))*100,1),
                "mean": round(float(np.mean(vals)),2),
            }
    return results

FWD_TAUS = [21,63,126,252]

# After 63d drawdown >= current
recovery_63 = recovery_dist(63, chg[63], FWD_TAUS)
# After 126d drawdown >= current
recovery_126 = recovery_dist(126, chg[126], FWD_TAUS)
# After any bottom decile drawdown (10th percentile)
p10_63 = float(np.percentile(gcf_63, 10))
recovery_extreme = recovery_dist(63, p10_63, FWD_TAUS)

print(f"  Recovery dates (>=current 63d draw): {sum(r['n'] for r in recovery_63.values() if r)//len(FWD_TAUS) if recovery_63 else 0}")

# ── GOLD AUTOCORRELATION CPE ──────────────────────────────────────────────────
print("Computing gold autocorrelation CPE...")
auto_cpe = {}
for tp in [21, 63, 126, 252]:
    lb = np.log(gcf/gcf.shift(tp)).dropna() * 100
    # Current return percentile
    curr_ret = current_inc.get(tp,{}).get("GC=F", None)
    if curr_ret is None: continue
    curr_ret_pct = curr_ret * 100
    q_now = float(np.mean(lb.values <= curr_ret_pct))
    auto_cpe[tp] = {"current_return_pct": round(curr_ret_pct,2),
                    "current_percentile": round(q_now*100,1)}
    for fwd in [21,63,126,252]:
        fwd_ret = np.log(gcf/gcf.shift(-fwd)).dropna() * 100
        # CPE: P(forward > 0 | past <= current percentile)
        past_below = lb[lb <= curr_ret_pct].index
        fwd_at_past = fwd_ret.reindex(past_below).dropna()
        if len(fwd_at_past) >= 20:
            cpe_up = float(np.mean(fwd_at_past > 0))
            cpe_large_up = float(np.mean(fwd_at_past > 5))
            auto_cpe[tp][f"fwd_{fwd}_pct_positive"] = round(cpe_up*100,1)
            auto_cpe[tp][f"fwd_{fwd}_pct_up5pct"] = round(cpe_large_up*100,1)
            auto_cpe[tp][f"fwd_{fwd}_n"] = int(len(fwd_at_past))
            auto_cpe[tp][f"fwd_{fwd}_median"] = round(float(fwd_at_past.median()),2)

# ── JOINT CPE SIGNALS FOR GOLD ───────────────────────────────────────────────
print("Computing joint CPE signals for gold...")
gold_joint = joint[joint["Y"].isin(GOLD_Y)].copy()
gold_joint = gold_joint[gold_joint["n_predictors"] <= 6].copy()

signal_rows = []
for _, row in gold_joint.iterrows():
    w = float(row["joint_CPE"]) * float(row["lift"]) * np.log(max(row["n_joint"],1))
    firing = fires(row["predictors"],row["tau_pasts"],row["q_Xs"],row["direction"])
    signal_rows.append({
        "Y":row["Y"],"direction":row["direction"],
        "tau_future":int(row["tau_future"]),"q_Y":float(row["q_Y"]),
        "n_predictors":int(row["n_predictors"]),
        "joint_CPE":float(row["joint_CPE"]),"lift":float(row["lift"]),
        "n_joint":int(row["n_joint"]),"weight":round(w,3),
        "firing":bool(firing),
        "predictors":list(row["predictors"]),
        "tau_pasts":[int(x) for x in row["tau_pasts"]],
        "q_Xs":[float(x) for x in row["q_Xs"]],
        "pred_str":" ∩ ".join([f"{x}(τ={tp},q={qx})"
                                for x,tp,qx in zip(row["predictors"],
                                                    row["tau_pasts"],
                                                    row["q_Xs"])]),
    })

# ── CPE SCORES PER HORIZON ────────────────────────────────────────────────────
scores = {}
for (y,tf), grp in pd.DataFrame(signal_rows).groupby(["Y","tau_future"]):
    bull = grp[grp["direction"]=="bullish"]
    bear = grp[grp["direction"]=="bearish"]
    tw   = bull["weight"].sum() + bear["weight"].sum()
    fb   = bull[bull["firing"]]["weight"].sum()
    fbr  = bear[bear["firing"]]["weight"].sum()
    sc   = (fb-fbr)/tw if tw>0 else 0
    scores[f"{y}_{tf}"] = {"score":round(sc,4),
                            "fired_bull":int(bull["firing"].sum()),
                            "fired_bear":int(bear["firing"].sum()),
                            "total_bull":len(bull),"total_bear":len(bear)}

# ── PREDICTOR PROXIMITY ───────────────────────────────────────────────────────
print("Computing predictor proximity to thresholds...")
KEY_PREDS = {
    "IBIT":      [(1,0.5),(5,0.5),(252,0.5),(126,0.6)],
    "FBTC":      [(1,0.5),(5,0.5),(252,0.5),(126,0.6)],
    "BITB":      [(1,0.5),(252,0.5),(126,0.6)],
    "SLV":       [(252,0.95),(300,0.95),(252,0.8)],
    "SI=F":      [(252,0.95),(300,0.95),(252,0.8)],
    "SGDUSD=X":  [(300,0.9),(252,0.9)],
    "GC=F":      [(63,0.10),(126,0.10),(252,0.10)],   # mean-reversion: lower tail
    # NEW: USD and Gold Volatility
    "DX-Y.NYB":  [(63,0.10),(252,0.10),(126,0.10)],   # weak USD (lower tail) = bullish gold
    "UUP":       [(63,0.10),(252,0.10),(126,0.10)],   # USD ETF backup — same signal
    "^GVZ":      [(21,0.10),(63,0.10)],                # falling vol (lower tail) = stabilising
}

pred_proximity = {}
for ticker, params in KEY_PREDS.items():
    if ticker not in prices.columns: continue
    rows = []
    for (tau,q) in params:
        curr = current_inc.get(tau,{}).get(ticker)
        if curr is None: continue
        # Lower tail signals: GC=F (mean-reversion), DXY (weak USD = bull gold),
        # GVZ (falling volatility = stabilising)
        is_lower = q <= 0.20  # q<=0.20 means lower tail condition
        if is_lower:
            # lower tail: condition fires when curr < q-th percentile (e.g. below 10th pct)
            # Use thresholds[(tau, q)] directly — the LOW end threshold
            th = thresholds.get((tau, q),{}).get(ticker, float("nan"))
            if np.isnan(th): continue
            in_tail = bool(curr < th)
            # dist_pct: positive means curr is above threshold (not yet in lower tail)
            dist_pct = (curr - th) / abs(th) * 100 if th != 0 else 0
            rows.append({"tau":tau,"q":q,"current":round(curr*100,3),
                         "threshold":round(th*100,3),
                         "in_tail":in_tail,"tail_type":"lower",
                         "dist_pct":round(dist_pct,2),
                         "proximity_score": max(0, min(100, 100*(1-abs(dist_pct)/100)))})
        else:
            th = thresholds.get((tau,q),{}).get(ticker,float("nan"))
            if np.isnan(th): continue
            in_tail = bool(curr > th)
            dist_pct = (curr - th) / abs(th) * 100 if th != 0 else 0
            rows.append({"tau":tau,"q":q,"current":round(curr*100,3),
                         "threshold":round(th*100,3),
                         "in_tail":in_tail,"tail_type":"upper",
                         "dist_pct":round(dist_pct,2),
                         "proximity_score": max(0, min(100, 100*(1-abs(dist_pct)/100) if not in_tail else 100))})
    if rows:
        pred_proximity[ticker] = rows

# ── COMPOSITE BUY SCORE ───────────────────────────────────────────────────────
print("Computing composite buy score...")

# Components (all 0-100):
# 1. Drawdown depth score — deeper drawdown = higher mean-reversion potential
draw_score = min(100, max(0, (-chg[63] / 20) * 100))  # -20% = 100, 0% = 0

# 2. Autocorrelation CPE score — % of time gold recovers after similar drawdown
auto_score = 0
if 63 in auto_cpe and "fwd_126_pct_positive" in auto_cpe[63]:
    auto_score = float(auto_cpe[63]["fwd_126_pct_positive"])

# 3. CPE predictor proximity score — how close are bull predictors to firing
prox_scores = []
for ticker in ["IBIT","FBTC","SLV","SI=F","DX-Y.NYB","UUP","^GVZ"]:
    rows = pred_proximity.get(ticker,[])
    for r in rows:
        # For lower tail predictors (DXY, GVZ, GC=F): in_tail IS the bull signal
        if r["tail_type"]=="lower":
            prox_scores.append(r["proximity_score"])
        elif r["tail_type"]=="upper":
            prox_scores.append(r["proximity_score"])
prox_score = float(np.mean(prox_scores)) if prox_scores else 50

# 4. Joint CPE signal score (normalised to 0-100)
gcf_252_score = scores.get("GC=F_252",{}).get("score",0)
cpe_score = max(0, min(100, (gcf_252_score + 1) / 2 * 100))

# Weighted composite
composite = round(
    0.35 * draw_score +
    0.35 * auto_score +
    0.20 * prox_score +
    0.10 * cpe_score, 1)

buy_label = ("STRONG BUY ZONE" if composite >= 70 else
             "BUY ZONE" if composite >= 55 else
             "WATCH — APPROACHING BUY" if composite >= 40 else
             "NEUTRAL — WAIT" if composite >= 25 else
             "NOT YET")

components = {
    "draw_score":  round(draw_score,1),
    "auto_score":  round(auto_score,1),
    "prox_score":  round(prox_score,1),
    "cpe_score":   round(cpe_score,1),
    "composite":   composite,
    "label":       buy_label,
}

print(f"  Composite buy score: {composite} — {buy_label}")

# ── PRICE CHART DATA ──────────────────────────────────────────────────────────
chart_dates  = [str(d.date()) for d in gcf.index[-365:]]
chart_prices = [round(float(p),2) for p in gcf.iloc[-365:]]
chart_sgd_g  = [round(float(p)*usd_per_sgd/31.1035,2) for p in gcf.iloc[-365:]]

# 52-week high and peak drawdown (already computed above)
peak_date_idx= gcf.iloc[-252:].idxmax() if len(gcf)>=252 else gcf.idxmax()

# ── FORWARD CONE ─────────────────────────────────────────────────────────────
# Use the recovery_63 distribution to project forward
cone_taus    = sorted(recovery_63.keys())
cone_p10     = [gold_usd * (1+recovery_63[t]["p10"]/100) for t in cone_taus]
cone_p25     = [gold_usd * (1+recovery_63[t]["p25"]/100) for t in cone_taus]
cone_p50     = [gold_usd * (1+recovery_63[t]["p50"]/100) for t in cone_taus]
cone_p75     = [gold_usd * (1+recovery_63[t]["p75"]/100) for t in cone_taus]
cone_p90     = [gold_usd * (1+recovery_63[t]["p90"]/100) for t in cone_taus]
# Cone projections in SGD per gram (spot)
cone_sgd_p10 = [p*usd_per_sgd/31.1035 for p in cone_p10]
cone_sgd_p25 = [p*usd_per_sgd/31.1035 for p in cone_p25]
cone_sgd_p50 = [p*usd_per_sgd/31.1035 for p in cone_p50]
cone_sgd_p75 = [p*usd_per_sgd/31.1035 for p in cone_p75]
cone_sgd_p90 = [p*usd_per_sgd/31.1035 for p in cone_p90]

# Historical return distribution for histogram
hist_63_vals = [round(float(v),2) for v in gcf_63]

# ── PAIRWISE SIGNALS ─────────────────────────────────────────────────────────
gold_pw = pair[pair["Y"].isin(GOLD_Y)].sort_values("CPE",ascending=False).head(60)

# ── DATA BUNDLE ───────────────────────────────────────────────────────────────
data = {
    "generated":     datetime.now().strftime("%Y-%m-%d %H:%M"),
    "latest_date":   str(latest_date.date()),
    "gold_usd":      round(gold_usd,2),
    "gold_sgd_oz":   round(gold_sgd_oz,2),
    "gold_sgd_g":    round(gold_sgd_g,4),
    "spot_sgd_g":    round(gold_spot_sgd_g,2),
    "spot_sgd_oz":   round(gold_spot_sgd_oz,2),
    "usd_per_sgd":   round(usd_per_sgd,4),
    "chg":              chg,
    "dd_from_peak":     dd_from_peak,
    "peak_252":         round(peak_252,2),
    "peak_spot_sgd_g":     peak_spot_sgd_g,
    "gold_spot_usd":    round(gold_spot_usd,2),
    "gold_spot_sgd_g":  round(gold_spot_sgd_g,4),
    "spot_sgd_g":       round(gold_spot_sgd_g,2),
    "spot_sgd_oz":      round(gold_spot_sgd_oz,2),
    "bar_sub_text":     bar_sub_text,
    "high_252":      round(float(gcf.iloc[-252:].max()) if len(gcf)>=252 else gold_usd,2),
    "low_252":       round(float(gcf.iloc[-252:].min()) if len(gcf)>=252 else gold_usd,2),
    "curr_pct_63":   round(curr_pct_63,1),
    "curr_pct_126":  round(curr_pct_126,1),
    "recovery_63":   {str(k):v for k,v in recovery_63.items()},
    "recovery_126":  {str(k):v for k,v in recovery_126.items()},
    "recovery_extreme": {str(k):v for k,v in recovery_extreme.items()},
    "auto_cpe":      {str(k):v for k,v in auto_cpe.items()},
    "components":    components,
    "scores":        scores,
    "signals":       signal_rows,
    "pred_proximity":pred_proximity,
    "chart_dates":   chart_dates,
    "chart_prices":  chart_prices,
    "chart_sgd_g":   chart_sgd_g,
    "cone_taus":     cone_taus,
    "cone_p10":      [round(v,2) for v in cone_p10],
    "cone_p25":      [round(v,2) for v in cone_p25],
    "cone_p50":      [round(v,2) for v in cone_p50],
    "cone_p75":      [round(v,2) for v in cone_p75],
    "cone_p90":      [round(v,2) for v in cone_p90],
    "cone_sgd_p50":  [round(v,2) for v in cone_sgd_p50],
    "cone_sgd_p25":  [round(v,2) for v in cone_sgd_p25],
    "cone_sgd_p75":  [round(v,2) for v in cone_sgd_p75],
    "cone_sgd_p10":  [round(v,2) for v in cone_sgd_p10],
    "cone_sgd_p90":  [round(v,2) for v in cone_sgd_p90],
    "hist_63_vals":  hist_63_vals,
    "pair_signals":  gold_pw[["Y","X","direction","tau_past","tau_future",
                               "q_X","q_Y","CPE","lift","n_condition"]].to_dict("records"),
}

data_json = json.dumps(data, allow_nan=False)
print(f"  Data bundle: {len(data_json)/1e3:.1f} KB")

# ── HTML ──────────────────────────────────────────────────────────────────────
# ── SIMPLE BANNER: computed in Python, injected as static text ───────────────
_days_old = (datetime.now() - latest_date.to_pydatetime().replace(tzinfo=None)).days
_sgt_h = (datetime.utcnow().hour + 8) % 24
_wday = datetime.now().weekday()  # 0=Mon, 6=Sun
if _days_old >= 1 and (_wday >= 5 or _sgt_h < 21):
    _banner = ("<div style=\"background:#1C1A0E;border:1px solid #4A4020;"
               "border-radius:8px;padding:14px 24px;font-family:sans-serif;"
               "font-size:13px;color:#C8A84A;text-align:center;"
               "margin-bottom:20px;line-height:1.7\">"
               "Gold markets are currently closed. Showing last available close: "
               + str(latest_date.date()) +
               ". In Singapore, fresh prices are available after 9:30 PM on trading days."
               "</div>")
else:
    _banner = ""

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gold Buy Signal Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --gold:#C9A84C; --gold2:#E8C97A;
  --bg:#0C0E0D; --s1:#141614; --s2:#1B1E1B; --s3:#222522;
  --bdr:#2C302C; --bdr2:#3A3F3A;
  --bull:#4DB87A; --bear:#E05555; --warn:#E8A020; --neut:#6B7D6B;
  --text:#DDE8DD; --text2:#7A8F7A; --text3:#4A5A4A;
  --mono:'IBM Plex Mono',monospace; --sans:'IBM Plex Sans',sans-serif;
  --r:10px; --r-sm:6px;
  --gap:16px; --pad:20px;
}
*{box-sizing:border-box;margin:0;padding:0;}
html{font-size:15px;}
body{background:var(--bg);color:var(--text);font-family:var(--sans);
     font-size:14px;line-height:1.6;min-height:100vh;}

/* ── HEADER ── */
header{
  background:linear-gradient(160deg,#0C0E0D,#161A14,#0C0E0D);
  border-bottom:1px solid var(--bdr);
  padding:18px var(--pad);
  display:flex;align-items:center;justify-content:space-between;
  gap:12px;flex-wrap:wrap;
}
.h-brand{display:flex;align-items:center;gap:12px;}
.h-icon{width:40px;height:40px;border-radius:10px;flex-shrink:0;
  background:linear-gradient(135deg,#8B6914,#C9A84C,#E8C97A);
  display:flex;align-items:center;justify-content:center;font-size:18px;}
.h-title{font-family:var(--mono);font-size:clamp(13px,2vw,17px);
         font-weight:600;color:var(--gold);}
.h-sub{font-family:var(--mono);font-size:clamp(9px,1.5vw,11px);
       color:var(--text2);letter-spacing:.06em;text-transform:uppercase;margin-top:2px;}
.h-meta{font-family:var(--mono);font-size:11px;color:var(--text2);
        text-align:right;line-height:1.8;flex-shrink:0;}

/* ── MAIN ── */
main{padding:var(--pad);max-width:1540px;margin:0 auto;}

/* ── CARD ── */
.card{background:var(--s1);border:1px solid var(--bdr);
      border-radius:var(--r);padding:var(--pad);margin-bottom:var(--gap);}
.ct{font-family:var(--mono);font-size:10px;font-weight:600;color:var(--gold);
    text-transform:uppercase;letter-spacing:.1em;margin-bottom:14px;
    display:flex;align-items:center;gap:8px;}
.ct::before{content:'';width:2px;height:12px;
            background:var(--gold);border-radius:1px;flex-shrink:0;}

/* ── BUY HERO ── */
.buy-hero{
  background:linear-gradient(135deg,#141614,#1B1E1B);
  border:1px solid var(--bdr2);border-radius:var(--r);
  padding:var(--pad);margin-bottom:var(--gap);
  display:grid;
  grid-template-columns:1fr auto 1fr;
  gap:24px;align-items:start;
}
.buy-score-ring{position:relative;width:140px;height:140px;margin:0 auto;}
.buy-score-ring svg{width:100%;height:100%;}
.buy-score-inner{position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;}
.buy-num{font-family:var(--mono);font-size:32px;font-weight:600;}
.buy-label-small{font-family:var(--mono);font-size:9px;color:var(--text2);
  text-transform:uppercase;letter-spacing:.1em;margin-top:2px;}
.buy-verdict{font-family:var(--mono);font-size:14px;font-weight:600;
  text-align:center;margin-top:8px;letter-spacing:.03em;}
.component-row{display:flex;flex-direction:column;gap:8px;}
.comp-item{background:var(--s3);border-radius:var(--r-sm);padding:10px 14px;}
.comp-label{font-family:var(--mono);font-size:9px;color:var(--text2);
  text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px;}
.comp-bar-wrap{height:5px;background:var(--bdr);border-radius:3px;overflow:hidden;}
.comp-bar{height:100%;border-radius:3px;}

/* ── DECISION PANEL ── */
.decision-panel{
  background:linear-gradient(135deg,#141614,#1a1e18);
  border:2px solid var(--bdr2);border-radius:var(--r);
  padding:var(--pad);margin-bottom:var(--gap);
}
.decision-verdict{display:flex;align-items:flex-start;gap:16px;margin-bottom:20px;flex-wrap:wrap;}
.verdict-icon{width:56px;height:56px;border-radius:50%;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;font-size:22px;}
.verdict-title{font-family:var(--mono);font-size:clamp(18px,3vw,24px);
  font-weight:600;letter-spacing:.02em;}
.verdict-sub{font-family:var(--mono);font-size:12px;color:var(--text2);margin-top:4px;}
.decision-body{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:var(--gap);
}
.dec-col h4{
  font-family:var(--mono);font-size:10px;font-weight:600;
  text-transform:uppercase;letter-spacing:.1em;
  margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid var(--bdr);
}

/* ── EVIDENCE CARDS ── */
.ev-card{
  background:var(--s2);
  border-radius:var(--r-sm);
  padding:12px 14px;
  margin-bottom:8px;
  font-size:12px;
  line-height:1.7;
  color:var(--text2);
  border:1px solid var(--bdr);
  border-left-width:3px;
  transition:border-color .2s;
}
.ev-card:last-child{margin-bottom:0;}
.ev-label{
  font-family:var(--mono);font-size:9px;font-weight:600;
  text-transform:uppercase;letter-spacing:.08em;
  margin-bottom:5px;display:block;
}

/* ── TRIGGER CARDS ── */
.trigger-list{display:flex;flex-direction:column;gap:8px;}
.trigger-item{background:var(--s2);border:1px solid var(--bdr);
  border-radius:var(--r-sm);padding:10px 12px;}
.trigger-name{font-family:var(--mono);font-size:11px;font-weight:600;margin-bottom:3px;}
.trigger-status{font-family:var(--mono);font-size:10px;color:var(--text2);}
.trigger-bar-wrap{height:3px;background:var(--bdr);border-radius:2px;margin-top:6px;}
.trigger-bar-fill{height:100%;border-radius:2px;}

/* ── STATS ── */
.stats-grid{
  display:grid;
  grid-template-columns:repeat(6,1fr);
  gap:12px;margin-bottom:var(--gap);
}
.stat{background:var(--s1);border:1px solid var(--bdr);
  border-radius:var(--r-sm);padding:12px 14px;}
.stat-l{font-family:var(--mono);font-size:9px;color:var(--text2);
  text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px;}
.stat-v{font-family:var(--mono);font-size:clamp(15px,2vw,19px);font-weight:600;}
.stat-s{font-size:11px;color:var(--text2);margin-top:2px;}

/* ── GRIDS ── */
.g2{display:grid;grid-template-columns:1fr 1fr;gap:var(--gap);margin-bottom:var(--gap);}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:var(--gap);margin-bottom:var(--gap);}

/* ── TABS ── */
.tabs{display:flex;gap:0;margin-bottom:14px;border-bottom:1px solid var(--bdr);
  overflow-x:auto;-webkit-overflow-scrolling:touch;}
.tb{padding:10px 16px;font-family:var(--mono);font-size:11px;font-weight:600;
  color:var(--text2);cursor:pointer;border:none;background:none;
  border-bottom:2px solid transparent;text-transform:uppercase;
  letter-spacing:.05em;transition:all .2s;white-space:nowrap;flex-shrink:0;}
.tb:hover{color:var(--text);}
.tb.active{color:var(--gold);border-bottom-color:var(--gold);}
.tp{display:none;}.tp.active{display:block;}

/* ── TABLES ── */
.rt{width:100%;border-collapse:collapse;font-size:12px;}
.rt th{font-family:var(--mono);font-size:9px;color:var(--text2);
  text-transform:uppercase;letter-spacing:.08em;
  padding:8px 10px;border-bottom:1px solid var(--bdr);text-align:left;white-space:nowrap;}
.rt td{padding:8px 10px;border-bottom:1px solid var(--bdr);font-size:11px;}
.rt tr:hover td{background:var(--s2);}
.tbl-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;border-radius:var(--r-sm);}

/* ── PREDICTOR CARDS ── */
.pred-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;}
.pred-card{background:var(--s2);border:1px solid var(--bdr);
  border-radius:var(--r-sm);padding:14px;}
.pred-ticker{font-family:var(--mono);font-size:13px;font-weight:600;
  color:var(--gold2);margin-bottom:10px;}
.pred-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;}
.pred-p{font-family:var(--mono);font-size:9px;color:var(--text2);}
.pred-v{font-family:var(--mono);font-size:11px;font-weight:600;}
.in-bull{color:var(--bull);}.in-bear{color:var(--bear);}.near{color:var(--warn);}.far{color:var(--neut);}
.prog-wrap{height:4px;background:var(--bdr);border-radius:2px;margin:4px 0 8px;overflow:hidden;}
.prog-fill{height:100%;border-radius:2px;}

/* ── AUTO CPE ── */
.auto-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;}
.auto-card{background:var(--s2);border:1px solid var(--bdr);
  border-radius:var(--r-sm);padding:14px;text-align:center;}
.auto-tau{font-family:var(--mono);font-size:10px;color:var(--text2);
  text-transform:uppercase;margin-bottom:6px;}
.auto-pct{font-family:var(--mono);font-size:11px;color:var(--text2);}

/* ── SIGNAL LIST ── */
.sig-list{display:flex;flex-direction:column;gap:6px;}
.sig-item{background:var(--s2);border:1px solid var(--bdr);
  border-radius:var(--r-sm);padding:10px 12px;
  display:flex;gap:10px;align-items:flex-start;}
.sig-item.fb{border-color:#4DB87A44;background:#4DB87A0A;}
.sig-item.br{border-color:#E0555544;background:#E055550A;}
.sig-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;margin-top:4px;}
.sig-dot.b{background:var(--bull);}.sig-dot.r{background:var(--bear);}
.sig-dot.o{background:var(--bdr2);}
.sig-body{flex:1;min-width:0;}
.sig-preds{font-family:var(--mono);font-size:10px;color:var(--text2);
  line-height:1.7;word-break:break-word;}
.sig-meta{font-size:11px;color:var(--text3);margin-top:2px;}
.badge{display:inline-block;padding:2px 8px;border-radius:3px;
  font-family:var(--mono);font-size:9px;font-weight:600;
  text-transform:uppercase;letter-spacing:.05em;}
.b-bull{background:#4DB87A22;color:var(--bull);border:1px solid #4DB87A44;}
.b-bear{background:#E0555522;color:var(--bear);border:1px solid #E0555544;}

/* ── GOLD SPOT PRICE ── */
.bullion-price{font-family:var(--mono);font-size:clamp(24px,4vw,32px);
  font-weight:600;color:var(--gold);}

/* ── MISC ── */
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--bdr2);border-radius:2px;}
.up{color:var(--bull);}.dn{color:var(--bear);}.wa{color:var(--warn);}
.disc{font-size:11px;color:var(--text2);padding:14px 18px;
  background:var(--s1);border:1px solid var(--bdr);
  border-radius:var(--r-sm);margin-top:var(--gap);line-height:1.8;}

/* ════════════════════════════════════════════════════
   RESPONSIVE BREAKPOINTS
   ════════════════════════════════════════════════════ */

/* Tablet: <= 1024px */
@media (max-width:1024px){
  .buy-hero{grid-template-columns:1fr 1fr;grid-template-rows:auto auto;}
  .buy-hero > :nth-child(2){grid-column:1;grid-row:1;justify-self:start;}
  .buy-hero > :nth-child(1){grid-column:2;grid-row:1;}
  .buy-hero > :nth-child(3){grid-column:1/-1;grid-row:2;}
  .stats-grid{grid-template-columns:repeat(3,1fr);}
  .decision-body{grid-template-columns:1fr 1fr;}
  .decision-body > :nth-child(3){grid-column:1/-1;}
  .g2{grid-template-columns:1fr;}
  .g3{grid-template-columns:1fr 1fr;}
  .auto-grid{grid-template-columns:repeat(2,1fr);}
}

/* Mobile: <= 640px */
@media (max-width:640px){
  :root{--pad:14px;--gap:12px;}
  header{padding:14px var(--pad);}
  .h-meta{display:none;}
  .buy-hero{grid-template-columns:1fr;grid-template-rows:auto;}
  .buy-hero > *{grid-column:1!important;grid-row:auto!important;}
  .buy-score-ring{width:120px;height:120px;}
  .stats-grid{grid-template-columns:repeat(2,1fr);}
  .decision-body{grid-template-columns:1fr;}
  .g2,.g3{grid-template-columns:1fr;}
  .auto-grid{grid-template-columns:1fr 1fr;}
  .pred-grid{grid-template-columns:1fr;}
  .tabs{margin-bottom:10px;}
  .rt th,.rt td{padding:6px 8px;font-size:10px;}
  .bullion-price{font-size:26px;}
}
</style>
</head>
<body>
<header>
  <div class="h-brand">
    <div class="h-icon">⬡</div>
    <div>
      <div class="h-title">GOLD BUY SIGNAL DASHBOARD</div>
      <div class="h-sub">CPE Multi-Asset Framework · Gold Spot · Singapore</div>
    </div>
  </div>
  <div class="h-meta">
    <div>Updated: <span id="hgen"></span></div>
    <div>Data: <span id="hdat"></span></div>
    <div style="color:var(--gold);margin-top:2px">USD/SGD: <span id="hfx"></span></div>
  </div>
</header>

<main>
""" + _banner + """


<!-- DECISION PANEL -->
<div class="decision-panel" id="decision-panel">
  <div class="decision-verdict" id="verdict-row"></div>
  <div class="decision-body">
    <div class="dec-col">
      <h4 style="color:var(--bull)">&#10003; Evidence For Buying</h4>
      <div id="for-list"></div>
    </div>
    <div class="dec-col">
      <h4 style="color:var(--bear)">&#10007; Evidence Against Buying Now</h4>
      <div id="against-list"></div>
    </div>
    <div class="dec-col">
      <h4 style="color:var(--warn)">&#11044; What Would Change This</h4>
      <div class="trigger-list" id="trigger-list"></div>
    </div>
  </div>
</div>

<!-- BUY SCORE HERO -->
<div class="buy-hero">
  <div>
    <div style="font-family:var(--mono);font-size:11px;color:var(--text2);
                text-transform:uppercase;letter-spacing:.08em;margin-bottom:16px">
      Composite Buy Score Components
    </div>
    <div class="component-row" id="comp-rows"></div>
  </div>
  <div>
    <div class="buy-score-ring">
      <svg viewBox="0 0 120 120">
        <circle cx="60" cy="60" r="52" fill="none" stroke="#2C302C" stroke-width="10"/>
        <circle cx="60" cy="60" r="52" fill="none" stroke-width="10"
                stroke-linecap="round" stroke-dasharray="327" stroke-dashoffset="327"
                id="score-arc" style="transform:rotate(-90deg);transform-origin:60px 60px;
                transition:stroke-dashoffset 1s ease,stroke .5s"/>
      </svg>
      <div class="buy-score-inner">
        <div class="buy-num" id="score-num">—</div>
        <div class="buy-label-small">/ 100</div>
      </div>
    </div>
    <div class="buy-verdict" id="score-verdict">—</div>
  </div>
  <div>
    <div style="font-family:var(--mono);font-size:11px;color:var(--text2);
                text-transform:uppercase;letter-spacing:.08em;margin-bottom:16px">
      Gold Spot — SGD per gram
    </div>
    <div style="font-family:var(--mono);font-size:32px;font-weight:600;
                color:var(--gold)" id="bar-price">SGD —</div>
    <div style="font-family:var(--mono);font-size:11px;color:var(--text2);
                margin-top:6px" id="bar-sub">—</div>
    <div style="margin-top:16px;font-family:var(--mono);font-size:11px">
      <div style="color:var(--text2);margin-bottom:6px">Peak-to-now:</div>
      <div style="font-size:18px;font-weight:600" id="dd-peak">—</div>
      <div style="color:var(--text2);font-size:10px;margin-top:2px" id="dd-peak-sub">—</div>
    </div>
  </div>
</div>

<!-- STATS -->
<div class="stats-grid" id="stats-grid"></div>
<div style="font-family:var(--mono);font-size:10px;color:var(--text2);margin-bottom:var(--gap);padding:8px 12px;background:var(--s2);border-radius:6px;border:1px solid var(--bdr)">
  ⓘ Prices reflect last available US close: """ + _us_close_label + """ (GLD ETF). Spot price derived from GLD ETF and USD/SGD rate.
</div>

<!-- PRICE CHARTS -->
<div class="g2">
  <div class="card">
    <div class="ct">GC=F Gold Futures — USD/oz (Last 365 Days)</div>
    <div id="chart-usd" style="height:260px"></div>
  </div>
  <div class="card">
    <div class="ct">Gold Price — SGD per gram (Last 365 Days)</div>
    <div id="chart-sgd" style="height:260px"></div>
  </div>
</div>

<!-- DRAWDOWN CONTEXT -->
<div class="g2">
  <div class="card">
    <div class="ct">Current 63-Day Return vs Historical Distribution</div>
    <div id="chart-hist" style="height:260px"></div>
    <div style="font-family:var(--mono);font-size:11px;color:var(--text2);margin-top:10px"
         id="hist-context"></div>
  </div>
  <div class="card">
    <div class="ct">Gold Autocorrelation CPE — Recovery Probability After Current Drawdown</div>
    <div class="auto-grid" id="auto-grid" style="margin-bottom:12px"></div>
    <div style="font-family:var(--mono);font-size:10px;color:var(--text2)">
      % of historical dates with similar past return where gold was positive at each forward horizon
    </div>
  </div>
</div>

<!-- FORWARD CONE -->
<div class="card" style="margin-bottom:22px">
  <div class="ct">Forward Price Cone — Based on Historical Recovery After Similar Drawdowns</div>
  <div class="tabs" style="margin-bottom:12px">
    <button class="tb active" onclick="showCone('usd',this)">USD / oz</button>
    <button class="tb" onclick="showCone('sgd',this)">SGD / gram</button>
  </div>
  <div id="cone-usd" style="height:300px"></div>
  <div id="cone-sgd" style="height:300px;display:none"></div>
  <div style="font-family:var(--mono);font-size:10px;color:var(--text2);margin-top:10px"
       id="cone-context"></div>
</div>

<!-- RECOVERY TABLE -->
<div class="card" style="margin-bottom:22px">
  <div class="ct">Historical Forward Return Distribution — After Drawdowns ≥ Current</div>
  <div class="tabs">
    <button class="tb active" onclick="showRec('r63',this)">After 63d Draw ≥ Current</button>
    <button class="tb" onclick="showRec('r126',this)">After 126d Draw ≥ Current</button>
    <button class="tb" onclick="showRec('rext',this)">After Extreme Drawdowns (P10)</button>
  </div>
  <div id="rec-r63" class="tp active"><table class="rt" id="tbl-r63"><thead><tr>
    <th>Forward Horizon</th><th>N</th><th>% Positive</th>
    <th>P10</th><th>P25</th><th>Median</th><th>P75</th><th>P90</th><th>Mean</th>
  </tr></thead><tbody id="tbody-r63"></tbody></table></div>
  <div id="rec-r126" class="tp"><table class="rt" id="tbl-r126"><thead><tr>
    <th>Forward Horizon</th><th>N</th><th>% Positive</th>
    <th>P10</th><th>P25</th><th>Median</th><th>P75</th><th>P90</th><th>Mean</th>
  </tr></thead><tbody id="tbody-r126"></tbody></table></div>
  <div id="rec-rext" class="tp"><table class="rt" id="tbl-rext"><thead><tr>
    <th>Forward Horizon</th><th>N</th><th>% Positive</th>
    <th>P10</th><th>P25</th><th>Median</th><th>P75</th><th>P90</th><th>Mean</th>
  </tr></thead><tbody id="tbody-rext"></tbody></table></div>
</div>

<!-- PREDICTOR PROXIMITY -->
<div class="card" style="margin-bottom:22px">
  <div class="ct">CPE Predictor Status — Distance to Buy Signal Threshold</div>
  <div style="font-family:var(--mono);font-size:10px;color:var(--text2);margin-bottom:14px">
    <span class="up">■</span> In tail (signal firing) &nbsp;
    <span class="wa">■</span> Within 20% of threshold &nbsp;
    <span class="far">■</span> Far from threshold
  </div>
  <div class="pred-grid" id="pred-grid"></div>
</div>

<!-- CPE SIGNALS TABS -->
<div class="tabs">
  <button class="tb active" onclick="showSig('joint',this)">Joint CPE Signals</button>
  <button class="tb" onclick="showSig('pair',this)">Pairwise CPE Signals</button>
</div>
<div id="sig-joint" class="tp active">
  <div class="g2">
    <div class="card">
      <div class="ct">Firing Bullish Signals</div>
      <div class="sig-list" id="sl-bull"></div>
    </div>
    <div class="card">
      <div class="ct">Firing Bearish Signals</div>
      <div class="sig-list" id="sl-bear"></div>
    </div>
  </div>
  <div class="card">
    <div class="ct">All Joint CPE Signals — Gold</div>
    <div class="tbl-wrap">
      <table class="rt"><thead><tr>
        <th>Y</th><th>Dir</th><th>τf</th><th>qY</th><th>K</th>
        <th>CPE</th><th>Lift</th><th>n</th><th>Firing</th><th>Predictors</th>
      </tr></thead><tbody id="joint-tbody"></tbody></table>
    </div>
  </div>
</div>
<div id="sig-pair" class="tp">
  <div class="card">
    <div class="ct">Top Pairwise CPE Signals — Gold</div>
    <div class="tbl-wrap">
      <table class="rt"><thead><tr>
        <th>Y</th><th>X</th><th>Dir</th><th>τp</th><th>τf</th>
        <th>qX</th><th>qY</th><th>CPE</th><th>Lift</th><th>n</th>
      </tr></thead><tbody id="pair-tbody"></tbody></table>
    </div>
  </div>
</div>

<div class="disc">
  <strong style="color:var(--gold)">Research Disclaimer:</strong>
  All CPE values are within-sample empirical frequencies. The composite buy score,
  recovery distributions, and forward price cone are summaries of historical patterns —
  not forecasts. Past statistical structure does not guarantee future behaviour.
  This dashboard is for personal research only and does not constitute investment advice.
  Always consult a licensed financial adviser before making investment decisions.
</div>
</main>

<script>
const D = """ + data_json + """;
const PL = {
  paper_bgcolor:'transparent',plot_bgcolor:'#141614',
  font:{family:'IBM Plex Mono,monospace',color:'#7A8F7A',size:10},
  margin:{l:60,r:20,t:20,b:50},
  xaxis:{gridcolor:'#2C302C',linecolor:'#2C302C',zerolinecolor:'#2C302C'},
  yaxis:{gridcolor:'#2C302C',linecolor:'#2C302C',zerolinecolor:'#2C302C'},
};
function pl(id,traces,layout){
  Plotly.newPlot(id,traces,Object.assign({},PL,layout),{responsive:true,displayModeBar:false});
}
function chgColor(v){return v>0?'var(--bull)':v<0?'var(--bear)':'var(--neut)';}
function chgStr(v){return (v>0?'+':'')+v.toFixed(2)+'%';}
function fmtChg(v){return '<span style="color:'+chgColor(v)+'">'+chgStr(v)+'</span>';}

function showCone(id,btn){
  ['usd','sgd'].forEach(x=>{
    document.getElementById('cone-'+x).style.display='none';
  });
  document.getElementById('cone-'+id).style.display='block';
  document.querySelectorAll('.card .tabs .tb').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
}
function showRec(id,btn){
  document.querySelectorAll('[id^="rec-"]').forEach(p=>p.classList.remove('active'));
  document.getElementById('rec-'+id).classList.add('active');
  btn.parentElement.querySelectorAll('.tb').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
}
function showSig(id,btn){
  ['joint','pair'].forEach(x=>{
    document.getElementById('sig-'+x).classList.remove('active');
  });
  document.getElementById('sig-'+id).classList.add('active');
  btn.parentElement.querySelectorAll('.tb').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
}

function init(){
  document.getElementById('hgen').textContent = D.generated;
  document.getElementById('hdat').textContent = D.latest_date;
  document.getElementById('hfx').textContent  = D.usd_per_sgd.toFixed(4);

  

  renderBuyScore();
  renderStats();
  renderPriceCharts();
  renderHistogram();
  renderAutoGrid();
  renderCone();
  renderRecoveryTables();
  renderPredictorProximity();
  renderSignals();
  renderPairTable();
  renderDecision();
}

document.addEventListener('DOMContentLoaded', init);

// ── BUY SCORE ────────────────────────────────────────────────────────────────
function renderBuyScore(){
  const c = D.components;
  const score = c.composite;
  const col = score>=70?'var(--bull)':score>=40?'var(--warn)':'var(--bear)';

  // Animate arc
  const circ = 327;
  const offset = circ * (1 - score/100);
  const arc = document.getElementById('score-arc');
  arc.setAttribute('stroke-dashoffset', offset);
  arc.setAttribute('stroke', col);

  document.getElementById('score-num').textContent = score;
  document.getElementById('score-num').style.color = col;
  document.getElementById('score-verdict').textContent = c.label;
  document.getElementById('score-verdict').style.color = col;

  // Bar price
  document.getElementById('bar-price').textContent =
    'S$ ' + (D.spot_sgd_g||D.gold_spot_sgd_g).toLocaleString('en-SG',{minimumFractionDigits:2,maximumFractionDigits:2}) + ' /g';
  document.getElementById('bar-sub').textContent =
    D.bar_sub_text || ('Spot price in SGD per gram, derived from GLD ETF and USD/SGD');

  document.getElementById('dd-peak').innerHTML =
    '<span style="color:var(--bear)">'+D.dd_from_peak.toFixed(1)+'%</span> from 252d peak';
  document.getElementById('dd-peak-sub').textContent =
    'Peak: USD '+D.peak_252.toLocaleString()+'/oz / S$'+(D.peak_spot_sgd_g||Math.round(D.peak_252*D.usd_per_sgd/31.1035))+'/g';

  // Components
  const comps = [
    {label:'Drawdown Depth (35%)',   val:c.draw_score,  col:'var(--bear)',
     desc:'Deeper fall = higher mean-reversion potential'},
    {label:'Historical Recovery % (35%)', val:c.auto_score, col:'var(--bull)',
     desc:'% of similar drawdowns that recovered over 126 days'},
    {label:'Predictor Proximity (20%)', val:c.prox_score, col:'var(--warn)',
     desc:'How close CPE triggers (IBIT,SLV) are to firing'},
    {label:'CPE Signal Score (10%)',  val:c.cpe_score,  col:'var(--gold)',
     desc:'Current joint CPE weighted score (normalised)'},
  ];
  document.getElementById('comp-rows').innerHTML = comps.map(cp=>`
    <div class="comp-item">
      <div class="comp-label">${cp.label} <span style="float:right;color:${cp.col};font-weight:600">${cp.val.toFixed(0)}</span></div>
      <div class="comp-bar-wrap">
        <div class="comp-bar" style="width:${cp.val}%;background:${cp.col}"></div>
      </div>
      <div style="font-family:var(--mono);font-size:9px;color:var(--text2);margin-top:4px">${cp.desc}</div>
    </div>`).join('');
}

// ── STATS ────────────────────────────────────────────────────────────────────
function renderStats(){
  const s = [
    {l:'GC=F (USD/oz)', v:'$'+D.gold_usd.toLocaleString('en-US',{minimumFractionDigits:2}),
     s:'1d: '+chgStr(D.chg[1])},
    {l:'Price (SGD/oz)', v:'S$'+D.gold_sgd_oz.toLocaleString('en-SG',{minimumFractionDigits:2}),
     s:'5d: '+chgStr(D.chg[5])},
    {l:'Price (SGD/g)',  v:'S$'+D.gold_sgd_g.toFixed(2),
     s:'21d: '+chgStr(D.chg[21])},
    {l:'63d Return',    v:chgStr(D.chg[63]),
     s:'Currently at P'+D.curr_pct_63+'ile historically'},
    {l:'126d Return',   v:chgStr(D.chg[126]),
     s:'Currently at P'+D.curr_pct_126+'ile historically'},
    {l:'252d Range',    v:'$'+Math.round(D.low_252)+'–'+Math.round(D.high_252),
     s:'63d: '+chgStr(D.chg[63])},
  ];
  document.getElementById('stats-grid').innerHTML = s.map(x=>`
    <div class="stat">
      <div class="stat-l">${x.l}</div>
      <div class="stat-v" style="color:${x.v.includes('-')?'var(--bear)':x.v.includes('+')?'var(--bull)':'var(--text)'}">${x.v}</div>
      <div class="stat-s">${x.s}</div>
    </div>`).join('');
}

// ── PRICE CHARTS ─────────────────────────────────────────────────────────────
function renderPriceCharts(){
  const goldColor = '#C9A84C';
  pl('chart-usd',[{
    x:D.chart_dates,y:D.chart_prices,type:'scatter',mode:'lines',
    line:{color:goldColor,width:2},fill:'tozeroy',fillcolor:goldColor+'18',
    hovertemplate:'%{x}<br>$%{y:,.2f}<extra></extra>',name:'GC=F',
  }],{yaxis:{title:'USD/oz',tickformat:'$,.0f'},xaxis:{type:'date'}});

  pl('chart-sgd',[{
    x:D.chart_dates,y:D.chart_sgd_g,type:'scatter',mode:'lines',
    line:{color:'#E8C97A',width:2},fill:'tozeroy',fillcolor:'#E8C97A18',
    hovertemplate:'%{x}<br>S$%{y:,.2f}/g<extra></extra>',name:'SGD per gram',
  }],{yaxis:{title:'SGD per gram',tickformat:'S$,.2f'},xaxis:{type:'date'}});
}

// ── HISTOGRAM ────────────────────────────────────────────────────────────────
function renderHistogram(){
  const vals = D.hist_63_vals;
  const curr = D.chg[63];
  pl('chart-hist',[
    {x:vals,type:'histogram',nbinsx:60,name:'Historical 63d returns',
     marker:{color:'#C9A84C55',line:{color:'#C9A84C88',width:0.5}}},
    {x:[curr,curr],y:[0,200],type:'scatter',mode:'lines',name:'Current ('+curr.toFixed(1)+'%)',
     line:{color:'var(--bear)',width:2,dash:'dash'}},
  ],{
    xaxis:{title:'63-day log return (%)'},
    yaxis:{title:'Frequency'},
    showlegend:true,
    annotations:[{x:curr,y:150,text:curr.toFixed(1)+'%<br>P'+D.curr_pct_63+'ile',
                  showarrow:true,arrowcolor:'var(--bear)',
                  font:{color:'var(--bear)',family:'IBM Plex Mono',size:11}}],
  });
  document.getElementById('hist-context').textContent =
    'Current 63d return ('+curr.toFixed(1)+'%) is at the '+
    D.curr_pct_63+'th percentile of all historical 63d returns for GC=F. '+
    (D.curr_pct_63 < 15 ? 'This is an UNUSUALLY LARGE drawdown — rare historically.' :
     D.curr_pct_63 < 30 ? 'This is a significant but not extreme drawdown.' :
     'This is within the normal range of volatility.');
}

// ── AUTO CPE GRID ─────────────────────────────────────────────────────────────
function renderAutoGrid(){
  const auto = D.auto_cpe;
  const pastTaus = Object.keys(auto).sort();
  const fwdTaus  = [21,63,126,252];
  let html = '';
  for (const pt of pastTaus) {
    const a = auto[pt];
    html += `<div class="auto-card">
      <div class="auto-tau">τ_past = ${pt}d</div>
      <div style="font-family:var(--mono);font-size:11px;color:var(--text2);margin-bottom:8px">
        Current: <span style="color:var(--bear)">${a.current_return_pct.toFixed(1)}%</span>
        (P${a.current_percentile})
      </div>`;
    for (const fv of fwdTaus) {
      const key = 'fwd_'+fv+'_pct_positive';
      if (a[key] !== undefined) {
        const pct = a[key];
        const col = pct>60?'var(--bull)':pct>45?'var(--warn)':'var(--bear)';
        html += `<div class="auto-pct" style="margin-bottom:3px">
          τf=${fv}d: <span style="color:${col};font-weight:600">${pct}% positive</span>
          <span style="color:var(--text2);font-size:9px"> (n=${a['fwd_'+fv+'_n']}, med=${a['fwd_'+fv+'_median']}%)</span>
        </div>`;
      }
    }
    html += '</div>';
  }
  document.getElementById('auto-grid').innerHTML = html;
}

// ── FORWARD CONE ─────────────────────────────────────────────────────────────
function renderCone(){
  const taus = D.cone_taus;
  const curr = D.gold_usd;
  const currSgd = D.spot_sgd_g||D.gold_spot_sgd_g;

  // Historical last 63 days for context
  const hist_n = 63;
  const histX = D.chart_dates.slice(-hist_n).map((_,i)=>-(hist_n-1-i));
  const histY = D.chart_prices.slice(-hist_n);
  const histYSgd = D.chart_sgd_g.slice(-hist_n);

  function coneTraces(p10,p25,p50,p75,p90,histYArr,currency,tickfmt,startVal){
    return [
      {x:histX,y:histYArr,mode:'lines',name:'Historical (63d)',
       line:{color:'#C9A84C',width:2.5},
       hovertemplate:'Day %{x}<br>'+currency+'%{y:,.2f}<extra>Historical</extra>'},
      {x:[0,...taus,...[...taus].reverse()],
       y:[startVal,...p10,...[...p90].reverse()],
       fill:'toself',fillcolor:'#E0555511',line:{color:'transparent'},
       name:'P10–P90',hoverinfo:'skip'},
      {x:[0,...taus,...[...taus].reverse()],
       y:[startVal,...p25,...[...p75].reverse()],
       fill:'toself',fillcolor:'#E0555533',line:{color:'transparent'},
       name:'P25–P75',hoverinfo:'skip'},
      {x:[0,...taus],y:[startVal,...p50],mode:'lines+markers',name:'P50 Median',
       line:{color:'#E05555',width:2.5},marker:{size:7},
       hovertemplate:'Day %{x}<br>'+currency+'%{y:,.2f}<extra>P50 Median</extra>'},
      {x:[0,...taus],y:[startVal,...p90],mode:'lines',name:'P90 Optimistic',
       line:{color:'#4DB87A',width:1.5,dash:'dot'},
       hovertemplate:'Day %{x}<br>'+currency+'%{y:,.2f}<extra>P90 Optimistic</extra>'},
    ];
  }

  pl('cone-usd',
    coneTraces(D.cone_p10,D.cone_p25,D.cone_p50,D.cone_p75,D.cone_p90,histY,'$','$,.0f',curr),
    {xaxis:{title:'Trading days from today (0 = current)',
            zeroline:true,zerolinecolor:'#C9A84C44',zerolinewidth:1.5},
     yaxis:{title:'GC=F (USD/oz)',tickformat:'$,.0f'},
     showlegend:true,
     shapes:[{type:'line',x0:0,x1:0,y0:0,y1:1,xref:'x',yref:'paper',
              line:{color:'#C9A84C55',width:1.5,dash:'dot'}}]});

  pl('cone-sgd',
    coneTraces(D.cone_sgd_p10,D.cone_sgd_p25,D.cone_sgd_p50,D.cone_sgd_p75,D.cone_sgd_p90,histYSgd,'S$','S$,.2f',currSgd),
    {xaxis:{title:'Trading days from today (0 = current)',
            zeroline:true,zerolinecolor:'#C9A84C44',zerolinewidth:1.5},
     yaxis:{title:'SGD per gram',tickformat:'S$,.2f'},
     showlegend:true,
     shapes:[{type:'line',x0:0,x1:0,y0:0,y1:1,xref:'x',yref:'paper',
              line:{color:'#C9A84C55',width:1.5,dash:'dot'}}]});

  const r = D.recovery_63;
  const keys = Object.keys(r).sort((a,b)=>+a-+b);
  if(keys.length>0){
    const last = r[keys[keys.length-1]];
    document.getElementById('cone-context').textContent =
      'Cone based on '+last.n+' historical episodes where GC=F fell at least '+
      D.chg[63].toFixed(1)+'% over 63 days. '+
      'At the 252-day horizon, median outcome: '+(last.p50>0?'+':'')+last.p50+'%, '+
      last.pct_positive+'% of cases were positive.';
  }
}

// ── RECOVERY TABLES ───────────────────────────────────────────────────────────
function renderRecTable(tbodyId, data){
  const tbody = document.getElementById(tbodyId);
  const keys = Object.keys(data).sort((a,b)=>+a-+b);
  tbody.innerHTML = keys.map(k=>{
    const r = data[k];
    const mc = r.p50>0?'var(--bull)':'var(--bear)';
    const pc = r.pct_positive>60?'var(--bull)':r.pct_positive>45?'var(--warn)':'var(--bear)';
    return `<tr>
      <td style="font-family:var(--mono)">${k}d (~${Math.round(+k/21)}mo)</td>
      <td style="font-family:var(--mono);color:var(--text2)">${r.n}</td>
      <td style="font-family:var(--mono);color:${pc};font-weight:600">${r.pct_positive}%</td>
      <td style="font-family:var(--mono);color:var(--bear)">${r.p10}%</td>
      <td style="font-family:var(--mono);color:var(--bear)">${r.p25}%</td>
      <td style="font-family:var(--mono);color:${mc};font-weight:600">${r.p50}%</td>
      <td style="font-family:var(--mono);color:var(--bull)">${r.p75}%</td>
      <td style="font-family:var(--mono);color:var(--bull)">${r.p90}%</td>
      <td style="font-family:var(--mono)">${r.mean}%</td>
    </tr>`;
  }).join('');
}
function renderRecoveryTables(){
  renderRecTable('tbody-r63',  D.recovery_63);
  renderRecTable('tbody-r126', D.recovery_126);
  renderRecTable('tbody-rext', D.recovery_extreme);
}

// ── PREDICTOR PROXIMITY ───────────────────────────────────────────────────────
function renderPredictorProximity(){
  const pp = D.pred_proximity;
  let html = '';
  for(const [ticker,rows] of Object.entries(pp)){
    let rowsHtml='';
    for(const r of rows){
      const in_t = r.in_tail;
      const near = !in_t && Math.abs(r.dist_pct) < 20;
      const col = in_t?'in-bull':near?'near':'far';
      // Plain English label
      const gap = Math.abs(r.current - r.threshold).toFixed(1);
      const direction = r.current >= r.threshold ? 'above' : 'below';
      const label = in_t && r.tail_type==='lower' ? 'FALLING ✓ SIGNAL ACTIVE' :
                    in_t ? 'SIGNAL FIRING ✓' :
                    near ? 'CLOSE — ' + gap + '% away' :
                    gap + '% away from firing';
      const barW = in_t ? 100 : Math.max(0, 100 - Math.min(Math.abs(r.dist_pct), 50)*2);
      const barCol = in_t?'var(--bull)':near?'var(--warn)':'var(--neut)';
      const tailLabel = r.tail_type==='lower' ? 
  'Mean-reversion signal (gold needs to be falling)' : 
  'Bull signal (needs to be rising/strong)';
      // Plain English param description
      const windowDesc = r.tau === 1 ? 'Today vs yesterday' :
                         r.tau === 5 ? 'Past 5 days' :
                         r.tau === 21 ? 'Past 1 month' :
                         r.tau === 63 ? 'Past 3 months' :
                         r.tau === 126 ? 'Past 6 months' :
                         r.tau === 252 ? 'Past 1 year' :
                         r.tau === 300 ? 'Past 14 months' : 'Past ' + r.tau + 'd';
      const pctileDesc = r.tail_type === 'lower' ?
        (r.q <= 0.10 ? 'needs to be in its lowest 10% historically' :
         r.q <= 0.20 ? 'needs to be in its lowest 20% historically' : 'must be falling') :
        (r.q >= 0.95 ? 'needs to be in top 5%' :
         r.q >= 0.90 ? 'needs to be in top 10%' :
         r.q >= 0.80 ? 'needs to be in top 20%' :
         r.q >= 0.60 ? 'needs to be above 60th pct' :
         r.q >= 0.50 ? 'needs to be above average' : '');
      rowsHtml += `<div class="pred-row">
        <span class="pred-p">${windowDesc} · ${pctileDesc}</span>
        <span class="pred-v ${col}">${label}</span>
      </div>
      <div class="prog-wrap">
        <div class="prog-fill" style="width:${barW}%;background:${barCol}"></div>
      </div>
      <div style="font-family:var(--mono);font-size:9px;color:var(--text2);margin-bottom:8px">
        ${in_t && r.tail_type==='lower' ? 'Currently falling at '+r.current.toFixed(1)+'% — below the '+r.threshold.toFixed(1)+'% lower threshold ✓' : in_t ? 'Currently at '+r.current.toFixed(1)+'% — above the '+r.threshold.toFixed(1)+'% trigger ✓' : r.tail_type==='lower' ? 'Currently at '+r.current.toFixed(1)+'% — needs to fall below '+r.threshold.toFixed(1)+'% to fire' : 'Currently at '+r.current.toFixed(1)+'% — needs to reach '+r.threshold.toFixed(1)+'% to fire'}
      </div>`;
    }
    html += `<div class="pred-card">
      <div class="pred-ticker">${ticker}</div>
      ${rowsHtml}
    </div>`;
  }
  document.getElementById('pred-grid').innerHTML = html || '<div style="color:var(--text2);padding:20px;text-align:center;font-family:var(--mono)">No predictor data</div>';
}

// ── SIGNALS ───────────────────────────────────────────────────────────────────
function renderSignals(){
  const sigs = D.signals;
  function sigItem(s){
    const dc = s.direction==='bullish'?'b':'r';
    const fc = s.firing ? (s.direction==='bullish'?'fb':'br') : '';
    return `<div class="sig-item ${fc}">
      <div class="sig-dot ${s.firing?dc:'o'}"></div>
      <div class="sig-body">
        <div style="display:flex;gap:8px;align-items:center;margin-bottom:3px">
          <span class="badge ${s.direction==='bullish'?'b-bull':'b-bear'}">${s.direction}</span>
          <span style="font-family:var(--mono);font-size:9px;color:var(--text2)">
            ${s.Y} · τf=${s.tau_future}d · qY=${s.q_Y} · K=${s.n_predictors}
          </span>
        </div>
        <div class="sig-preds">${s.pred_str}</div>
        <div class="sig-meta">CPE=${s.joint_CPE.toFixed(3)} · Lift=${s.lift.toFixed(2)}× · n=${s.n_joint} · w=${s.weight.toFixed(2)}</div>
      </div>
    </div>`;
  }

  const firingBull = sigs.filter(s=>s.firing&&s.direction==='bullish');
  const firingBear = sigs.filter(s=>s.firing&&s.direction==='bearish');

  document.getElementById('sl-bull').innerHTML = firingBull.length ?
    firingBull.map(sigItem).join('') :
    '<div style="color:var(--text2);font-family:var(--mono);font-size:11px;padding:20px;text-align:center">No bullish signals currently firing<br><span style="font-size:10px">Check Predictor Proximity panel to see what needs to happen</span></div>';

  document.getElementById('sl-bear').innerHTML = firingBear.length ?
    firingBear.map(sigItem).join('') :
    '<div style="color:var(--text2);font-family:var(--mono);font-size:11px;padding:20px;text-align:center">No bearish signals currently firing</div>';

  // All signals table
  document.getElementById('joint-tbody').innerHTML = sigs.sort((a,b)=>b.weight-a.weight).map(s=>{
    const dc = s.direction==='bullish'?'var(--bull)':'var(--bear)';
    const fc = s.firing ? (s.direction==='bullish'?'background:#4DB87A0A':'background:#E055550A') : '';
    return `<tr style="${fc}">
      <td style="font-family:var(--mono);color:var(--gold)">${s.Y}</td>
      <td style="font-family:var(--mono);color:${dc}">${s.direction}</td>
      <td style="font-family:var(--mono)">${s.tau_future}d</td>
      <td style="font-family:var(--mono)">${s.q_Y}</td>
      <td style="font-family:var(--mono)">${s.n_predictors}</td>
      <td style="font-family:var(--mono)">${s.joint_CPE.toFixed(3)}</td>
      <td style="font-family:var(--mono)">${s.lift.toFixed(2)}×</td>
      <td style="font-family:var(--mono);color:var(--text2)">${s.n_joint}</td>
      <td style="font-family:var(--mono);color:${s.firing?dc:'var(--text2)'}">${s.firing?'⬤ YES':'○ no'}</td>
      <td style="font-family:var(--mono);font-size:9px;color:var(--text2)">${s.pred_str}</td>
    </tr>`;
  }).join('');
}

function renderPairTable(){
  document.getElementById('pair-tbody').innerHTML = D.pair_signals.map(s=>{
    const dc = s.direction==='bullish'?'var(--bull)':'var(--bear)';
    return `<tr>
      <td style="font-family:var(--mono);color:var(--gold)">${s.Y}</td>
      <td style="font-family:var(--mono)">${s.X}</td>
      <td style="font-family:var(--mono);color:${dc}">${s.direction}</td>
      <td style="font-family:var(--mono)">${s.tau_past}d</td>
      <td style="font-family:var(--mono)">${s.tau_future}d</td>
      <td style="font-family:var(--mono)">${s.q_X}</td>
      <td style="font-family:var(--mono)">${s.q_Y}</td>
      <td style="font-family:var(--mono);color:${dc}">${s.CPE.toFixed(3)}</td>
      <td style="font-family:var(--mono)">${s.lift.toFixed(2)}×</td>
      <td style="font-family:var(--mono);color:var(--text2)">${s.n_condition}</td>
    </tr>`;
  }).join('');
}


function renderDecision() {
  const c   = D.components;
  const r63 = D.recovery_63;
  const pp  = D.pred_proximity;
  const score    = c.composite;
  const chg63    = D.chg['63'];
  const chg126   = D.chg['126'];
  const pct63    = D.curr_pct_63;
  const slv_on   = (pp['SLV']  ||[]).some(r=>r.in_tail && r.q>=0.9);
  const sif_on   = (pp['SI=F'] ||[]).some(r=>r.in_tail && r.q>=0.9);
  const gcf_low  = (pp['GC=F'] ||[]).some(r=>r.in_tail);
  const pct126   = r63['126'] ? r63['126'].pct_positive : 50;
  const med126   = r63['126'] ? r63['126'].p50 : 0;
  const med252   = r63['252'] ? r63['252'].p50 : 0;
  const n_hist   = r63['126'] ? r63['126'].n   : 0;

  // Verdict
  let verdict, vcolor, vicon, vsub;
  if (score>=70 && pct126>=55) {
    verdict='BUY NOW'; vcolor='var(--bull)'; vicon='&#10003;';
    vsub='Multiple signals aligned — historical evidence supports entry';
  } else if (score>=55 && pct126>=50) {
    verdict='BUY GRADUALLY'; vcolor='#8FD4A0'; vicon='&#8599;';
    vsub='Consider staged entry — not all signals aligned but conditions improving';
  } else if (score>=40 && pct126>=43) {
    verdict='WAIT &amp; WATCH'; vcolor='var(--warn)'; vicon='&#11044;';
    vsub='Approaching buy zone — monitor triggers below before committing';
  } else if (pct126<40 && med252<0) {
    verdict='TOO EARLY'; vcolor='var(--bear)'; vicon='&#10007;';
    vsub='Historical data shows continued weakness likely — preserve capital for now';
  } else {
    verdict='WAIT &amp; WATCH'; vcolor='var(--warn)'; vicon='&#11044;';
    vsub='Mixed signals — no clear entry point yet';
  }

  document.getElementById('verdict-row').innerHTML =
    '<div class="verdict-icon" style="background:'+vcolor+'22;border:2px solid '+vcolor+'66">'+vicon+'</div>'+
    '<div>'+
      '<div class="verdict-title" style="color:'+vcolor+'">'+verdict+'</div>'+
      '<div class="verdict-sub">'+vsub+'</div>'+
      '<div style="font-family:var(--mono);font-size:11px;margin-top:8px;color:var(--text2)">'+
        'Composite Score: <span style="color:'+vcolor+';font-weight:600">'+score+'/100</span> &nbsp;&middot;&nbsp; '+
        '63d return: <span style="color:var(--bear)">'+chg63.toFixed(1)+'%</span> (P'+pct63+'ile) &nbsp;&middot;&nbsp; '+
        'Recovery rate (126d): <span style="color:'+(pct126>=50?'var(--bull)':'var(--bear)')+'">'+pct126+'% positive</span> from '+n_hist+' historical episodes'+
      '</div>'+
    '</div>';

  // For / Against
  const FOR=[], AGN=[];

  if (pct63<5) {
    FOR.push({col:'var(--bull)',
      txt:'Extreme drawdown: '+chg63.toFixed(1)+'% over 63d is at the '+pct63+'th percentile — only '+pct63+'% of history was worse. Rare conditions like this historically precede eventual recovery.'});
  } else if (pct63<15) {
    FOR.push({col:'var(--warn)',
      txt:'Significant drawdown: '+chg63.toFixed(1)+'% over 63d is at the '+pct63+'th percentile — in the lower tail of historical returns.'});
  } else {
    AGN.push({col:'var(--neut)',
      txt:'Drawdown ('+chg63.toFixed(1)+'% over 63d at P'+pct63+') is not extreme by historical standards.'});
  }

  if (slv_on && sif_on) {
    FOR.push({col:'var(--bull)',
      txt:'Silver (SLV & SI=F) is in its top 5% over 252 days. Our CPE analysis shows this condition historically preceded large upward moves in long-duration bonds and gold-correlated assets (CPE 0.84-1.00, lift 3-5x).'});
  }

  if (gcf_low) {
    const fwd252 = D.auto_cpe['63'] && D.auto_cpe['63']['fwd_252_pct_positive'];
    FOR.push({col:'var(--warn)',
      txt:'Gold itself is in its lower 10th percentile regime (63d & 126d). Autocorrelation CPE: '+(fwd252||'—')+'% of similar episodes saw recovery at 252 days.'});
  }

  if (pct126>=50) {
    FOR.push({col:'var(--bull)',
      txt:'In '+n_hist+' historical episodes with this level of drawdown, gold was positive at 126 days '+pct126+'% of the time — above the 50% threshold.'});
  } else {
    AGN.push({col:'var(--bear)',
      txt:'In '+n_hist+' historical episodes with a 63d fall this large, gold was POSITIVE at 126 days only '+pct126+'% of the time. Continued weakness is the modal historical outcome.'});
  }

  if (med126<0) {
    AGN.push({col:'var(--bear)',
      txt:'Median 126-day return after similar drawdowns: '+med126+'% (negative). The most likely single outcome based on history is further weakness.'});
  }
  if (med252<0) {
    AGN.push({col:'var(--bear)',
      txt:'Even at 252 days (1 year), the median outcome after similar drawdowns is '+med252+'% — gold has historically taken a long time to recover from falls of this magnitude.'});
  }

  const ibit252=(pp['IBIT']||[]).find(r=>r.tau===252);
  if (ibit252 && !ibit252.in_tail) {
    AGN.push({col:'var(--bear)',
      txt:'Bitcoin ETFs (IBIT) are well below their 252-day median — the strongest gold bull CPE signals (IBIT+FBTC above median) are far from firing. Crypto remains in a bearish regime.'});
  }

  function pts(arr,el){
    const label = el==='for-list' ? 'FOR' : 'AGAINST';
    const defCol = el==='for-list' ? 'var(--bull)' : 'var(--bear)';
    document.getElementById(el).innerHTML = arr.length ?
      arr.map(p=>'<div class="ev-card" style="border-left-color:'+p.col+';border-color:'+p.col+'33">'+
        '<span class="ev-label" style="color:'+p.col+'">'+label+'</span>'+p.txt+'</div>'
      ).join('') :
      '<div style="color:var(--text2);font-size:12px;padding:12px;text-align:center;font-family:var(--mono)">No evidence in this direction</div>';
  }
  pts(FOR,'for-list');
  pts(AGN,'against-list');

  // Triggers
  const ibit5  = (pp['IBIT']||[]).find(r=>r.tau===5);
  const sgd    = (pp['SGDUSD=X']||[])[0];
  const trigs = [
    { name:'Historical recovery rate &gt; 50%',
      desc:'When % of similar episodes that were positive at 126d crosses 50%, odds favour buying',
      cur:'Currently '+pct126+'% (need &gt;50%)',
      prog:pct126, col:pct126>=50?'var(--bull)':'var(--warn)', fired:pct126>=50 },
    { name:'Gold 21-day return turns positive',
      desc:'When short-term momentum stabilises, the acute selling phase is likely over',
      cur:'Currently '+D.chg['21'].toFixed(1)+'% over 21 days',
      prog:Math.max(0,Math.min(100,50+D.chg['21']*4)),
      col:D.chg['21']>=0?'var(--bull)':'var(--warn)', fired:D.chg['21']>=0 },
    { name:'Bitcoin ETFs turn positive (5-day)',
      desc:'IBIT+FBTC both above their 5-day median fires the short-term gold bull signal',
      cur:ibit5 ? (ibit5.in_tail?'FIRING ✓':
  'Currently '+(ibit5.current).toFixed(1)+'% — needs to reach '+(ibit5.threshold).toFixed(1)+'%') : 'N/A',
      prog:ibit5 ? Math.max(0,Math.min(100, ibit5.in_tail ? 100 : Math.max(0,(ibit5.current-Math.min(ibit5.current,ibit5.threshold))/(Math.abs(ibit5.threshold)+0.01)*50))) : 0,
      col:'var(--bull)', fired:!!(ibit5&&ibit5.in_tail) },
    { name:'Composite score &ge; 60',
      desc:'When score crosses 60 the balance of evidence tips toward buying',
      cur:score+'/100 — need 60',
      prog:score, col:score>=60?'var(--bull)':'var(--warn)', fired:score>=60 },
    { name:'SGD strengthens vs USD (300d)',
      desc:'Stronger SGD reduces the SGD cost of gold — also a CPE predictor',
      cur:sgd ? (sgd.in_tail?'FIRING ✓':
  'Currently '+(sgd.current).toFixed(1)+'% — needs to reach '+(sgd.threshold).toFixed(1)+'%') : 'N/A',
      prog:sgd ? Math.max(0,Math.min(100, sgd.in_tail ? 100 : Math.max(0,sgd.current/sgd.threshold*80))) : 0,
      col:'var(--gold)', fired:!!(sgd&&sgd.in_tail) },
  ];

  document.getElementById('trigger-list').innerHTML = trigs.map(t=>
    '<div class="trigger-item" style="border-color:'+(t.fired?t.col+'66':'var(--bdr)')+'">'+
      '<div class="trigger-name" style="color:'+(t.fired?t.col:'var(--text)')+'">'+
        (t.fired?'&#10003; ':'&#9711; ')+t.name+'</div>'+
      '<div class="trigger-status">'+t.cur+'</div>'+
      '<div class="trigger-bar-wrap"><div class="trigger-bar-fill" style="width:'+t.prog+'%;background:'+t.col+'"></div></div>'+
      '<div style="font-family:var(--mono);font-size:9px;color:var(--text2);margin-top:4px">'+t.desc+'</div>'+
    '</div>'
  ).join('');
}

</script>
</body>
</html>"""
out = "gold_dashboard.html"
with open(out,"w",encoding="utf-8") as f:
    f.write(html)
print(f"\nSaved: {out}  ({os.path.getsize(out)/1e3:.0f} KB)")
print("Open gold_dashboard.html in Chrome/Firefox.")
print("Re-run this script to refresh prices.")
