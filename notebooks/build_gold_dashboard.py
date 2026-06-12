"""
Gold CPE Buy Signal Dashboard — Enhanced
=========================================
Comprehensive gold buy/sell signal dashboard using:
  1. Current drawdown context vs historical distribution
  2. Forward return distributions after similar drawdowns
  3. All CPE predictors for gold (pairwise + joint)
  4. Gold autocorrelation CPE (mean-reversion signal)
  5. Composite buy score
  6. 20g bar price cone in SGD

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
RATE_TICKERS = {"^VIX","^VXN","^OVX","^GVZ","^EVZ","^VVIX","^SKEW",
                "^TNX","^TYX","^FVX","^IRX"}
price_tickers = [t for t in prices.columns if t not in RATE_TICKERS]
rate_tickers  = [t for t in prices.columns if t in RATE_TICKERS]

print("Fetching latest prices from Yahoo Finance...")
try:
    import yfinance as yf
    fetch_list = list(set(GOLD_Y + [FX_TICKER,"SLV","SI=F","IBIT","FBTC","BITB",
                                    "SGDUSD=X","SOXX","XLK","QQQ","VUG","EWY","XAUUSD=X"]))
    raw = yf.download(fetch_list, period="400d", auto_adjust=True, progress=False)["Close"]
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    for col in raw.columns:
        if col in prices.columns:
            new = raw[[col]].loc[raw.index > prices.index.max()]
            if not new.empty:
                prices = pd.concat([prices, new])
    prices = prices.sort_index().loc[~prices.index.duplicated(keep="last")]
    print(f"  Latest date: {prices.index.max().date()}")
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

# Use XAUUSD=X spot price for bar calculation if available
# GC=F (futures) carries ~$10-15/oz above spot due to storage/financing
if "XAUUSD=X" in prices.columns:
    spot_series = prices["XAUUSD=X"].dropna()
    if len(spot_series) > 0:
        gold_spot_usd = float(spot_series.iloc[-1])
        print(f"  Using XAUUSD=X spot: ${gold_spot_usd:.2f}/oz (GC=F futures: ${float(gcf.iloc[-1]):.2f}/oz)")
    else:
        gold_spot_usd = float(gcf.iloc[-1]) - 12.0
        print(f"  XAUUSD=X empty, using GC=F - $12 carry adjustment")
else:
    gold_spot_usd = float(gcf.iloc[-1]) - 12.0
    print(f"  XAUUSD=X not available, using GC=F - $12 carry adjustment")

gold_usd = float(gcf.iloc[-1])  # keep futures price for chart/analysis
gold_sgd_oz = gold_usd * usd_per_sgd
gold_sgd_g  = gold_sgd_oz / 31.1035

# Bar price uses SPOT price (not futures) with 0.8% BullionStar dealer premium
# 0.8% is calibrated to Britannia 20g bar — a competitive major-brand bar
DEALER_PREMIUM = 1.008
gold_spot_sgd_oz = gold_spot_usd * usd_per_sgd
gold_spot_sgd_g  = gold_spot_sgd_oz / 31.1035
bar_sgd = gold_spot_sgd_g * 20 * DEALER_PREMIUM
print(f"  Spot SGD/g: S${gold_spot_sgd_g:.2f} | 20g bar est: S${bar_sgd:.2f} (0.8% premium)")
# Always define these for the data bundle
bar_sub_text = f"Spot S${gold_spot_sgd_g:.2f}/g × 20g × 0.8% BullionStar dealer premium"
peak_252 = float(gcf.iloc[-252:].max()) if len(gcf) >= 252 else float(gcf.max())
dd_from_peak = round((gold_usd / peak_252 - 1) * 100, 2)
peak_bar_sgd = round(peak_252 * usd_per_sgd / 31.1035 * 20 * DEALER_PREMIUM, 2)

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
    "IBIT":    [(1,0.5),(5,0.5),(252,0.5),(126,0.6)],
    "FBTC":    [(1,0.5),(5,0.5),(252,0.5),(126,0.6)],
    "BITB":    [(1,0.5),(252,0.5),(126,0.6)],
    "SLV":     [(252,0.95),(300,0.95),(252,0.8)],
    "SI=F":    [(252,0.95),(300,0.95),(252,0.8)],
    "SGDUSD=X":[(300,0.9),(252,0.9)],
    "GC=F":    [(63,0.10),(126,0.10),(252,0.10)],  # mean-reversion: lower tail
}

pred_proximity = {}
for ticker, params in KEY_PREDS.items():
    if ticker not in prices.columns: continue
    rows = []
    for (tau,q) in params:
        curr = current_inc.get(tau,{}).get(ticker)
        if curr is None: continue
        is_lower = q <= 0.20  # for GC=F mean-reversion, lower tail
        if is_lower:
            # lower tail: condition fires when curr < lower threshold
            th = thresholds.get((tau,round(1-q,10)),{}).get(ticker,float("nan"))
            if np.isnan(th): continue
            in_tail = bool(curr < th)
            # dist: how close to threshold (negative = already past it)
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
for ticker in ["IBIT","FBTC","SLV","SI=F"]:
    rows = pred_proximity.get(ticker,[])
    for r in rows:
        if r["tail_type"]=="upper":
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
chart_sgd_g  = [round(float(p)*usd_per_sgd/31.1035*DEALER_PREMIUM*20,2) for p in gcf.iloc[-365:]]

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
# Use 0.8% premium for cone projections (consistent with bar price calculation)
cone_sgd_p10 = [p*usd_per_sgd/31.1035*20*DEALER_PREMIUM for p in cone_p10]
cone_sgd_p25 = [p*usd_per_sgd/31.1035*20*DEALER_PREMIUM for p in cone_p25]
cone_sgd_p50 = [p*usd_per_sgd/31.1035*20*DEALER_PREMIUM for p in cone_p50]
cone_sgd_p75 = [p*usd_per_sgd/31.1035*20*DEALER_PREMIUM for p in cone_p75]
cone_sgd_p90 = [p*usd_per_sgd/31.1035*20*DEALER_PREMIUM for p in cone_p90]

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
    "bar_sgd":       round(bar_sgd,2),
    "usd_per_sgd":   round(usd_per_sgd,4),
    "chg":              chg,
    "dd_from_peak":     dd_from_peak,
    "peak_252":         round(peak_252,2),
    "peak_bar_sgd":     peak_bar_sgd,
    "gold_spot_usd":    round(gold_spot_usd,2),
    "gold_spot_sgd_g":  round(gold_spot_sgd_g,4),
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
  --gold:#C9A84C; --gold2:#E8C97A; --goldlt:#FDF3D0;
  --bg:#0C0E0D; --s1:#141614; --s2:#1B1E1B; --s3:#222522;
  --bdr:#2C302C; --bdr2:#3A3F3A;
  --bull:#4DB87A; --bear:#E05555; --warn:#E8A020; --neut:#6B7D6B;
  --text:#DDE8DD; --text2:#7A8F7A; --text3:#4A5A4A;
  --mono:'IBM Plex Mono',monospace; --sans:'IBM Plex Sans',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;}

header{
  background:linear-gradient(160deg,#0C0E0D,#161A14,#0C0E0D);
  border-bottom:1px solid var(--bdr);padding:22px 36px;
  display:flex;align-items:center;justify-content:space-between;gap:20px;
}
.h-brand{display:flex;align-items:center;gap:14px;}
.h-icon{
  width:44px;height:44px;border-radius:10px;
  background:linear-gradient(135deg,#8B6914,#C9A84C,#E8C97A);
  display:flex;align-items:center;justify-content:center;
  font-size:20px;flex-shrink:0;
}
.h-title{font-family:var(--mono);font-size:17px;font-weight:600;color:var(--gold);}
.h-sub{font-family:var(--mono);font-size:10px;color:var(--text2);
       letter-spacing:.08em;text-transform:uppercase;margin-top:3px;}
.h-meta{font-family:var(--mono);font-size:11px;color:var(--text2);
        text-align:right;line-height:1.9;}

main{padding:24px 36px;max-width:1540px;margin:0 auto;}

/* BUY SCORE HERO */
.buy-hero{
  background:linear-gradient(135deg,#141614,#1B1E1B);
  border:1px solid var(--bdr2);border-radius:14px;
  padding:28px 32px;margin-bottom:22px;
  display:grid;grid-template-columns:1fr auto 1fr;gap:32px;align-items:center;
}
.buy-score-ring{
  position:relative;width:160px;height:160px;margin:0 auto;
}
.buy-score-ring svg{width:100%;height:100%;}
.buy-score-inner{
  position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;
}
.buy-num{font-family:var(--mono);font-size:36px;font-weight:600;}
.buy-label-small{font-family:var(--mono);font-size:9px;color:var(--text2);
                  text-transform:uppercase;letter-spacing:.1em;margin-top:2px;}
.buy-verdict{font-family:var(--mono);font-size:16px;font-weight:600;
             text-align:center;margin-top:10px;letter-spacing:.03em;}
.component-row{display:flex;flex-direction:column;gap:10px;}
.comp-item{background:var(--s3);border-radius:6px;padding:10px 14px;}
.comp-label{font-family:var(--mono);font-size:9px;color:var(--text2);
             text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;}
.comp-bar-wrap{height:5px;background:var(--bdr);border-radius:3px;
               position:relative;overflow:hidden;}
.comp-bar{height:100%;border-radius:3px;transition:width .6s;}
.comp-val{font-family:var(--mono);font-size:11px;font-weight:600;
          float:right;margin-top:-16px;}

/* STATS GRID */
.stats-grid{
  display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:22px;
}
.stat{background:var(--s1);border:1px solid var(--bdr);
      border-radius:8px;padding:14px 16px;}
.stat-l{font-family:var(--mono);font-size:9px;color:var(--text2);
         text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;}
.stat-v{font-family:var(--mono);font-size:19px;font-weight:600;}
.stat-s{font-size:11px;color:var(--text2);margin-top:3px;}

/* CARD */
.card{background:var(--s1);border:1px solid var(--bdr);
      border-radius:10px;padding:20px;}
.ct{font-family:var(--mono);font-size:10px;font-weight:600;color:var(--gold);
    text-transform:uppercase;letter-spacing:.1em;margin-bottom:16px;
    display:flex;align-items:center;gap:8px;}
.ct::before{content:'';width:2px;height:12px;
            background:var(--gold);border-radius:1px;}

.g2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:22px;}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;margin-bottom:22px;}

/* TABS */
.tabs{display:flex;gap:0;margin-bottom:16px;border-bottom:1px solid var(--bdr);}
.tb{padding:10px 18px;font-family:var(--mono);font-size:11px;font-weight:600;
    color:var(--text2);cursor:pointer;border:none;background:none;
    border-bottom:2px solid transparent;text-transform:uppercase;
    letter-spacing:.05em;transition:all .2s;}
.tb:hover{color:var(--text);}
.tb.active{color:var(--gold);border-bottom-color:var(--gold);}
.tp{display:none;}.tp.active{display:block;}

/* RECOVERY TABLE */
.rt{width:100%;border-collapse:collapse;font-size:12px;}
.rt th{font-family:var(--mono);font-size:9px;color:var(--text2);
       text-transform:uppercase;letter-spacing:.08em;
       padding:8px 10px;border-bottom:1px solid var(--bdr);text-align:left;}
.rt td{padding:8px 10px;border-bottom:1px solid var(--bdr);}
.rt tr:hover td{background:var(--s2);}

/* PREDICTOR CARDS */
.pred-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;}
.pred-card{background:var(--s2);border:1px solid var(--bdr);border-radius:8px;padding:14px;}
.pred-ticker{font-family:var(--mono);font-size:13px;font-weight:600;
             color:var(--gold2);margin-bottom:10px;}
.pred-row{display:flex;justify-content:space-between;align-items:center;
          margin-bottom:4px;}
.pred-p{font-family:var(--mono);font-size:9px;color:var(--text2);}
.pred-v{font-family:var(--mono);font-size:11px;font-weight:600;}
.in-bull{color:var(--bull);}
.in-bear{color:var(--bear);}
.near{color:var(--warn);}
.far{color:var(--neut);}
.prog-wrap{height:4px;background:var(--bdr);border-radius:2px;
           margin:4px 0 8px;overflow:hidden;}
.prog-fill{height:100%;border-radius:2px;}

/* SIGNAL LIST */
.sig-list{display:flex;flex-direction:column;gap:6px;}
.sig-item{
  background:var(--s2);border:1px solid var(--bdr);
  border-radius:6px;padding:10px 12px;
  display:flex;gap:10px;align-items:flex-start;
}
.sig-item.fb{border-color:#4DB87A44;background:#4DB87A0A;}
.sig-item.br{border-color:#E0555544;background:#E055550A;}
.sig-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;margin-top:4px;}
.sig-dot.b{background:var(--bull);}
.sig-dot.r{background:var(--bear);}
.sig-dot.o{background:var(--bdr2);}
.sig-body{flex:1;}
.sig-preds{font-family:var(--mono);font-size:10px;color:var(--text2);line-height:1.7;}
.sig-meta{font-size:11px;color:var(--text3);margin-top:2px;}

/* BADGE */
.badge{display:inline-block;padding:2px 8px;border-radius:3px;
       font-family:var(--mono);font-size:9px;font-weight:600;
       text-transform:uppercase;letter-spacing:.05em;}
.b-bull{background:#4DB87A22;color:var(--bull);border:1px solid #4DB87A44;}
.b-bear{background:#E0555522;color:var(--bear);border:1px solid #E0555544;}

/* AUTO CPE */
.auto-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;}
.auto-card{background:var(--s2);border:1px solid var(--bdr);
           border-radius:8px;padding:14px;text-align:center;}
.auto-tau{font-family:var(--mono);font-size:10px;color:var(--text2);
           text-transform:uppercase;margin-bottom:6px;}
.auto-pct{font-family:var(--mono);font-size:11px;color:var(--text2);}

::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--bdr2);border-radius:3px;}

/* ── DECISION PANEL ── */
.decision-panel {
  background:linear-gradient(135deg,#141614,#1a1e18);
  border:2px solid var(--bdr2);border-radius:14px;
  padding:28px 32px;margin-bottom:22px;
}
.decision-verdict {
  display:flex;align-items:center;gap:20px;margin-bottom:24px;
}
.verdict-icon {
  width:64px;height:64px;border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  font-size:28px;flex-shrink:0;
}
.verdict-title {
  font-family:var(--mono);font-size:26px;font-weight:600;letter-spacing:.02em;
}
.verdict-sub {
  font-family:var(--mono);font-size:12px;color:var(--text2);margin-top:4px;
}
.decision-body {
  display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;
}
.dec-col h4 {
  font-family:var(--mono);font-size:10px;font-weight:600;
  text-transform:uppercase;letter-spacing:.1em;
  margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--bdr);
}
.dec-point {
  display:flex;gap:8px;margin-bottom:8px;
  font-size:12px;line-height:1.6;color:var(--text2);
}
.dec-dot {width:6px;height:6px;border-radius:50%;flex-shrink:0;margin-top:6px;}
.trigger-list {display:flex;flex-direction:column;gap:8px;}
.trigger-item {
  background:var(--s3);border:1px solid var(--bdr);
  border-radius:6px;padding:10px 12px;
}
.trigger-name {
  font-family:var(--mono);font-size:11px;font-weight:600;margin-bottom:4px;
}
.trigger-status {font-family:var(--mono);font-size:10px;color:var(--text2);}
.trigger-bar-wrap {height:3px;background:var(--bdr);border-radius:2px;margin-top:6px;}
.trigger-bar-fill {height:100%;border-radius:2px;}

.disc{font-size:11px;color:var(--text2);padding:14px 18px;
      background:var(--s1);border:1px solid var(--bdr);
      border-radius:8px;margin-top:20px;line-height:1.8;}
.up{color:var(--bull);}.dn{color:var(--bear);}
.wa{color:var(--warn);}
</style>
</head>
<body>
<header>
  <div class="h-brand">
    <div class="h-icon">⬡</div>
    <div>
      <div class="h-title">GOLD BUY SIGNAL DASHBOARD</div>
      <div class="h-sub">CPE Multi-Asset Framework · BullionStar Singapore · 20g Bar</div>
    </div>
  </div>
  <div class="h-meta">
    <div>Updated: <span id="hgen"></span></div>
    <div>Data: <span id="hdat"></span></div>
    <div style="color:var(--gold);margin-top:2px">USD/SGD: <span id="hfx"></span></div>
  </div>
</header>

<main>

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
      20g Bar — BullionStar Singapore
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

<!-- PRICE CHARTS -->
<div class="g2">
  <div class="card">
    <div class="ct">GC=F Gold Futures — USD/oz (Last 365 Days)</div>
    <div id="chart-usd" style="height:260px"></div>
  </div>
  <div class="card">
    <div class="ct">20g Bar Estimated Price — SGD (Last 365 Days)</div>
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
    <button class="tb" onclick="showCone('sgd',this)">SGD / 20g bar</button>
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
    <div style="overflow-x:auto">
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
    <div style="overflow-x:auto">
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
    'S$ ' + D.bar_sgd.toLocaleString('en-SG',{minimumFractionDigits:2,maximumFractionDigits:2});
  document.getElementById('bar-sub').textContent =
    D.bar_sub_text || ('Spot S$'+(D.gold_spot_sgd_g||D.gold_sgd_g).toFixed(2)+'/g × 20g × 0.8% premium');

  document.getElementById('dd-peak').innerHTML =
    '<span style="color:var(--bear)">'+D.dd_from_peak.toFixed(1)+'%</span> from 252d peak';
  document.getElementById('dd-peak-sub').textContent =
    'Peak: USD '+D.peak_252.toLocaleString()+' / S$'+(D.peak_bar_sgd||Math.round(D.peak_252*D.usd_per_sgd/31.1035*20*1.008));

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
    hovertemplate:'%{x}<br>S$%{y:,.2f}<extra></extra>',name:'20g bar SGD',
  }],{yaxis:{title:'SGD (20g bar)',tickformat:'S$,.0f'},xaxis:{type:'date'}});
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
  const currSgd = D.bar_sgd;

  // Historical last 63 days for context
  const hist_n = 63;
  const histX = D.chart_dates.slice(-hist_n).map((_,i)=>-(hist_n-1-i));
  const histY = D.chart_prices.slice(-hist_n);
  const histYSgd = D.chart_sgd_g.slice(-hist_n);

  function coneTraces(p10,p25,p50,p75,p90,histYArr,currency,tickfmt){
    return [
      {x:histX,y:histYArr,mode:'lines',name:'Historical (63d)',
       line:{color:'#C9A84C',width:2.5},
       hovertemplate:'Day %{x}<br>'+currency+'%{y:,.2f}<extra>Historical</extra>'},
      {x:[0,...taus,...[...taus].reverse()],
       y:[curr,...p10,...[...p90].reverse()],
       fill:'toself',fillcolor:'#E0555511',line:{color:'transparent'},
       name:'P10–P90',hoverinfo:'skip'},
      {x:[0,...taus,...[...taus].reverse()],
       y:[curr,...p25,...[...p75].reverse()],
       fill:'toself',fillcolor:'#E0555533',line:{color:'transparent'},
       name:'P25–P75',hoverinfo:'skip'},
      {x:[0,...taus],y:[curr,...p50],mode:'lines+markers',name:'P50 Median',
       line:{color:'#E05555',width:2.5},marker:{size:7},
       hovertemplate:'Day %{x}<br>'+currency+'%{y:,.2f}<extra>P50 Median</extra>'},
      {x:[0,...taus],y:[curr,...p90],mode:'lines',name:'P90 Optimistic',
       line:{color:'#4DB87A',width:1.5,dash:'dot'},
       hovertemplate:'Day %{x}<br>'+currency+'%{y:,.2f}<extra>P90 Optimistic</extra>'},
    ];
  }

  pl('cone-usd',
    coneTraces(D.cone_p10,D.cone_p25,D.cone_p50,D.cone_p75,D.cone_p90,histY,'$','$,.0f'),
    {xaxis:{title:'Trading days from today (0 = current)',
            zeroline:true,zerolinecolor:'#C9A84C44',zerolinewidth:1.5},
     yaxis:{title:'GC=F (USD/oz)',tickformat:'$,.0f'},
     showlegend:true,
     shapes:[{type:'line',x0:0,x1:0,y0:0,y1:1,xref:'x',yref:'paper',
              line:{color:'#C9A84C55',width:1.5,dash:'dot'}}]});

  pl('cone-sgd',
    coneTraces(D.cone_sgd_p10,D.cone_sgd_p25,D.cone_sgd_p50,D.cone_sgd_p75,D.cone_sgd_p90,histYSgd,'S$','S$,.0f'),
    {xaxis:{title:'Trading days from today (0 = current)',
            zeroline:true,zerolinecolor:'#C9A84C44',zerolinewidth:1.5},
     yaxis:{title:'20g Bar (SGD)',tickformat:'S$,.0f'},
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
      const label = in_t?'IN TAIL ✓':near?'NEAR ('+r.dist_pct.toFixed(1)+'%)':''+r.dist_pct.toFixed(1)+'%';
      const barW = in_t ? 100 : Math.max(0, 100 - Math.abs(r.dist_pct)*2);
      const barCol = in_t?'var(--bull)':near?'var(--warn)':'var(--neut)';
      const tailLabel = r.tail_type==='lower'?'lower tail (mean-reversion)':'upper tail (bull signal)';
      rowsHtml += `<div class="pred-row">
        <span class="pred-p">τ=${r.tau}d · q=${r.q} · ${tailLabel}</span>
        <span class="pred-v ${col}">${label}</span>
      </div>
      <div class="prog-wrap">
        <div class="prog-fill" style="width:${barW}%;background:${barCol}"></div>
      </div>
      <div style="font-family:var(--mono);font-size:9px;color:var(--text2);margin-bottom:8px">
        curr=${r.current.toFixed(2)}% · threshold=${r.threshold.toFixed(2)}%
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
    document.getElementById(el).innerHTML = arr.length ?
      arr.map(p=>'<div class="dec-point"><div class="dec-dot" style="background:'+p.col+'"></div><div>'+p.txt+'</div></div>').join('') :
      '<div style="color:var(--text2);font-size:12px;padding:8px">No clear evidence in this direction</div>';
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
      cur:ibit5 ? (ibit5.in_tail?'FIRING':'Currently '+ibit5.dist_pct.toFixed(1)+'% from threshold') : 'N/A',
      prog:ibit5 ? Math.max(0,Math.min(100,100+ibit5.dist_pct/5)) : 0,
      col:'var(--bull)', fired:!!(ibit5&&ibit5.in_tail) },
    { name:'Composite score &ge; 60',
      desc:'When score crosses 60 the balance of evidence tips toward buying',
      cur:score+'/100 — need 60',
      prog:score, col:score>=60?'var(--bull)':'var(--warn)', fired:score>=60 },
    { name:'SGD strengthens vs USD (300d)',
      desc:'Stronger SGD reduces the SGD cost of gold bars — also a CPE predictor',
      cur:sgd ? (sgd.in_tail?'FIRING':''+sgd.dist_pct.toFixed(1)+'% from threshold') : 'N/A',
      prog:sgd ? Math.max(0,Math.min(100,50+sgd.dist_pct/2)) : 0,
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
