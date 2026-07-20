"""
Causal validation of Fed Funds futures -- a third candidate genuinely-
forecasted field, per the user's suggestion. CME 30-Day Federal Funds futures
(ZQ=F) are a real, free, publicly-traded market forecast of the average
effective fed funds rate for the contract's delivery month -- the same kind
of aggregated forward-looking object VIX futures are for volatility, just
for monetary policy instead.

fed_funds_implied_change_21d = 21-trading-day change in (100 - ZQ=F close),
i.e. how the market's OWN rate-path forecast has been revised over the past
month -- the analogue of tracking how an NWP forecast run has updated from
the previous run, not just reading one static forecasted value. The raw
level is a slow-moving, highly serially-correlated object (mostly already
priced in); the *change* captures genuine new information reaching the
market.

Also runs a 3-way multivariate VAR test (fed_funds_implied_change_21d,
vix_term_slope, credit_spread_regime) against SPY realized vol -- does the
Fed-funds-based forecast add independent information beyond the other two
already-validated candidates?

Run: python 13_fed_funds_causal.py
Requires network access (yfinance, ZQ=F) -- caches to zq_futures_cache.parquet
Output: fed_funds_causal_results.json
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.api import VAR

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LAGS = [1, 5, 21]
CCF_MAX_LAG = 30
CACHE_PATH = os.path.join(OUT_DIR, "zq_futures_cache.parquet")

print("=" * 60)
print("  FED FUNDS FUTURES (ZQ=F) -- CAUSAL VALIDATION")
print("=" * 60)

if os.path.exists(CACHE_PATH):
    zq = pd.read_parquet(CACHE_PATH)["Close"]
    print(f"Loaded cached ZQ=F data: {len(zq)} rows")
else:
    import yfinance as yf
    d = yf.download("ZQ=F", period="max", progress=False, auto_adjust=True)
    zq = d["Close"]["ZQ=F"] if isinstance(d.columns, pd.MultiIndex) else d["Close"]
    zq.to_frame("Close").to_parquet(CACHE_PATH)
    print(f"Fetched and cached ZQ=F data: {len(zq)} rows, {zq.index.min().date()} .. {zq.index.max().date()}")

zq = zq[(zq >= 80) & (zq <= 100)]  # sanity filter -- realistic implied-rate range
fed_funds_implied = (100 - zq).rename("fed_funds_implied")
fed_funds_change21 = fed_funds_implied.diff(21).rename("fed_funds_implied_change_21d")
print(f"fed_funds_implied_change_21d: {len(fed_funds_change21.dropna())} obs")

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
spy = prices["SPY"].dropna()
spy_ret = np.log(spy / spy.shift(1)).rename("spy_ret")
spy_rvol21 = spy_ret.rolling(21).std().rename("spy_rvol21")
TARGETS = {"spy_ret": spy_ret, "spy_rvol21": spy_rvol21}

CANDIDATE = fed_funds_change21


def adf_verdict(series, name):
    s = series.dropna()
    stat, pval, *_ = adfuller(s, autolag="AIC")
    stationary = pval < 0.05
    print(f"  ADF {name}: stat={stat:.3f} p={pval:.4f} -> {'stationary' if stationary else 'NON-stationary'}")
    return stationary


print("\n--- Stationarity ---")
cand_stationary = adf_verdict(CANDIDATE, "fed_funds_implied_change_21d")
targ_stationary = {name: adf_verdict(s, name) for name, s in TARGETS.items()}

results = {"stationarity": {"fed_funds_implied_change_21d": bool(cand_stationary),
                             **{k: bool(v) for k, v in targ_stationary.items()}},
           "granger": {}, "ccf": {}, "multivariate": {}}


def maybe_diff(series, is_stationary):
    return series if is_stationary else series.diff()


cand_s = maybe_diff(CANDIDATE, cand_stationary).dropna()

print("\n--- Granger causality (both directions, lags {1,5,21}) ---")
for targ_name, targ_series in TARGETS.items():
    targ_s = maybe_diff(targ_series, targ_stationary[targ_name]).dropna()
    aligned = pd.concat([targ_s.rename("target"), cand_s.rename("cand")], axis=1).dropna()
    results["granger"][targ_name] = {"n": int(len(aligned)), "lags": {}}
    for lag in LAGS:
        r_fwd = grangercausalitytests(aligned[["target", "cand"]].values, maxlag=lag, verbose=False)
        r_rev = grangercausalitytests(aligned[["cand", "target"]].values, maxlag=lag, verbose=False)
        p_fwd = r_fwd[lag][0]["ssr_ftest"][1]
        p_rev = r_rev[lag][0]["ssr_ftest"][1]
        results["granger"][targ_name]["lags"][lag] = {"cand_causes_target_p": float(p_fwd),
                                                        "target_causes_cand_p": float(p_rev)}
        verdict = ("cand LEADS (mechanism-consistent)" if p_fwd < 0.05 <= p_rev else
                   "REVERSE (contradicts mechanism)" if p_rev < 0.05 <= p_fwd else
                   "BOTH significant (simultaneity)" if p_fwd < 0.05 and p_rev < 0.05 else
                   "neither significant")
        print(f"  fed_funds_implied_change_21d vs {targ_name} lag={lag}: p(cand->target)={p_fwd:.4f}, "
              f"p(target->cand)={p_rev:.4f} -> {verdict}")

print("\n--- Cross-correlation function (lags -30..+30) ---")
for targ_name, targ_series in TARGETS.items():
    aligned = pd.concat([targ_series.rename("target"), CANDIDATE.rename("cand")], axis=1).dropna()
    n = len(aligned)
    ci = 1.96 / np.sqrt(n)
    ccf_vals = {}
    for lag in range(-CCF_MAX_LAG, CCF_MAX_LAG + 1):
        if lag >= 0:
            a, b = aligned["cand"].values[:n - lag], aligned["target"].values[lag:]
        else:
            a, b = aligned["cand"].values[-lag:], aligned["target"].values[:n + lag]
        if len(a) < 100:
            continue
        ccf_vals[lag] = float(np.corrcoef(a, b)[0, 1])
    peak_lag = max(ccf_vals, key=lambda k: abs(ccf_vals[k]))
    results["ccf"][targ_name] = {"peak_lag": peak_lag, "peak_corr": ccf_vals[peak_lag], "sig_threshold": float(ci)}
    print(f"  fed_funds_implied_change_21d vs {targ_name}: peak |corr|={abs(ccf_vals[peak_lag]):.4f} "
          f"at lag={peak_lag}, sig_threshold={ci:.4f}")

# ── 3-way multivariate VAR: fed funds + vix term structure + credit spread, vs realized vol ──
print("\n--- 3-way multivariate VAR vs spy_rvol21 ---")
vix_res = json.load(open(os.path.join(OUT_DIR, "vix_term_structure_causal_results.json")))
macro = pd.read_parquet(os.path.join(OUT_DIR, "features_macro_interaction.parquet"))
macro["date"] = pd.to_datetime(macro["date"])
credit_spread_regime = macro[["date", "credit_spread_regime"]].drop_duplicates("date").set_index("date")["credit_spread_regime"]

vixy = prices["VIXY"].dropna()
vixm = prices["VIXM"].dropna()
common = vixy.index.intersection(vixm.index)
raw_ratio = np.log(vixm.reindex(common) / vixy.reindex(common))
vix_term_slope = ((raw_ratio - raw_ratio.rolling(200, min_periods=100).mean())
                   / raw_ratio.rolling(200, min_periods=100).std())

targ_s = maybe_diff(TARGETS["spy_rvol21"], targ_stationary["spy_rvol21"]).rename("spy_rvol21")
c1 = cand_s.rename("fed_funds_chg")
c2 = vix_term_slope.rename("vix_term_slope")  # already stationary by construction (z-scored)
c3 = credit_spread_regime.rename("credit_spread_regime")  # already stationary
joint = pd.concat([targ_s, c1, c2, c3], axis=1).dropna()
print(f"  joint sample: n={len(joint)}, {joint.index.min().date()} .. {joint.index.max().date()}")

var_res = VAR(joint).fit(5)
for causing in ["fed_funds_chg", "vix_term_slope", "credit_spread_regime"]:
    others = [c for c in ["fed_funds_chg", "vix_term_slope", "credit_spread_regime"] if c != causing]
    test = var_res.test_causality("spy_rvol21", [causing], kind="f")
    print(f"  {causing} -> spy_rvol21 controlling for the other two: F={test.test_statistic:.3f} "
          f"p={test.pvalue:.4f} ({'significant' if test.pvalue < 0.05 else 'NOT significant'})")
    results["multivariate"][causing] = {"F": float(test.test_statistic), "p": float(test.pvalue)}

out_path = os.path.join(OUT_DIR, "fed_funds_causal_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved results to {out_path}")
print("\nDone.")
