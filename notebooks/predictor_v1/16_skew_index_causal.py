"""
Causal validation of the CBOE SKEW Index -- the candidate that actually
matches what the user is asking for: not a ratio I constructed from two
ETFs, but a professionally-designed, purpose-built, freely and publicly
published index (CBOE, updated daily, 36 years of history) whose entire
purpose is to price the market's own view of tail/crash risk from
out-of-the-money S&P 500 options. Higher SKEW = market pricing fatter left
tail risk. This is a genuine forecast object in the same sense VIX is, but
specifically for tail asymmetry -- directly relevant to the leverage-effect
question (14_leverage_effect_test.py) rather than needing to be inferred
from an unstable, DIY-constructed ratio.

Same rigor as every other candidate: ADF stationarity, Granger causality
both directions at lags {1,5,21} against SPY returns and 21d realized vol,
CCF, and a 3-way multivariate VAR test alongside credit_spread_regime and
vix_term_slope (both already validated) to see whether SKEW adds
independent information or is redundant with what's already been found.

Run: python 16_skew_index_causal.py
Output: skew_index_causal_results.json
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

print("=" * 60)
print("  CBOE SKEW INDEX -- CAUSAL VALIDATION")
print("=" * 60)

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
spy = prices["SPY"].dropna()
spy_ret = np.log(spy / spy.shift(1)).rename("spy_ret")
spy_rvol21 = spy_ret.rolling(21).std().rename("spy_rvol21")
TARGETS = {"spy_ret": spy_ret, "spy_rvol21": spy_rvol21}

skew_idx = prices["^SKEW"].dropna().rename("skew_idx")
print(f"^SKEW: {len(skew_idx)} obs, {skew_idx.index.min().date()} .. {skew_idx.index.max().date()}")
print(f"  mean={skew_idx.mean():.2f}, min={skew_idx.min():.2f}, max={skew_idx.max():.2f}")


def adf_verdict(series, name):
    s = series.dropna()
    stat, pval, *_ = adfuller(s, autolag="AIC")
    stationary = pval < 0.05
    print(f"  ADF {name}: stat={stat:.3f} p={pval:.4f} -> {'stationary' if stationary else 'NON-stationary'}")
    return stationary


print("\n--- Stationarity ---")
skew_stationary = adf_verdict(skew_idx, "skew_idx (raw level)")
target_stationary = {name: adf_verdict(s, name) for name, s in TARGETS.items()}


def maybe_diff(series, is_stationary):
    return series if is_stationary else series.diff()


results = {"stationarity": {"skew_idx": bool(skew_stationary),
                             **{k: bool(v) for k, v in target_stationary.items()}},
           "granger": {}, "ccf": {}, "multivariate": {}}

skew_s = maybe_diff(skew_idx, skew_stationary).dropna()

print("\n--- Granger causality (both directions, lags {1,5,21}) ---")
for targ_name, targ_series in TARGETS.items():
    targ_s = maybe_diff(targ_series, target_stationary[targ_name]).dropna()
    aligned = pd.concat([targ_s.rename("target"), skew_s.rename("cand")], axis=1).dropna()
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
        print(f"  skew_idx vs {targ_name} lag={lag}: p(cand->target)={p_fwd:.4f}, "
              f"p(target->cand)={p_rev:.4f} -> {verdict}")

print("\n--- Cross-correlation function (lags -30..+30) ---")
for targ_name, targ_series in TARGETS.items():
    aligned = pd.concat([targ_series.rename("target"), skew_idx.rename("cand")], axis=1).dropna()
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
    print(f"  skew_idx vs {targ_name}: peak |corr|={abs(ccf_vals[peak_lag]):.4f} at lag={peak_lag}, "
          f"sig_threshold={ci:.4f}")

# ── Does SKEW add independent info beyond the two already-validated candidates? ──
print("\n--- 3-way multivariate VAR vs spy_rvol21 (skew_idx + vix_term_slope + credit_spread_regime) ---")
macro = pd.read_parquet(os.path.join(OUT_DIR, "features_macro_interaction.parquet"))
macro["date"] = pd.to_datetime(macro["date"])
credit_spread_regime = macro[["date", "credit_spread_regime"]].drop_duplicates("date").set_index("date")["credit_spread_regime"]

vixy = prices["VIXY"].dropna()
vixm = prices["VIXM"].dropna()
common = vixy.index.intersection(vixm.index)
raw_ratio = np.log(vixm.reindex(common) / vixy.reindex(common))
vix_term_slope = ((raw_ratio - raw_ratio.rolling(200, min_periods=100).mean())
                   / raw_ratio.rolling(200, min_periods=100).std())

targ_s = maybe_diff(TARGETS["spy_rvol21"], target_stationary["spy_rvol21"]).rename("spy_rvol21")
c1 = skew_s.rename("skew_idx")
c2 = vix_term_slope.rename("vix_term_slope")
c3 = credit_spread_regime.rename("credit_spread_regime")
joint = pd.concat([targ_s, c1, c2, c3], axis=1).dropna()
print(f"  joint sample: n={len(joint)}, {joint.index.min().date()} .. {joint.index.max().date()}")

var_res = VAR(joint).fit(5)
for causing in ["skew_idx", "vix_term_slope", "credit_spread_regime"]:
    test = var_res.test_causality("spy_rvol21", [causing], kind="f")
    print(f"  {causing} -> spy_rvol21 controlling for the other two: F={test.test_statistic:.3f} "
          f"p={test.pvalue:.4f} ({'significant' if test.pvalue < 0.05 else 'NOT significant'})")
    results["multivariate"][causing] = {"F": float(test.test_statistic), "p": float(test.pvalue)}

out_path = os.path.join(OUT_DIR, "skew_index_causal_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved results to {out_path}")
print("\nDone.")
