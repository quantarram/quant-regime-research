"""
Causal validation of a genuinely FORECASTED field, not a nowcast -- the gap
identified after 11_causal_validation.py: everything tested so far used
today's observed value of a driver to predict tomorrow's SPY, implicitly
assuming today's value persists. VIX futures term structure is different in
kind: it's the market's own aggregated forward-looking forecast (from the
whole options chain), directly analogous to reading precipitation off an
NWP model's forecasted fields rather than nowcasting from today's obs.

vix_term_slope = log(VIXM / VIXY) -- VIXM tracks ~4th-7th month VIX futures,
VIXY tracks ~1st-2nd month. Positive = contango (market expects vol similar
or lower further out, "normal"/complacent term structure); negative =
backwardation (near-term fear priced above the longer-dated forecast, the
classic stress signature). This is the market's own forecast of how vol
will evolve, read directly off traded futures-based ETF prices -- not
something we estimate from historical data the way every other candidate
in this program has been.

Per the user's explicit instruction: credit_spread_regime is RETAINED
alongside this, not discarded -- it passed causal validation cleanly
(11_causal_validation.py: leads SPY realized vol at all 3 lags, non-reverse-
dominated). This script (a) validates vix_term_slope with the same rigor,
and (b) runs a multivariate VAR-based joint test to see whether the two
carry independent information or one subsumes the other.

Run: python 12_vix_term_structure_causal.py
Output: vix_term_structure_causal_results.json
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
print("  VIX TERM STRUCTURE (genuinely forecasted field) -- CAUSAL VALIDATION")
print("=" * 60)

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
spy = prices["SPY"].dropna()
spy_ret = np.log(spy / spy.shift(1)).rename("spy_ret")
spy_rvol21 = spy_ret.rolling(21).std().rename("spy_rvol21")
TARGETS = {"spy_ret": spy_ret, "spy_rvol21": spy_rvol21}

vixy = prices["VIXY"].dropna()
vixm = prices["VIXM"].dropna()
common = vixy.index.intersection(vixm.index)
raw_ratio = np.log(vixm.reindex(common) / vixy.reindex(common))
# VIXY/VIXM are both real ETFs that structurally bleed value from contango roll costs over their
# lifetime (VIXY, short-dated futures, bleeds faster than VIXM, mid-dated) -- their raw price ratio
# is dominated by this secular differential-decay trend, not day-to-day term-structure fluctuation
# (confirmed: VIXY fell from 633,840 to 23.46 over the sample, a real, well-documented structural
# effect, not a data error -- the 4 largest single-day moves are all genuine vol-spike days: Feb
# 2018 Volmageddon, Mar/Jun 2020 COVID, Aug 2024 carry-trade unwind). Z-score against the ratio's
# own 200d trailing mean to strip the secular decay trend and isolate genuine contango/backwardation
# shifts, the same fix already applied to the CL=F/USO and GC=F/GLD basis features.
vix_term_slope = ((raw_ratio - raw_ratio.rolling(200, min_periods=100).mean())
                   / raw_ratio.rolling(200, min_periods=100).std()).rename("vix_term_slope")
print(f"vix_term_slope: {len(vix_term_slope.dropna())} obs, "
      f"{vix_term_slope.dropna().index.min().date()} .. {vix_term_slope.dropna().index.max().date()}")
print(f"  mean={vix_term_slope.mean():.4f}, pct negative={100*(vix_term_slope.dropna()<0).mean():.1f}%")

macro = pd.read_parquet(os.path.join(OUT_DIR, "features_macro_interaction.parquet"))
macro["date"] = pd.to_datetime(macro["date"])
credit_spread_regime = macro[["date", "credit_spread_regime"]].drop_duplicates("date").set_index("date")["credit_spread_regime"]

CANDIDATES = {"vix_term_slope": vix_term_slope, "credit_spread_regime": credit_spread_regime}

results = {"stationarity": {}, "granger": {}, "ccf": {}, "multivariate": {}}


def adf_verdict(series, name):
    s = series.dropna()
    stat, pval, *_ = adfuller(s, autolag="AIC")
    stationary = pval < 0.05
    print(f"  ADF {name}: stat={stat:.3f} p={pval:.4f} -> {'stationary' if stationary else 'NON-stationary'}")
    return stationary


print("\n--- Stationarity ---")
target_stationary = {name: adf_verdict(s, name) for name, s in TARGETS.items()}
candidate_stationary = {name: adf_verdict(s, name) for name, s in CANDIDATES.items()}
for name, ok in {**target_stationary, **candidate_stationary}.items():
    results["stationarity"][name] = bool(ok)


def maybe_diff(series, is_stationary):
    return series if is_stationary else series.diff()


print("\n--- Granger causality (both directions, lags {1,5,21}) ---")
for cand_name, cand_series in CANDIDATES.items():
    cand_s = maybe_diff(cand_series, candidate_stationary[cand_name]).dropna()
    results["granger"][cand_name] = {}
    for targ_name, targ_series in TARGETS.items():
        targ_s = maybe_diff(targ_series, target_stationary[targ_name]).dropna()
        aligned = pd.concat([targ_s.rename("target"), cand_s.rename("cand")], axis=1).dropna()
        if len(aligned) < 300:
            print(f"  {cand_name} vs {targ_name}: insufficient overlap, skipping")
            continue
        results["granger"][cand_name][targ_name] = {"n": int(len(aligned)), "lags": {}}
        for lag in LAGS:
            r_fwd = grangercausalitytests(aligned[["target", "cand"]].values, maxlag=lag, verbose=False)
            r_rev = grangercausalitytests(aligned[["cand", "target"]].values, maxlag=lag, verbose=False)
            f_fwd, p_fwd = r_fwd[lag][0]["ssr_ftest"][0], r_fwd[lag][0]["ssr_ftest"][1]
            f_rev, p_rev = r_rev[lag][0]["ssr_ftest"][0], r_rev[lag][0]["ssr_ftest"][1]
            results["granger"][cand_name][targ_name]["lags"][lag] = {
                "cand_causes_target_p": float(p_fwd), "target_causes_cand_p": float(p_rev)}
            verdict = ("cand LEADS (mechanism-consistent)" if p_fwd < 0.05 <= p_rev else
                       "REVERSE (contradicts mechanism)" if p_rev < 0.05 <= p_fwd else
                       "BOTH significant (simultaneity)" if p_fwd < 0.05 and p_rev < 0.05 else
                       "neither significant")
            print(f"  {cand_name} vs {targ_name} lag={lag}: p(cand->target)={p_fwd:.4f}, "
                  f"p(target->cand)={p_rev:.4f} -> {verdict}")

print("\n--- Cross-correlation function (lags -30..+30) ---")
for cand_name, cand_series in CANDIDATES.items():
    results["ccf"][cand_name] = {}
    for targ_name, targ_series in TARGETS.items():
        aligned = pd.concat([targ_series.rename("target"), cand_series.rename("cand")], axis=1).dropna()
        if len(aligned) < 300:
            continue
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
        if not ccf_vals:
            continue
        peak_lag = max(ccf_vals, key=lambda k: abs(ccf_vals[k]))
        results["ccf"][cand_name][targ_name] = {"peak_lag": peak_lag, "peak_corr": ccf_vals[peak_lag],
                                                  "sig_threshold": float(ci)}
        print(f"  {cand_name} vs {targ_name}: peak |corr|={abs(ccf_vals[peak_lag]):.4f} at lag={peak_lag}, "
              f"sig_threshold={ci:.4f}")

# ── Multivariate: does each candidate add independent information beyond the other? ──
print("\n--- Multivariate VAR: does each candidate Granger-cause spy_rvol21 controlling for the other? ---")
targ_s = maybe_diff(TARGETS["spy_rvol21"], target_stationary["spy_rvol21"]).rename("spy_rvol21")
c1 = maybe_diff(CANDIDATES["vix_term_slope"], candidate_stationary["vix_term_slope"]).rename("vix_term_slope")
c2 = maybe_diff(CANDIDATES["credit_spread_regime"], candidate_stationary["credit_spread_regime"]).rename("credit_spread_regime")
joint = pd.concat([targ_s, c1, c2], axis=1).dropna()
print(f"  joint sample: n={len(joint)}, {joint.index.min().date()} .. {joint.index.max().date()}")

for lag in [5]:
    var_model = VAR(joint)
    var_res = var_model.fit(lag)
    for causing in ["vix_term_slope", "credit_spread_regime"]:
        test = var_res.test_causality("spy_rvol21", [causing], kind="f")
        print(f"  lag={lag}: {causing} -> spy_rvol21 controlling for the other predictor: "
              f"F={test.test_statistic:.3f} p={test.pvalue:.4f} "
              f"({'significant' if test.pvalue < 0.05 else 'NOT significant'})")
        results["multivariate"][f"{causing}_controlling_for_other_lag{lag}"] = {
            "F": float(test.test_statistic), "p": float(test.pvalue)}
    joint_test = var_res.test_causality("spy_rvol21", ["vix_term_slope", "credit_spread_regime"], kind="f")
    print(f"  lag={lag}: BOTH jointly -> spy_rvol21: F={joint_test.test_statistic:.3f} p={joint_test.pvalue:.4f}")
    results["multivariate"][f"both_joint_lag{lag}"] = {"F": float(joint_test.test_statistic), "p": float(joint_test.pvalue)}

out_path = os.path.join(OUT_DIR, "vix_term_structure_causal_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved results to {out_path}")
print("\nDone.")
