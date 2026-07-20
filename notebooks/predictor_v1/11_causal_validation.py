"""
Causal validation of candidate predictors -- BEFORE any more ML.

Everything else in predictor_v1/ tested candidates by adding them to a
LightGBM model and checking whether a tail-aware metric improved. That's a
test of correlation-with-model-skill, not of causality. This script asks the
econometric equivalent of "what does the physics say" for four candidates
that came out of Wave 1/2/Phase C, each checked against a citable economic
transmission mechanism -- pure statsmodels time-series analysis, no ML,
no walk-forward, no train/test split.

Candidates and the mechanism each is checked against:
  credit_spread_regime (HYG/LQD ratio z-score)
    -> funding-liquidity / intermediary-leverage spirals
       (Adrian & Shin 2010; Longstaff, Mithal & Neis 2005)
  uup_trend_regime (dollar trend z-score)
    -> dollar as a global risk-taking/leverage constraint
       (Avdjiev, Du, Koch & Shin 2019)
  curve_curvature_10ybelly (yield curve butterfly)
    -> yield curve shape as a forward-looking risk-appetite signal
       (Estrella & Mishkin 1996)
  hmm_prob_high_vol (regime-switching state)
    -> volatility clustering / regime persistence
       (Hamilton 1989; Ang & Bekaert 2002)

Each mechanism predicts the candidate should LEAD SPY (Granger-cause it,
positive-lag CCF peak) more than the reverse. Bidirectional or
reverse-dominant causality contradicts the mechanism, regardless of
whatever hit-rate number the ML grids produced for that feature.

Run: python 11_causal_validation.py
Output: causal_validation_results.json
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import grangercausalitytests, adfuller

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LAGS = [1, 5, 21]
CCF_MAX_LAG = 30

print("=" * 60)
print("  CAUSAL VALIDATION (statsmodels only, no ML)")
print("=" * 60)


# ── Sanity check: does grangercausalitytests actually detect a planted ──
# ── lead-lag relationship, before trusting it on real data? ─────────────
def sanity_check_granger():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=n)
    noise = rng.normal(scale=0.3, size=n)
    y = np.zeros(n)
    y[1:] = 0.5 * x[:-1] + noise[1:]  # y[t] = 0.5*x[t-1] + noise -- x should Granger-cause y at lag>=1
    data_fwd = np.column_stack([y, x])   # tests x -> y (real relationship, should be significant)
    data_rev = np.column_stack([x, y])   # tests y -> x (should NOT be significant)

    res_fwd = grangercausalitytests(data_fwd, maxlag=1, verbose=False)
    res_rev = grangercausalitytests(data_rev, maxlag=1, verbose=False)
    p_fwd = res_fwd[1][0]["ssr_ftest"][1]
    p_rev = res_rev[1][0]["ssr_ftest"][1]
    print(f"Sanity check (planted y[t]=0.5*x[t-1]+noise): p(x->y)={p_fwd:.6f} (expect <0.01), "
          f"p(y->x)={p_rev:.6f} (expect >0.05)")
    assert p_fwd < 0.01, "Granger test failed to detect a planted causal relationship"
    assert p_rev > 0.05, "Granger test found spurious reverse causality in planted data"
    print("Granger causality mechanics verified on synthetic data.\n")


sanity_check_granger()

# ── Load SPY and build both target framings ──────────────────────────────
prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
spy = prices["SPY"].dropna()
spy_ret = np.log(spy / spy.shift(1)).rename("spy_ret")
spy_rvol21 = spy_ret.rolling(21).std().rename("spy_rvol21")

TARGETS = {"spy_ret": spy_ret, "spy_rvol21": spy_rvol21}

# ── Load candidates (dedup to one row per date -- these are market-wide) ──
macro = pd.read_parquet(os.path.join(OUT_DIR, "features_macro_interaction.parquet"))
macro["date"] = pd.to_datetime(macro["date"])
macro = macro[["date", "uup_trend_regime", "credit_spread_regime"]].drop_duplicates("date").set_index("date")

term = pd.read_parquet(os.path.join(OUT_DIR, "features_term_structure.parquet"))
term["date"] = pd.to_datetime(term["date"])
term = term[["date", "curve_curvature_10ybelly"]].set_index("date")

regime = pd.read_parquet(os.path.join(OUT_DIR, "features_regime_switching.parquet"))
regime["date"] = pd.to_datetime(regime["date"])
regime = regime[["date", "hmm_prob_high_vol"]].set_index("date")

CANDIDATES = {
    "credit_spread_regime": macro["credit_spread_regime"],
    "uup_trend_regime": macro["uup_trend_regime"],
    "curve_curvature_10ybelly": term["curve_curvature_10ybelly"],
    "hmm_prob_high_vol": regime["hmm_prob_high_vol"],
}

MECHANISMS = {
    "credit_spread_regime": "funding-liquidity/intermediary-leverage spiral (Adrian & Shin 2010)",
    "uup_trend_regime": "dollar as global risk-taking/leverage constraint (Avdjiev, Du, Koch & Shin 2019)",
    "curve_curvature_10ybelly": "yield curve shape as forward-looking risk signal (Estrella & Mishkin 1996)",
    "hmm_prob_high_vol": "volatility clustering/regime persistence (Hamilton 1989)",
}


def adf_verdict(series, name):
    s = series.dropna()
    stat, pval, *_ = adfuller(s, autolag="AIC")
    stationary = pval < 0.05
    print(f"  ADF {name}: stat={stat:.3f} p={pval:.4f} -> {'stationary' if stationary else 'NON-stationary'}")
    return stationary, float(pval)


print("--- Stationarity checks ---")
print("Reference checks (known priors):")
adf_verdict(spy, "raw SPY price (expect non-stationary)")
adf_verdict(spy_ret, "SPY daily return (expect stationary)")
print("Targets:")
target_stationary = {}
for name, s in TARGETS.items():
    target_stationary[name], _ = adf_verdict(s, name)
print("Candidates:")
candidate_stationary = {}
for name, s in CANDIDATES.items():
    candidate_stationary[name], _ = adf_verdict(s, name)

results = {"mechanisms": MECHANISMS, "stationarity": {}, "granger": {}, "ccf": {}}
for name, ok in {**target_stationary, **candidate_stationary}.items():
    results["stationarity"][name] = bool(ok)


def maybe_difference(series, is_stationary):
    return series if is_stationary else series.diff()


print("\n--- Granger causality (both directions, lags {1,5,21}) ---")
for cand_name, cand_series in CANDIDATES.items():
    cand_s = maybe_difference(cand_series, candidate_stationary[cand_name]).dropna()
    results["granger"][cand_name] = {}
    for targ_name, targ_series in TARGETS.items():
        targ_s = maybe_difference(targ_series, target_stationary[targ_name]).dropna()
        aligned = pd.concat([targ_s.rename("target"), cand_s.rename("cand")], axis=1).dropna()
        if len(aligned) < 300:
            print(f"  {cand_name} vs {targ_name}: insufficient overlap ({len(aligned)} rows), skipping")
            continue

        results["granger"][cand_name][targ_name] = {"n": int(len(aligned)), "lags": {}}
        for lag in LAGS:
            data_fwd = aligned[["target", "cand"]].values  # tests cand -> target
            data_rev = aligned[["cand", "target"]].values  # tests target -> cand
            try:
                r_fwd = grangercausalitytests(data_fwd, maxlag=lag, verbose=False)
                r_rev = grangercausalitytests(data_rev, maxlag=lag, verbose=False)
                f_fwd, p_fwd = r_fwd[lag][0]["ssr_ftest"][0], r_fwd[lag][0]["ssr_ftest"][1]
                f_rev, p_rev = r_rev[lag][0]["ssr_ftest"][0], r_rev[lag][0]["ssr_ftest"][1]
            except Exception as e:
                print(f"  {cand_name} vs {targ_name} lag={lag}: FAILED ({e})")
                continue
            results["granger"][cand_name][targ_name]["lags"][lag] = {
                "cand_causes_target_F": float(f_fwd), "cand_causes_target_p": float(p_fwd),
                "target_causes_cand_F": float(f_rev), "target_causes_cand_p": float(p_rev),
            }
            verdict = ("cand LEADS (mechanism-consistent)" if p_fwd < 0.05 <= p_rev else
                       "REVERSE (target leads cand -- contradicts mechanism)" if p_rev < 0.05 <= p_fwd else
                       "BOTH directions significant (likely simultaneity, not clean causality)" if p_fwd < 0.05 and p_rev < 0.05 else
                       "neither direction significant")
            print(f"  {cand_name} vs {targ_name} lag={lag}: p(cand->target)={p_fwd:.4f}, "
                  f"p(target->cand)={p_rev:.4f} -> {verdict}")

print("\n--- Cross-correlation function (lags -30..+30, positive lag = candidate leads) ---")
for cand_name, cand_series in CANDIDATES.items():
    results["ccf"][cand_name] = {}
    for targ_name, targ_series in TARGETS.items():
        aligned = pd.concat([targ_series.rename("target"), cand_series.rename("cand")], axis=1).dropna()
        if len(aligned) < 300:
            continue
        n = len(aligned)
        ci = 1.96 / np.sqrt(n)  # approximate (iid) significance band -- these series are autocorrelated,
        # so treat this as an indicative rather than exact threshold
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
        results["ccf"][cand_name][targ_name] = {
            "peak_lag": peak_lag, "peak_corr": ccf_vals[peak_lag], "sig_threshold": float(ci),
            "significant": bool(abs(ccf_vals[peak_lag]) > ci),
        }
        print(f"  {cand_name} vs {targ_name}: peak |corr|={abs(ccf_vals[peak_lag]):.4f} at lag={peak_lag} "
              f"({'candidate leads' if peak_lag > 0 else 'candidate lags/contemporaneous' if peak_lag <= 0 else ''}), "
              f"sig_threshold={ci:.4f}")

out_path = os.path.join(OUT_DIR, "causal_validation_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved results to {out_path}")
print("\nDone.")
