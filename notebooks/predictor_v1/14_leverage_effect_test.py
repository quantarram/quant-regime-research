"""
Leverage-effect asymmetry test -- the bridge from "predicted vol regime"
to an actual price signal.

credit_spread_regime and vix_term_slope are both validated (11/12_*.py) to
Granger-cause future SPY realized volatility. That's a magnitude statement,
not a direction. This tests whether elevated predicted stress from these
same two variables asymmetrically skews forward RETURNS toward the downside
specifically (the well-documented leverage effect: Black 1976, Christie
1982) rather than just widening the distribution symmetrically -- the
mechanism that would make "volatility is about to rise" translate into an
actual tradeable directional signal, the finance analogue of turning
"atmosphere is unstable" into an actual rain forecast via the full
convergence+moisture mechanism.

Two complementary tests per predictor, forward 21-trading-day SPY return
(matching the validated spy_rvol21 window):
  1. Decile sort: mean / skew / %negative of forward return by predictor
     decile -- does the top (highest predicted stress) decile show a more
     negative mean AND more negative skew than the bottom decile?
  2. Quantile regression at q=0.1 (downside tail), q=0.5 (median), q=0.9
     (upside tail): is the predictor's slope on the downside tail more
     negative than its slope on the upside tail is positive -- i.e. does it
     pull the left tail down more than it pushes the right tail up?

Also computed as context: the classical (unconditional) leverage effect --
correlation between contemporaneous returns and volatility changes -- to
confirm the mechanism is present in this data at all before looking for it
conditional on our predictors.

Pure statistics (statsmodels), no ML.

Run: python 14_leverage_effect_test.py
Output: leverage_effect_results.json
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from scipy.stats import skew

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FWD_HORIZON = 21

print("=" * 60)
print("  LEVERAGE-EFFECT ASYMMETRY TEST")
print("=" * 60)

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
spy = prices["SPY"].dropna()
spy_ret = np.log(spy / spy.shift(1))
spy_rvol21 = spy_ret.rolling(21).std()
fwd_ret = np.log(spy.shift(-FWD_HORIZON) / spy).rename("fwd_ret")

# ── Context: classical unconditional leverage effect ─────────────────────
rvol_chg = spy_rvol21.diff(1)
context = pd.concat([spy_ret.rename("ret"), rvol_chg.rename("rvol_chg")], axis=1).dropna()
classical_corr = context["ret"].corr(context["rvol_chg"])
print(f"\nClassical leverage effect check: corr(same-day return, change in 21d realized vol) = "
      f"{classical_corr:.4f} (expect negative -- down days coincide with rising vol)")

# ── Load the two validated predictors ─────────────────────────────────────
macro = pd.read_parquet(os.path.join(OUT_DIR, "features_macro_interaction.parquet"))
macro["date"] = pd.to_datetime(macro["date"])
credit_spread_regime = macro[["date", "credit_spread_regime"]].drop_duplicates("date").set_index("date")["credit_spread_regime"]

vixy = prices["VIXY"].dropna()
vixm = prices["VIXM"].dropna()
common = vixy.index.intersection(vixm.index)
raw_ratio = np.log(vixm.reindex(common) / vixy.reindex(common))
vix_term_slope = ((raw_ratio - raw_ratio.rolling(200, min_periods=100).mean())
                   / raw_ratio.rolling(200, min_periods=100).std()).rename("vix_term_slope")

PREDICTORS = {"credit_spread_regime": credit_spread_regime, "vix_term_slope": vix_term_slope}

results = {"classical_leverage_corr": float(classical_corr), "predictors": {}}

for name, pred in PREDICTORS.items():
    print(f"\n--- {name} ---")
    df = pd.concat([pred.rename("x"), fwd_ret], axis=1).dropna()
    df["decile"] = pd.qcut(df["x"], 10, labels=False, duplicates="drop")

    print("Decile sort (0=lowest predictor value, 9=highest -> highest predicted stress):")
    decile_stats = []
    for d in sorted(df["decile"].unique()):
        sub = df[df["decile"] == d]["fwd_ret"]
        decile_stats.append({"decile": int(d), "n": int(len(sub)), "mean": float(sub.mean()),
                              "skew": float(skew(sub)), "pct_negative": float((sub < 0).mean())})
        print(f"  decile {int(d)}: n={len(sub)}, mean_fwd_ret={sub.mean():.4f}, "
              f"skew={skew(sub):.3f}, pct_negative={100*(sub < 0).mean():.1f}%")
    results["predictors"][name] = {"decile_sort": decile_stats}

    top = df[df["decile"] == df["decile"].max()]["fwd_ret"]
    bot = df[df["decile"] == df["decile"].min()]["fwd_ret"]
    print(f"  Top decile (highest {name}) vs bottom decile: "
          f"mean {top.mean():.4f} vs {bot.mean():.4f}, skew {skew(top):.3f} vs {skew(bot):.3f}")
    results["predictors"][name]["top_vs_bottom"] = {
        "top_mean": float(top.mean()), "bottom_mean": float(bot.mean()),
        "top_skew": float(skew(top)), "bottom_skew": float(skew(bot)),
    }

    print("Quantile regression (fwd_ret ~ predictor), slope at each quantile:")
    X = sm.add_constant(df["x"])
    qr_results = {}
    for q in [0.1, 0.5, 0.9]:
        model = QuantReg(df["fwd_ret"], X).fit(q=q)
        slope = model.params["x"]
        pval = model.pvalues["x"]
        qr_results[q] = {"slope": float(slope), "p": float(pval)}
        print(f"  q={q}: slope={slope:.5f} (p={pval:.4f})")
    results["predictors"][name]["quantile_regression"] = qr_results

    downside_slope = qr_results[0.1]["slope"]
    upside_slope = qr_results[0.9]["slope"]
    asymmetric = downside_slope < 0 and abs(downside_slope) > abs(upside_slope)
    print(f"  Asymmetry check: |downside slope|={abs(downside_slope):.5f} vs "
          f"|upside slope|={abs(upside_slope):.5f} -> "
          f"{'ASYMMETRIC, downside-dominant (leverage-effect-consistent)' if asymmetric else 'not clearly asymmetric in the expected direction'}")
    results["predictors"][name]["asymmetric_downside_dominant"] = bool(asymmetric)

out_path = os.path.join(OUT_DIR, "leverage_effect_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved results to {out_path}")
print("\nDone.")
