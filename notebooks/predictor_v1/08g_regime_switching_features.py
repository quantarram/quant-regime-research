"""
Wave-2 predictor panel: regime-switching state (Hamilton 1989).

A 2-state Gaussian HMM fit on SPY's daily log returns, refit every 21
trading days on a trailing 2-year window (point-in-time -- each refit only
sees data up to that date), giving a discrete steering-regime label as an
alternative to the continuous multiplicative-interaction framing used
elsewhere. The regime state is held constant between refits (a "slow"
variable, consistent with the steering-flow framing throughout this plan).

Feature: hmm_prob_high_vol = posterior probability of being in the
higher-variance of the two states, as of the most recent refit date.

Run: python 08g_regime_switching_features.py
Output: features_regime_switching.parquet
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from hmmlearn.hmm import GaussianHMM

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOOKBACK = 504
STRIDE = 21

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
# dropna BEFORE shift -- see 08a_vrp_features.py's comment: shifting on the raw column before
# dropping the weekend NaN rows (introduced once crypto joined the shared calendar ~2014-09-17)
# silently discards every Monday's return instead of computing it correctly against Friday's close.
spy = prices["SPY"].dropna()
spy_ret = np.log(spy / spy.shift(1)).dropna()
idx = spy_ret.index
n = len(idx)

rows = []
n_fail = 0
for pos in range(LOOKBACK - 1, n, STRIDE):
    window = spy_ret.values[pos - LOOKBACK + 1: pos + 1].reshape(-1, 1)
    try:
        model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=0)
        model.fit(window)
        variances = model.covars_.flatten()
        high_vol_state = int(np.argmax(variances))
        post = model.predict_proba(window)
        prob_high_vol = float(post[-1, high_vol_state])
    except Exception:
        prob_high_vol = np.nan
        n_fail += 1
    rows.append({"date": idx[pos], "hmm_prob_high_vol": prob_high_vol})

sparse = pd.DataFrame(rows)
print(f"Fit {len(sparse)} HMM refits ({n_fail} failed and left NaN)")

full_dates = pd.DataFrame({"date": idx})
feat = full_dates.merge(sparse, on="date", how="left").ffill()

out_path = os.path.join(OUT_DIR, "features_regime_switching.parquet")
feat.to_parquet(out_path)
print(f"Saved {len(feat)} rows x {len(feat.columns)} cols to {out_path}")
print(feat.dropna().tail(5))
print(f"\nDistribution: mean={feat['hmm_prob_high_vol'].mean():.3f}, "
      f"pct>0.5={100*(feat['hmm_prob_high_vol']>0.5).mean():.1f}%")
