"""
Trace Moment / Double Trace Moment analysis of the raw SPY price field.

Estimates the universal-multifractal parameters (alpha, C1, H) via the
standard TM/DTM procedure (Lavallee et al. 1993; Schertzer & Lovejoy 1987),
following the same methodology used in Ramanathan et al. (2022, HESS) for
rainfall rate: normalized trace moments of the raw (untransformed) field,
block-averaged over a dyadic cascade, with alpha extracted from the slope
of log K(q_ref, eta) vs log eta.

Field: raw daily SPY closing price (no log, no return, no differencing
beyond what the cascade/structure-function analysis itself computes).
"""
import pandas as pd
import numpy as np
import json
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
spy = prices["SPY"].dropna()
flux_full = spy.values  # raw price field, untransformed, positive by construction

N = len(flux_full)
n_levels = int(np.floor(np.log2(N)))
N_use = 2 ** n_levels
flux = flux_full[-N_use:]


def build_pyramid(field):
    """lambda (scale ratio) -> array of block-averaged values R_lambda(j)."""
    pyramid = {}
    cur = field.copy()
    lam = len(cur)
    pyramid[lam] = cur.copy()
    while len(cur) > 1:
        cur = cur.reshape(-1, 2).mean(axis=1)
        lam = len(cur)
        pyramid[lam] = cur.copy()
    return pyramid


pyramid = build_pyramid(flux)
lambdas = sorted(pyramid.keys())
fit_lambdas = [l for l in lambdas if l >= 4]

qs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
Kq, Kq_r2 = {}, {}
for q in qs:
    logT, logL = [], []
    for lam in fit_lambdas:
        vals = pyramid[lam]
        mean_q = np.mean(vals ** q)
        mean_1_q = np.mean(vals) ** q
        if mean_q <= 0 or mean_1_q <= 0:
            continue
        TM = mean_q / mean_1_q
        if TM <= 0 or not np.isfinite(TM):
            continue
        logT.append(np.log(TM))
        logL.append(np.log(lam))
    b, a = np.polyfit(logL, logT, 1)
    yhat = a + b * np.array(logL)
    ss_res = np.sum((np.array(logT) - yhat) ** 2)
    ss_tot = np.sum((np.array(logT) - np.mean(logT)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    Kq[q], Kq_r2[q] = b, r2

# H via unnormalized first moment
logT, logL = [], []
for lam in fit_lambdas:
    m1 = np.mean(pyramid[lam])
    if m1 <= 0:
        continue
    logT.append(np.log(m1))
    logL.append(np.log(lam))
b, a = np.polyfit(logL, logT, 1)
H_est = -b
yhat = a + b * np.array(logL)
denom = np.sum((np.array(logT) - np.mean(logT)) ** 2)
r2_H = 1 - np.sum((np.array(logT) - yhat) ** 2) / denom if denom > 0 else np.nan

# DTM: raise finest-resolution field to power eta, rebuild pyramid, TM at fixed q_ref
q_ref = 2.0
etas = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0]
K_of_eta = {}
for eta in etas:
    flux_eta = flux ** eta
    pyr_eta = build_pyramid(flux_eta)
    logT, logL = [], []
    for lam in fit_lambdas:
        vals = pyr_eta[lam]
        mean_q = np.mean(vals ** q_ref)
        mean_1_q = np.mean(vals) ** q_ref
        if mean_q <= 0 or mean_1_q <= 0:
            continue
        TM = mean_q / mean_1_q
        if TM <= 0 or not np.isfinite(TM):
            continue
        logT.append(np.log(TM))
        logL.append(np.log(lam))
    b, a = np.polyfit(logL, logT, 1)
    K_of_eta[eta] = b

valid_etas = [e for e in etas if K_of_eta[e] > 0]
logE = np.log(valid_etas)
logK = np.log([K_of_eta[e] for e in valid_etas])
alpha_est, intercept = np.polyfit(logE, logK, 1)
yhat = intercept + alpha_est * logE
r2_alpha = 1 - np.sum((logK - yhat) ** 2) / np.sum((logK - np.mean(logK)) ** 2)

K_qref_direct = Kq[q_ref]
if abs(alpha_est - 1.0) > 1e-6:
    C1_est = K_qref_direct * (alpha_est - 1) / (q_ref ** alpha_est - q_ref)
else:
    C1_est = K_qref_direct / (q_ref * np.log(q_ref))

results = {
    "field": "raw SPY daily closing price (untransformed)",
    "n_obs": int(N),
    "n_dyadic": int(N_use),
    "K_q": {str(q): Kq[q] for q in qs},
    "K_q_r2": {str(q): Kq_r2[q] for q in qs},
    "H_estimate": H_est,
    "H_r2": r2_H,
    "K_of_eta": {str(e): K_of_eta[e] for e in etas},
    "alpha": alpha_est,
    "alpha_r2": r2_alpha,
    "C1": C1_est,
    "q_ref": q_ref,
}

print(json.dumps(results, indent=2))
with open(os.path.join(OUT_DIR, "results_dtm.json"), "w") as f:
    json.dump(results, f, indent=2)
