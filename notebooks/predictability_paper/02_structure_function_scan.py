"""
Structure-function scaling exponent xi(q) for raw SPY price.

Computes <|Delta f(tau)|^q> ~ tau^xi(q) directly and empirically (the
two-point, lag-based moment -- distinct from the one-point trace moment
K(q) computed in 01_dtm_multifractal_analysis.py), on the raw
(untransformed) daily closing price series.
"""
import pandas as pd
import numpy as np
import json
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
spy = prices["SPY"].dropna()

taus = [1, 2, 4, 8, 16, 32, 64, 128, 256]
qs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]

zeta, r2s = {}, {}
for q in qs:
    logS, logT = [], []
    for tau in taus:
        inc = (spy - spy.shift(tau)).dropna()
        Sq = np.mean(np.abs(inc.values) ** q)
        logS.append(np.log(Sq))
        logT.append(np.log(tau))
    b, a = np.polyfit(logT, logS, 1)
    yhat = a + b * np.array(logT)
    ss_res = np.sum((np.array(logS) - yhat) ** 2)
    ss_tot = np.sum((np.array(logS) - np.mean(logS)) ** 2)
    zeta[q] = b
    r2s[q] = 1 - ss_res / ss_tot

results = {
    "field": "raw SPY daily closing price (untransformed)",
    "taus": taus,
    "xi_q": {str(q): zeta[q] for q in qs},
    "xi_q_r2": {str(q): r2s[q] for q in qs},
}
print(json.dumps(results, indent=2))
with open(os.path.join(OUT_DIR, "results_structure_function.json"), "w") as f:
    json.dump(results, f, indent=2)
