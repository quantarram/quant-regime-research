"""
Correlated / decorrelated structure-function decomposition across the
multi-asset universe.

For each instrument, computes at every lag tau in [1, 300] trading days:
  D(tau, q) = <(f(t+tau) - f(t))^q>                          (decorrelated)
  C(tau, q) = sum_{n=1}^{q-1} (-1)^(n+1) C(q,n) <f(t+tau)^(q-n) f(t)^n>  (correlated)

on the raw (untransformed) absolute daily price increment |Delta p|, for
q = 2 and q = 4. Predictability lags are ranked by local maxima of the gap
G(tau,q) = C(tau,q) - D(tau,q), with boundary lags (tau=1, tau=300)
correctly included as candidate peaks (see paper Section 5.1 for the
methodological note on why this matters).
"""
import pandas as pd
import numpy as np
from scipy.special import comb
import json
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

TICKERS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "AAPL", "MSFT", "JPM", "XOM",
           "GLD", "BTC-USD", "TLT", "EURUSD=X", "^VIX"]
TAUS = list(range(1, 301))
TRADEABLE_MIN_TAU = 21  # see paper Sec 5.4: CPE finds no validated signal below this


def decorrelated_moment(f, tau, q):
    diff = f[tau:] - f[:-tau]
    return np.mean(np.abs(diff) ** q)


def correlated_moment(f, tau, q):
    a, b = f[tau:], f[:-tau]
    total = 0.0
    for n in range(1, int(q)):
        coeff = ((-1) ** (n + 1)) * comb(q, n)
        total += coeff * np.mean((a ** (q - n)) * (b ** n))
    return total


def find_peaks(gap, taus, min_tau=1):
    """Boundary-inclusive local-maximum search (fixes the tau=1/2 exclusion bug)."""
    peaks = []
    n = len(gap)
    for i in range(n):
        if gap[i] <= 0 or taus[i] < min_tau:
            continue
        left_ok = (i == 0) or (gap[i] >= gap[i - 1])
        right_ok = (i == n - 1) or (gap[i] >= gap[i + 1])
        strictly_a_peak = (i > 0 and gap[i] > gap[i - 1]) or (i < n - 1 and gap[i] > gap[i + 1])
        if left_ok and right_ok and strictly_a_peak:
            peaks.append((taus[i], gap[i]))
    peaks.sort(key=lambda x: -x[1])
    return peaks


def main():
    prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
    all_results = {}

    for tk in TICKERS:
        s = prices[tk].dropna()
        f = np.abs(np.diff(s.values))
        if len(f) < 400:
            print(f"{tk}: insufficient data, skipping")
            continue

        tk_data = {}
        for q in [2, 4]:
            D = np.array([decorrelated_moment(f, t, q) for t in TAUS])
            C = np.array([correlated_moment(f, t, q) for t in TAUS])
            gap = C - D
            ratio = C / (C + D)

            all_peaks = find_peaks(gap, TAUS)
            tradeable_peaks = find_peaks(gap, TAUS, min_tau=TRADEABLE_MIN_TAU)

            tk_data[str(q)] = dict(
                D=[round(x, 6) for x in D.tolist()],
                C=[round(x, 6) for x in C.tolist()],
                ratio=[round(x, 6) for x in ratio.tolist()],
                ratio_min=float(ratio.min()),
                ratio_max=float(ratio.max()),
                top5_all=[[int(t), round(g, 6)] for t, g in all_peaks[:5]],
                top5_tradeable=[[int(t), round(g, 6)] for t, g in tradeable_peaks[:5]],
            )
            print(f"{tk} q={q}: top5={tk_data[str(q)]['top5_all'][:3]}..."
                  f"  tradeable={tk_data[str(q)]['top5_tradeable'][:3]}...")

        all_results[tk] = tk_data

    with open(os.path.join(OUT_DIR, "results_correlated_decorrelated.json"), "w") as f:
        json.dump(all_results, f)
    print(f"\nSaved -> results_correlated_decorrelated.json ({len(all_results)} instruments)")


if __name__ == "__main__":
    main()
