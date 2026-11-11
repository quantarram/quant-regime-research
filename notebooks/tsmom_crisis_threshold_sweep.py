import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from math import comb, sqrt
import tsmom_crisis_alpha_check as c
t, p = c.build_series()
eq = np.exp(p.fillna(0)).cumprod(); dd = eq / eq.cummax() - 1
def wilson(k, n, z=1.96):
    ph = k / n; d = 1 + z * z / n
    ctr = (ph + z * z / (2 * n)) / d
    h = z * sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return ctr - h, ctr + h
def binom_p(k, n): return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
print(f"{'thr':>6} {'minD':>4} {'episodes':>8} {'wins':>5} {'win%':>6} {'p(>=k|H0=.5)':>13} {'Wilson95':>14}")
rows = []
for thr in (-0.05, -0.075, -0.10, -0.125, -0.15, -0.20):
    for md in (10, 15, 30):
        eps = c.cluster_episodes(dd <= thr, md)
        w = 0
        for s, e in eps:
            m = (t.index >= s) & (t.index <= e)
            tc = np.exp(t[m].fillna(0)).prod() - 1; pc = np.exp(p[m].fillna(0)).prod() - 1
            w += tc > pc
        n = len(eps)
        if n:
            lo, hi = wilson(w, n)
            print(f"{thr:>6.1%} {md:>4} {n:>8} {w:>5} {w/n:>6.0%} {binom_p(w, n):>13.3f} [{lo:.2f}, {hi:.2f}]")
