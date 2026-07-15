"""
Classifies each instrument's correlated/decorrelated relationship by how
many times C(tau) and D(tau) swap dominance across the 300-day lag window --
a direct, quantitative test of whether the shape resembles the atmospheric
monotonic single-crossing picture (Figure 1) or something else.

Three regimes:
  - "persistent"     : zero crossings -- one side dominates for the entire window
  - "single-crossing" : 1-5 crossings -- effectively one transition (atmosphere-like shape)
  - "oscillating"     : >5 crossings -- repeated pocket structure, no single crossing time
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "results_correlated_decorrelated.json")) as f:
    d = json.load(f)

TICKERS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "AAPL", "MSFT", "JPM", "XOM",
           "GLD", "BTC-USD", "TLT", "EURUSD=X", "^VIX"]


def classify(n):
    if n == 0:
        return "persistent"
    elif n <= 5:
        return "single-crossing"
    else:
        return "oscillating"


results = {}
for tk in TICKERS:
    tk_out = {}
    for q in ["2", "4"]:
        e = d[tk][q]
        C, D = e["C"], e["D"]
        signs = [1 if C[i] > D[i] else -1 for i in range(len(C))]
        crossings = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
        dominant_side = "correlated" if sum(signs) > 0 else "decorrelated" if sum(signs) < 0 else "mixed"
        regime = classify(crossings)
        tk_out[q] = dict(crossings=crossings, regime=regime, dominant_side=dominant_side)
        print(f"{tk:10s} q={q}  crossings={crossings:4d}  regime={regime:16s}  dominant={dominant_side}")
    results[tk] = tk_out

with open(os.path.join(HERE, "results_crossing_typology.json"), "w") as f:
    json.dump(results, f, indent=2)

# summary counts at q=2
from collections import Counter
counts = Counter(results[tk]["2"]["regime"] for tk in TICKERS)
print("\nq=2 regime counts across 15 instruments:", dict(counts))
counts4 = Counter(results[tk]["4"]["regime"] for tk in TICKERS)
print("q=4 regime counts across 15 instruments:", dict(counts4))
