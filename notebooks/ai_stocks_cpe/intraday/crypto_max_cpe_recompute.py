"""
crypto_max_cpe_recompute.py
========================================
The original crypto intraday feasibility test (files/cpe_engine_intraday_btc.py)
reported "max CPE ~0.545" as an approximate figure from a broader sweep
(full q-grid 0.50-0.99, both directions, 73,728 combinations). For a
precise, directly comparable figure against the equities test in this
paper (which restricts to genuine tail events, q>=0.80, bullish side
only -- see intraday_cpe_feasibility_test.py), this recomputes the same
restricted statistic from the original cached BTC/ETH/SOL/BNB 1-minute
data, rather than citing a differently-scoped historical figure.

Run: python crypto_max_cpe_recompute.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import json
import os

HORIZONS_MIN = [1, 5, 15, 30, 60, 240]
MIN_N = 100
REPO_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
SYMBOLS = {"BTC": "intraday_btc_1m.parquet", "ETH": "intraday_eth_1m.parquet",
           "SOL": "intraday_sol_1m.parquet", "BNB": "intraday_bnb_1m.parquet"}

prices = {}
for label, fname in SYMBOLS.items():
    df = pd.read_parquet(os.path.join(REPO_DATA_DIR, fname))
    s = df.set_index("open_time")["close"].astype(float)
    s.index = pd.to_datetime(s.index, unit="ms")
    prices[label] = s[~s.index.duplicated(keep="last")].sort_index()

df_prices = pd.DataFrame(prices).dropna(how="all")
print("Combined crypto panel:", df_prices.shape)

increments = {h: np.log(df_prices / df_prices.shift(h)) for h in HORIZONS_MIN}
future_inc = {h: increments[h].shift(-h) for h in HORIZONS_MIN}
thresholds = {(h, q): increments[h].quantile(q) for h in HORIZONS_MIN for q in [0.80, 0.90, 0.95, 0.99]}

all_cpes = []
for y in SYMBOLS:
    for h_f in HORIZONS_MIN:
        fy = future_inc[h_f][y].dropna()
        for h_p in HORIZONS_MIN:
            px_all = increments[h_p]
            common_idx = fy.index.intersection(px_all.dropna(how="all").index)
            if len(common_idx) < MIN_N:
                continue
            fy_vals = fy.loc[common_idx].values
            px_aligned = px_all.loc[common_idx]
            for q_y in [0.80, 0.90, 0.95, 0.99]:
                thresh_y_up = thresholds[(h_f, q_y)].get(y, np.nan)
                if np.isnan(thresh_y_up):
                    continue
                event_bull = fy_vals > thresh_y_up
                for x in SYMBOLS:
                    px_vals = px_aligned[x].values
                    valid = ~np.isnan(px_vals)
                    if valid.sum() < MIN_N:
                        continue
                    for q_x in [0.80, 0.90, 0.95, 0.99]:
                        thresh_x_up = thresholds[(h_p, q_x)].get(x, np.nan)
                        if np.isnan(thresh_x_up):
                            continue
                        cond = valid & (px_vals > thresh_x_up)
                        if cond.sum() >= MIN_N:
                            all_cpes.append(float(event_bull[cond].mean()))

result = {"max_cpe_tail_only_bullish": float(max(all_cpes)), "median_cpe_tail_only_bullish": float(np.median(all_cpes)),
          "n_combos_tested": len(all_cpes), "symbols": list(SYMBOLS), "horizons_min": HORIZONS_MIN,
          "note": "Bullish-side, q>=0.80-only restriction, recomputed to match the equities test's exact methodology for direct comparability -- supersedes the broader-grid '~0.545' figure in project memory for this specific statistic."}
print(json.dumps(result, indent=2))
json.dump(result, open("crypto_max_cpe_recompute.json", "w"), indent=2)
