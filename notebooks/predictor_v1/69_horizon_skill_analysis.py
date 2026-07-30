"""
Explains Section 6.2's finding: why IWM and QQQ show flat or reversing
error-vs-training-window-size curves in 68_window_sweep_error_curve.py's
sweep, while the other ten instruments show a clear, roughly-doubling climb.

For each instrument, computes:
  - the error of the most naive possible forecast (assume no price change
    at all over the horizon), as a percentage of mean price -- a reference
    floor that scales with horizon length under approximately random-walk
    dynamics, since a longer horizon gives price more time to move;
  - climatology's own error at the freshest, safely-within-limit training
    window (0.5x tau_star), as a percentage of mean price;
  - "skill": how much better than the naive floor climatology's
    fresh-trained forecast manages to do;
  - "growth": how much climatology's error increases from the freshest
    (0.5x) to the stalest (8x) swept training-window size.

Only climatology is computed here (not all five architectures from
66_/68_) -- climatology alone is enough to establish the skill-vs-horizon
relationship, and skipping the RL/GAN/VAE fits makes this diagnostic cheap
enough to run in well under a minute for all twelve instruments.

Run: python 69_horizon_skill_analysis.py
Output: prints Table 2 (this paper's Section 6.2) to stdout.
"""
import json
import os

import numpy as np
import pandas as pd

NB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTABILITY_JSON = os.path.join(NB_DIR, "predictability_paper", "results_correlated_decorrelated.json")

INSTRUMENTS = {
    "GLD": 189, "JPM": 252, "AAPL": 252, "XLK": 189, "EURUSD=X": 189, "IWM": 21,
    "MSFT": 189, "QQQ": 21, "SPY": 189, "XLE": 252, "XLF": 189, "XOM": 63,
}
N_LAGS = 10
SWEEP_MULTIPLES = [0.5, 1, 1.5, 2, 3, 4, 6, 8]


def load_tau_star():
    with open(PREDICTABILITY_JSON) as f:
        d = json.load(f)
    return {t: d[t]["2"]["top5_tradeable"][0][0] for t in INSTRUMENTS}


def build_price_frame(price_series, horizon, n_lags=N_LAGS):
    vals = price_series.values
    dates = price_series.index
    n = len(vals)
    rets = np.log(vals[1:] / vals[:-1])
    rets = np.concatenate([[np.nan], rets])
    rows = []
    for i in range(n_lags, n - horizon):
        lags = [rets[i - lag + 1] for lag in range(1, n_lags + 1)]
        if any(np.isnan(lags)):
            continue
        base_price = vals[i]
        target_price = vals[i + horizon]
        fwd_ret = np.log(target_price / base_price)
        rows.append([dates[i], dates[i + horizon], base_price, target_price, fwd_ret] + lags)
    cols = ["base_date", "target_date", "base_price", "target_price", "fwd_ret"] + \
           [f"lag_ret_{k}" for k in range(1, n_lags + 1)]
    return pd.DataFrame(rows, columns=cols)


def climatology_mae(frame, half_window, train_window, p_start):
    frame = frame.reset_index(drop=True)
    n = len(frame)
    errs = []
    p = p_start
    while p + half_window <= n:
        train_df = frame.iloc[max(0, p - train_window):p]
        test_df = frame.iloc[p:p + half_window]
        mu = float(train_df["fwd_ret"].mean())
        pred_price = test_df["base_price"].values * np.exp(mu)
        errs.extend(np.abs(pred_price - test_df["target_price"].values).tolist())
        p += half_window
    return float(np.mean(errs))


if __name__ == "__main__":
    prices = pd.read_parquet(os.path.join(NB_DIR, "multiasset_prices.parquet"))
    tau_star = load_tau_star()

    rows = []
    for tkr, horizon in INSTRUMENTS.items():
        window = tau_star[tkr]
        half_window = window // 2
        series = prices[tkr].dropna()
        frame = build_price_frame(series, horizon)

        train_windows = [max(int(round(m * window)), 4) for m in SWEEP_MULTIPLES]
        p_start = max(train_windows) + half_window
        mean_price = float(frame.iloc[p_start:]["target_price"].mean())

        naive_mae = (frame["target_price"] - frame["base_price"]).abs().mean()
        naive_pct = naive_mae / frame["target_price"].mean() * 100

        clim_pct = [climatology_mae(frame, half_window, tw, p_start) / mean_price * 100 for tw in train_windows]
        skill = (naive_pct - clim_pct[0]) / naive_pct * 100
        growth = clim_pct[-1] / clim_pct[0]

        rows.append((tkr, horizon, naive_pct, clim_pct[0], skill, growth))
        print(f"{tkr:10s} horizon={horizon:3d}  naive={naive_pct:5.1f}%  "
              f"clim@0.5x={clim_pct[0]:5.1f}%  skill={skill:5.1f}%  growth={growth:.2f}x")

    rows.sort(key=lambda r: r[1])
    print("\nSorted by horizon (Table 2):")
    for tkr, horizon, naive_pct, clim0, skill, growth in rows:
        print(f"  {tkr:10s} {horizon:3d}d  naive={naive_pct:5.1f}%  clim@0.5x={clim0:5.1f}%  "
              f"skill={skill:5.1f}%  growth={growth:.2f}x")
