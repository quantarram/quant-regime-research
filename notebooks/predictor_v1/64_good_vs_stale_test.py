"""
Fresh-vs-stale training comparison, per the user's own design (see memory):
for each instrument with a real, already-published Paper 11 predictability
limit (tau_star), fit a deliberately unregularized model on the immediately
preceding tau_star/2-day window ("fresh") and, separately, on an equally-
sized window drawn from >= 2*tau_star days earlier ("stale"), then predict
on the SAME test window and overlay both predictions against the actual
outcome curve directly -- no significance test, no color-coded pass/fail
label, per the user's explicit standing preference (feedback-prefers-plots,
feedback-no-randomization-testing).

Corrected window sizing (a real methodological fix caught by the user):
naive train-on-tau_star/test-on-next-tau_star allows train-test gaps up to
~2*tau_star for some point pairs, already beyond the predictability limit
for much of the comparison. Halving both windows to tau_star/2 keeps every
train-test pair within tau_star days of each other.

Two independent feature variants, tested separately:
  (a) features-based: the same multifractal + credit/vix-regime-interaction
      features feature_lib.py builds for the live deployed system.
  (b) price-only: only the instrument's own 10 lagged daily returns, no
      exogenous features at all.

Covers all 12 of the 22 predictor_v1 instruments with a real Paper 11
predictability limit (results_correlated_decorrelated.json covers a
15-instrument sample, not all 22 -- the other 10 are not estimated here).

Run: python 64_good_vs_stale_test.py
Output: 64_good_vs_stale_{TICKER}.png (one per instrument, 2 panels each)
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import feature_lib as fl

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.dirname(OUT_DIR)
PREDICTABILITY_JSON = os.path.join(NB_DIR, "predictability_paper", "results_correlated_decorrelated.json")
PROXY_TICKERS = ("IYR", "VOX")

INSTRUMENTS = {
    "GLD": {"horizon": 189, "winner": "vix_only"},
    "JPM": {"horizon": 252, "winner": "credit_only"},
    "AAPL": {"horizon": 252, "winner": "credit_only"},
    "XLK": {"horizon": 189, "winner": "climatology"},
    "EURUSD=X": {"horizon": 189, "winner": "credit_only"},
    "IWM": {"horizon": 21, "winner": "climatology"},
    "MSFT": {"horizon": 189, "winner": "climatology"},
    "QQQ": {"horizon": 21, "winner": "climatology"},
    "SPY": {"horizon": 189, "winner": "climatology"},
    "XLE": {"horizon": 252, "winner": "credit_only"},
    "XLF": {"horizon": 189, "winner": "climatology"},
    "XOM": {"horizon": 63, "winner": "credit_only"},
}
N_LAG_RETURNS = 10


def load_predictability_limits():
    with open(PREDICTABILITY_JSON) as f:
        d = json.load(f)
    return {t: d[t]["2"]["top5_tradeable"][0][0] for t in INSTRUMENTS}


def build_regime_series(prices):
    hyg, lqd = prices["HYG"], prices["LQD"]
    vixm, vixy = prices["VIXM"], prices["VIXY"]
    credit = fl.credit_spread_regime(hyg, lqd).rename("credit_spread_regime")
    vix = fl.vix_term_slope(vixm, vixy).rename("vix_term_slope")
    return pd.concat([credit, vix], axis=1).reset_index().rename(columns={"index": "date"})


def build_option_a_frame(ticker, horizon, winner, panel, price_series, regimes):
    df = panel[panel["ticker"] == ticker].merge(regimes, on="date", how="left").copy()
    df["interact_gap21q4_credit"] = df["gap_tau21_q4_z"] * df["credit_spread_regime"]
    df["interact_xiq4_credit"] = df["xi_q4_z"] * df["credit_spread_regime"]
    df["interact_gap21q4_vix"] = df["gap_tau21_q4_z"] * df["vix_term_slope"]
    df["interact_xiq4_vix"] = df["xi_q4_z"] * df["vix_term_slope"]

    baseline_cols = [c for c in panel.columns if c.endswith("_z")]
    if ticker in fl.ORIG_GROUP_TICKERS:
        baseline_cols = baseline_cols + [c for c in panel.columns if c.startswith("ctx_")] + ["self_ref_score"]
    # A climatology winner has no natural interaction-term "variant" of its
    # own (this diagnostic is independent of the instrument's own deployed
    # model type) -- use the full set (both regimes) rather than picking one.
    variant_key = "both" if winner == "climatology" else winner
    feature_cols = baseline_cols + fl.VARIANT_EXTRA_COLS[variant_key]

    date_pos = {d: i for i, d in enumerate(price_series.index)}
    df["pos"] = df["date"].map(date_pos)
    df = df.dropna(subset=["pos"]).copy()
    df["pos"] = df["pos"].astype(int)

    vals = price_series.values
    n = len(vals)
    has_label = df["pos"].values + horizon < n
    fwd_ret = np.full(len(df), np.nan)
    fwd_ret[has_label] = np.log(vals[df["pos"].values[has_label] + horizon] / vals[df["pos"].values[has_label]])
    df["fwd_ret"] = fwd_ret
    return df[["date", "fwd_ret"] + feature_cols].dropna().sort_values("date").reset_index(drop=True), feature_cols


def build_option_b_frame(horizon, price_series, n_lags=N_LAG_RETURNS):
    rets = np.log(price_series / price_series.shift(1))
    df = pd.DataFrame({"date": price_series.index})
    for lag in range(1, n_lags + 1):
        df[f"lag_ret_{lag}"] = rets.shift(lag - 1).values
    vals = price_series.values
    n = len(vals)
    fwd_ret = np.full(len(df), np.nan)
    valid_idx = np.arange(n - horizon)
    fwd_ret[valid_idx] = np.log(vals[valid_idx + horizon] / vals[valid_idx])
    df["fwd_ret"] = fwd_ret
    feature_cols = [f"lag_ret_{lag}" for lag in range(1, n_lags + 1)]
    return df[["date", "fwd_ret"] + feature_cols].dropna().sort_values("date").reset_index(drop=True), feature_cols


def fit_predict(train_df, test_df, cols):
    if len(train_df) < 4 or len(test_df) < 1:
        return None
    m = DecisionTreeRegressor(max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=0)
    m.fit(train_df[cols].values, train_df["fwd_ret"].values)
    return m.predict(test_df[cols].values)


def run_good_vs_stale(frame, cols, half_window, window):
    """half_window = floor(tau_star / 2), window = tau_star. Fresh training
    uses the immediately preceding half_window days; stale training uses an
    equally-sized window ending >= 2*window days before the test window."""
    frame = frame.reset_index(drop=True)
    n = len(frame)
    actual_all, good_all, stale_all, boundaries = [], [], [], [0]
    p = half_window
    while p + half_window <= n:
        test_df = frame.iloc[p:p + half_window]
        good_train = frame.iloc[p - half_window:p]
        stale_train_end = p - 2 * window
        stale_train_start = stale_train_end - half_window
        if stale_train_start < 0:
            p += half_window
            continue
        stale_train = frame.iloc[stale_train_start:stale_train_end]

        good_pred = fit_predict(good_train, test_df, cols)
        stale_pred = fit_predict(stale_train, test_df, cols)
        if good_pred is None or stale_pred is None:
            p += half_window
            continue
        actual_all.extend(test_df["fwd_ret"].values.tolist())
        good_all.extend(good_pred.tolist())
        stale_all.extend(stale_pred.tolist())
        boundaries.append(boundaries[-1] + len(test_df))
        p += half_window
    return np.array(actual_all), np.array(good_all), np.array(stale_all), boundaries


if __name__ == "__main__":
    prices = pd.read_parquet(os.path.join(NB_DIR, "multiasset_prices.parquet"))
    panel = pd.read_parquet(os.path.join(OUT_DIR, "features_daily_panel.parquet"))
    regimes = build_regime_series(prices)
    limits = load_predictability_limits()

    for tkr, cfg in INSTRUMENTS.items():
        horizon, winner = cfg["horizon"], cfg["winner"]
        window = limits[tkr]
        half_window = window // 2
        series = prices[tkr].dropna()

        frame_a, cols_a = build_option_a_frame(tkr, horizon, winner, panel, series, regimes)
        frame_b, cols_b = build_option_b_frame(horizon, series)

        actual_a, good_a, stale_a, bounds_a = run_good_vs_stale(frame_a, cols_a, half_window, window)
        actual_b, good_b, stale_b, bounds_b = run_good_vs_stale(frame_b, cols_b, half_window, window)

        fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=False)
        for ax, actual, good, stale, bounds, label in [
            (axes[0], actual_a, good_a, stale_a, bounds_a, f"{tkr} -- features-based"),
            (axes[1], actual_b, good_b, stale_b, bounds_b, f"{tkr} -- price-only"),
        ]:
            x = np.arange(len(actual))
            ax.plot(x, actual, color="#333333", lw=1.1, label="actual forward return", zorder=3)
            ax.plot(x, good, color="#5B8DBE", lw=1.0, alpha=0.85,
                    label=f"trained on immediate prior {half_window}d (within predictability limit)", zorder=2)
            ax.plot(x, stale, color="#C0392B", lw=1.0, alpha=0.75,
                    label=f"trained on stale {half_window}d, >2x{window}d={2*window}d earlier", zorder=1)
            for b in bounds[1:-1]:
                ax.axvline(b, color="#e0e0e0", lw=0.4, zorder=0)
            ax.set_title(label, fontsize=10, loc="left")
            ax.set_xlabel("sequential test-segment day (segments concatenated back-to-back)")
            ax.set_ylabel("forward log return")
            ax.legend(loc="upper right", fontsize=7.5)

        fig.suptitle(f"{tkr}: fresh (within predictability limit) vs stale (>2x predictability limit) "
                     f"training -- half-window={half_window}d, full window={window}d", fontsize=11)
        fig.tight_layout()
        safe_tkr = tkr.replace("=", "")
        fig.savefig(os.path.join(OUT_DIR, f"64_good_vs_stale_{safe_tkr}.png"), dpi=140)
        plt.close(fig)
        print(f"{tkr}: half_window={half_window} window={window} n_points={len(actual_a)}")

    print("Saved: 64_good_vs_stale_{" + ",".join(t.replace("=", "") for t in INSTRUMENTS) + "}.png")
