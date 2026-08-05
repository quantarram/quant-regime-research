"""
Follow-up to 65_architecture_bakeoff.py. Direct question: Paper 14 found
that every architecture's POINT forecast collapses onto climatology within
the predictability limit -- but all five of those models (climatology's
mean, the tree's leaf averages, RL's squared-error reward, the GAN/VAE's
Gaussian-equivalent reconstruction terms) are, explicitly or implicitly,
L2-optimal estimators. An L2 objective is minimized by the CONDITIONAL
MEAN, which is exactly the quantity that gets pulled toward zero/flat as
noise dominates a tiny training window -- it has no mechanism to preserve
a genuinely wide or skewed conditional distribution, only its center.

This script holds everything else in Experiment 1's design fixed -- same
three instruments (MSFT, EURUSD=X, XLF), same half-window =
floor(tau_star/2) training budget, same walk-forward evaluation, same
price-only 10-lagged-return features, same linear model class (no added
capacity) -- and swaps ONLY the loss function: a linear model trained
separately at 5 quantile levels (0.1/0.25/0.5/0.75/0.9) via pinball loss
instead of MSE. This isolates the ONE variable the user's original
question was actually about (does the loss function, not the model
family, determine whether extremes survive), rather than re-testing
model complexity again.

Pinball loss for quantile alpha: L(y,yhat) = max(alpha*(y-yhat), (alpha-1)*(y-yhat))
(same convention as loss_functions.pinball_loss, reused here for the
reported loss value). Subgradient wrt yhat:
    dL/dyhat = -alpha        if y >= yhat  (under-prediction)
             = (1-alpha)     if y <  yhat  (over-prediction)
Note this gradient's MAGNITUDE does not grow with the size of the miss
(unlike L2's r-proportional gradient) -- it only ever pulls yhat up or
down at a constant rate set by alpha, which is precisely the mechanism
that lets extreme quantiles (alpha=0.1 or 0.9) settle far from the mean
without being tugged back the way an L2 point estimate would be.

Fit in standardized target space (ys = (y-mean)/std) for numerical
sanity, converted back to real return units at prediction time -- an
affine transform preserves quantile order, so this is exact, not an
approximation. Quantile crossing (a well-known artifact of fitting each
quantile level independently) is fixed with monotone rearrangement
(Chernozhukov, Fernandez-Val & Galichon, 2010) -- same technique already
used in live_train_predict.py's production quantile pipeline.

No metrics, no significance tests, per this whole experimental line's
standing instruction. Output is purely visual: actual price, climatology
(the established L2 baseline all five original models collapsed onto),
the overfit tree (the one model that did something other than converge
to climatology, for context), and this quantile-loss model's median line
PLUS a shaded 10-90% band -- the direct visual test of whether the band
brackets real extreme price moves that the point-forecast lines miss.

Run: python 73_quantile_loss_bakeoff.py
Output: 73_quantile_loss_bakeoff_{TICKER}.png (MSFT, EURUSDX, XLF)
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loss_functions import pinball_loss

NB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTABILITY_JSON = os.path.join(NB_DIR, "predictability_paper", "results_correlated_decorrelated.json")

INSTRUMENTS = {"MSFT": 189, "EURUSD=X": 189, "XLF": 189}
N_LAGS = 10
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


def load_tau_star():
    with open(PREDICTABILITY_JSON) as f:
        d = json.load(f)
    return {t: d[t]["2"]["top5_tradeable"][0][0] for t in INSTRUMENTS}


def build_price_frame(price_series, horizon, n_lags=N_LAGS):
    """Identical to 65_architecture_bakeoff.py's version -- one row per base
    date with full lag-feature history plus base_price/target_price so
    predicted returns convert back into predicted PRICE for the plot."""
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


FEATURE_COLS = [f"lag_ret_{k}" for k in range(1, N_LAGS + 1)]


def _standardize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < 1e-8, 1e-8, std)
    return (X - mean) / std, mean, std


class Climatology:
    name = "climatology"

    def fit(self, X, y):
        self.mu = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mu)


class OverfitTree:
    name = "overfit tree (Paper 13 diagnostic)"

    def fit(self, X, y):
        self.m = DecisionTreeRegressor(max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=0)
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)


class QuantileLossForecaster:
    """Linear model, one weight vector per quantile level, trained
    independently via full-batch gradient descent on pinball loss in
    standardized target space. No added capacity vs. the L2 models in
    65_architecture_bakeoff.py -- same 10 standardized features, same
    linear form -- the only thing that changes is the loss function."""
    name = "quantile-loss (linear, pinball)"

    def __init__(self, quantiles=QUANTILES, lr=0.3, epochs=400, seed=0):
        self.quantiles, self.lr, self.epochs, self.seed = quantiles, lr, epochs, seed

    def fit(self, X, y):
        Xs, mean, std = _standardize(X)
        self.mean, self.std = mean, std
        y_mean = float(np.mean(y))
        y_std = max(float(np.std(y)), 1e-6)
        self.y_mean, self.y_std = y_mean, y_std
        ys = (y - y_mean) / y_std

        n, d = Xs.shape
        self.w, self.b = {}, {}
        for alpha in self.quantiles:
            w = np.zeros(d)
            b = float(np.quantile(ys, alpha))  # sane starting point, mirrors Climatology's mean-init
            for _ in range(self.epochs):
                yhat = Xs @ w + b
                # dL/dyhat: -alpha where under-predicting (ys>=yhat), (1-alpha) where over-predicting
                g = np.where(ys >= yhat, -alpha, 1.0 - alpha)
                grad_w = (g[:, None] * Xs).mean(axis=0)
                grad_b = g.mean()
                w = w - self.lr * grad_w
                b = b - self.lr * grad_b
            self.w[alpha], self.b[alpha] = w, b
        return self

    def predict_quantiles(self, X):
        """Returns {alpha: array of predicted forward returns}, monotone-
        rearranged across the quantile grid per row (Chernozhukov, Fernandez-
        Val & Galichon 2010) so quantile crossing cannot occur."""
        Xs = (X - self.mean) / self.std
        n = len(Xs)
        raw = np.column_stack([Xs @ self.w[a] + self.b[a] for a in self.quantiles])  # (n, n_q), standardized-y space
        raw_sorted = np.sort(raw, axis=1)  # monotone rearrangement, per row
        out_std = {a: raw_sorted[:, i] for i, a in enumerate(self.quantiles)}
        return {a: v * self.y_std + self.y_mean for a, v in out_std.items()}  # back to real return units

    def predict(self, X):
        return self.predict_quantiles(X)[0.5]


def run_bakeoff(frame, half_window):
    frame = frame.reset_index(drop=True)
    n = len(frame)
    target_dates, actual_price = [], []
    clima_price, tree_price = [], []
    q_price = {a: [] for a in QUANTILES}
    p = half_window
    while p + half_window <= n:
        train_df = frame.iloc[p - half_window:p]
        test_df = frame.iloc[p:p + half_window]
        Xtr, ytr = train_df[FEATURE_COLS].values, train_df["fwd_ret"].values
        Xte = test_df[FEATURE_COLS].values

        target_dates.extend(test_df["target_date"].tolist())
        actual_price.extend(test_df["target_price"].tolist())
        base_price = test_df["base_price"].values

        clima = Climatology().fit(Xtr, ytr)
        clima_price.extend((base_price * np.exp(clima.predict(Xte))).tolist())

        tree = OverfitTree().fit(Xtr, ytr)
        tree_price.extend((base_price * np.exp(tree.predict(Xte))).tolist())

        qmodel = QuantileLossForecaster().fit(Xtr, ytr)
        qpreds = qmodel.predict_quantiles(Xte)
        for a in QUANTILES:
            q_price[a].extend((base_price * np.exp(qpreds[a])).tolist())

        p += half_window
    return target_dates, actual_price, clima_price, tree_price, q_price


if __name__ == "__main__":
    prices = pd.read_parquet(os.path.join(NB_DIR, "multiasset_prices.parquet"))
    tau_star = load_tau_star()

    for tkr, horizon in INSTRUMENTS.items():
        window = tau_star[tkr]
        half_window = window // 2
        series = prices[tkr].dropna()
        frame = build_price_frame(series, horizon)

        dates, actual, clima, tree, q_price = run_bakeoff(frame, half_window)

        dates_arr = np.array(dates)
        recent_mask = dates_arr >= (dates_arr[-1] - pd.Timedelta(days=730))

        # pinball loss on the median (q=0.5) vs actual, reported once per
        # instrument in the console only -- not plotted, per the standing
        # visual-only instruction for this experimental line.
        actual_arr = np.array(actual)
        med_pb = pinball_loss(actual_arr, np.array(q_price[0.5]), 0.5)
        clima_pb = pinball_loss(actual_arr, np.array(clima), 0.5)
        print(f"{tkr}: median pinball loss -- quantile model {med_pb:.5f}, climatology {clima_pb:.5f}")

        fig, axes = plt.subplots(2, 1, figsize=(16, 11))
        for ax, mask, is_log, subtitle in [
            (axes[0], slice(None), True, "full history, log scale"),
            (axes[1], recent_mask, False, "most recent 2 years, linear scale"),
        ]:
            d = dates_arr[mask] if isinstance(mask, np.ndarray) else dates_arr
            a = actual_arr[mask] if isinstance(mask, np.ndarray) else actual_arr
            c = np.array(clima)[mask] if isinstance(mask, np.ndarray) else np.array(clima)
            t = np.array(tree)[mask] if isinstance(mask, np.ndarray) else np.array(tree)
            q10 = np.array(q_price[0.1])[mask] if isinstance(mask, np.ndarray) else np.array(q_price[0.1])
            q25 = np.array(q_price[0.25])[mask] if isinstance(mask, np.ndarray) else np.array(q_price[0.25])
            q50 = np.array(q_price[0.5])[mask] if isinstance(mask, np.ndarray) else np.array(q_price[0.5])
            q75 = np.array(q_price[0.75])[mask] if isinstance(mask, np.ndarray) else np.array(q_price[0.75])
            q90 = np.array(q_price[0.9])[mask] if isinstance(mask, np.ndarray) else np.array(q_price[0.9])

            ax.fill_between(d, q10, q90, color="#D68910", alpha=0.15, label="quantile-loss 10-90% band", zorder=1)
            ax.fill_between(d, q25, q75, color="#D68910", alpha=0.25, label="quantile-loss 25-75% band", zorder=2)
            ax.plot(d, t, color="#5B8DBE", lw=1.0, alpha=0.8, label="overfit tree (Paper 13 diagnostic)", zorder=3)
            ax.plot(d, q50, color="#D68910", lw=1.6, label="quantile-loss median", zorder=4)
            ax.plot(d, a, color="#222222", lw=1.3, label="actual price", zorder=5)
            ax.plot(d, c, color="#8E8E8E", lw=1.8, linestyle="--", alpha=0.95, label="climatology", zorder=6)

            if is_log:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_title(subtitle, fontsize=9, loc="left")
            ax.set_xlabel("target date")
            ax.set_ylabel("price" + (" (log scale)" if is_log else ""))
            ax.legend(loc="upper left", fontsize=8, ncol=2)

        fig.suptitle(
            f"{tkr}: native-resolution quantile (pinball) loss vs. climatology/tree, all trained ONLY on the "
            f"immediately preceding {half_window}d (half of the {window}d predictability limit) -- "
            f"{horizon}d-ahead price forecast",
            fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        safe_tkr = tkr.replace("=", "")
        fig.savefig(os.path.join(OUT_DIR, f"73_quantile_loss_bakeoff_{safe_tkr}.png"), dpi=140)
        plt.close(fig)
        print(f"{tkr}: half_window={half_window} window={window} n_points={len(actual)}")

    print("Saved: 73_quantile_loss_bakeoff_{" + ",".join(t.replace("=", "") for t in INSTRUMENTS) + "}.png")
