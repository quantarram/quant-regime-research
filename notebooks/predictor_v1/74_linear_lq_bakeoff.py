"""
Direct follow-up to 73_quantile_loss_bakeoff.py, closing the one variant
left open by 71_cascade_loss_escalation.py's Lq investigation: every prior
attempt at escalating (q>2) loss used a model flexible enough to memorize
individual rows (LightGBM leaves, an MLP) and blew up catastrophically
(pooled OOS R2 in the -1e14 to -1e29 range even after every numerics
safeguard this project validates elsewhere -- adaptive y-scaling, gradient
clipping, bounded tanh activation, correct tau_star-scaled windows). The
diagnosed mechanism was tail-chasing: minimizing E[|r-c|^q] for large q
pulls the fit toward whichever single row has the largest residual, and a
flexible model can move all the way to that row and memorize it.

This script isolates whether that instability is really about model
FLEXIBILITY (as diagnosed) or the loss order itself, by fitting a LINEAR
model -- same 10 standardized lagged-return features, same half_window =
floor(tau_star/2) training budget, same 3 instruments (MSFT, EURUSD=X,
XLF) as every other script in this line -- via plain full-batch gradient
descent on Lq loss, q in {2, 4, 6, 8}. A linear model has only 11
parameters (10 weights + bias) and cannot carve out a leaf around one row
the way a tree can, so if it still blows up, the mechanism isn't really
about flexibility; if it stays bounded, flexibility was in fact the
culprit and linear Lq is a clean, stable way to test whether higher q
pulls predictions usefully toward genuine extremes.

Lq loss: L(r) = |r|^q, r = yhat - y (same convention as
loss_functions.make_lq_objective). Gradient wrt yhat: q*sign(r)*|r|^(q-1).
Fit in standardized target space (ys = (y-mean)/std, an affine transform,
exact not approximate) so the loss operates on O(1)-scale residuals
regardless of the instrument's raw return scale. Elementwise gradient
clipping to +-10.0 is retained -- the same numerics-only safety valve
already validated in 66_/70_/71_ (bounds one update step's size, does not
touch the loss itself, so it is not a form of outlier suppression).

No metrics-driven verdict, no significance tests -- purely visual, same
standing instruction as every script in this line. Output: does the q=4/6/8
line diverge/spike wildly (flexibility wasn't the real culprit) or does it
stay bounded and merely shift away from climatology (flexibility was)?

Run: python 74_linear_lq_bakeoff.py
Output: 74_linear_lq_bakeoff_{TICKER}.png (MSFT, EURUSDX, XLF)
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTABILITY_JSON = os.path.join(NB_DIR, "predictability_paper", "results_correlated_decorrelated.json")

INSTRUMENTS = {"MSFT": 189, "EURUSD=X": 189, "XLF": 189}
N_LAGS = 10
QS = [2, 4, 6, 8]
GRAD_CLIP = 10.0  # matches 66_/70_/71_'s already-validated elementwise clip


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


class LinearLqForecaster:
    """Linear model (11 parameters total), fit via full-batch gradient
    descent on pure Lq loss in standardized target space. No tree/MLP
    flexibility to carve a leaf/region around one outlier row -- the whole
    point of this script is to test whether that flexibility, not the
    loss order itself, was what made 71_'s cascade blow up."""

    def __init__(self, q, lr=0.05, epochs=400, seed=0):
        self.q, self.lr, self.epochs, self.seed = q, lr, epochs, seed

    def fit(self, X, y):
        Xs, mean, std = _standardize(X)
        self.mean, self.std = mean, std
        y_mean = float(np.mean(y))
        y_std = max(float(np.std(y)), 1e-6)
        self.y_mean, self.y_std = y_mean, y_std
        ys = (y - y_mean) / y_std

        n, d = Xs.shape
        w = np.zeros(d)
        b = float(np.mean(ys))
        q = self.q
        for _ in range(self.epochs):
            yhat = Xs @ w + b
            r = yhat - ys
            g = q * np.sign(r) * np.abs(r) ** (q - 1)
            g = np.clip(g, -GRAD_CLIP, GRAD_CLIP)
            grad_w = (g[:, None] * Xs).mean(axis=0)
            grad_b = g.mean()
            w = w - self.lr * grad_w
            b = b - self.lr * grad_b
        self.w, self.b = w, b
        return self

    def predict(self, X):
        Xs = (X - self.mean) / self.std
        yhat_std = Xs @ self.w + self.b
        return yhat_std * self.y_std + self.y_mean


def run_bakeoff(frame, half_window):
    frame = frame.reset_index(drop=True)
    n = len(frame)
    target_dates, actual_price = [], []
    clima_price, tree_price = [], []
    lq_price = {q: [] for q in QS}
    n_nonfinite = {q: 0 for q in QS}
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

        for q in QS:
            model = LinearLqForecaster(q=q).fit(Xtr, ytr)
            pred_ret = model.predict(Xte)
            n_nonfinite[q] += int(np.sum(~np.isfinite(pred_ret)))
            pred_ret = np.nan_to_num(pred_ret, nan=0.0, posinf=50.0, neginf=-50.0)  # visibility only, not silent
            lq_price[q].extend((base_price * np.exp(np.clip(pred_ret, -50, 50))).tolist())

        p += half_window
    return target_dates, actual_price, clima_price, tree_price, lq_price, n_nonfinite


if __name__ == "__main__":
    prices = pd.read_parquet(os.path.join(NB_DIR, "multiasset_prices.parquet"))
    tau_star = load_tau_star()

    q_colors = {2: "#F4C542", 4: "#E67E22", 6: "#C0392B", 8: "#6C1414"}  # light -> dark as q escalates

    for tkr, horizon in INSTRUMENTS.items():
        window = tau_star[tkr]
        half_window = window // 2
        series = prices[tkr].dropna()
        frame = build_price_frame(series, horizon)

        dates, actual, clima, tree, lq_price, n_nonfinite = run_bakeoff(frame, half_window)
        print(f"{tkr}: half_window={half_window} window={window} n_points={len(actual)} "
              f"non-finite raw predictions per q: {n_nonfinite}")

        dates_arr = np.array(dates)
        actual_arr = np.array(actual)
        recent_mask = dates_arr >= (dates_arr[-1] - pd.Timedelta(days=730))

        fig, axes = plt.subplots(2, 1, figsize=(16, 11))
        for ax, mask, is_log, subtitle in [
            (axes[0], slice(None), True, "full history, log scale"),
            (axes[1], recent_mask, False, "most recent 2 years, linear scale"),
        ]:
            d = dates_arr[mask] if isinstance(mask, np.ndarray) else dates_arr
            a = actual_arr[mask] if isinstance(mask, np.ndarray) else actual_arr
            c = np.array(clima)[mask] if isinstance(mask, np.ndarray) else np.array(clima)
            t = np.array(tree)[mask] if isinstance(mask, np.ndarray) else np.array(tree)

            ax.plot(d, t, color="#5B8DBE", lw=0.9, alpha=0.6, label="overfit tree (Paper 13 diagnostic)", zorder=2)
            for q in QS:
                v = np.array(lq_price[q])[mask] if isinstance(mask, np.ndarray) else np.array(lq_price[q])
                ax.plot(d, v, color=q_colors[q], lw=1.2, alpha=0.85, label=f"linear L{q}", zorder=3)
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
            f"{tkr}: linear model, Lq loss q={{2,4,6,8}}, all trained ONLY on the immediately preceding "
            f"{half_window}d (half of the {window}d predictability limit) -- {horizon}d-ahead price forecast",
            fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        safe_tkr = tkr.replace("=", "")
        fig.savefig(os.path.join(OUT_DIR, f"74_linear_lq_bakeoff_{safe_tkr}.png"), dpi=140)
        plt.close(fig)

    print("Saved: 74_linear_lq_bakeoff_{" + ",".join(t.replace("=", "") for t in INSTRUMENTS) + "}.png")
