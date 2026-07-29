"""
Compact, paper-embeddable companion to 66_window_sweep_bakeoff.py. The full
8-stacked-panel figures that script produces are the complete, honest record
(all 8 multiples, all 12 instruments) and stay in the repo as supplementary
detail -- but at that aspect ratio they do not fit a single printed page and
were clipping badly when embedded directly in the paper PDF.

This script re-runs the identical model/data/windowing logic for a curated
subset of instruments and exactly three of the eight swept multiples
(0.5x, 2x, 8x tau_star -- within the limit, just past it, and deep beyond
it), laid out as three panels SIDE BY SIDE rather than stacked, which scales
down to a page width cleanly without needing to shrink each panel into
illegibility.

Run: python 67_window_sweep_highlights.py
Output: 67_window_sweep_highlight_{TICKER}.png
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

HIGHLIGHT_INSTRUMENTS = {"MSFT": 189, "EURUSD=X": 189, "JPM": 252, "XLF": 189, "QQQ": 21}
N_LAGS = 10
HIGHLIGHT_MULTIPLES = [0.5, 2, 8]


def load_tau_star():
    with open(PREDICTABILITY_JSON) as f:
        d = json.load(f)
    return {t: d[t]["2"]["top5_tradeable"][0][0] for t in HIGHLIGHT_INSTRUMENTS}


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


class RLPolicyForecaster:
    name = "RL (policy gradient)"

    def __init__(self, lr=0.05, epochs=400, sigma0=0.3, sigma_min=0.05, seed=0):
        self.lr, self.epochs, self.sigma0, self.sigma_min, self.seed = lr, epochs, sigma0, sigma_min, seed

    def fit(self, X, y):
        Xs, mean, std = _standardize(X)
        self.mean, self.std = mean, std
        y_scale = max(float(np.std(y)), 1e-8)
        self.y_scale = y_scale
        ys = y / y_scale
        rng = np.random.default_rng(self.seed)
        n, d = Xs.shape
        w = np.zeros(d)
        b = float(np.mean(ys))
        baseline = 0.0
        for ep in range(self.epochs):
            sigma = max(self.sigma0 * (1 - ep / self.epochs), self.sigma_min)
            mu = Xs @ w + b
            noise = rng.normal(0.0, sigma, size=n)
            a = mu + noise
            reward = -(a - ys) ** 2
            baseline = 0.9 * baseline + 0.1 * reward.mean()
            adv = np.clip(reward - baseline, -5.0, 5.0)
            grad_w = np.clip((adv * noise)[:, None] * Xs, -10.0, 10.0)
            grad_b = np.clip(adv * noise, -10.0, 10.0)
            w = w + self.lr * grad_w.mean(axis=0)
            b = b + self.lr * grad_b.mean()
        self.w, self.b = w, b
        return self

    def predict(self, X):
        Xs = (X - self.mean) / self.std
        return (Xs @ self.w + self.b) * self.y_scale


class ConditionalGANForecaster:
    name = "conditional GAN"

    def __init__(self, z_dim=2, lr=0.1, epochs=400, k_samples=200, seed=0):
        self.z_dim, self.lr, self.epochs, self.k_samples, self.seed = z_dim, lr, epochs, k_samples, seed

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        Xs, mean, std = _standardize(X)
        self.mean, self.std = mean, std
        y_scale = max(float(np.std(y)), 1e-8)
        self.y_scale = y_scale
        ys = y / y_scale
        rng = np.random.default_rng(self.seed)
        n, d = Xs.shape
        zd = self.z_dim

        wg = np.zeros(d)
        vg = rng.normal(0, 0.01, size=zd)
        bg = float(np.mean(ys))

        wd = np.zeros(d)
        ud = 0.0
        bd = 0.0

        for ep in range(self.epochs):
            z = rng.normal(0.0, 1.0, size=(n, zd))
            y_fake = Xs @ wg + z @ vg + bg

            s_real = Xs @ wd + ud * ys + bd
            s_fake = Xs @ wd + ud * y_fake + bd
            D_real = self._sigmoid(s_real)
            D_fake = self._sigmoid(s_fake)

            dL_ds_real = -(1 - D_real)
            dL_ds_fake = D_fake
            gwd = np.clip(((dL_ds_real[:, None] * Xs) + (dL_ds_fake[:, None] * Xs)).mean(axis=0), -10.0, 10.0)
            gud = np.clip((dL_ds_real * ys + dL_ds_fake * y_fake).mean(), -10.0, 10.0)
            gbd = np.clip((dL_ds_real + dL_ds_fake).mean(), -10.0, 10.0)
            wd = wd - self.lr * gwd
            ud = ud - self.lr * gud
            bd = bd - self.lr * gbd

            s_fake_g = Xs @ wd + ud * y_fake + bd
            D_fake_g = self._sigmoid(s_fake_g)
            dLg_dyfake = -(1 - D_fake_g) * ud
            gwg = np.clip((dLg_dyfake[:, None] * Xs).mean(axis=0), -10.0, 10.0)
            gvg = np.clip((dLg_dyfake[:, None] * z).mean(axis=0), -10.0, 10.0)
            gbg = np.clip(dLg_dyfake.mean(), -10.0, 10.0)
            wg = wg - self.lr * gwg
            vg = vg - self.lr * gvg
            bg = bg - self.lr * gbg

        self.wg, self.vg, self.bg = wg, vg, bg
        self._rng = rng
        return self

    def predict(self, X):
        Xs = (X - self.mean) / self.std
        n = len(Xs)
        acc = np.zeros(n)
        for _ in range(self.k_samples):
            z = self._rng.normal(0.0, 1.0, size=(n, self.z_dim))
            acc += Xs @ self.wg + z @ self.vg + self.bg
        return (acc / self.k_samples) * self.y_scale


class ConditionalVAEForecaster:
    name = "conditional VAE (generative/stochastic)"

    def __init__(self, lr=0.1, epochs=400, beta=0.1, k_samples=200, seed=0):
        self.lr, self.epochs, self.beta, self.k_samples, self.seed = lr, epochs, beta, k_samples, seed

    def fit(self, X, y):
        Xs, mean, std = _standardize(X)
        self.mean, self.std = mean, std
        y_scale = max(float(np.std(y)), 1e-8)
        self.y_scale = y_scale
        ys = y / y_scale
        rng = np.random.default_rng(self.seed)
        n, d = Xs.shape

        We = np.zeros(d); be_y = 0.0; be = 0.0
        Ue = np.zeros(d); ue_y = 0.0; ue = 0.0
        Wdx = np.zeros(d); wdz = 0.1; bd = float(np.mean(ys))

        for ep in range(self.epochs):
            eps = rng.normal(0.0, 1.0, size=n)
            mu_e = Xs @ We + be_y * ys + be
            log_sigma_e = np.clip(Xs @ Ue + ue_y * ys + ue, -5, 5)
            sigma_e = np.exp(log_sigma_e)
            z = mu_e + sigma_e * eps

            mu_d = Xs @ Wdx + wdz * z + bd
            resid = mu_d - ys

            dL_dmu_e = resid * wdz + self.beta * mu_e
            dL_dsigma_e = resid * wdz * eps + self.beta * (sigma_e - 1.0 / sigma_e)
            dL_dlogsig_e = dL_dsigma_e * sigma_e

            gWdx = np.clip((resid[:, None] * Xs).mean(axis=0), -10.0, 10.0)
            gwdz = np.clip((resid * z).mean(), -10.0, 10.0)
            gbd = np.clip(resid.mean(), -10.0, 10.0)

            gWe = np.clip((dL_dmu_e[:, None] * Xs).mean(axis=0), -10.0, 10.0)
            gbe_y = np.clip((dL_dmu_e * ys).mean(), -10.0, 10.0)
            gbe = np.clip(dL_dmu_e.mean(), -10.0, 10.0)

            gUe = np.clip((dL_dlogsig_e[:, None] * Xs).mean(axis=0), -10.0, 10.0)
            gue_y = np.clip((dL_dlogsig_e * ys).mean(), -10.0, 10.0)
            gue = np.clip(dL_dlogsig_e.mean(), -10.0, 10.0)

            Wdx -= self.lr * gWdx; wdz -= self.lr * gwdz; bd -= self.lr * gbd
            We -= self.lr * gWe; be_y -= self.lr * gbe_y; be -= self.lr * gbe
            Ue -= self.lr * gUe; ue_y -= self.lr * gue_y; ue -= self.lr * gue

        self.Wdx, self.wdz, self.bd = Wdx, wdz, bd
        self._rng = rng
        return self

    def predict(self, X):
        Xs = (X - self.mean) / self.std
        n = len(Xs)
        acc = np.zeros(n)
        for _ in range(self.k_samples):
            z = self._rng.normal(0.0, 1.0, size=n)
            acc += Xs @ self.Wdx + self.wdz * z + self.bd
        return (acc / self.k_samples) * self.y_scale


MODEL_FACTORIES = [
    lambda: Climatology(),
    lambda: OverfitTree(),
    lambda: RLPolicyForecaster(),
    lambda: ConditionalGANForecaster(),
    lambda: ConditionalVAEForecaster(),
]

COLORS = {
    "climatology": "#8E8E8E",
    "overfit tree (Paper 13 diagnostic)": "#5B8DBE",
    "RL (policy gradient)": "#2E8B57",
    "conditional GAN": "#C0392B",
    "conditional VAE (generative/stochastic)": "#8E44AD",
}


def run_sweep_window(frame, half_window, train_window, p_start):
    frame = frame.reset_index(drop=True)
    n = len(frame)
    target_dates, actual_price = [], []
    pred_price = {m().name: [] for m in MODEL_FACTORIES}
    p = p_start
    while p + half_window <= n:
        train_df = frame.iloc[max(0, p - train_window):p]
        test_df = frame.iloc[p:p + half_window]
        Xtr, ytr = train_df[FEATURE_COLS].values, train_df["fwd_ret"].values
        Xte = test_df[FEATURE_COLS].values

        target_dates.extend(test_df["target_date"].tolist())
        actual_price.extend(test_df["target_price"].tolist())
        base_price = test_df["base_price"].values

        for factory in MODEL_FACTORIES:
            model = factory()
            model.fit(Xtr, ytr)
            pred_ret = model.predict(Xte)
            pred_price[model.name].extend((base_price * np.exp(pred_ret)).tolist())
        p += half_window
    return target_dates, actual_price, pred_price


if __name__ == "__main__":
    prices = pd.read_parquet(os.path.join(NB_DIR, "multiasset_prices.parquet"))
    tau_star = load_tau_star()

    for tkr, horizon in HIGHLIGHT_INSTRUMENTS.items():
        window = tau_star[tkr]
        half_window = window // 2
        series = prices[tkr].dropna()
        frame = build_price_frame(series, horizon)

        all_train_windows = [max(int(round(m * window)), 4) for m in [0.5, 1, 1.5, 2, 3, 4, 6, 8]]
        p_start = max(all_train_windows) + half_window  # identical anchor to 66_'s full sweep

        highlight_windows = [max(int(round(m * window)), 4) for m in HIGHLIGHT_MULTIPLES]

        fig, axes = plt.subplots(1, len(HIGHLIGHT_MULTIPLES), figsize=(5.2 * len(HIGHLIGHT_MULTIPLES), 3.6))

        for ax, mult, tw in zip(axes, HIGHLIGHT_MULTIPLES, highlight_windows):
            dates, actual, preds = run_sweep_window(frame, half_window, tw, p_start)
            dates_arr = np.array(dates)
            recent_mask = dates_arr >= (dates_arr[-1] - pd.Timedelta(days=730))

            d = dates_arr[recent_mask]
            a = np.array(actual)[recent_mask]
            ax.plot(d, a, color="#222222", lw=1.1, label="actual price", zorder=5)
            for name, vals in preds.items():
                if name == "climatology":
                    continue
                v = np.array(vals)[recent_mask]
                ax.plot(d, v, color=COLORS[name], lw=0.85, alpha=0.85, label=name, zorder=3)
            clim = np.array(preds["climatology"])[recent_mask]
            ax.plot(d, clim, color=COLORS["climatology"], lw=1.4, linestyle="--",
                    alpha=0.95, label="climatology", zorder=6)

            beyond = "BEYOND limit" if mult > 1 else ("AT limit" if mult == 1 else "within limit")
            ax.set_title(f"{tw}d = {mult}x tau_star ({window}d) -- {beyond}", fontsize=8.5)
            ax.tick_params(axis="x", labelsize=7, rotation=30)
            ax.tick_params(axis="y", labelsize=7)
            if ax is axes[0]:
                ax.set_ylabel("price", fontsize=8)
                ax.legend(loc="upper left", fontsize=6, ncol=1)

        fig.suptitle(f"{tkr}: train-window sweep highlight (full 8-point sweep in repo)", fontsize=9.5)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        safe_tkr = tkr.replace("=", "")
        fig.savefig(os.path.join(OUT_DIR, f"67_window_sweep_highlight_{safe_tkr}.png"), dpi=150)
        plt.close(fig)
        print(f"{tkr}: highlight_windows(days)={highlight_windows}")

    print("Saved: 67_window_sweep_highlight_{" +
          ",".join(t.replace("=", "") for t in HIGHLIGHT_INSTRUMENTS) + "}.png")
