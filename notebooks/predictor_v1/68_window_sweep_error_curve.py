"""
Quantitative companion to 66_window_sweep_bakeoff.py's visual sweep. That
script showed, panel by panel, that all five architectures' predicted-price
curves visibly detach from actual price as training-window size grows past
the Paper 11 predictability limit. This script puts a number on that same
detachment: for each of the same eight swept training-window multiples of
tau_star, and each of the same five architectures, it computes the mean
absolute error across every walk-forward test point in an instrument's
full available history -- not just the recent-2-year slice used for the
visual panels -- expressed as a percentage of that instrument's own mean
price over the test period (not raw price units, which are incomparable
across a $1.20 currency pair and a $500 stock), and plots that percentage
against the swept multiple, one curve per model, one figure per instrument.
Where 66_'s figures make the decay something a reader sees, this script's
figures make it something a reader can read off an axis directly, in units
ordinary common sense can judge as acceptable or not.

This is a direct, descriptive summary statistic (a mean absolute error),
not a significance test, a p-value, or a resampling procedure -- consistent
with this research program's standing rejection of the latter, but not in
tension with it; a plotted MAE curve is the same kind of object as Paper
11's own G(tau,q) structure-function curve, just for this paper's own
architectures instead of Paper 11's non-parametric decorrelation measure.

Run: python 68_window_sweep_error_curve.py
Output: 68_window_sweep_error_curve_{TICKER}.png (one per instrument, 12 total)
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
    """Same fixed test segmentation across every train_window size; returns
    per-model mean absolute price error over EVERY walk-forward test point
    in the instrument's full available history (not just a recent slice),
    plus the mean actual price over those same test points (for expressing
    error as a percentage of price rather than raw price units)."""
    frame = frame.reset_index(drop=True)
    n = len(frame)
    mae = {}
    abs_err_accum = {m().name: [] for m in MODEL_FACTORIES}
    actual_price_accum = []
    p = p_start
    while p + half_window <= n:
        train_df = frame.iloc[max(0, p - train_window):p]
        test_df = frame.iloc[p:p + half_window]
        Xtr, ytr = train_df[FEATURE_COLS].values, train_df["fwd_ret"].values
        Xte = test_df[FEATURE_COLS].values
        actual_price = test_df["target_price"].values
        base_price = test_df["base_price"].values
        actual_price_accum.extend(actual_price.tolist())

        for factory in MODEL_FACTORIES:
            model = factory()
            model.fit(Xtr, ytr)
            pred_ret = model.predict(Xte)
            pred_price = base_price * np.exp(pred_ret)
            abs_err_accum[model.name].extend(np.abs(pred_price - actual_price).tolist())
        p += half_window
    for name, errs in abs_err_accum.items():
        mae[name] = float(np.mean(errs))
    mean_price = float(np.mean(actual_price_accum))
    return mae, mean_price


def make_plot(tkr, window, mae_by_model, mean_price, out_path):
    """Shared plotting logic, usable both from a fresh run and from a
    replot of already-computed MAE values (e.g. parsed back out of a prior
    run's log, to avoid re-paying the ~25-30 minute full compute cost for a
    presentation-only change).

    Two things beyond the raw curves: (1) the y-axis is expressed as mean
    absolute error relative to the instrument's own mean price over the
    test period, not raw price units -- a $9 error means something very
    different for a $20 instrument than a $500 one, and only the
    percentage form lets a reader judge, using ordinary common sense
    ("an error this large relative to the price is clearly too big"),
    whether a given window size is still acceptable, without an
    arbitrarily chosen numeric threshold; and (2) a shaded "within/at
    predictability limit" region for x in [0.5, 1.0] -- not a new, invented
    boundary, but a direct restatement of the one Ramanathan (2026a)
    already established independently, so the reader can see directly
    whether the point where a curve's percentage error visibly worsens
    coincides with it.
    """
    x = np.array(SWEEP_MULTIPLES)
    pct_by_model = {name: (np.array(vals) / mean_price) * 100.0 for name, vals in mae_by_model.items()}

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.axvspan(x.min(), 1.0, color="#4C9A5A", alpha=0.10, zorder=0,
               label="within / at predictability limit (Ramanathan, 2026a)")

    # Climatology is drawn last, with a distinct diamond marker and thicker
    # dashed line, so it stays visible even where other curves sit directly
    # on top of it -- a plain circle marker in the same grey nearly
    # disappears into the bunched-together cluster in most instruments.
    non_clima = {k: v for k, v in pct_by_model.items() if k != "climatology"}
    for name, vals in non_clima.items():
        ax.plot(x, vals, marker="o", markersize=4, color=COLORS[name], lw=1.6,
                linestyle="-", label=name, zorder=3)
    ax.plot(x, pct_by_model["climatology"], marker="D", markersize=5.5,
            color=COLORS["climatology"], lw=2.0, linestyle="--",
            label="climatology", zorder=4)

    ax.set_xlabel("training-window size (multiple of tau_star)")
    ax.set_ylabel("mean absolute error, as % of mean price over the test period")
    ax.set_title(f"{tkr}: error (% of mean price) vs. training-window size (tau_star={window}d)",
                 fontsize=10, loc="left")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    prices = pd.read_parquet(os.path.join(NB_DIR, "multiasset_prices.parquet"))
    tau_star = load_tau_star()

    for tkr, horizon in INSTRUMENTS.items():
        window = tau_star[tkr]
        half_window = window // 2
        series = prices[tkr].dropna()
        frame = build_price_frame(series, horizon)

        train_windows = [max(int(round(mult * window)), 4) for mult in SWEEP_MULTIPLES]
        p_start = max(train_windows) + half_window  # identical anchor to 66_'s full sweep

        mae_by_model = {m().name: [] for m in MODEL_FACTORIES}
        mean_prices = []
        for tw in train_windows:
            mae, mean_price = run_sweep_window(frame, half_window, tw, p_start)
            for name, val in mae.items():
                mae_by_model[name].append(val)
            mean_prices.append(mean_price)
        mean_price = float(np.mean(mean_prices))  # identical test segments every multiple; sanity-averaged

        safe_tkr = tkr.replace("=", "")
        make_plot(tkr, window, mae_by_model, mean_price,
                   os.path.join(OUT_DIR, f"68_window_sweep_error_curve_{safe_tkr}.png"))
        print(f"{tkr}: window(tau_star)={window} mean_price={mean_price} MAE_by_model={mae_by_model}")

    print("Saved: 68_window_sweep_error_curve_{" + ",".join(t.replace("=", "") for t in INSTRUMENTS) + "}.png")
