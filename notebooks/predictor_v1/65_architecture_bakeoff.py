"""
Architecture bake-off, per the user's own design (see memory): given that
predictability limits are a genuine, irreducible ceiling (chaos theory /
stochastic dynamics, not a data-scarcity artifact -- Lorenz 1963/1969), the
interesting question is no longer "can a sophisticated AI/ML architecture
beat the predictability limit" (it can't, by hypothesis) but "which
architecture gets closest to the skill that IS achievable within it,
when every architecture is given the exact same fair, non-stale, half-window
training budget."

Five competing "models," all trained ONLY on the immediately preceding
half_window = floor(tau_star/2) days (same fresh-window design as
64_good_vs_stale_test.py), all sharing the identical 10-lagged-return
price-only feature set, walked forward across full history:

  1. Climatology       -- mean of the training window's forward returns.
     A real competing model, not a strawman (feedback-respect-climatology).
  2. Overfit tree       -- the Paper 13 diagnostic, DecisionTreeRegressor
     with zero regularization (one leaf per training row).
  3. RL (policy gradient) -- linear-Gaussian policy, continuous action =
     predicted forward return, reward = -squared error, trained via
     REINFORCE with a running baseline (bandit formulation of forecasting).
  4. Conditional GAN    -- linear generator G(x,z) vs. linear discriminator
     D(x,y), trained adversarially (manual gradients, non-saturating G
     loss); forecast = mean of K generator samples at test time.
  5. Conditional VAE (generative/stochastic) -- linear encoder/decoder with
     the reparameterization trick, fixed-variance Gaussian decoder (so
     reconstruction loss reduces to MSE) + KL regularization to a unit
     Gaussian prior; forecast = mean of K decoder samples from the prior
     at test time.

No torch in this environment -- all three iterative models (RL/GAN/VAE) are
small enough (10 features, half_window in {16,21,31} training rows) that a
from-scratch numpy implementation with manual gradients is both sufficient
and, given how little data each window has, more honest than reaching for a
deep net that couldn't possibly be justified by the sample size.

Per the user's explicit instruction: NO metrics, no skill scores, no
significance tests. Output is purely visual -- actual price vs. each
model's forecast price, overlaid, per instrument. If a sophisticated
architecture's curve is indistinguishable from climatology's, or from the
tree's, that is visible directly in the plot; it does not need a number
attached to it.

Instruments: the top 3 of 12 by Paper 11 predictability limit -- MSFT (63d),
EURUSD=X (43d), XLF (33d) -- chosen because a data-hungry architecture needs
as many real rows in its fair, valid training window as possible.

Run: python 65_architecture_bakeoff.py
Output: 65_architecture_bakeoff_{TICKER}.png (one per instrument)
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

INSTRUMENTS = {"MSFT": 189, "EURUSD=X": 189, "XLF": 189}  # ticker -> horizon (days ahead)
N_LAGS = 10


def load_tau_star():
    with open(PREDICTABILITY_JSON) as f:
        d = json.load(f)
    return {t: d[t]["2"]["top5_tradeable"][0][0] for t in INSTRUMENTS}


def build_price_frame(price_series, horizon, n_lags=N_LAGS):
    """One row per base date with a full label + full lag-feature history.
    Carries base_price/target_price explicitly so predicted returns can be
    converted back into predicted PRICE for the visual comparison."""
    vals = price_series.values
    dates = price_series.index
    n = len(vals)
    rets = np.log(vals[1:] / vals[:-1])
    rets = np.concatenate([[np.nan], rets])  # align to vals, rets[i] = log(vals[i]/vals[i-1])

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
    """Continuous-action bandit: policy pi(a|x) = N(w.x+b, sigma), reward =
    -(a-y)^2, trained via REINFORCE with a running reward baseline."""
    name = "RL (policy gradient)"

    def __init__(self, lr=0.05, epochs=400, sigma0=0.05, sigma_min=0.02, seed=0):
        self.lr, self.epochs, self.sigma0, self.sigma_min, self.seed = lr, epochs, sigma0, sigma_min, seed

    def fit(self, X, y):
        Xs, mean, std = _standardize(X)
        self.mean, self.std = mean, std
        rng = np.random.default_rng(self.seed)
        n, d = Xs.shape
        w = np.zeros(d)
        b = float(np.mean(y))
        baseline = 0.0
        y_scale = max(float(np.std(y)), 1e-6)
        for ep in range(self.epochs):
            sigma = max(self.sigma0 * (1 - ep / self.epochs), self.sigma_min) * y_scale
            mu = Xs @ w + b
            noise = rng.normal(0.0, sigma, size=n)
            a = mu + noise
            reward = -((a - y) / y_scale) ** 2
            baseline = 0.9 * baseline + 0.1 * reward.mean()
            adv = np.clip(reward - baseline, -5.0, 5.0)
            # Natural-gradient step for a Gaussian policy's mean: proportional
            # to advantage * raw noise, without the 1/sigma^2 score-function
            # rescaling that otherwise blows up as sigma is annealed toward 0.
            grad_w = (adv * noise)[:, None] * Xs
            grad_b = adv * noise
            w = w + self.lr * grad_w.mean(axis=0)
            b = b + self.lr * grad_b.mean()
        self.w, self.b = w, b
        return self

    def predict(self, X):
        Xs = (X - self.mean) / self.std
        return Xs @ self.w + self.b


class ConditionalGANForecaster:
    """Linear generator G(x,z) vs. linear discriminator D(x,y), trained
    adversarially with manual gradients (non-saturating G loss). Forecast
    is the mean of K generator draws at test time."""
    name = "conditional GAN"

    def __init__(self, z_dim=2, lr=0.1, epochs=400, k_samples=200, seed=0):
        self.z_dim, self.lr, self.epochs, self.k_samples, self.seed = z_dim, lr, epochs, k_samples, seed

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        Xs, mean, std = _standardize(X)
        self.mean, self.std = mean, std
        rng = np.random.default_rng(self.seed)
        n, d = Xs.shape
        zd = self.z_dim

        wg = np.zeros(d)
        vg = rng.normal(0, 0.01, size=zd)
        bg = float(np.mean(y))

        wd = np.zeros(d)
        ud = 0.0
        bd = 0.0

        for ep in range(self.epochs):
            z = rng.normal(0.0, 1.0, size=(n, zd))
            y_fake = Xs @ wg + z @ vg + bg

            s_real = Xs @ wd + ud * y + bd
            s_fake = Xs @ wd + ud * y_fake + bd
            D_real = self._sigmoid(s_real)
            D_fake = self._sigmoid(s_fake)

            # discriminator: minimize -(log D_real + log(1-D_fake))
            dL_ds_real = -(1 - D_real)
            dL_ds_fake = D_fake
            gwd = ((dL_ds_real[:, None] * Xs) + (dL_ds_fake[:, None] * Xs)).mean(axis=0)
            gud = (dL_ds_real * y + dL_ds_fake * y_fake).mean()
            gbd = (dL_ds_real + dL_ds_fake).mean()
            wd = wd - self.lr * gwd
            ud = ud - self.lr * gud
            bd = bd - self.lr * gbd

            # generator: minimize -log D(x, y_fake)  (non-saturating)
            s_fake_g = Xs @ wd + ud * y_fake + bd
            D_fake_g = self._sigmoid(s_fake_g)
            dLg_dyfake = -(1 - D_fake_g) * ud
            gwg = (dLg_dyfake[:, None] * Xs).mean(axis=0)
            gvg = (dLg_dyfake[:, None] * z).mean(axis=0)
            gbg = dLg_dyfake.mean()
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
        return acc / self.k_samples


class ConditionalVAEForecaster:
    """Linear encoder q(z|x,y)=N(mu_e,sigma_e^2) / linear decoder
    p(y|x,z)=N(mu_d, fixed variance) with the reparameterization trick.
    Fixed decoder variance reduces reconstruction loss to MSE (a standard
    VAE simplification). Forecast = mean of K decoder draws from the prior
    z~N(0,1) at test time (no y is available at test time to encode)."""
    name = "conditional VAE (generative/stochastic)"

    def __init__(self, lr=0.1, epochs=400, beta=0.1, k_samples=200, seed=0):
        self.lr, self.epochs, self.beta, self.k_samples, self.seed = lr, epochs, beta, k_samples, seed

    def fit(self, X, y):
        Xs, mean, std = _standardize(X)
        self.mean, self.std = mean, std
        rng = np.random.default_rng(self.seed)
        n, d = Xs.shape

        We = np.zeros(d); be_y = 0.0; be = float(np.mean(y)) * 0.0  # mu_e = We.x + be_y*y + be
        Ue = np.zeros(d); ue_y = 0.0; ue = 0.0                       # log_sigma_e = Ue.x + ue_y*y + ue
        Wdx = np.zeros(d); wdz = 0.1; bd = float(np.mean(y))         # mu_d = Wdx.x + wdz*z + bd

        for ep in range(self.epochs):
            eps = rng.normal(0.0, 1.0, size=n)
            mu_e = Xs @ We + be_y * y + be
            log_sigma_e = Xs @ Ue + ue_y * y + ue
            sigma_e = np.exp(np.clip(log_sigma_e, -10, 10))
            z = mu_e + sigma_e * eps

            mu_d = Xs @ Wdx + wdz * z + bd
            resid = mu_d - y  # d(recon)/d(mu_d)

            dL_dmu_e = resid * wdz + self.beta * mu_e
            dL_dsigma_e = resid * wdz * eps + self.beta * (sigma_e - 1.0 / sigma_e)
            dL_dlogsig_e = dL_dsigma_e * sigma_e

            gWdx = (resid[:, None] * Xs).mean(axis=0)
            gwdz = (resid * z).mean()
            gbd = resid.mean()

            gWe = (dL_dmu_e[:, None] * Xs).mean(axis=0)
            gbe_y = (dL_dmu_e * y).mean()
            gbe = dL_dmu_e.mean()

            gUe = (dL_dlogsig_e[:, None] * Xs).mean(axis=0)
            gue_y = (dL_dlogsig_e * y).mean()
            gue = dL_dlogsig_e.mean()

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
        return acc / self.k_samples


MODEL_FACTORIES = [
    lambda: Climatology(),
    lambda: OverfitTree(),
    lambda: RLPolicyForecaster(),
    lambda: ConditionalGANForecaster(),
    lambda: ConditionalVAEForecaster(),
]


def run_bakeoff(frame, half_window):
    frame = frame.reset_index(drop=True)
    n = len(frame)
    target_dates, actual_price = [], []
    pred_price = {m().name: [] for m in MODEL_FACTORIES}
    p = half_window
    while p + half_window <= n:
        train_df = frame.iloc[p - half_window:p]
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

    colors = {
        "climatology": "#8E8E8E",
        "overfit tree (Paper 13 diagnostic)": "#5B8DBE",
        "RL (policy gradient)": "#2E8B57",
        "conditional GAN": "#C0392B",
        "conditional VAE (generative/stochastic)": "#8E44AD",
    }

    for tkr, horizon in INSTRUMENTS.items():
        window = tau_star[tkr]
        half_window = window // 2
        series = prices[tkr].dropna()
        frame = build_price_frame(series, horizon)

        dates, actual, preds = run_bakeoff(frame, half_window)

        dates_arr = np.array(dates)
        recent_mask = dates_arr >= (dates_arr[-1] - pd.Timedelta(days=730))

        fig, axes = plt.subplots(2, 1, figsize=(16, 11))
        for ax, mask, is_log, subtitle in [
            (axes[0], slice(None), True, "full history, log scale"),
            (axes[1], recent_mask, False, "most recent 2 years, linear scale"),
        ]:
            d = dates_arr[mask] if isinstance(mask, np.ndarray) else dates_arr
            a = np.array(actual)[mask] if isinstance(mask, np.ndarray) else np.array(actual)
            ax.plot(d, a, color="#222222", lw=1.3, label="actual price", zorder=5)
            # Climatology is drawn last, dashed, so it stays visible even
            # where its curve sits directly under the other, more textured
            # models -- the whole point of including it is to be able to
            # SEE when a sophisticated model's curve collapses onto it.
            non_clima = {k: v for k, v in preds.items() if k != "climatology"}
            for name, vals in non_clima.items():
                v = np.array(vals)[mask] if isinstance(mask, np.ndarray) else np.array(vals)
                ax.plot(d, v, color=colors[name], lw=1.1, alpha=0.85, label=name, zorder=3)
            clim = np.array(preds["climatology"])[mask] if isinstance(mask, np.ndarray) else np.array(preds["climatology"])
            ax.plot(d, clim, color=colors["climatology"], lw=1.8,
                    linestyle="--", alpha=0.95, label="climatology", zorder=6)
            if is_log:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_title(subtitle, fontsize=9, loc="left")
            ax.set_xlabel("target date")
            ax.set_ylabel("price" + (" (log scale)" if is_log else ""))
            ax.legend(loc="upper left", fontsize=8, ncol=2)

        fig.suptitle(
            f"{tkr}: architecture bake-off, all models trained ONLY on the immediately preceding "
            f"{half_window}d (half of the {window}d predictability limit) -- {horizon}d-ahead price forecast",
            fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        safe_tkr = tkr.replace("=", "")
        fig.savefig(os.path.join(OUT_DIR, f"65_architecture_bakeoff_{safe_tkr}.png"), dpi=140)
        plt.close(fig)
        print(f"{tkr}: half_window={half_window} window={window} n_points={len(actual)}")

    print("Saved: 65_architecture_bakeoff_{" + ",".join(t.replace("=", "") for t in INSTRUMENTS) + "}.png")
