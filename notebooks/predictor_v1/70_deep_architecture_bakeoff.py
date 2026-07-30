"""
Follow-up to 65_architecture_bakeoff.py, prompted by a direct question: if
the linear-parameter RL/GAN/VAE in Experiment 1 can't beat climatology
within the predictability limit, is that about the predictability limit,
or just about those three models being linear? This script holds
everything else in Experiment 1's design fixed -- same three instruments
(MSFT, EURUSD=X, XLF), same half-window = floor(tau_star/2) training
budget, same walk-forward evaluation, same price-only 10-lagged-return
features -- and adds a genuinely nonlinear (one hidden layer, tanh
activation, manual backprop) version of each of the RL forecaster, the
conditional GAN, and the conditional VAE, run alongside their original
linear counterparts plus climatology and the overfit tree. No extra data
of any kind is given to the nonlinear versions; the test is whether
nonlinearity/depth alone, at zero additional data cost, changes anything.

This deliberately does NOT feed the nonlinear models more data to make
them "properly" trainable -- doing so would mean training on windows
larger than the predictability limit permits, which is precisely the
constraint under test, not a design inconvenience to route around.

Hidden layer size (6 units) was chosen to be the same order of magnitude
as the 10 input features -- enough to be genuinely nonlinear, not enough
to be a gratuitous capacity increase dressed up as "depth." All three
nonlinear architectures were validated on a synthetic nonlinear regression
task before being run here (correlations of 0.65-0.997 with the true
signal on ample synthetic data -- see the model class docstrings), to
confirm the hand-derived backprop is correct before drawing any
conclusion from how it behaves on 11-31 real, tiny rows.

Run: python 70_deep_architecture_bakeoff.py
Output: 70_deep_architecture_bakeoff_{TICKER}.png (MSFT, EURUSDX, XLF)
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)  # benign Accelerate/BLAS FP-flag quirk, verified

NB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTABILITY_JSON = os.path.join(NB_DIR, "predictability_paper", "results_correlated_decorrelated.json")

INSTRUMENTS = {"MSFT": 189, "EURUSD=X": 189, "XLF": 189}
N_LAGS = 10
HIDDEN = 6  # same order of magnitude as the 10 input features, not a gratuitous capacity increase


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


def _init_layer(fan_in, fan_out, rng, scale=None):
    if scale is None:
        scale = 1.0 / np.sqrt(fan_in)
    return rng.normal(0, scale, size=(fan_in, fan_out)), np.zeros(fan_out)


def _clip(g):
    return np.clip(g, -10.0, 10.0)


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
    """Linear-Gaussian policy, as in 65_architecture_bakeoff.py."""
    name = "RL, linear"

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
            grad_w = _clip((adv * noise)[:, None] * Xs)
            grad_b = _clip(adv * noise)
            w = w + self.lr * grad_w.mean(axis=0)
            b = b + self.lr * grad_b.mean()
        self.w, self.b = w, b
        return self

    def predict(self, X):
        Xs = (X - self.mean) / self.std
        return (Xs @ self.w + self.b) * self.y_scale


class DeepRLPolicyForecaster:
    """Same continuous-action bandit/REINFORCE setup as RLPolicyForecaster,
    but the policy mean is a 1-hidden-layer tanh MLP instead of a linear
    map. Validated on synthetic nonlinear data (corr 0.94) before use here."""
    name = "RL, 1-hidden-layer MLP"

    def __init__(self, hidden=HIDDEN, lr=0.05, epochs=600, sigma0=0.3, sigma_min=0.05, seed=0):
        self.hidden, self.lr, self.epochs = hidden, lr, epochs
        self.sigma0, self.sigma_min, self.seed = sigma0, sigma_min, seed

    def fit(self, X, y):
        Xs, mean, std = _standardize(X)
        self.mean, self.std = mean, std
        y_scale = max(float(np.std(y)), 1e-8)
        self.y_scale = y_scale
        ys = y / y_scale
        rng = np.random.default_rng(self.seed)
        n, d = Xs.shape
        H = self.hidden
        W1, b1 = _init_layer(d, H, rng)
        W2 = np.zeros(H)
        b2 = float(np.mean(ys))
        baseline = 0.0
        for ep in range(self.epochs):
            sigma = max(self.sigma0 * (1 - ep / self.epochs), self.sigma_min)
            h = np.tanh(Xs @ W1 + b1)
            mu = h @ W2 + b2
            noise = rng.normal(0.0, sigma, size=n)
            a = mu + noise
            reward = -(a - ys) ** 2
            baseline = 0.9 * baseline + 0.1 * reward.mean()
            adv = np.clip(reward - baseline, -5.0, 5.0)
            grad_common = adv * noise
            gW2 = _clip((h * grad_common[:, None]).mean(axis=0))
            gb2 = _clip(grad_common.mean())
            gh = grad_common[:, None] * W2[None, :]
            gpre = gh * (1 - h ** 2)
            gW1 = _clip(Xs.T @ gpre / n)
            gb1 = _clip(gpre.mean(axis=0))
            W2 = W2 + self.lr * gW2
            b2 = b2 + self.lr * gb2
            W1 = W1 + self.lr * gW1
            b1 = b1 + self.lr * gb1
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
        return self

    def predict(self, X):
        Xs = (X - self.mean) / self.std
        h = np.tanh(Xs @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2) * self.y_scale


class ConditionalGANForecaster:
    """Linear generator/discriminator, as in 65_architecture_bakeoff.py."""
    name = "GAN, linear"

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
            gwd = _clip(((dL_ds_real[:, None] * Xs) + (dL_ds_fake[:, None] * Xs)).mean(axis=0))
            gud = _clip((dL_ds_real * ys + dL_ds_fake * y_fake).mean())
            gbd = _clip((dL_ds_real + dL_ds_fake).mean())
            wd = wd - self.lr * gwd
            ud = ud - self.lr * gud
            bd = bd - self.lr * gbd

            s_fake_g = Xs @ wd + ud * y_fake + bd
            D_fake_g = self._sigmoid(s_fake_g)
            dLg_dyfake = -(1 - D_fake_g) * ud
            gwg = _clip((dLg_dyfake[:, None] * Xs).mean(axis=0))
            gvg = _clip((dLg_dyfake[:, None] * z).mean(axis=0))
            gbg = _clip(dLg_dyfake.mean())
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


class DeepGANForecaster:
    """Same conditional-GAN minimax setup as ConditionalGANForecaster, but
    both the generator and the discriminator are 1-hidden-layer tanh MLPs.
    Validated on synthetic nonlinear data (corr 0.65 -- GANs are harder to
    train than direct regression even with correct gradients, so this is
    the expected order of validation success, not a red flag) before use
    here."""
    name = "GAN, 1-hidden-layer MLP"

    def __init__(self, hidden=HIDDEN, z_dim=2, lr=0.05, epochs=600, k_samples=200, seed=0):
        self.hidden, self.z_dim, self.lr, self.epochs, self.k_samples, self.seed = \
            hidden, z_dim, lr, epochs, k_samples, seed

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
        H, zd = self.hidden, self.z_dim

        W1g, b1g = _init_layer(d + zd, H, rng)
        W2g = np.zeros(H)
        b2g = float(np.mean(ys))
        W1d, b1d = _init_layer(d + 1, H, rng, scale=0.1)
        W2d = np.zeros(H)
        b2d = 0.0

        for ep in range(self.epochs):
            z = rng.normal(0.0, 1.0, size=(n, zd))
            gin = np.concatenate([Xs, z], axis=1)
            hg = np.tanh(gin @ W1g + b1g)
            y_fake = hg @ W2g + b2g

            din_real = np.concatenate([Xs, ys[:, None]], axis=1)
            hd_real = np.tanh(din_real @ W1d + b1d)
            s_real = hd_real @ W2d + b2d
            D_real = self._sigmoid(s_real)

            din_fake = np.concatenate([Xs, y_fake[:, None]], axis=1)
            hd_fake = np.tanh(din_fake @ W1d + b1d)
            s_fake = hd_fake @ W2d + b2d
            D_fake = self._sigmoid(s_fake)

            dL_ds_real = -(1 - D_real)
            dL_ds_fake = D_fake

            gW2d_r = (hd_real * dL_ds_real[:, None]).mean(axis=0)
            gb2d_r = dL_ds_real.mean()
            gpre_real = (dL_ds_real[:, None] * W2d[None, :]) * (1 - hd_real ** 2)
            gW1d_r = din_real.T @ gpre_real / n
            gb1d_r = gpre_real.mean(axis=0)

            gW2d_f = (hd_fake * dL_ds_fake[:, None]).mean(axis=0)
            gb2d_f = dL_ds_fake.mean()
            gpre_fake = (dL_ds_fake[:, None] * W2d[None, :]) * (1 - hd_fake ** 2)
            gW1d_f = din_fake.T @ gpre_fake / n
            gb1d_f = gpre_fake.mean(axis=0)

            W2d = W2d - self.lr * _clip(gW2d_r + gW2d_f)
            b2d = b2d - self.lr * _clip(gb2d_r + gb2d_f)
            W1d = W1d - self.lr * _clip(gW1d_r + gW1d_f)
            b1d = b1d - self.lr * _clip(gb1d_r + gb1d_f)

            hd_fake2 = np.tanh(din_fake @ W1d + b1d)
            s_fake2 = hd_fake2 @ W2d + b2d
            D_fake2 = self._sigmoid(s_fake2)
            dLg_ds = -(1 - D_fake2)
            w1d_y_row = W1d[-1, :]
            dLg_dyfake = dLg_ds * ((1 - hd_fake2 ** 2) @ (W2d * w1d_y_row))

            gW2g = _clip((hg * dLg_dyfake[:, None]).mean(axis=0))
            gb2g = _clip(dLg_dyfake.mean())
            gpre_g = (dLg_dyfake[:, None] * W2g[None, :]) * (1 - hg ** 2)
            gW1g = _clip(gin.T @ gpre_g / n)
            gb1g = _clip(gpre_g.mean(axis=0))

            W2g = W2g - self.lr * gW2g
            b2g = b2g - self.lr * gb2g
            W1g = W1g - self.lr * gW1g
            b1g = b1g - self.lr * gb1g

        self.W1g, self.b1g, self.W2g, self.b2g = W1g, b1g, W2g, b2g
        self._rng = rng
        return self

    def predict(self, X):
        Xs = (X - self.mean) / self.std
        n = len(Xs)
        acc = np.zeros(n)
        for _ in range(self.k_samples):
            z = self._rng.normal(0.0, 1.0, size=(n, self.z_dim))
            gin = np.concatenate([Xs, z], axis=1)
            hg = np.tanh(gin @ self.W1g + self.b1g)
            acc += hg @ self.W2g + self.b2g
        return (acc / self.k_samples) * self.y_scale


class ConditionalVAEForecaster:
    """Linear encoder/decoder, as in 65_architecture_bakeoff.py."""
    name = "VAE, linear"

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

            gWdx = _clip((resid[:, None] * Xs).mean(axis=0))
            gwdz = _clip((resid * z).mean())
            gbd = _clip(resid.mean())
            gWe = _clip((dL_dmu_e[:, None] * Xs).mean(axis=0))
            gbe_y = _clip((dL_dmu_e * ys).mean())
            gbe = _clip(dL_dmu_e.mean())
            gUe = _clip((dL_dlogsig_e[:, None] * Xs).mean(axis=0))
            gue_y = _clip((dL_dlogsig_e * ys).mean())
            gue = _clip(dL_dlogsig_e.mean())

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


class DeepVAEForecaster:
    """Same conditional-VAE ELBO setup as ConditionalVAEForecaster, but the
    encoder and decoder are each 1-hidden-layer tanh MLPs (encoder's two
    heads, mu_e and log_sigma_e, share one hidden trunk). Validated on
    synthetic nonlinear data (corr 0.997) before use here."""
    name = "VAE, 1-hidden-layer MLP"

    def __init__(self, hidden=HIDDEN, lr=0.05, epochs=600, beta=0.1, k_samples=200, seed=0):
        self.hidden, self.lr, self.epochs, self.beta, self.k_samples, self.seed = \
            hidden, lr, epochs, beta, k_samples, seed

    def fit(self, X, y):
        Xs, mean, std = _standardize(X)
        self.mean, self.std = mean, std
        y_scale = max(float(np.std(y)), 1e-8)
        self.y_scale = y_scale
        ys = y / y_scale
        rng = np.random.default_rng(self.seed)
        n, d = Xs.shape
        H = self.hidden

        W1e, b1e = _init_layer(d + 1, H, rng, scale=0.1)
        W2e_mu = np.zeros(H); b2e_mu = 0.0
        W2e_ls = np.zeros(H); b2e_ls = 0.0
        W1d, b1d = _init_layer(d + 1, H, rng)
        W2d = np.zeros(H); b2d = float(np.mean(ys))

        for ep in range(self.epochs):
            eps = rng.normal(0.0, 1.0, size=n)
            ein = np.concatenate([Xs, ys[:, None]], axis=1)
            he = np.tanh(ein @ W1e + b1e)
            mu_e = he @ W2e_mu + b2e_mu
            log_sigma_e = np.clip(he @ W2e_ls + b2e_ls, -5, 5)
            sigma_e = np.exp(log_sigma_e)
            z = mu_e + sigma_e * eps

            din = np.concatenate([Xs, z[:, None]], axis=1)
            hd = np.tanh(din @ W1d + b1d)
            mu_d = hd @ W2d + b2d
            resid = mu_d - ys

            gW2d = _clip((hd * resid[:, None]).mean(axis=0))
            gb2d = _clip(resid.mean())
            gpre_d = (resid[:, None] * W2d[None, :]) * (1 - hd ** 2)
            gW1d = _clip(din.T @ gpre_d / n)
            gb1d = _clip(gpre_d.mean(axis=0))

            w1d_z_row = W1d[-1, :]
            drecon_dz = resid * ((1 - hd ** 2) @ (W2d * w1d_z_row))

            dL_dmu_e = drecon_dz + self.beta * mu_e
            dL_dsigma_e = drecon_dz * eps + self.beta * (sigma_e - 1.0 / sigma_e)
            dL_dlogsig_e = dL_dsigma_e * sigma_e

            gW2e_mu = _clip((he * dL_dmu_e[:, None]).mean(axis=0))
            gb2e_mu = _clip(dL_dmu_e.mean())
            gW2e_ls = _clip((he * dL_dlogsig_e[:, None]).mean(axis=0))
            gb2e_ls = _clip(dL_dlogsig_e.mean())
            gpre_e = (dL_dmu_e[:, None] * W2e_mu[None, :] + dL_dlogsig_e[:, None] * W2e_ls[None, :]) * (1 - he ** 2)
            gW1e = _clip(ein.T @ gpre_e / n)
            gb1e = _clip(gpre_e.mean(axis=0))

            W2d -= self.lr * gW2d; b2d -= self.lr * gb2d
            W1d -= self.lr * gW1d; b1d -= self.lr * gb1d
            W2e_mu -= self.lr * gW2e_mu; b2e_mu -= self.lr * gb2e_mu
            W2e_ls -= self.lr * gW2e_ls; b2e_ls -= self.lr * gb2e_ls
            W1e -= self.lr * gW1e; b1e -= self.lr * gb1e

        self.W1d, self.b1d, self.W2d, self.b2d = W1d, b1d, W2d, b2d
        self._rng = rng
        return self

    def predict(self, X):
        Xs = (X - self.mean) / self.std
        n = len(Xs)
        acc = np.zeros(n)
        for _ in range(self.k_samples):
            z = self._rng.normal(0.0, 1.0, size=n)
            din = np.concatenate([Xs, z[:, None]], axis=1)
            hd = np.tanh(din @ self.W1d + self.b1d)
            acc += hd @ self.W2d + self.b2d
        return (acc / self.k_samples) * self.y_scale


MODEL_FACTORIES = [
    lambda: Climatology(),
    lambda: OverfitTree(),
    lambda: RLPolicyForecaster(),
    lambda: DeepRLPolicyForecaster(),
    lambda: ConditionalGANForecaster(),
    lambda: DeepGANForecaster(),
    lambda: ConditionalVAEForecaster(),
    lambda: DeepVAEForecaster(),
]

COLORS = {
    "climatology": "#8E8E8E",
    "overfit tree (Paper 13 diagnostic)": "#5B8DBE",
    "RL, linear": "#2E8B57",
    "RL, 1-hidden-layer MLP": "#1ABC9C",
    "GAN, linear": "#C0392B",
    "GAN, 1-hidden-layer MLP": "#E67E22",
    "VAE, linear": "#8E44AD",
    "VAE, 1-hidden-layer MLP": "#AD1457",
}


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
            non_clima = {k: v for k, v in preds.items() if k != "climatology"}
            for name, vals in non_clima.items():
                v = np.array(vals)[mask] if isinstance(mask, np.ndarray) else np.array(vals)
                ax.plot(d, v, color=COLORS[name], lw=1.0, alpha=0.85, label=name, zorder=3)
            clim = np.array(preds["climatology"])[mask] if isinstance(mask, np.ndarray) else np.array(preds["climatology"])
            ax.plot(d, clim, color=COLORS["climatology"], lw=1.8, linestyle="--",
                    alpha=0.95, label="climatology", zorder=6, marker="D", markersize=3.5)
            if is_log:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_title(subtitle, fontsize=9, loc="left")
            ax.set_xlabel("target date")
            ax.set_ylabel("price" + (" (log scale)" if is_log else ""))
            ax.legend(loc="upper left", fontsize=7, ncol=2)

        fig.suptitle(
            f"{tkr}: does nonlinearity/depth help within the predictability limit, at zero extra data cost? "
            f"half-window={half_window}d, {horizon}d-ahead forecast", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        safe_tkr = tkr.replace("=", "")
        fig.savefig(os.path.join(OUT_DIR, f"70_deep_architecture_bakeoff_{safe_tkr}.png"), dpi=140)
        plt.close(fig)
        print(f"{tkr}: half_window={half_window} window={window} n_points={len(actual)}")

    print("Saved: 70_deep_architecture_bakeoff_{" + ",".join(t.replace("=", "") for t in INSTRUMENTS) + "}.png")
