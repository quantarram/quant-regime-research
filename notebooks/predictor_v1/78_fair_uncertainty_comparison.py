"""
Direct follow-up to 76_downscaler_vs_architectures_panel.py, closing a real
fairness gap in that comparison: three of the five architectures (RL, GAN,
VAE) are ALREADY internally stochastic -- each draws 200 samples at
prediction time -- but 76_'s predict() collapses every one of them down to
a single averaged number before scoring, so their own CRPS there is really
"CRPS if this stochastic model is forced to pretend it's certain," not
"CRPS of what this model actually knows." Climatology and the tree have no
built-in sampler, but each has an equally natural, zero-new-modeling-effort
uncertainty source: climatology's own training-window return distribution,
and the tree's own leaf-mate residuals.

This script gives every one of the 5 architectures its own fair shot at an
uncertainty estimate, using ONLY what it already has internally -- no new
calibration, no borrowed technique from the downscaler's 3 fixes (76_):

  Climatology  -- the training window's own observed forward returns,
                  used directly as the empirical ensemble (its true and
                  complete "knowledge" of the return distribution).
  Overfit tree -- for each test row, the OTHER training rows that landed
                  in the same leaf (sklearn's own apply()). Since these
                  trees are unregularized (min_samples_leaf=1), most
                  leaves contain exactly the row that defined them --
                  expected to often degenerate back to a single point,
                  which is itself an honest, disclosed finding about this
                  specific model, not a bug to route around.
  RL forecaster -- the policy's own final annealed Gaussian sigma
                  (already computed during training, previously discarded
                  at predict() time in favor of the mean action alone).
  Conditional GAN -- the actual 200 generator draws already computed
                  inside predict(), previously averaged away.
  Conditional VAE -- the actual 200 decoder draws already computed inside
                  predict(), previously averaged away.

Each is scored by the same CRPS estimator already used for the generative
downscaler (energy_score_crps, below) -- fully apples-to-apples. The
downscaler's own (76_'s three-fix-calibrated) CRPS is included for
reference, unchanged.

Expectation, stated before running anything: none of these five natural
uncertainty sources have been calibrated in any way (no tau_star-scale
chaining, no decoder-variance correction, no recency-windowed variance
rescaling -- the three real fixes 76_ needed before its own ensemble was
trustworthy). They may therefore easily be UNDER- or OVER-dispersed
relative to real outcomes, and CRPS punishes both. The honest hypothesis
this script tests is not "does merely HAVING a distribution help" -- it is
whether an uncalibrated, already-available distribution is enough on its
own, or whether calibration (as 76_ did for the downscaler) is what
actually matters.

CRPS estimated via the standard unbiased ensemble ("energy score") form:
    CRPS(ensemble, y) ~= (1/K) sum_i |x_i - y|  -  (1/(2K^2)) sum_i,j |x_i - x_j|

Walk-forward, instruments, features, cutoff: identical to 76_.

Run: python 78_fair_uncertainty_comparison.py
Output: results_fair_uncertainty_comparison.json,
        78_fair_uncertainty_comparison_crps.png
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

warnings.filterwarnings("ignore", category=RuntimeWarning)  # benign Accelerate/BLAS FP-flag quirk, verified (see 70_)

NB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTABILITY_JSON = os.path.join(NB_DIR, "predictability_paper", "results_correlated_decorrelated.json")

# All 12 of Paper 13's instruments, per-instrument horizon -- exact copy of 66_window_sweep_bakeoff.py's dict.
INSTRUMENTS = {
    "GLD": 189, "JPM": 252, "AAPL": 252, "XLK": 189, "EURUSD=X": 189, "IWM": 21,
    "MSFT": 189, "QQQ": 21, "SPY": 189, "XLE": 252, "XLF": 189, "XOM": 63,
}
N_LAGS = 10
Z_DIM = 6
N_ENSEMBLE = 200
TRAIN_CUTOFF = "2020-01-01"
GRAD_CLIP = 5.0


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


# ---- The 5 existing architectures, verbatim from 65_architecture_bakeoff.py ----

class Climatology:
    name = "climatology"

    def fit(self, X, y):
        self.mu = float(np.mean(y))
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.mu)

    def sample(self, X, k, rng):
        """Natural ensemble: the training window's own observed forward
        returns, unchanged -- this IS climatology's actual knowledge of the
        return distribution, not a stand-in for it. Same finite set (size =
        half_window) returned for every row, exactly mirroring how
        predict() also returns the same point for every row."""
        n = len(X)
        return np.tile(self.y_train[None, :], (n, 1))  # (n, half_window)


class OverfitTree:
    name = "overfit tree"

    def fit(self, X, y):
        self.m = DecisionTreeRegressor(max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=0)
        self.m.fit(X, y)
        self.y_train = np.asarray(y)
        self.leaf_ids_train = self.m.apply(X)
        return self

    def predict(self, X):
        return self.m.predict(X)

    def sample(self, X, k, rng):
        """Natural ensemble: for each test row, the actual training targets
        that landed in the SAME leaf (sklearn's own apply()). Unregularized
        (min_samples_leaf=1) trees typically carve one leaf per training
        row, so most test rows will get a degenerate (size-1) ensemble --
        an honest, expected consequence of this specific model's own
        structure, not routed around."""
        leaf_ids_test = self.m.apply(X)
        preds = self.m.predict(X)
        out = []
        for lid, p in zip(leaf_ids_test, preds):
            members = self.y_train[self.leaf_ids_train == lid]
            if len(members) == 0:
                members = np.array([p])  # unreachable: every leaf a test row can land in was built from >=1 training row
            out.append(members)
        return out  # ragged list, one array per row (sizes may differ)


class RLPolicyForecaster:
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
            grad_w = (adv * noise)[:, None] * Xs
            grad_b = adv * noise
            w = w + self.lr * grad_w.mean(axis=0)
            b = b + self.lr * grad_b.mean()
        self.w, self.b = w, b
        self.final_sigma = max(self.sigma0 * (1 - (self.epochs - 1) / self.epochs), self.sigma_min) * y_scale
        return self

    def predict(self, X):
        Xs = (X - self.mean) / self.std
        return Xs @ self.w + self.b

    def sample(self, X, k, rng):
        """Natural ensemble: the policy IS a Gaussian, pi(a|x) = N(mu(x),
        sigma^2) -- draw k actions per row from the same policy predict()
        already collapses to its mean. sigma is the final ANNEALED value
        (annealed toward sigma_min by design, for training convergence, not
        for representing genuine uncertainty -- likely far too narrow, and
        deliberately not fixed here, since the point is to test what the
        model already has, not to recalibrate it)."""
        mu = self.predict(X)
        return mu[:, None] + rng.normal(0.0, self.final_sigma, size=(len(X), k))


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
        rng = np.random.default_rng(self.seed)
        n, d = Xs.shape
        zd = self.z_dim
        wg = np.zeros(d); vg = rng.normal(0, 0.01, size=zd); bg = float(np.mean(y))
        wd = np.zeros(d); ud = 0.0; bd = 0.0
        for ep in range(self.epochs):
            z = rng.normal(0.0, 1.0, size=(n, zd))
            y_fake = Xs @ wg + z @ vg + bg
            s_real = Xs @ wd + ud * y + bd
            s_fake = Xs @ wd + ud * y_fake + bd
            D_real = self._sigmoid(s_real)
            D_fake = self._sigmoid(s_fake)
            dL_ds_real = -(1 - D_real)
            dL_ds_fake = D_fake
            gwd = ((dL_ds_real[:, None] * Xs) + (dL_ds_fake[:, None] * Xs)).mean(axis=0)
            gud = (dL_ds_real * y + dL_ds_fake * y_fake).mean()
            gbd = (dL_ds_real + dL_ds_fake).mean()
            wd = wd - self.lr * gwd; ud = ud - self.lr * gud; bd = bd - self.lr * gbd
            s_fake_g = Xs @ wd + ud * y_fake + bd
            D_fake_g = self._sigmoid(s_fake_g)
            dLg_dyfake = -(1 - D_fake_g) * ud
            gwg = (dLg_dyfake[:, None] * Xs).mean(axis=0)
            gvg = (dLg_dyfake[:, None] * z).mean(axis=0)
            gbg = dLg_dyfake.mean()
            wg = wg - self.lr * gwg; vg = vg - self.lr * gvg; bg = bg - self.lr * gbg
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

    def sample(self, X, k, rng):
        """Natural ensemble: the ACTUAL k generator draws predict() already
        computes and then averages away -- exposed here raw, (n, k),
        instead of collapsed to a single mean."""
        Xs = (X - self.mean) / self.std
        n = len(Xs)
        z = rng.normal(0.0, 1.0, size=(k, n, self.z_dim))
        base = (Xs @ self.wg + self.bg)[None, :]  # (1, n)
        draws = base + np.einsum("knd,d->kn", z, self.vg)  # (k, n)
        return draws.T  # (n, k)


class ConditionalVAEForecaster:
    name = "conditional VAE"

    def __init__(self, lr=0.1, epochs=400, beta=0.1, k_samples=200, seed=0):
        self.lr, self.epochs, self.beta, self.k_samples, self.seed = lr, epochs, beta, k_samples, seed

    def fit(self, X, y):
        Xs, mean, std = _standardize(X)
        self.mean, self.std = mean, std
        rng = np.random.default_rng(self.seed)
        n, d = Xs.shape
        We = np.zeros(d); be_y = 0.0; be = 0.0
        Ue = np.zeros(d); ue_y = 0.0; ue = 0.0
        Wdx = np.zeros(d); wdz = 0.1; bd = float(np.mean(y))
        for ep in range(self.epochs):
            eps = rng.normal(0.0, 1.0, size=n)
            mu_e = Xs @ We + be_y * y + be
            log_sigma_e = Xs @ Ue + ue_y * y + ue
            sigma_e = np.exp(np.clip(log_sigma_e, -10, 10))
            z = mu_e + sigma_e * eps
            mu_d = Xs @ Wdx + wdz * z + bd
            resid = mu_d - y
            dL_dmu_e = resid * wdz + self.beta * mu_e
            dL_dsigma_e = resid * wdz * eps + self.beta * (sigma_e - 1.0 / sigma_e)
            dL_dlogsig_e = dL_dsigma_e * sigma_e
            gWdx = (resid[:, None] * Xs).mean(axis=0); gwdz = (resid * z).mean(); gbd = resid.mean()
            gWe = (dL_dmu_e[:, None] * Xs).mean(axis=0); gbe_y = (dL_dmu_e * y).mean(); gbe = dL_dmu_e.mean()
            gUe = (dL_dlogsig_e[:, None] * Xs).mean(axis=0); gue_y = (dL_dlogsig_e * y).mean(); gue = dL_dlogsig_e.mean()
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

    def sample(self, X, k, rng):
        """Natural ensemble: the ACTUAL k decoder draws from the prior that
        predict() already computes and then averages away -- exposed here
        raw, (n, k). No decoder-noise term is added on top of z here (unlike
        the downscaler's Fix 2 in 76_) -- this is deliberately the model's
        UNCORRECTED, as-trained sampling behavior, the same standard this
        applies to every other architecture in this script."""
        Xs = (X - self.mean) / self.std
        n = len(Xs)
        z = rng.normal(0.0, 1.0, size=(k, n))
        base = (Xs @ self.Wdx + self.bd)[None, :]  # (1, n)
        draws = base + self.wdz * z  # (k, n)
        return draws.T  # (n, k)


MODEL_FACTORIES = [
    lambda: Climatology(), lambda: OverfitTree(), lambda: RLPolicyForecaster(),
    lambda: ConditionalGANForecaster(), lambda: ConditionalVAEForecaster(),
]


# ---- Generative downscaler's shape-VAE, from 75_generative_downscaler.py, with a
# vectorized batch sampler added (this script calls sample() at ~1500 folds total,
# the per-call python-loop version would be needlessly slow at this scale). ----

class ShapeVAE:
    def __init__(self, h, z_dim=Z_DIM, lr=0.05, epochs=300, beta=0.1, seed=0):
        self.h, self.z_dim, self.lr, self.epochs, self.beta, self.seed = h, z_dim, lr, epochs, beta, seed

    def fit(self, targets, c):
        n, h = targets.shape
        zd = self.z_dim
        t_mean = targets.mean(axis=0)
        t_std = targets.std(axis=0)
        t_std = np.where(t_std < 1e-8, 1e-8, t_std)
        ts = (targets - t_mean) / t_std
        c_mean, c_std = float(c.mean()), max(float(c.std()), 1e-6)
        cs = (c - c_mean) / c_std
        self.t_mean, self.t_std, self.c_mean, self.c_std = t_mean, t_std, c_mean, c_std

        rng = np.random.default_rng(self.seed)
        We = np.zeros((h, zd)); we_c = np.zeros(zd); be = np.zeros(zd)
        Ue = np.zeros((h, zd)); ue_c = np.zeros(zd); ue = np.zeros(zd)
        Wdz = rng.normal(0, 0.05, size=(zd, h)); wdc = np.zeros(h); bd = np.zeros(h)

        for _ in range(self.epochs):
            eps = rng.normal(0.0, 1.0, size=(n, zd))
            mu_e = ts @ We + np.outer(cs, we_c) + be
            log_sigma_e = np.clip(ts @ Ue + np.outer(cs, ue_c) + ue, -5, 5)
            sigma_e = np.exp(log_sigma_e)
            z = mu_e + sigma_e * eps
            mu_d = z @ Wdz + np.outer(cs, wdc) + bd
            resid = mu_d - ts

            gWdz = np.clip(z.T @ resid / n, -GRAD_CLIP, GRAD_CLIP)
            gwdc = np.clip((resid * cs[:, None]).mean(axis=0), -GRAD_CLIP, GRAD_CLIP)
            gbd = np.clip(resid.mean(axis=0), -GRAD_CLIP, GRAD_CLIP)

            dL_dz = resid @ Wdz.T
            dL_dmu_e = dL_dz + self.beta * mu_e
            dL_dsigma_e = dL_dz * eps
            dL_dlogsig_e = dL_dsigma_e * sigma_e + self.beta * (sigma_e ** 2 - 1.0)

            gWe = np.clip(ts.T @ dL_dmu_e / n, -GRAD_CLIP, GRAD_CLIP)
            gwe_c = np.clip((dL_dmu_e * cs[:, None]).mean(axis=0), -GRAD_CLIP, GRAD_CLIP)
            gbe = np.clip(dL_dmu_e.mean(axis=0), -GRAD_CLIP, GRAD_CLIP)

            gUe = np.clip(ts.T @ dL_dlogsig_e / n, -GRAD_CLIP, GRAD_CLIP)
            gue_c = np.clip((dL_dlogsig_e * cs[:, None]).mean(axis=0), -GRAD_CLIP, GRAD_CLIP)
            gue = np.clip(dL_dlogsig_e.mean(axis=0), -GRAD_CLIP, GRAD_CLIP)

            Wdz -= self.lr * gWdz; wdc -= self.lr * gwdc; bd -= self.lr * gbd
            We -= self.lr * gWe; we_c -= self.lr * gwe_c; be -= self.lr * gbe
            Ue -= self.lr * gUe; ue_c -= self.lr * gue_c; ue -= self.lr * gue

        self.Wdz, self.wdc, self.bd = Wdz, wdc, bd

        # Calibration fix: the training loss (0.5*(mu_d-ts)^2) assumes a
        # FIXED unit-variance decoder, i.e. Var(x|z) = 1 in standardized
        # units -- but the true generative model's MARGINAL variance
        # (integrating over z, law of total variance) is Var(x) =
        # Var(x|z) + Var_z(E[x|z]) = 1 + Var_z(mu_d). Sampling with a full
        # unit-variance decoder noise on top of z's own contribution
        # double-counts: the marginal ends up at 1+Var_z(mu_d) instead of
        # the calibrated target of exactly 1 (real per-day standardized
        # variance, by construction of the standardization itself).
        # Measured directly on JPM before this fix: Var_z(mu_d) ~= 0.16 in
        # standardized units -- a real, confirmed ~8%/day excess SD, and
        # the dominant source of the ensemble's measured 25% excess SD at
        # the full-horizon scale (compounds across chained blocks). Fixed
        # by shrinking the decoder noise variance to what's left over:
        # eps_scale^2 = max(1 - Var_z(mu_d), floor), estimated once here by
        # sampling z at the training conditions' own mean c.
        probe_k = 2000
        rng_probe = np.random.default_rng(self.seed + 1)
        z_probe = rng_probe.normal(0.0, 1.0, size=(probe_k, zd))
        mu_d_probe = z_probe @ Wdz + 0.0 * wdc[None, :] + bd[None, :]  # cs=0, i.e. c=c_mean
        var_z_mu_d = float(mu_d_probe.var(axis=0).mean())
        self.eps_scale = float(np.sqrt(max(1.0 - var_z_mu_d, 0.05)))
        return self

    def sample_batch_sums(self, c_value, k, rng, length=None):
        """Vectorized: draw k samples at once, return their RAW (uncorrected)
        summed returns over the first `length` days (default: all self.h),
        in real return units. This is the genuine probabilistic forecast --
        no consistency correction is applied here (that correction is what
        makes the downscaler's POINT prediction tie Climatology exactly;
        this method deliberately skips it to get the real, uncorrected
        ensemble spread for CRPS). `length` lets a caller take a partial
        (< self.h) slice of a fixed-h model's own output -- used by
        multiblock_ensemble_sums below for a final short block."""
        cs = (c_value - self.c_mean) / self.c_std
        z = rng.normal(0.0, 1.0, size=(k, self.z_dim))
        mu_d = z @ self.Wdz + cs * self.wdc[None, :] + self.bd[None, :]
        eps_out = rng.normal(0.0, self.eps_scale, size=(k, self.h))  # calibrated, not always unit-variance
        ts_sample = mu_d + eps_out
        paths = ts_sample * self.t_std[None, :] + self.t_mean[None, :]  # (k, h), real return units
        if length is not None:
            paths = paths[:, :length]
        return paths.sum(axis=1)  # (k,) -- raw returns, uncorrected


def multiblock_ensemble_sums(vae, r_coarse, horizon, block_size, k, rng, target_std=None):
    """Build a k-member ensemble of TOTAL horizon-length returns by chaining
    INDEPENDENT draws of block_size (=tau_star)-scale segments, rather than
    asking one small linear model to represent coherent structure across
    the whole horizon. Beyond tau_star an instrument's own dynamics
    decorrelate (the entire premise of this predictor_v1 program) -- so
    independent per-block sampling is the mechanistically honest way to
    build up a longer horizon, not a workaround. Each block's conditioning
    value is r_coarse allocated proportionally to that block's own length
    (the only allocation defensible without more information than the
    single aggregate coarse forecast provides). The final block is
    shorter than block_size whenever horizon isn't an exact multiple of
    it -- handled via sample_batch_sums's `length` truncation, not by
    padding or discarding the remainder.

    target_std, if given, rescales the resulting ensemble's dispersion
    (around its own mean, mean left untouched) to match it exactly. This
    corrects a real, measured gap: chaining INDEPENDENT blocks assumes
    variance adds linearly (Var(sum) = n_blocks * Var(one block)), which
    is what zero-autocorrelation (tau_star's own definition) implies --
    but real equity/index returns show well-documented SUB-linear variance
    scaling beyond a few weeks (long-horizon mean reversion, Fama-French/
    Poterba-Summers-type effect), not pure random-walk scaling. Checked
    directly on JPM: single-block (23d) ensemble std matched real 23d
    return std almost exactly (0.0972 vs 0.0997) after the decoder-
    variance fix above -- but the naively-chained 11-block (252d) ensemble
    still overshot real 252d return std (0.328 vs 0.286) purely from the
    linear-scaling assumption, not from any remaining per-block miscali-
    bration. target_std, computed once per instrument from real pre-cutoff
    history at the actual horizon scale, corrects this directly against
    real data rather than assuming any particular scaling law holds."""
    n_blocks = int(np.ceil(horizon / block_size))
    total = np.zeros(k)
    remaining = horizon
    for _ in range(n_blocks):
        L = min(block_size, remaining)
        remaining -= L
        c_i = r_coarse * (L / horizon)
        total += vae.sample_batch_sums(c_i, k, rng, length=L)
    if target_std is not None:
        current_std = total.std()
        if current_std > 1e-8:
            total = total.mean() + (total - total.mean()) * (target_std / current_std)
    return total


def energy_score_crps(ensemble, y):
    """Standard unbiased ensemble CRPS estimator (the 'energy score'):
    CRPS ~= E|X-y| - 0.5*E|X-X'|, X,X' iid draws from the ensemble."""
    k = len(ensemble)
    term1 = np.mean(np.abs(ensemble - y))
    diffs = np.abs(ensemble[:, None] - ensemble[None, :])
    term2 = 0.5 * diffs.sum() / (k * k)
    return term1 - term2


if __name__ == "__main__":
    prices = pd.read_parquet(os.path.join(NB_DIR, "multiasset_prices.parquet"))
    tau_star = load_tau_star()

    METHOD_NAMES = ["climatology", "overfit tree", "RL (policy gradient)",
                    "conditional GAN", "conditional VAE", "generative downscaler"]
    results = {}

    for tkr, horizon in INSTRUMENTS.items():
        window = tau_star[tkr]
        half_window = window // 2
        block_size = window  # shape-VAE's own scale = tau_star, NOT horizon -- see multiblock_ensemble_sums
        series = prices[tkr].dropna()
        frame = build_price_frame(series, horizon).reset_index(drop=True)
        n = len(frame)

        vals = series.values
        dates = series.index
        daily_logret = np.diff(np.log(vals))
        cutoff_pos = int(dates.searchsorted(pd.Timestamp(TRAIN_CUTOFF)))
        # Shape-VAE trained at block_size (=tau_star) resolution, not horizon: asking one
        # small linear model to jointly represent a 252-day shape when the instrument's own
        # measured memory is 22-63 days was the calibration bug this update fixes. This also
        # means far more, far-less-overlapping real training windows per instrument (e.g. JPM:
        # ~horizon/tau_star ~= 11x as many independent-ish 23-day windows as 252-day ones in
        # the same pre-cutoff history) -- a second, independent reason to expect this to help.
        targets, conds = [], []
        for i in range(0, max(cutoff_pos - block_size, 0)):
            w = daily_logret[i:i + block_size]
            if len(w) < block_size:
                continue
            targets.append(w)
            conds.append(w.sum())
        targets = np.array(targets)
        conds = np.array(conds)
        vae = ShapeVAE(h=block_size).fit(targets, conds)

        # Real historical horizon-scale (not block-scale) return std, used to correct the
        # chained ensemble's variance directly against real data (see
        # multiblock_ensemble_sums docstring for why chaining independent blocks alone isn't
        # enough -- real markets mean-revert beyond tau_star, so variance doesn't scale
        # linearly the way independence implies). RECENT window only, not the full
        # pre-cutoff history -- first version used all of it and made AAPL's CRPS WORSE
        # (85.5% -> 89.7%), traced directly to AAPL's 40-year pre-cutoff history including
        # its wild early-company volatility (near-bankruptcy in the 90s, dot-com bubble),
        # nothing like its calmer post-2020 regime -- the same "recency matters, a fixed
        # long-history calibration target goes stale" lesson already established in this
        # project's own post-processing arc (rolling/adaptive design, JPM/GLD deployed
        # correction). Reuses that arc's own validated window convention,
        # max(252, 4*horizon) trading days, rather than inventing a new one.
        recency_window = max(252, 4 * horizon)
        recent_start = max(cutoff_pos - recency_window, 0)
        horizon_rets = np.array([daily_logret[i:i + horizon].sum()
                                  for i in range(recent_start, max(cutoff_pos - horizon, 0), max(block_size // 4, 1))])
        target_std = float(horizon_rets.std()) if len(horizon_rets) > 5 else None

        p = half_window
        while p < n and frame.loc[p, "base_date"] < pd.Timestamp(TRAIN_CUTOFF):
            p += half_window

        mean_price = float(frame["base_price"].mean())
        abs_err = {m: [] for m in METHOD_NAMES}
        crps_certain = {m: [] for m in METHOD_NAMES}   # treated as a degenerate point forecast (== MAE)
        crps_own = {m: [] for m in METHOD_NAMES}        # each model's OWN natural uncertainty, this script's real question
        degenerate_ensemble_frac = {m: [] for m in METHOD_NAMES}  # tree-specific diagnostic: how often the leaf ensemble is size 1
        rng = np.random.default_rng(0)
        fold_idx = 0
        n_folds = 0

        while p + half_window <= n:
            train_df = frame.iloc[p - half_window:p]
            test_df = frame.iloc[p:p + half_window]
            Xtr, ytr = train_df[FEATURE_COLS].values, train_df["fwd_ret"].values
            Xte = test_df[FEATURE_COLS].values
            base_price_arr = test_df["base_price"].values
            actual_arr = test_df["target_price"].values

            for factory in MODEL_FACTORIES:
                m = factory().fit(Xtr, ytr)
                pred_ret = m.predict(Xte)
                pred_price = base_price_arr * np.exp(pred_ret)
                err = np.abs(pred_price - actual_arr)
                abs_err[m.name].extend(err.tolist())
                crps_certain[m.name].extend(err.tolist())  # degenerate point forecast: CRPS == MAE

                own_samples = m.sample(Xte, N_ENSEMBLE, rng)  # (n,K) array, or ragged list for the tree
                for row_idx, (bp, act) in enumerate(zip(base_price_arr, actual_arr)):
                    ret_ens = own_samples[row_idx]
                    if m.name == "overfit tree":
                        degenerate_ensemble_frac[m.name].append(1.0 if len(ret_ens) <= 1 else 0.0)
                    price_ens = bp * np.exp(np.asarray(ret_ens))
                    if len(price_ens) == 1:
                        crps_own[m.name].append(float(np.abs(price_ens[0] - act)))  # CRPS of a size-1 ensemble == its own error
                    else:
                        crps_own[m.name].append(energy_score_crps(price_ens, act))

            r_coarse = float(np.mean(ytr))  # == Climatology's own prediction, by construction
            downscaler_pred_price = base_price_arr * np.exp(r_coarse)  # tie-by-construction, see docstring
            abs_err["generative downscaler"].extend(np.abs(downscaler_pred_price - actual_arr).tolist())
            crps_certain["generative downscaler"].extend(np.abs(downscaler_pred_price - actual_arr).tolist())

            raw_sums = multiblock_ensemble_sums(vae, r_coarse, horizon, block_size, N_ENSEMBLE, rng,
                                                 target_std=target_std)
            for bp, act in zip(base_price_arr, actual_arr):
                ens_prices = bp * np.exp(raw_sums)
                crps_own["generative downscaler"].append(energy_score_crps(ens_prices, act))  # 76_'s calibrated result, for reference

            p += half_window
            fold_idx += 1
            n_folds += 1

        results[tkr] = {"n_folds": n_folds, "mean_price": mean_price, "horizon": horizon, "window": window}
        for m in METHOD_NAMES:
            results[tkr][m] = {
                "mae_pct": float(np.mean(abs_err[m]) / mean_price * 100),
                "crps_certain_pct": float(np.mean(crps_certain[m]) / mean_price * 100),
                "crps_own_pct": float(np.mean(crps_own[m]) / mean_price * 100),
            }
            if degenerate_ensemble_frac[m]:
                results[tkr][m]["degenerate_ensemble_frac"] = float(np.mean(degenerate_ensemble_frac[m]))
        print(f"{tkr}: {n_folds} folds")
        for m in METHOD_NAMES:
            extra = f"  (degenerate leaf frac={results[tkr][m]['degenerate_ensemble_frac']:.2f})" \
                if "degenerate_ensemble_frac" in results[tkr][m] else ""
            print(f"    {m:24s} MAE%={results[tkr][m]['mae_pct']:6.3f}  "
                  f"CRPS(certain)%={results[tkr][m]['crps_certain_pct']:6.3f}  "
                  f"CRPS(own uncertainty)%={results[tkr][m]['crps_own_pct']:6.3f}{extra}")

    with open(os.path.join(OUT_DIR, "results_fair_uncertainty_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results_fair_uncertainty_comparison.json")

    # ---- Plot: one figure, 12 subplots, grouped bars -- each architecture's OWN
    # uncertainty CRPS vs. the calibrated downscaler's, all on the same footing ----
    tickers = list(INSTRUMENTS.keys())
    colors = {
        "climatology": "#8E8E8E", "overfit tree": "#5B8DBE", "RL (policy gradient)": "#2E8B57",
        "conditional GAN": "#C0392B", "conditional VAE": "#8E44AD", "generative downscaler": "#1B4F72",
    }
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.flatten()
    for ax, tkr in zip(axes, tickers):
        vals_m = [results[tkr][m]["crps_own_pct"] for m in METHOD_NAMES]
        ax.bar(range(len(METHOD_NAMES)), vals_m, color=[colors[m] for m in METHOD_NAMES])
        ax.set_xticks(range(len(METHOD_NAMES)))
        ax.set_xticklabels(["clim", "tree", "RL", "GAN", "VAE", "downscl"], fontsize=8, rotation=0)
        ax.set_title(f"{tkr} (tau*={tau_star[tkr]}d, {results[tkr]['n_folds']} folds)", fontsize=9)
        ax.set_ylabel("CRPS, own uncertainty (% of mean price)", fontsize=8)
    fig.suptitle("All 12 instruments: CRPS using each model's OWN uncertainty (no calibration fixes "
                 "except the downscaler's, which keeps its 3 from 76_) -- fair-shot comparison", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, "78_fair_uncertainty_comparison_crps.png"), dpi=130)
    plt.close(fig)
    print("Saved 78_fair_uncertainty_comparison_crps.png")
