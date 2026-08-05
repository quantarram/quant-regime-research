"""
Generative downscaler, per the user's original idea: aggregate (average)
a fine, daily return series up to a much coarser quantity -- here, the
existing Climatology model's own whole-horizon forward return, the most
extreme form of "upscaling" and already what every L2 model in this
research line effectively predicts -- fit THAT coarse quantity with a
plain L2 estimator (appropriate, since a summed/averaged quantity is far
closer to Gaussian via the CLT than the fine daily series it's built
from, exactly the reasoning behind real statistical/dynamical downscaling
in meteorology), then LEARN a stochastic map back down to daily
resolution that reproduces realistic day-by-day texture and extremes
conditional on that coarse value.

This deliberately does NOT use a deterministic (L2-trained) downscaler --
that would just regress to the conditional mean at the fine scale too,
the same blurry-output failure mode flagged before building this (the
well-known problem with L2-trained image super-resolution/downscaling
nets). Instead:

  Stage 1 -- Coarse forecaster: Climatology, mean of the immediately
  preceding half_window = floor(tau_star/2) days' forward returns.
  Unchanged from every other script in this line -- deliberately the
  plainest, most validated L2-consistent point estimate of the aggregate
  return (feedback-respect-climatology: a real model, not a strawman).

  Stage 2 -- Generative downscaler: a conditional VAE (linear
  encoder/decoder, reparameterization trick -- same family as
  65_architecture_bakeoff.py's ConditionalVAEForecaster, generalized here
  from a SCALAR to a full H-day daily-return VECTOR output), trained ONCE
  per instrument on real historical H-day windows. Condition c = that
  window's own realized aggregate log return (exactly the sum of its
  daily log returns, by construction, not an approximation). Target =
  the full vector of daily log returns within it. This is a STATISTICAL-
  SHAPE model -- what does a realistic day-by-day path look like, given
  it sums to some total return c? -- not a forecasting model: it never
  sees or predicts WHICH c will occur, only how to texture one once
  given one. Trained on all windows ending strictly before DEMO_CUTOFF,
  so every illustrative window shown below (drawn from after the cutoff)
  is honestly out-of-sample for the downscaler too, not just for stage 1.

At generation time, the decoder's own reconstruction only approximately
preserves "sum of sampled path == c" (it's an ELBO objective, not a hard
linear constraint) -- a disclosed consistency correction (shift every
sampled path by a constant so it sums EXACTLY to the conditioning value)
is applied before converting to price, so the generated path is always
consistent with the coarse forecast by construction.

For a handful of illustrative out-of-sample windows per instrument: stage
1's forecast becomes the conditioning value, K samples are drawn from the
trained decoder, and the resulting daily PRICE paths are plotted against
(a) the actual daily price path and (b) the flat/linear-interpolation path
a deterministic downscaler would produce (same endpoint, zero fine-scale
texture, by definition) -- the direct visual test of whether a generative
downscaler recovers realistic pattern and extremes that a deterministic
one structurally cannot.

No metrics, no significance tests -- purely visual, matching every other
script in this experimental line.

Run: python 75_generative_downscaler.py
Output: 75_generative_downscaler_{TICKER}.png (MSFT, EURUSDX, XLF)
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)  # benign Accelerate/BLAS FP-flag quirk, verified (see 70_)

NB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTABILITY_JSON = os.path.join(NB_DIR, "predictability_paper", "results_correlated_decorrelated.json")

INSTRUMENTS = {"MSFT": 189, "EURUSD=X": 189, "XLF": 189}
N_LAGS = 10
Z_DIM = 6
K_SAMPLES = 5
N_MEDIAN_ENSEMBLE = 200  # larger, separate draw used only to compute the "most likely" composite path
# BLOCK_SIZE is set per-instrument, below, to that instrument's own tau_star (the
# predictability limit) -- not a fixed round number. The composite's block size answers
# "how long can one scenario draw be trusted to represent a single coherent regime before
# it's fair to let a different draw take over" -- that's exactly what tau_star measures
# (the horizon over which the instrument's own dynamics stay self-correlated before
# decorrelating into noise). Switching scenarios more often than tau_star injects fresh
# randomness faster than the real process would justify; switching less often forces
# artificial coherence past the point the dynamics can actually sustain. An earlier
# version used a fixed BLOCK_SIZE=21 (~1 month) for every instrument -- disclosed at the
# time as "not derived from tau_star" -- replaced here with the principled, per-instrument
# choice once the connection was pointed out.
DEMO_CUTOFF = "2020-01-01"  # shape-VAE trained only on windows ending strictly before this
N_DEMO_WINDOWS = 5
GRAD_CLIP = 5.0


def load_tau_star():
    with open(PREDICTABILITY_JSON) as f:
        d = json.load(f)
    return {t: d[t]["2"]["top5_tradeable"][0][0] for t in INSTRUMENTS}


def build_price_frame(price_series, horizon, n_lags=N_LAGS):
    """Same as 65_/73_/74_ -- one row per base date, base_price/target_price
    plus fwd_ret (the aggregate log return over the whole horizon)."""
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


class ShapeVAE:
    """Conditional VAE mapping (z, c) -> a full H-day vector of daily log
    returns. Linear encoder/decoder, fixed-variance Gaussian decoder (so
    reconstruction reduces to MSE, the same standard VAE simplification
    used in 65_'s ConditionalVAEForecaster) + KL regularization to a unit
    Gaussian prior -- generalized here from scalar to h-dimensional output.
    The decoder only ever sees (z, c); it has no access to anything else
    at generation time, matching what's actually available when
    conditioning on a forecast rather than a known historical outcome."""

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

            mu_d = z @ Wdz + np.outer(cs, wdc) + bd  # (n, h)
            resid = mu_d - ts  # d(recon)/d(mu_d)

            gWdz = np.clip(z.T @ resid / n, -GRAD_CLIP, GRAD_CLIP)
            gwdc = np.clip((resid * cs[:, None]).mean(axis=0), -GRAD_CLIP, GRAD_CLIP)
            gbd = np.clip(resid.mean(axis=0), -GRAD_CLIP, GRAD_CLIP)

            dL_dz = resid @ Wdz.T  # (n, zd), reconstruction term only
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
        return self

    def sample(self, c_value, k=K_SAMPLES, seed=1):
        """Returns a list of k arrays, each (h,), in REAL daily-log-return
        units. c_value is a single scalar conditioning value (the coarse
        forecast). Consistency correction NOT applied here -- done by the
        caller, since it needs the exact conditioning value used.

        Generation draws z ~ prior AND decoder output noise ~ N(0,1) in
        standardized-target space -- the fixed decoder variance=1 that the
        training loss already assumes (0.5*(mu_d-x)^2 IS the negative log
        likelihood of a unit-variance Gaussian decoder), but which is only
        actually a distribution to sample from at generation time, not
        just a loss weighting. Returning mu_d alone (what the first version
        of this method did) discards that per-day noise term entirely --
        confirmed by direct comparison to real data before this fix: the
        per-day std across mu_d-only samples was ~14% of real historical
        daily-return std, and less than 20% of the empirical per-day shape
        diversity among real historical windows sharing a similar total
        return. Sampling the decoder's own assumed noise term restores it."""
        rng = np.random.default_rng(seed)
        cs = (c_value - self.c_mean) / self.c_std
        paths = []
        for _ in range(k):
            z = rng.normal(0.0, 1.0, size=(1, self.z_dim))
            mu_d = z @ self.Wdz + cs * self.wdc[None, :] + self.bd[None, :]
            eps_out = rng.normal(0.0, 1.0, size=(1, self.h))  # decoder's own fixed-variance noise
            ts_sample = mu_d + eps_out
            path = ts_sample[0] * self.t_std + self.t_mean
            paths.append(path)
        return paths


if __name__ == "__main__":
    prices = pd.read_parquet(os.path.join(NB_DIR, "multiasset_prices.parquet"))
    tau_star = load_tau_star()

    for tkr, horizon in INSTRUMENTS.items():
        window = tau_star[tkr]
        half_window = window // 2
        block_size = window  # composite's block size = this instrument's own tau_star
        series = prices[tkr].dropna()
        frame = build_price_frame(series, horizon)

        vals = series.values
        dates = series.index
        daily_logret = np.diff(np.log(vals))  # daily_logret[j] = log(vals[j+1]/vals[j])

        cutoff_pos = int(dates.searchsorted(pd.Timestamp(DEMO_CUTOFF)))
        targets, conds = [], []
        for i in range(0, max(cutoff_pos - horizon, 0)):
            w = daily_logret[i:i + horizon]
            if len(w) < horizon:
                continue
            targets.append(w)
            conds.append(w.sum())
        targets = np.array(targets)
        conds = np.array(conds)
        print(f"{tkr}: training shape-VAE on {len(targets)} windows ending before {DEMO_CUTOFF}")

        vae = ShapeVAE(h=horizon).fit(targets, conds)
        # sanity check: does the decoder's own (uncorrected) reconstruction
        # sum land near its conditioning value, on the TRAINING data itself?
        check_paths = vae.sample(float(conds[len(conds) // 2]), k=20, seed=0)
        check_sums = [p.sum() for p in check_paths]
        print(f"{tkr}: sanity check -- target c={conds[len(conds)//2]:.4f}, "
              f"20 uncorrected sample sums range [{min(check_sums):.4f}, {max(check_sums):.4f}], "
              f"mean {np.mean(check_sums):.4f}")

        demo_frame = frame[frame["base_date"] >= pd.Timestamp(DEMO_CUTOFF)].reset_index(drop=True)
        idxs = np.linspace(0, len(demo_frame) - 1, N_DEMO_WINDOWS).astype(int)

        fig, axes = plt.subplots(N_DEMO_WINDOWS, 1, figsize=(14, 3.2 * N_DEMO_WINDOWS), squeeze=False)
        axes = axes[:, 0]
        for ax, idx in zip(axes, idxs):
            row = demo_frame.iloc[idx]
            base_date, target_date = row["base_date"], row["target_date"]
            base_price = row["base_price"]

            pos_match = frame.index[frame["base_date"] == base_date]
            pos = pos_match[0]
            train_df = frame.iloc[max(0, pos - half_window):pos]
            r_coarse = float(train_df["fwd_ret"].mean())  # Climatology, stage 1

            actual_slice = series.loc[base_date:target_date]
            actual_dates = actual_slice.index
            actual_path = actual_slice.values
            h_actual = len(actual_dates) - 1  # number of return-steps actually available

            t = np.arange(len(actual_dates))
            flat_path = base_price * np.exp(r_coarse * t / max(h_actual, 1))

            samples = vae.sample(r_coarse, k=K_SAMPLES)
            for s_i, daily in enumerate(samples):
                daily = daily[:h_actual]
                daily = daily + (r_coarse - daily.sum()) / max(h_actual, 1)  # consistency correction
                price_path = base_price * np.exp(np.concatenate([[0.0], np.cumsum(daily)]))
                ax.plot(actual_dates, price_path, color="tab:purple", lw=1.0, alpha=0.5,
                        label="generative downscaler sample" if s_i == 0 else None, zorder=3)

            # "Most likely" scenario -- third and current design. First
            # attempt: pointwise median return across a large ensemble;
            # collapsed onto the flat/deterministic path (median of
            # symmetric noise washes texture out). Second attempt: a single
            # ensemble MEDOID (the one whole-path member closest to that
            # median reference) -- real texture, but forces ONE draw's
            # idiosyncratic noise to stand for the entire horizon, even
            # though a different member might be locally more representative
            # at a different point in time. Third (this version), per the
            # user's own suggestion: a PIECEWISE composite -- split the
            # horizon into BLOCK_SIZE-day blocks, and in each block
            # independently pick whichever ensemble member is closest to
            # that block's OWN local median reference (not the whole-path
            # one), then concatenate the chosen members' return SEGMENTS.
            # Built in return space, not by switching between pre-cumulated
            # price paths, so there is no level-matching discontinuity to
            # patch -- concatenating real-valued returns and cumulating once
            # at the end is automatically continuous regardless of which
            # member each segment came from. block_size = this instrument's
            # own tau_star (set above, per-instrument) -- the natural
            # coherence timescale for "how long can one scenario draw be
            # trusted to represent a single regime," not an arbitrary round
            # number (an earlier version used a fixed 21 days for every
            # instrument; replaced once the tau_star connection was pointed
            # out). Visible small kinks at block boundaries (where the
            # chosen member switches) are left in, not smoothed away -- an
            # honest signature of this being a stitched composite, not a
            # single continuous draw.
            #
            # Real bug caught before trusting this: comparing each block to
            # that block's own POINTWISE MEDIAN across only 200 noisy samples
            # let the composite drift systematically off-pace (an early dip
            # below the flat path, then a catch-up, in most windows) --
            # checked directly against real historical windows sharing a
            # similar total return (roughly linear-to-front-loaded pacing,
            # e.g. ~65% of the gain realized by the 55%-through mark here)
            # and against the decoder's own smooth mean-response shape
            # (mu_shape below, matching that same pacing almost exactly) --
            # neither showed any such dip, confirming it was small-sample
            # noise in the per-block median, not a real pattern. Fixed by
            # comparing each block to the decoder's own smooth, already-
            # computed mu_shape reference instead of a noisy resampled
            # local median -- ties every block's selection to the same
            # globally-consistent pacing, while still choosing a genuine,
            # textured ensemble member for that block, not a smoothed value.
            ensemble = vae.sample(r_coarse, k=N_MEDIAN_ENSEMBLE, seed=7)
            ens_daily = np.array([e[:h_actual] + (r_coarse - e[:h_actual].sum()) / max(h_actual, 1)
                                   for e in ensemble])  # (N_MEDIAN_ENSEMBLE, h_actual)
            cs = (r_coarse - vae.c_mean) / vae.c_std
            mu_shape = (cs * vae.wdc + vae.bd)[:h_actual]  # decoder's own smooth mean-response reference
            composite_daily = np.empty(h_actual)
            for blk_start in range(0, h_actual, block_size):
                blk_end = min(blk_start + block_size, h_actual)
                blk = ens_daily[:, blk_start:blk_end]
                local_ref = mu_shape[blk_start:blk_end]
                blk_distances = np.sum((blk - local_ref[None, :]) ** 2, axis=1)
                best_member = np.argmin(blk_distances)
                composite_daily[blk_start:blk_end] = blk[best_member]
            composite_daily = composite_daily + (r_coarse - composite_daily.sum()) / max(h_actual, 1)
            median_path = base_price * np.exp(np.concatenate([[0.0], np.cumsum(composite_daily)]))

            ax.plot(actual_dates, actual_path, color="black", lw=1.6, label="actual price", zorder=5)
            ax.plot(actual_dates, flat_path, color="#8E8E8E", lw=1.8, linestyle="--",
                     label="flat/deterministic implied path (same coarse forecast)", zorder=4)
            ax.plot(actual_dates, median_path, color="#1B4F72", lw=2.0,
                     label=f"most likely scenario (best-matching member per {block_size}d [=tau*] block, "
                           f"{N_MEDIAN_ENSEMBLE}-sample ensemble)", zorder=6)
            ax.set_title(
                f"{tkr}: {pd.Timestamp(base_date).date()} -> {pd.Timestamp(target_date).date()} "
                f"({horizon}d), coarse (climatology) forecast r={r_coarse:+.4f}",
                fontsize=9, loc="left")
            ax.legend(fontsize=7, loc="upper left")

        fig.suptitle(
            f"{tkr}: generative downscaler -- {K_SAMPLES} sampled daily paths + most-likely "
            f"(piecewise-composite, {block_size}d [=tau*] blocks) path per window, conditioned on "
            f"stage-1 climatology's coarse forecast, vs. actual and vs. a flat deterministic path "
            f"with the identical coarse forecast",
            fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        safe_tkr = tkr.replace("=", "")
        fig.savefig(os.path.join(OUT_DIR, f"75_generative_downscaler_{safe_tkr}.png"), dpi=140)
        plt.close(fig)

    print("Saved: 75_generative_downscaler_{" + ",".join(t.replace("=", "") for t in INSTRUMENTS) + "}.png")
