"""
Custom LightGBM objectives for going beyond MSE/L2 training, matching the
higher-moment feature philosophy used throughout predictor_v1 (features are
built from q=2/q=4 structure functions; training should not then collapse
back to a q=2-only loss).

L_q loss:  L(r) = |r|^q,  r = y_pred - y_true
  grad = q * sign(r) * |r|^(q-1)
  hess = q * (q-1) * |r|^(q-2)   (floored -- degenerates toward 0 near r=0
                                   for q>2, a known instability with
                                   higher-order losses in gradient boosting)

Tail weights: sample weights proportional to |y_true|, so LightGBM's native
weighted-loss machinery penalizes errors on large realized moves more than
errors on small ones, without needing a custom weighted objective.
"""
import numpy as np
import pandas as pd


def make_lq_objective(q, hess_floor=1e-3):
    if q <= 1:
        raise ValueError("q must be > 1 for a well-defined Hessian")

    def objective(y_true, y_pred):
        r = np.asarray(y_pred) - np.asarray(y_true)
        grad = q * np.sign(r) * np.abs(r) ** (q - 1)
        hess = q * (q - 1) * np.abs(r) ** (q - 2)
        hess = np.maximum(hess, hess_floor)
        return grad, hess

    objective.__name__ = f"lq_q{q}"
    return objective


def lq_loss_value(r, q):
    """L(r) = |r|^q -- for the finite-difference sanity check and for eval."""
    return np.abs(r) ** q


def tail_weights(y_true, floor=0.1):
    """Sample weights proportional to |y_true|, floored so near-zero-return
    rows still contribute a minimum amount rather than being ignored."""
    w = np.abs(np.asarray(y_true))
    w = w / (w.mean() + 1e-12)
    return np.maximum(w, floor)


def pinball_loss(y_true, y_pred, alpha):
    r = np.asarray(y_true) - np.asarray(y_pred)
    return np.mean(np.maximum(alpha * r, (alpha - 1) * r))


def crps_from_quantiles(y_true, quantile_preds, alphas):
    """Finite-quantile-grid approximation to CRPS: average pinball loss
    across the quantile grid. Standard approximation when a full continuous
    predictive distribution isn't directly available."""
    losses = [pinball_loss(y_true, quantile_preds[a], a) for a in alphas]
    return float(np.mean(losses))


def _component_grad_hess(y_pred, y_true, spec):
    """One term of a composite objective. r = y_pred - y_true throughout,
    matching make_lq_objective's convention."""
    r = np.asarray(y_pred) - np.asarray(y_true)
    kind = spec["type"]
    if kind == "l2":
        return r, np.ones_like(r)
    elif kind == "lq":
        q = spec["q"]
        hess_floor = spec.get("hess_floor", 1e-3)
        grad = q * np.sign(r) * np.abs(r) ** (q - 1)
        hess = np.maximum(q * (q - 1) * np.abs(r) ** (q - 2), hess_floor)
        return grad, hess
    elif kind == "pinball":
        # Derived directly from this module's own pinball_loss(y_true, y_pred, alpha)
        # convention (u = y_true - y_pred): dL/dy_pred = (1-alpha) for r=y_pred-y_true >= 0
        # (over-predicting), -alpha for r < 0 (under-predicting). Piecewise-linear, so the
        # true Hessian is 0 a.e. -- use a constant pseudo-Hessian (same scale as L2's) so a
        # blended objective's Newton step stays well-defined.
        alpha = spec["alpha"]
        grad = np.where(r >= 0, 1 - alpha, -alpha)
        hess = np.ones_like(r)
        return grad, hess
    elif kind == "tail_l2":
        # L2 with per-row weight proportional to |y_true| (the same intent as
        # tail_weights()'s sample_weight usage, folded into an additive loss term instead).
        w = tail_weights(np.asarray(y_true))
        return w * r, w * np.ones_like(r)
    else:
        raise ValueError(f"unknown composite loss component: {kind}")


def make_composite_objective(components):
    """components: list of {"type": "l2"|"lq"|"pinball"|"tail_l2", "weight": float, ...}.
    Gradients and Hessians are linear in the loss, so a weighted sum of components' (grad,
    hess) pairs is exactly the (grad, hess) of the weighted-sum loss -- no approximation."""
    total_weight = sum(c["weight"] for c in components)

    def objective(y_true, y_pred):
        grad_total = np.zeros(len(np.asarray(y_pred)), dtype=float)
        hess_total = np.zeros(len(np.asarray(y_pred)), dtype=float)
        for spec in components:
            g, h = _component_grad_hess(y_pred, y_true, spec)
            grad_total += spec["weight"] * g
            hess_total += spec["weight"] * h
        return grad_total / total_weight, hess_total / total_weight

    objective.__name__ = "composite_" + "_".join(c["type"] for c in components)
    return objective


def calibration_check(y_true, quantile_preds, alphas):
    """For a well-calibrated quantile alpha, P(y_true <= pred_alpha) should
    equal alpha. Returns {alpha: empirical_coverage}."""
    out = {}
    for a in alphas:
        pred = quantile_preds[a]
        out[a] = float(np.mean(np.asarray(y_true) <= pred))
    return out


# ── Tail-aware metrics (added after the user rejected median/pinball-loss's ──
# ── R2/dir_acc "win" in Phase A on the grounds that it misses the extremes) ──

def extreme_mask(y_true, lower_q=0.1, upper_q=0.9):
    y_true = np.asarray(y_true)
    lo, hi = np.quantile(y_true, [lower_q, upper_q])
    return (y_true <= lo) | (y_true >= hi)


def extreme_hit_rate(y_true, y_pred, lower_q=0.1, upper_q=0.9):
    """Rank-based: of the rows where the ACTUAL outcome was in this fold's own
    top/bottom decile, how often did the PREDICTION also land in its own
    top/bottom decile? Not a magnitude-matching metric -- a model can't game
    this just by being well-scaled, it has to actually rank extreme rows
    correctly relative to everything else in the fold."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_lo, true_hi = np.quantile(y_true, [lower_q, upper_q])
    pred_lo, pred_hi = np.quantile(y_pred, [lower_q, upper_q])
    true_hi_mask, true_lo_mask = y_true >= true_hi, y_true <= true_lo
    pred_hi_mask, pred_lo_mask = y_pred >= pred_hi, y_pred <= pred_lo
    n_extreme = int(true_hi_mask.sum() + true_lo_mask.sum())
    if n_extreme == 0:
        return np.nan
    hits = int(np.sum(true_hi_mask & pred_hi_mask) + np.sum(true_lo_mask & pred_lo_mask))
    return float(hits / n_extreme)


def tail_pinball_loss(y_true, y_pred, alpha, lower_q=0.1, upper_q=0.9):
    """Pinball loss at `alpha`, restricted to rows where the actual outcome
    was itself extreme (top/bottom decile) -- unlike Phase A's pooled pinball
    loss, this can't be made to look good by nailing the calm majority of
    rows while missing the tails."""
    mask = extreme_mask(y_true, lower_q, upper_q)
    if mask.sum() == 0:
        return np.nan
    y_true, y_pred = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
    return pinball_loss(y_true, y_pred, alpha)


def crps_from_quantiles_weighted(y_true, quantile_preds, alphas, tail_weight=3.0):
    """Threshold-weighted CRPS (Gneiting & Ranjan 2011): the same finite-grid
    pinball-average as crps_from_quantiles, but the outermost (tail) alphas
    are weighted more heavily than the central ones, so the aggregate score
    is itself tail-sensitive rather than dominated by how well the model fits
    the bulk of the distribution."""
    lo_a, hi_a = min(alphas), max(alphas)
    weights = {a: (tail_weight if a in (lo_a, hi_a) else 1.0) for a in alphas}
    losses = [pinball_loss(y_true, quantile_preds[a], a) * weights[a] for a in alphas]
    return float(sum(losses) / sum(weights.values()))


# ── Fractional Skill Score (Roberts & Lean 2008) adapted from spatial ──
# ── neighborhoods to rolling time windows. Requested as a replacement ──
# ── for R2/RMSE-style point accuracy: FSS asks whether the forecast   ──
# ── gets the RATE of threshold-exceedance right over a window, not    ──
# ── whether it nails the exact-day value -- the same "double penalty" ──
# ── problem RMSE has with displaced events, FSS was built to avoid.   ──

def quantile_exceedance_prob(quantile_preds, alphas, threshold, direction="above"):
    """Predicted probability that y exceeds `threshold`, estimated by linear
    interpolation of the predicted quantile function (piecewise-linear CDF
    through the (q_alpha, alpha) points). direction='above' -> P(y > threshold);
    'below' -> P(y < threshold). Vectorized over rows.

    quantile_preds: {alpha: array of predicted values at that quantile}, same
    row order across alphas. Quantiles are sorted by predicted VALUE (not by
    alpha) before interpolation, since independently-fit quantile regressions
    can cross; np.interp needs its x-array increasing. Flat-extrapolates
    beyond the outermost fitted quantile (e.g. threshold above q_0.9 predicts
    the same CDF as at q_0.9 itself) rather than guessing a tail shape.
    """
    alphas_sorted = sorted(alphas)
    values = np.stack([np.asarray(quantile_preds[a]) for a in alphas_sorted], axis=1)  # (n, n_alpha)
    order = np.argsort(values, axis=1)
    values_sorted = np.take_along_axis(values, order, axis=1)
    alpha_arr = np.array(alphas_sorted)
    alphas_sorted_per_row = alpha_arr[order]  # (n, n_alpha)

    cdf_at_threshold = np.array([
        np.interp(threshold, values_sorted[i], alphas_sorted_per_row[i])
        for i in range(values.shape[0])
    ])
    return (1.0 - cdf_at_threshold) if direction == "above" else cdf_at_threshold


def fractional_skill_score(obs_indicator, pred_fraction, window=63, min_periods=20):
    """Core FSS: obs_indicator is a 0/1 array (did y exceed the threshold at
    each date); pred_fraction is either a matching 0/1 array (binary/point
    forecast) or a [0,1] probability array (probabilistic forecast). Both are
    aggregated over a trailing rolling window (the time-series analogue of
    FSS's spatial neighborhood) into an observed fraction O_t and a forecast
    fraction F_t, then scored as:
        FSS = 1 - mean((F_t - O_t)^2) / (mean(F_t^2) + mean(O_t^2))
    Bounded [0, 1] (can dip slightly negative for anti-skillful forecasts);
    0 = no skill (equivalent to a climatological/constant forecast), 1 =
    perfect. Unlike R2, a forecast that gets the right EVENT RATE within the
    window but is off by a few days on exact timing still scores well here.
    """
    obs = pd.Series(np.asarray(obs_indicator, dtype=float))
    pred = pd.Series(np.asarray(pred_fraction, dtype=float))
    O = obs.rolling(window, min_periods=min_periods).mean()
    F = pred.rolling(window, min_periods=min_periods).mean()
    valid = O.notna() & F.notna()
    O, F = O[valid].values, F[valid].values
    if len(O) == 0:
        return np.nan
    numerator = np.mean((F - O) ** 2)
    denominator = np.mean(F ** 2) + np.mean(O ** 2)
    if denominator == 0:
        return np.nan
    return float(1.0 - numerator / denominator)


def fss_from_quantiles(y_true, quantile_preds, alphas, threshold, direction="above", window=63, min_periods=20):
    """FSS for a probabilistic (quantile-forecast) model: predicted fraction
    per date is the interpolated exceedance probability, not a hard 0/1."""
    y_true = np.asarray(y_true)
    obs_indicator = (y_true > threshold) if direction == "above" else (y_true < threshold)
    pred_prob = quantile_exceedance_prob(quantile_preds, alphas, threshold, direction)
    return fractional_skill_score(obs_indicator.astype(float), pred_prob, window, min_periods)


def fss_from_point_pred(y_true, y_pred, threshold_true, threshold_pred, direction="above", window=63, min_periods=20):
    """FSS for a point-forecast model (e.g. plain LightGBM regression, no
    quantiles): both obs and forecast collapse to 0/1 indicators. Thresholds
    for y_true and y_pred are passed separately since a point predictor's
    output scale/distribution need not match the target's (e.g. rank-based
    decile thresholds computed independently on each, as extreme_hit_rate
    already does elsewhere in this module)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    obs_indicator = (y_true > threshold_true) if direction == "above" else (y_true < threshold_true)
    pred_indicator = (y_pred > threshold_pred) if direction == "above" else (y_pred < threshold_pred)
    return fractional_skill_score(obs_indicator.astype(float), pred_indicator.astype(float), window, min_periods)


if __name__ == "__main__":
    # Sanity check: analytical grad/hess of L_q vs. central finite differences,
    # at a range of residuals including near-zero (where the Hessian floor
    # matters) -- confirms the implementation before it's trusted on real data.
    eps = 1e-4
    rs = np.array([-2.0, -0.5, -0.05, 0.0, 0.05, 0.5, 2.0])
    for q in [3, 4, 5, 6]:
        obj = make_lq_objective(q, hess_floor=0.0)  # no floor, testing raw math
        y_true = np.zeros_like(rs)
        grad, hess = obj(y_true, rs)  # r = y_pred - y_true = rs
        num_grad = (lq_loss_value(rs + eps, q) - lq_loss_value(rs - eps, q)) / (2 * eps)
        num_hess = (lq_loss_value(rs + eps, q) - 2 * lq_loss_value(rs, q) + lq_loss_value(rs - eps, q)) / (eps ** 2)
        grad_err = np.abs(grad - num_grad)
        # skip r=0 for hessian check (true hess=0 there for q>2, floor irrelevant to the math check)
        mask = rs != 0
        hess_err = np.abs(hess[mask] - num_hess[mask])
        print(f"q={q}: max |grad_analytic - grad_numeric| = {grad_err.max():.6f}, "
              f"max |hess_analytic - hess_numeric| = {hess_err.max():.6f}")
        assert grad_err.max() < 1e-2, f"gradient mismatch for q={q}"
        assert hess_err.max() < 1e-1, f"hessian mismatch for q={q}"
    print("All L_q gradient/Hessian checks passed.")

    # Sanity check: extreme_hit_rate on hand-constructed perfect vs. reversed
    # rankings (deterministic, exact expected values).
    y_true = np.arange(100.0)
    hit_perfect = extreme_hit_rate(y_true, y_true)
    hit_reversed = extreme_hit_rate(y_true, 99.0 - y_true)
    print(f"extreme_hit_rate: perfect ranking = {hit_perfect:.3f} (expect 1.0), "
          f"reversed ranking = {hit_reversed:.3f} (expect 0.0)")
    assert abs(hit_perfect - 1.0) < 1e-9
    assert abs(hit_reversed - 0.0) < 1e-9

    # Sanity check: tail_pinball_loss on a perfect prediction (must be exactly 0).
    y_true_tp = np.arange(1.0, 11.0)
    tp = tail_pinball_loss(y_true_tp, y_true_tp, alpha=0.9)
    print(f"tail_pinball_loss (perfect prediction) = {tp:.6f} (expect 0.0)")
    assert abs(tp) < 1e-9

    # Sanity check: crps_from_quantiles_weighted's arithmetic against an
    # independently-computed weighted average of the (already-tested) pinball_loss.
    alphas = [0.1, 0.5, 0.9]
    y_true_w = np.array([0.0])
    quantile_preds_w = {a: np.array([1.0]) for a in alphas}
    expected_pb = {a: pinball_loss(y_true_w, quantile_preds_w[a], a) for a in alphas}
    tw = 3.0
    expected_weighted = sum(expected_pb[a] * (tw if a in (0.1, 0.9) else 1.0) for a in alphas) / (tw + 1 + tw)
    got_weighted = crps_from_quantiles_weighted(y_true_w, quantile_preds_w, alphas, tail_weight=tw)
    print(f"crps_from_quantiles_weighted = {got_weighted:.6f} (expect {expected_weighted:.6f})")
    assert abs(got_weighted - expected_weighted) < 1e-9

    print("All tail-aware metric checks passed.")

    # Sanity check: a composite objective with one component (weight=1.0) must exactly
    # reproduce that component's own standalone grad/hess; a 50/50 blend must equal the
    # manually-averaged grad/hess of the two components computed independently.
    y_true_c = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred_c = np.array([-1.0, -0.1, 0.1, 1.0])

    solo = make_composite_objective([{"type": "l2", "weight": 1.0}])
    g_solo, h_solo = solo(y_true_c, y_pred_c)
    r_c = y_pred_c - y_true_c
    assert np.allclose(g_solo, r_c) and np.allclose(h_solo, 1.0), "solo l2 composite mismatch"

    blend = make_composite_objective([{"type": "l2", "weight": 0.5}, {"type": "lq", "q": 4, "weight": 0.5}])
    g_blend, h_blend = blend(y_true_c, y_pred_c)
    g_l2, h_l2 = _component_grad_hess(y_pred_c, y_true_c, {"type": "l2"})
    g_lq4, h_lq4 = _component_grad_hess(y_pred_c, y_true_c, {"type": "lq", "q": 4})
    expected_g = 0.5 * g_l2 + 0.5 * g_lq4
    expected_h = 0.5 * h_l2 + 0.5 * h_lq4
    assert np.allclose(g_blend, expected_g) and np.allclose(h_blend, expected_h), "l2+lq4 blend mismatch"

    print("All composite-objective checks passed.")

    # Sanity check: FSS on a perfect point forecast (obs and pred indicators
    # identical) must be exactly 1.0; on a forecast that's exactly inverted
    # (predicts exceedance only when it doesn't happen) must be close to 0.
    rng = np.random.default_rng(0)
    y_true_fss = rng.normal(size=2000).cumsum()  # autocorrelated, like a real return series
    y_perfect = y_true_fss.copy()
    thr = np.quantile(y_true_fss, 0.8)
    fss_perfect = fss_from_point_pred(y_true_fss, y_perfect, thr, thr, direction="above", window=63)
    print(f"FSS (perfect point forecast) = {fss_perfect:.4f} (expect 1.0)")
    assert abs(fss_perfect - 1.0) < 1e-9

    y_inverted = -y_true_fss  # deliberately anti-correlated
    fss_inverted = fss_from_point_pred(y_true_fss, y_inverted, thr, np.quantile(y_inverted, 0.8), direction="above", window=63)
    print(f"FSS (inverted point forecast) = {fss_inverted:.4f} (expect << 1.0, near/below 0)")
    assert fss_inverted < 0.3

    # Sanity check: quantile_exceedance_prob recovers ~alpha at the fitted
    # quantile points themselves, on a simple synthetic quantile function.
    alphas_test = [0.1, 0.5, 0.9]
    qp_test = {0.1: np.array([1.0, 1.0]), 0.5: np.array([5.0, 5.0]), 0.9: np.array([9.0, 9.0])}
    p_above_5 = quantile_exceedance_prob(qp_test, alphas_test, threshold=5.0, direction="above")
    print(f"quantile_exceedance_prob(threshold=median) = {p_above_5} (expect ~[0.5, 0.5])")
    assert np.allclose(p_above_5, 0.5, atol=1e-9)

    # FSS from quantile forecasts: with a genuinely regime-clustered target
    # (autocorrelated latent driver z_t, like real vol regimes) a forecast
    # conditioned on the true driver should track the WINDOWED exceedance
    # rate much better than a flat/climatological forecast -- this is
    # exactly the situation FSS is meant to reward (getting clustered event
    # RATES right), unlike IID data where no windowed metric has anything
    # to grab onto.
    n = 3000
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = 0.97 * z[t - 1] + rng.normal(scale=0.3)  # persistent latent regime driver
    y_cond = z + rng.normal(scale=0.5, size=n)
    noise_q = {a: np.quantile(rng.normal(scale=0.5, size=200000), a) for a in alphas_test}
    good_qp = {a: z + noise_q[a] for a in alphas_test}
    flat_qp = {a: np.full(n, np.quantile(y_cond, a)) for a in alphas_test}
    thr2 = np.quantile(y_cond, 0.85)
    fss_good = fss_from_quantiles(y_cond, good_qp, alphas_test, thr2, direction="above", window=63)
    fss_flat = fss_from_quantiles(y_cond, flat_qp, alphas_test, thr2, direction="above", window=63)
    print(f"FSS good conditional forecast = {fss_good:.4f}, FSS flat/climatological forecast = {fss_flat:.4f}")
    assert fss_good > fss_flat

    print("All FSS checks passed.")
