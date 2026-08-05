"""
Order-Escalating Residual Cascade.

Not a single model with a blended/composite loss (that's what
make_composite_objective + Phase A's ablation already cover). This is a
sequence of SEPARATELY trained models, each fit only on what the previous
stage left behind, each using a pure (unrobustified, unclamped) L_q
objective with q strictly increasing stage over stage:

    Stage 1: model_1 fit on y with q=2 (plain L2, LightGBM) -> climatology
    Stage 2: model_2 fit on r1 = y - model_1(x) with q=4     -> next layer of extremes
    Stage 3: model_3 fit on r2 = r1 - model_2(x) with q=6    -> more extreme still
    Stage 4: model_4 fit on r3 = r2 - model_3(x) with q=8    -> diminishing-returns check

    final prediction = model_1(x) + model_2(x) + model_3(x) + model_4(x)

Why raising q at each stage actually targets "more extreme" and isn't just a
label: the minimizer of E[|r - c|^q] moves continuously from the conditional
mean (q=2) toward the conditional midrange as q grows, which for a
heavy-tailed one-sided residual sits much closer to the tail than the mean
does. Each stage is mathematically pulled toward whatever large-magnitude
structure is left in the residual, not toward re-explaining the part earlier
stages already got right.

Model family, v1 vs. v2: v1 used LightGBM for every stage and the cascade
made OOS R2 and extreme-hit-rate WORSE at every horizon. Diagnosis: a tree
can "solve" an unclamped high-q loss for free by carving out a leaf around a
single outlier row and predicting it almost exactly -- memorization of that
fold's specific extreme dates, not a learned feature-based extremity
mechanism. So stages 2+ use a small MLP (torch) instead: a smooth function
can't isolate individual rows the way a leaf can. Stage 1 (climatology,
q=2) stays on LightGBM, matching the existing baseline exactly.

Regularization, v2 vs. v3: v2 added weight_decay and early-stopping on a
held-out time-slice to the MLP stages. That measurably stabilized things,
but it is ALSO outlier suppression -- stopping training before the model
converges on the loss is a way of not letting the largest residuals
dominate the fit, just dressed up as "generalization control" instead of
"robustification." Rejected on those grounds. v3 removed weight_decay and
the validation split entirely and trained each stage to full convergence of
the training loss -- which blew up to R2 values like -1e14 to -1e23 and
eventually a literal float64 infinity, because the network (two hidden
layers, ~3,777 parameters) had MORE parameters than rows in the smallest
training folds (~980), so it could and did interpolate individual outlier
rows exactly regardless of how it was trained.

Capacity, v3 vs. v4 (this version): the loss stays exactly as in v3 -- no
weight_decay, no validation split, trained to full convergence, largest
residuals dominate for as long as gradient descent takes it. What changed
is SmallMLP itself: one hidden layer, width 8, ~209 parameters -- an order
of magnitude below even the smallest fold's row count, so the network is
structurally incapable of memorizing individual rows no matter how long or
how aggressively it's trained on them. This bounds capacity, not the loss:
the largest residuals still dominate every gradient step, they just can't
be perfectly interpolated by a function this constrained.

Result, and the final decision on it: this reduced the blowups (1d pooled
R2 went from ~-1e23 in v3 down to the hundreds/low-thousands in v4) but did
NOT eliminate them -- several folds, worst on 2008 across every horizon
(21d/2008 R2 = -1.7e29), still explode. Root cause: bounding parameter
COUNT doesn't bound weight MAGNITUDE, and a tiny network with a few
arbitrarily large weights still extrapolates without limit on any OOS point
that differs from training data -- routine across a regime shift like 2008.
The direct fix would be a hard weight-norm cap (a projection after each
step, bounding the hypothesis space itself, unlike weight_decay which fights
the loss during optimization) -- offered as an option and explicitly
declined: the decision was to accept the blowups on folds like 2008 as the
honest, literal consequence of "unclamped loss, bounded parameter count,
nothing else," rather than add any further constraint. The v4 numbers in
results_cascade_loss.json / the three PNG plots are that accepted result,
not a bug to be chased further.

Training window, v4 vs. v5 (this version): every prior version used an
EXPANDING window (all data since panel inception, 1964, up to the test
year) -- silently ignoring this project's own established predictability
limit tau_star and the validated 0.5*tau_star rolling-window default already
in use elsewhere in this repo (66_window_sweep_bakeoff.py). Per-instrument
tau_star is ~22-63 trading days here, so every v1-v4 run trained on
100-1000x more history than that per instrument, most of it from regimes
with no claim to relevance -- plausibly the single largest contributor to
folds like 2008 breaking so badly, ahead of anything about loss functions
or model capacity. v5 replaces the expanding window with a rolling,
per-ticker window of 0.5*tau_star trading days ending immediately before
each test date, loaded from predictability_paper/results_correlated_decorrelated.json.
Note this shrinks typical pooled per-fold training sets from thousands of
rows to roughly the 200-500 range (15 tickers x ~11-32 days each), which
changes the capacity-vs-data-size picture the v4 MLP sizing (209 params)
was reasoned about -- flagged here, not silently re-tuned.

Pooling, v5 vs v6a: v5 still pooled all 15 instruments into one shared
model, a convention silently carried over from 02_train_predict_daily.py
that was never asked for here and directly conflicts with tau_star being
an INSTRUMENT-level limit (66_window_sweep_bakeoff.py trains one model per
ticker). v6a switched to one model per instrument, walked forward with
window = step = 0.5*tau_star (mirroring 66_'s own train/test cadence
exactly), sampling a few random windows per instrument rather than an
exhaustive multi-decade walk (~30k folds, ~11.5hrs) for no methodological
reason once pooling was removed.

Numerics, v6a vs v6b (this version): per-instrument windows are ~11-32
rows, and the v4/v5 capacity-vs-data reasoning (209 params, an order of
magnitude below fold size) no longer holds at this scale -- v6a still
diverged on 10-25% of folds. But this project's OWN already-validated
architectures (RLPolicyForecaster, ConditionalGANForecaster in
66_window_sweep_bakeoff.py) train on these exact same tau_star-scaled
windows without diverging, using two numerics devices neither of which
touches the loss's sensitivity to outliers: (1) target rescaling by that
WINDOW's own std (y_scale = std(y), adaptive, not a fixed global constant
-- v6a's RESID_SCALE=100 was fixed and wrong for this reason), and (2)
elementwise gradient clipping to +-10.0, explicitly documented there as "a
second line of defense" against blow-up, not a penalty on the objective.
v6b adopts both, replacing the fixed RESID_SCALE with an adaptive
per-window scale and adding torch's clip_grad_value_ at the same +-10.0
bound. The largest residuals still produce the largest gradients, right up
to that bound -- this stops one catastrophic step from destroying the fit,
which is the same role it already plays in this project's own validated
architectures, not a new form of the outlier-suppression that was rejected.

Residuals are rescaled via a signed-log-magnitude transform before the Lq
loss is computed -- this is retained because it's pure numerics (q=8
gradients on O(1e-2)-magnitude returns would otherwise underflow float32 to
exactly zero before any outlier gets a chance to dominate anything), not a
damping of the outlier-domination effect: it's a strictly monotonic
transform, so the largest residual is still the largest after transforming,
and still drives the biggest loss/gradient.

Escalating order, v6c vs v7: every version through v6c used LightGBM for the
climatology stage (carried over from the unrelated pooled
02_train_predict_daily.py pipeline, never asked for here) and escalating
Lq loss (q=4/6/8) for the residual stages. v7 dropped both -- but v7 STILL
treated stage 1 as a learned tanh-MLP fit, which is wrong: climatology is
architecture-independent by definition (65_/66_/70_'s own Climatology
class: self.mu = mean(y), predict = full(len(X), self.mu), no features, no
model). It cannot change with model choice because no model is involved.

Climatology, v7 vs v7b (this version): stage 1 is now TRUE climatology --
mean(y_train), a constant, computed once per fold, no features, no
LightGBM, no MLP. Stages 2+ are the AI models: the same tanh MLP from
70_deep_architecture_bakeoff.py, trained the same way (adaptive y_scale,
elementwise grad clip, fixed 400-epoch budget, plain L2), each fit on the
residual left by the stage before it. This is, in the user's words, "a
cascade of residual AI models but all L2 based" -- a residual-boosting
cascade of already-validated nonlinear AI architecture sitting on top of
true climatology, not replacing it. CASCADE_QS values are retained only as
distinct dict/column keys, not loss orders.

Same walk-forward/purge folds, same feature panel as 02_train_predict_daily.py
/ 03_ablation_loss.py.

Run: python 71_cascade_loss_escalation.py
Requires: features_daily_panel.parquet, ../multiasset_prices.parquet
Output: results_cascade_loss.json, 71_cascade_loss_escalation_<horizon>d.png
"""
import os
# Must be set before lightgbm/torch import: each bundles its own OpenMP
# runtime, and repeated alternating fits between them (this script's whole
# design) deadlocks after a handful of iterations otherwise -- confirmed by
# a standalone stress test that hung indefinitely without this and finished
# 40 alternating fits in 3.9s with it.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

import torch
torch.set_num_threads(1)
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from loss_functions import extreme_hit_rate

torch.manual_seed(0)


class SmallMLP(nn.Module):
    """Single tiny hidden layer, width 8 -> 26*8+1 = 209 parameters (small
    relative to input dim, matching 70_deep_architecture_bakeoff.py's own
    "same order of magnitude as the input features, not a gratuitous
    capacity increase" convention -- their HIDDEN=6 against 10 features).

    Activation is TANH, not ReLU -- this is the change from v6b to v6c.
    Every prior version used ReLU, whose output is unbounded: even with
    weight-magnitude controls (gradient clipping, fixed epoch budget), a
    ReLU net can still extrapolate to an unbounded value on any OOS row
    that falls outside the ~11-32-row training window's feature range,
    which is exactly what kept diverging. 70_deep_architecture_bakeoff.py's
    own validated nonlinear architectures (DeepRLPolicyForecaster etc.) use
    tanh specifically -- output bounded to [-1,1] regardless of weight
    magnitude or input range, so the network structurally cannot blow up on
    an out-of-distribution test row, independent of any training-loop
    safeguard. This is an architectural property, not a numerics patch."""
    def __init__(self, n_in, hidden=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _to_log_magnitude(r, scale):
    """sign-preserving log-magnitude transform. scale is ADAPTIVE -- the
    training window's own std of r (matching RLPolicyForecaster/
    ConditionalGANForecaster's y_scale = std(y) in 66_window_sweep_bakeoff.py,
    not a fixed global constant), so an 11-row window and a 3700-row window
    each get rescaled to their OWN O(1) units before the Lq loss ever sees
    them. Ordering is still strictly preserved (bigger |r| -> bigger
    |transform(r)|), so the loss still preferentially matches the largest
    residuals -- this bounds representation, it doesn't change who dominates."""
    return np.sign(r) * np.log1p(np.abs(r) * scale)


def _from_log_magnitude(z, scale):
    return np.sign(z) * (np.expm1(np.abs(z)) / scale)


GRAD_CLIP_VALUE = 10.0  # matches the +-10.0 elementwise clip already used and
                         # validated in 66_window_sweep_bakeoff.py's RL/GAN
                         # forecasters -- a numerical safety valve on the SIZE
                         # of a single update step, not a penalty on the loss
                         # itself. The largest residuals still produce the
                         # largest gradients right up to this bound; this only
                         # stops one catastrophic step from destroying the fit,
                         # exactly the "second line of defense" role it plays
                         # in those already-validated architectures.


TRAIN_EPOCHS = 400  # fixed, matching RLPolicyForecaster/ConditionalGANForecaster's
                     # epochs=400 exactly. Not a validation-based early stop --
                     # this is a compute-budget cap, the same category as
                     # GRAD_CLIP_VALUE: per-step clipping bounds one step, but
                     # doesn't bound how far several thousand small-but-clipped
                     # steps (the old max_epochs=2000-with-convergence-check
                     # regime) can still walk a ReLU net's weights over an
                     # open-ended schedule, which is what kept diverging even
                     # with clipping alone. A fixed epoch count is the second
                     # half of the same already-validated stabilization pair.


def train_residual_stage(Xtr, r_train, q):
    """Fit one residual stage with a small MLP on a pure |pred - target|^q
    loss, target = signed-log-magnitude of r_train. NO weight decay, NO
    held-out validation split, NO early-stopping-for-generalization -- those
    were all reintroducing outlier-suppression under a different name, which
    is exactly what was rejected. Trains for TRAIN_EPOCHS (fixed, not
    validation-gated) so the largest residuals dominate the fit for exactly
    the same budget this project's own validated architectures use --
    including memorizing individual extreme rows, if that's what an
    unclamped fit to genuine outliers within that budget actually does.

    Returns (predict_fn, train_pred, final_train_loss) where predict_fn(X)
    and train_pred are both in ORIGINAL (raw return) residual units -- the
    log transform is purely internal numerics, inverted before anything
    leaves this function.
    """
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    scale = max(float(np.std(r_train)), 1e-8)  # adaptive, this window's own std
    r_log = _to_log_magnitude(r_train, scale)

    X_all_t = torch.tensor(Xtr_s, dtype=torch.float32)
    y_all_t = torch.tensor(r_log, dtype=torch.float32)

    model = SmallMLP(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)  # no weight_decay

    cur = float("nan")
    for epoch in range(TRAIN_EPOCHS):
        model.train()
        opt.zero_grad()
        pred = model(X_all_t)
        loss = torch.mean(torch.abs(pred - y_all_t) ** q)  # pure Lq
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), GRAD_CLIP_VALUE)  # numerics-only, see GRAD_CLIP_VALUE docstring
        opt.step()
        cur = loss.item()

    model.eval()
    final_train_loss = cur

    def predict_fn(X):
        Xs = scaler.transform(X)
        with torch.no_grad():
            out = model(torch.tensor(Xs, dtype=torch.float32)).numpy()
        return _from_log_magnitude(out, scale)

    # also return the fitted-training-set prediction (needed for the next
    # stage's residual), inverted back to raw residual units
    with torch.no_grad():
        train_pred = _from_log_magnitude(model(X_all_t).numpy(), scale)

    return predict_fn, train_pred, final_train_loss

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

HORIZONS = [1, 5, 21]
INITIAL_TRAIN_YEARS = 10
STEP_YEARS = 1

# v7: escalating statistical order abandoned entirely -- every stage below
# uses plain L2, matching the direct-extension-of-the-architecture-bakeoff
# spec exactly ("a cascade of residual AI models but all L2 based"). The
# values here are kept only as distinct stage LABELS (dict/column keys
# elsewhere in the script expect distinct ints) -- they are no longer loss
# orders; every stage's actual training loss is q=2.
CASCADE_QS = [2, 4, 6, 8]

print("=" * 60)
print("  ORDER-ESCALATING RESIDUAL CASCADE")
print("=" * 60)

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
panel = pd.read_parquet(os.path.join(OUT_DIR, "features_daily_panel.parquet"))
panel["date"] = pd.to_datetime(panel["date"])
TICKERS = sorted(panel["ticker"].unique())
date_pos = {d: i for i, d in enumerate(prices.index)}

# ── Training window: rolling 0.5*tau_star per instrument, NOT the expanding
# since-1964 window this script used through every previous run. tau_star
# is this project's own established predictability limit (predictability_paper/
# results_correlated_decorrelated.json), and 0.5*tau_star is the "half-window
# design from Papers 13 and 65" -- the validated, already-in-use default
# elsewhere in this repo (66_window_sweep_bakeoff.py). Every prior run of
# THIS script trained on 100-1000x more history than that, per instrument --
# a direct contradiction of the project's own core finding, not a stylistic
# choice, and the most likely dominant explanation (ahead of anything about
# loss functions or model capacity) for why folds like 2008 -- a regime the
# expanding window would have trained across, not just tested into -- broke
# so badly.
with open(os.path.join(REPO_DIR, "predictability_paper", "results_correlated_decorrelated.json")) as f:
    _tau_raw = json.load(f)
TAU_STAR = {}
for t in TICKERS:
    try:
        TAU_STAR[t] = _tau_raw[t]["2"]["top5_tradeable"][0][0]
    except (KeyError, IndexError):
        TAU_STAR[t] = None
_median_tau = int(np.median([v for v in TAU_STAR.values() if v is not None]))
for t in TICKERS:
    if TAU_STAR[t] is None:
        print(f"  tau_star missing for {t}, using panel median ({_median_tau}d) as fallback")
        TAU_STAR[t] = _median_tau
WINDOW_DAYS = {t: max(round(0.5 * TAU_STAR[t]), 4) for t in TICKERS}
print(f"Rolling training windows (0.5*tau_star, trading days): {WINDOW_DAYS}")

ret_frames = []
for t in TICKERS:
    s = prices[t]
    df = pd.DataFrame({"date": s.index, "ticker": t, "price_now": s.values})
    for h in HORIZONS:
        fwd_price = s.shift(-h)
        df[f"fwd_{h}"] = np.log(fwd_price.values / s.values)
    ret_frames.append(df)
ret_df = pd.concat(ret_frames, ignore_index=True)
panel = panel.merge(ret_df, on=["ticker", "date"], how="left")

z_cols = [c for c in panel.columns if c.endswith("_z")]
ctx_cols = [c for c in panel.columns if c.startswith("ctx_")]
for c in ctx_cols:
    panel[c] = np.sign(panel[c]) * np.log1p(np.abs(panel[c]))
FEATURE_COLS = z_cols + ctx_cols + ["self_ref_score"]
print(f"Feature columns ({len(FEATURE_COLS)}), unchanged from v1/Phase A control")

panel["year"] = panel["date"].dt.year
min_year, max_year = panel["year"].min(), panel["year"].max()
first_test_year = min_year + INITIAL_TRAIN_YEARS
print(f"Panel spans {min_year}-{max_year}; first OOS test year: {first_test_year}")


def sample_moments(x, q_list):
    """Raw central moments E[|x - mean(x)|^q] for a few q -- the S_q
    diagnostic from the structure-function discussion, at native resolution
    (no scale/lambda dimension, per the simplification the user settled on).
    Not robustified: this is meant to show exactly how much the tail
    dominates each stage's remaining residual, not hide it."""
    x = np.asarray(x)
    mu = x.mean()
    return {q: float(np.mean(np.abs(x - mu) ** q)) for q in q_list}


def excess_kurtosis(x):
    x = np.asarray(x)
    mu, sd = x.mean(), x.std()
    if sd == 0:
        return np.nan
    return float(np.mean(((x - mu) / sd) ** 4) - 3.0)


def skewness(x):
    x = np.asarray(x)
    mu, sd = x.mean(), x.std()
    if sd == 0:
        return np.nan
    return float(np.mean(((x - mu) / sd) ** 3))


MIN_TRAIN_ROWS = 8  # per-instrument windows are ~11-63 rows to begin with; a
                     # 200-row pooled-era floor is meaningless here
N_SAMPLES_PER_TICKER = 8  # a few random test windows per instrument, not an
                           # exhaustive walk through the whole 1964-2026
                           # history -- exhaustive was ~30k folds / ~11.5hrs
                           # for no methodological reason; a handful of
                           # random draws, each still trained on the correct
                           # 0.5*tau_star window, is what was actually asked for
SAMPLE_SEED = 0

results = {"horizons": {}, "cascade_qs": CASCADE_QS, "window_days": WINDOW_DAYS, "tau_star": TAU_STAR,
           "n_samples_per_ticker": N_SAMPLES_PER_TICKER}

for h in HORIZONS:
    print(f"\n--- Horizon {h}d ---")
    label_col = f"fwd_{h}"
    d_all = panel.dropna(subset=FEATURE_COLS + [label_col]).copy()

    fold_records = []
    oos_true = []
    oos_rows = []
    skipped_tickers = []

    # ── Per-instrument, a FEW RANDOM test windows -- not pooled across
    # instruments, not an exhaustive walk through 1964-2026. Each ticker
    # gets its own model; for each of N_SAMPLES_PER_TICKER randomly chosen
    # test windows, trained on a rolling window of exactly window=
    # 0.5*tau_star of ITS OWN preceding rows, tested on the next
    # `window`-sized chunk. Window length/recency still matches
    # 66_window_sweep_bakeoff.py's validated design -- only the WALK is
    # sampled rather than exhaustive, purely for tractability.
    rng = np.random.default_rng(SAMPLE_SEED)
    for t in TICKERS:
        window = WINDOW_DAYS[t]
        d_t = d_all[d_all["ticker"] == t].sort_values("date").reset_index(drop=True)
        n_t = len(d_t)
        p_start = window + h + 1
        if p_start + window > n_t:
            skipped_tickers.append((t, n_t))
            continue

        valid_p = np.arange(p_start, n_t - window + 1)
        sample_p = rng.choice(valid_p, size=min(N_SAMPLES_PER_TICKER, len(valid_p)), replace=False)
        sample_p.sort()

        n_folds_ticker = 0
        for p in sample_p:
            idx = np.arange(max(0, p - window), p)
            idx = idx[idx + h < p]  # purge: label window must not reach into the test chunk
            test_idx = np.arange(p, p + window)

            if len(idx) < MIN_TRAIN_ROWS:
                continue

            Xtr, ytr = d_t.loc[idx, FEATURE_COLS], d_t.loc[idx, label_col].values
            Xte, yte = d_t.loc[test_idx, FEATURE_COLS], d_t.loc[test_idx, label_col].values

            # ── Cascade: direct extension of the architecture bakeoff
            # (70_deep_architecture_bakeoff.py). Stage 1 is TRUE climatology
            # -- the training window's mean, full stop, no features, no
            # model, matching Climatology.fit in 65_/66_/70_ exactly
            # (self.mu = mean(y); predict = full(len(X), self.mu)).
            # Climatology cannot depend on architecture because it never
            # involves one -- treating it as "the first learned model"
            # (LightGBM, then a tanh MLP) was wrong in every prior version.
            # Stages 2+ are the AI models: the same tanh-MLP family from
            # that bakeoff, trained the same way (plain L2, adaptive
            # per-window y_scale, elementwise grad clip, fixed epoch
            # budget), each fit on the residual left by the stage before
            # it -- a cascade of residual AI models, all L2 based, sitting
            # on top of true climatology, not replacing it. Stage labels in
            # CASCADE_QS are kept only as distinct dict/column keys; the
            # actual loss order passed to every AI stage is always 2.
            climatology_pred = float(np.mean(ytr))
            r_train = ytr - climatology_pred
            pred_test_cum = np.full_like(yte, climatology_pred)
            stage_test_preds = {CASCADE_QS[0]: pred_test_cum.copy()}
            stage_increment_test = {CASCADE_QS[0]: pred_test_cum.copy()}
            stage_residual_moments = {CASCADE_QS[0]: sample_moments(r_train, [2, 4, 6])}

            for q in CASCADE_QS[1:]:
                predict_fn, stage_pred_train, final_train_loss = train_residual_stage(Xtr.values, r_train, 2)
                stage_pred_test = predict_fn(Xte.values)

                pred_test_cum = pred_test_cum + stage_pred_test
                r_train = r_train - stage_pred_train

                stage_test_preds[q] = pred_test_cum.copy()
                stage_increment_test[q] = stage_pred_test.copy()
                stage_residual_moments[q] = sample_moments(r_train, [2, 4, 6])

            for q in CASCADE_QS:
                pred = stage_test_preds[q]
                # The accepted design (unclamped loss, bounded params only)
                # produces literal non-finite predictions on some folds --
                # that outcome stays as-is, unhidden. What's fixed here is
                # only that the SCRIPT must not die on the first one: a
                # diverged fold is recorded as such (r2=-inf, not silently
                # dropped or clipped) so every instrument still gets a full
                # report rather than the run dying on whichever comes first.
                diverged = bool(np.any(~np.isfinite(pred)))
                if diverged:
                    fold_records.append(dict(
                        ticker=t, variant=f"cascade_upto_q{q}", test_start_date=str(d_t.loc[test_idx[0], "date"].date()),
                        n=len(yte), n_train=len(idx), diverged=True,
                        r2=float("-inf"), rmse=float("inf"), dir_acc=np.nan, extreme_hit_rate=np.nan,
                        pred_kurtosis=np.nan, pred_skew=np.nan,
                        resid_S2=stage_residual_moments[q][2], resid_S4=stage_residual_moments[q][4],
                        resid_S6=stage_residual_moments[q][6],
                    ))
                else:
                    fold_records.append(dict(
                        ticker=t, variant=f"cascade_upto_q{q}", test_start_date=str(d_t.loc[test_idx[0], "date"].date()),
                        n=len(yte), n_train=len(idx), diverged=False,
                        r2=r2_score(yte, pred), rmse=mean_squared_error(yte, pred) ** 0.5,
                        dir_acc=float(np.mean(np.sign(pred) == np.sign(yte))),
                        extreme_hit_rate=extreme_hit_rate(yte, pred),
                        pred_kurtosis=excess_kurtosis(pred), pred_skew=skewness(pred),
                        resid_S2=stage_residual_moments[q][2],
                        resid_S4=stage_residual_moments[q][4],
                        resid_S6=stage_residual_moments[q][6],
                    ))

            oos_true.append(yte)
            row_data = {
                "date": d_t.loc[test_idx, "date"].values,
                "ticker": t,
                "sample_id": f"{t}_{p}",  # distinguishes separate sampled windows -- NOT contiguous across samples
                "actual": yte,
                "l2_pred": stage_test_preds[CASCADE_QS[0]],
                "cascade_pred": stage_test_preds[CASCADE_QS[-1]],
            }
            for q in CASCADE_QS:
                row_data[f"cumulative_q{q}"] = stage_test_preds[q]
                row_data[f"layer_q{q}"] = stage_increment_test[q]
            oos_rows.append(pd.DataFrame(row_data))

            n_folds_ticker += 1

        print(f"  {t}: window={window}d, {n_folds_ticker}/{len(sample_p)} sampled folds valid "
              f"{'-- ALL SKIPPED (window too small for horizon after purge)' if n_folds_ticker == 0 else ''}")

    if skipped_tickers:
        print(f"  Skipped entirely (not enough total history for even one window): {skipped_tickers}")

    if not fold_records:
        print(f"  No valid folds at all for horizon {h}d -- skipping this horizon's summary/plots.")
        results["horizons"][str(h)] = {"error": "no valid folds"}
        continue

    fm = pd.DataFrame(fold_records)
    summary = {}
    for v in fm["variant"].unique():
        sub_all = fm[fm["variant"] == v]
        sub = sub_all[~sub_all["diverged"]]  # pooled stats computed on the FINITE folds only,
                                              # so one diverged fold doesn't wipe out the rest
        n_diverged = int(sub_all["diverged"].sum())
        if len(sub) == 0:
            summary[v] = {"n_folds": int(len(sub_all)), "n_diverged": n_diverged,
                          "note": "every fold diverged for this variant"}
            continue
        n_total = sub["n"].sum()
        summary[v] = {
            "n_folds": int(len(sub_all)), "n_diverged": n_diverged, "n_obs": int(n_total),
            "pooled_r2": float((sub["r2"] * sub["n"]).sum() / n_total),
            "pooled_rmse": float((sub["rmse"] * sub["n"]).sum() / n_total),
            "pooled_dir_acc": float((sub["dir_acc"] * sub["n"]).sum() / n_total),
            "pooled_extreme_hit_rate": float(sub["extreme_hit_rate"].mean()),
            "mean_pred_kurtosis": float(sub["pred_kurtosis"].mean()),
            "mean_pred_skew": float(sub["pred_skew"].mean()),
            "mean_resid_S4": float(sub["resid_S4"].mean()),
            "mean_resid_S6": float(sub["resid_S6"].mean()),
        }
    true_kurt = excess_kurtosis(np.concatenate(oos_true))
    true_skew = skewness(np.concatenate(oos_true))
    summary["_true_oos_kurtosis"] = true_kurt
    summary["_true_oos_skew"] = true_skew
    results["horizons"][str(h)] = summary

    print(f"\n--- Horizon {h}d pooled summary (true OOS excess kurtosis={true_kurt:.2f}, skew={true_skew:.2f}) ---")
    for v, s in summary.items():
        if v.startswith("_"):
            continue
        if "note" in s:
            print(f"  {v}: {s['note']} ({s['n_diverged']}/{s['n_folds']} folds)")
            continue
        print(f"  {v}: r2={s['pooled_r2']:.3f} hit_rate={s['pooled_extreme_hit_rate']:.3f} "
              f"pred_kurtosis={s['mean_pred_kurtosis']:.2f} pred_skew={s['mean_pred_skew']:.2f} "
              f"resid_S4={s['mean_resid_S4']:.3e} resid_S6={s['mean_resid_S6']:.3e} "
              f"diverged={s['n_diverged']}/{s['n_folds']}")

    # ── Plot: does the cascade close the kurtosis gap and improve extreme
    # hit-rate as q escalates, without degrading r2? One figure per horizon.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    qs = CASCADE_QS
    r2s = [summary[f"cascade_upto_q{q}"].get("pooled_r2", np.nan) for q in qs]
    hits = [summary[f"cascade_upto_q{q}"].get("pooled_extreme_hit_rate", np.nan) for q in qs]
    kurts = [summary[f"cascade_upto_q{q}"].get("mean_pred_kurtosis", np.nan) for q in qs]

    axes[0].plot(qs, r2s, marker="o")
    axes[0].set_title(f"OOS R2 vs cascade depth ({h}d)")
    axes[0].set_xlabel("cumulative stages through q="); axes[0].set_ylabel("pooled R2")

    axes[1].plot(qs, hits, marker="o", color="tab:orange")
    axes[1].set_title(f"Extreme hit-rate vs cascade depth ({h}d)")
    axes[1].set_xlabel("cumulative stages through q="); axes[1].set_ylabel("extreme_hit_rate")

    axes[2].plot(qs, kurts, marker="o", color="tab:green", label="prediction excess kurtosis")
    axes[2].axhline(true_kurt, color="black", linestyle="--", label="true OOS excess kurtosis")
    axes[2].set_title(f"Predicted kurtosis vs truth ({h}d)")
    axes[2].set_xlabel("cumulative stages through q="); axes[2].set_ylabel("excess kurtosis")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, f"71_cascade_loss_escalation_{h}d.png")
    fig.savefig(plot_path, dpi=110)
    plt.close(fig)
    print(f"  Saved plot: {plot_path}")

    # ── The actual predictions, not just summary stats: per-row OOS
    # actual/l2/cascade for every instrument, saved to disk, plus an overlay
    # plot for a few representative tickers (same set 02_train_predict_daily.py
    # already plots). Two panels per ticker: symlog full range (so the
    # accepted blowup folds, e.g. 2008, are actually visible rather than
    # hidden) and a zoomed linear range (so the typical-day behavior isn't
    # flattened to nothing by those same blowups).
    oos_df = pd.concat(oos_rows, ignore_index=True).sort_values(["ticker", "date"])
    oos_path = os.path.join(OUT_DIR, f"71_cascade_oos_predictions_{h}d.parquet")
    oos_df.to_parquet(oos_path)
    print(f"  Saved per-row OOS predictions: {oos_path} ({len(oos_df)} rows)")

    plot_tickers = [t for t in ["SPY", "GLD", "BTC-USD"] if t in oos_df["ticker"].unique()]
    if plot_tickers:
        fig, axes = plt.subplots(len(plot_tickers), 2, figsize=(14, 3.5 * len(plot_tickers)), squeeze=False)
        for i, tkr in enumerate(plot_tickers):
            sub = oos_df[oos_df["ticker"] == tkr].sort_values("date")
            for col, (ax, label) in enumerate(zip(axes[i], ["full range (symlog)", "zoomed typical range"])):
                # Each sample_id is its own short, genuinely-contiguous window
                # (~11-63 days) -- plotted as its own line segment with
                # markers, NOT connected to the next sample, since different
                # samples can be years apart and a single connecting line
                # across that gap would be a real, misleading artifact.
                first = True
                for sid, seg in sub.groupby("sample_id", sort=False):
                    lbl = dict(actual="actual", l2_pred="L2 (climatology)", cascade_pred="cascade (q8)") if first else {}
                    ax.plot(seg["date"], seg["actual"], color="black", lw=1.0, marker="o", ms=2, label=lbl.get("actual"))
                    ax.plot(seg["date"], seg["l2_pred"], color="tab:blue", lw=1.0, marker="o", ms=2, alpha=0.7, label=lbl.get("l2_pred"))
                    ax.plot(seg["date"], seg["cascade_pred"], color="tab:red", lw=1.0, marker="o", ms=2, alpha=0.7, label=lbl.get("cascade_pred"))
                    first = False
                if col == 0:
                    ax.set_yscale("symlog", linthresh=0.05)
                else:
                    typical = sub["actual"].abs().quantile(0.99)
                    ax.set_ylim(-3 * typical, 3 * typical)
                ax.set_title(f"{tkr} {h}d forward return, OOS -- {label} ({sub['sample_id'].nunique()} sampled windows, window={WINDOW_DAYS[tkr]}d each)")
                ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        pred_plot_path = os.path.join(OUT_DIR, f"71_cascade_pred_vs_actual_{h}d.png")
        fig.savefig(pred_plot_path, dpi=110)
        plt.close(fig)
        print(f"  Saved predicted-vs-actual plot: {pred_plot_path}")

        # ── Each cascade layer shown separately, for one representative
        # ticker: row per stage (q2 climatology, q4/q6/q8 residual layers).
        # Left column = that stage's OWN output in isolation (what that
        # layer alone is adding); right column = the CUMULATIVE prediction
        # through that stage (what you'd actually use if you stopped there).
        # Symlog throughout since later layers' isolated output can itself
        # be large on the same folds that blow up in the cumulative sum.
        tkr = plot_tickers[0]
        sub = oos_df[oos_df["ticker"] == tkr].sort_values("date")
        fig, axes = plt.subplots(len(CASCADE_QS), 2, figsize=(14, 3 * len(CASCADE_QS)), squeeze=False)
        stage_names = {CASCADE_QS[0]: "stage 1 (true climatology, mean(y), no model)"}
        stage_names.update({q: f"stage {i} residual AI layer (tanh MLP, L2)" for i, q in enumerate(CASCADE_QS[1:], start=2)})
        for row, q in enumerate(CASCADE_QS):
            ax_layer, ax_cum = axes[row]
            first = True
            for sid, seg in sub.groupby("sample_id", sort=False):
                lbl1 = "actual" if first else None
                lbl2 = f"layer output ({stage_names[q]})" if first else None
                ax_layer.plot(seg["date"], seg["actual"], color="black", lw=0.8, marker="o", ms=2, alpha=0.5, label=lbl1)
                ax_layer.plot(seg["date"], seg[f"layer_q{q}"], color="tab:purple", lw=1.0, marker="o", ms=2, label=lbl2)
                first = False
            ax_layer.set_yscale("symlog", linthresh=0.02)
            ax_layer.set_title(f"{tkr} {h}d -- stage {stage_names[q]}: THIS LAYER ALONE ({sub['sample_id'].nunique()} sampled windows)")
            ax_layer.legend(fontsize=7, loc="upper left")

            first = True
            for sid, seg in sub.groupby("sample_id", sort=False):
                lbl1 = "actual" if first else None
                lbl2 = f"cumulative through {stage_names[q]}" if first else None
                ax_cum.plot(seg["date"], seg["actual"], color="black", lw=0.8, marker="o", ms=2, alpha=0.5, label=lbl1)
                ax_cum.plot(seg["date"], seg[f"cumulative_q{q}"], color="tab:red", lw=1.0, marker="o", ms=2, label=lbl2)
                first = False
            ax_cum.set_yscale("symlog", linthresh=0.02)
            ax_cum.set_title(f"{tkr} {h}d -- CUMULATIVE through {stage_names[q]}")
            ax_cum.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        layers_plot_path = os.path.join(OUT_DIR, f"71_cascade_layers_{h}d.png")
        fig.savefig(layers_plot_path, dpi=110)
        plt.close(fig)
        print(f"  Saved per-layer breakdown plot: {layers_plot_path}")

out_path = os.path.join(OUT_DIR, "results_cascade_loss.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved results to {out_path}")
print("\nDone.")
