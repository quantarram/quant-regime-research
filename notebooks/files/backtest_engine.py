"""
backtest_engine.py
===================
Independent reimplementation of the Portfolio Tilt strategy's daily
walk-forward backtest, built ONLY from the prose specification in
SPEC.md (which is itself extracted from the Portfolio Tilt paper's
Sections 2.3-2.4, 7, and 10.3). The original code that produced the
paper's reported numbers no longer exists; this is a fresh, independent
implementation against the same written spec, not a recovered copy.

This module implements two distinct position-construction mechanisms,
matching the paper's own two specifications:
  - STATIC TILT  (paper Sections 2-9): daily-re-evaluated discrete tilt
    tiers, computed fresh each day from whichever configurations are
    currently firing.
  - HOLD-TO-HORIZON (paper Section 10.3): a newly-firing configuration's
    tilt is held for its full forward horizon, continuously
    conviction-scaled, regardless of whether the trigger condition
    persists.

Both mechanisms share the same underlying joint-CPE-configuration input,
the same neutral weights, the same train/test split, and the same
position-lag discipline -- only the day-to-day weight-construction logic
differs, exactly as in the paper.

Usage:
    from backtest_engine import run_backtest, JOINT_SCREEN_UNRESTRICTED, JOINT_SCREEN_PRIOR_GATED

    result = run_backtest(joint_path="joint_cpe_results_ORIGINAL.parquet",
                           mechanism="static")
    result = run_backtest(joint_path="joint_cpe_results_PRIOR_GATED.parquet",
                           mechanism="hold_to_horizon")
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG (from SPEC.md section A) ─────────────────────────────────────

TRAIN_CUTOFF = pd.Timestamp("2024-12-31")
EVAL_START   = pd.Timestamp("2025-01-01")
EVAL_END     = pd.Timestamp("2025-12-31")

SLEEVES = {
    "Equities": "SPY",
    "Gold":     "GC=F",
    "Bonds":    "TLT",
    "Crypto":   "BTC-USD",
    "FX":       "UUP",
}

# Neutral weights, exactly as stated in the paper (Section 2.4).
# These are quoted figures from the paper's prose, not recomputed here,
# since recomputing them would require the exact training-period Sharpe
# convention (lookback window, etc.) the paper used but did not fully
# specify down to the implementation. Using the paper's own stated
# numbers keeps this rebuild's neutral baseline identical to the
# original, isolating the test to the tilt mechanism and screen content.
NEUTRAL_WEIGHTS = {
    "Equities": 30.87,
    "Gold":     24.29,
    "Bonds":     4.74,
    "Crypto":   30.10,
    "FX":       10.00,
}

HORIZON_WEIGHTS = {21: 0.20, 63: 0.30, 126: 0.30, 252: 0.20}  # 300d excluded, per spec A.5

TILT_TIERS = [   # (threshold, delta_pp) — applied symmetrically, see _static_tilt_delta
    (0.30, 15.0),
    (0.05, 8.0),
]

HISTORY_RAMP_MIN_OBS = 100
HISTORY_RAMP_FULL_OBS = 756
HISTORY_RAMP_MIN_WEIGHT = 0.35

WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX = 0.0, 50.0


# ── HISTORY-LENGTH DOWN-WEIGHT h(Π) — spec A.4 ──────────────────────────

def _ticker_history_obs(ticker: str, prices: pd.DataFrame) -> int:
    """Number of training-period (<=2024-12-31) non-null observations."""
    s = prices[ticker].dropna()
    return int((s.index <= TRAIN_CUTOFF).sum())


def _history_weight(n_obs: int) -> float:
    if n_obs >= HISTORY_RAMP_FULL_OBS:
        return 1.0
    if n_obs <= HISTORY_RAMP_MIN_OBS:
        return HISTORY_RAMP_MIN_WEIGHT
    frac = (n_obs - HISTORY_RAMP_MIN_OBS) / (HISTORY_RAMP_FULL_OBS - HISTORY_RAMP_MIN_OBS)
    return HISTORY_RAMP_MIN_WEIGHT + frac * (1.0 - HISTORY_RAMP_MIN_WEIGHT)


def compute_h_for_joint_row(predictors: list, prices: pd.DataFrame) -> float:
    """h(Pi) = min over predictors in the set of the per-predictor history weight."""
    obs_counts = [_ticker_history_obs(p, prices) for p in predictors]
    return min(_history_weight(n) for n in obs_counts)


# ── EPISODE-INDEPENDENCE CONVICTION (replaces ln(n_joint)) ───────────────
#
# n_joint counts overlapping daily conditioning observations, which the
# Atlas paper's own Section 3.3 and every paper since has flagged as
# overstating independent support: a 252-day trailing window stepped one
# day at a time can turn 3-4 genuinely distinct historical episodes into
# a nominal count of 100+. ln(n_joint) rewards this directly. The
# replacement below clusters a configuration's firing dates into
# genuinely separated episodes (gap > 1.5x the conditioning window,
# same convention as the Portfolio Tilt paper's own Section 14 episode
# filter) and evaluates the target's outcome ONCE PER EPISODE rather
# than once per overlapping day, so a relationship is only rewarded for
# how many independent times it has actually recurred and how
# consistently those independent recurrences agree.

EPISODE_MIN_OBS_FOR_CONVICTION = 3  # hard floor: below this, conviction = 0, not discounted
EPISODE_GAP_MULTIPLIER = 1.5
Q_GRID_FOR_THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]


def _cluster_into_episodes(firing_dates: pd.DatetimeIndex, gap_trading_days: int) -> list:
    """Group sorted firing dates into episodes separated by a gap exceeding
    EPISODE_GAP_MULTIPLIER * gap_trading_days (converted to calendar days)."""
    if len(firing_dates) == 0:
        return []
    dates = pd.DatetimeIndex(sorted(firing_dates))
    gap_calendar_days = gap_trading_days * 1.45 * EPISODE_GAP_MULTIPLIER
    episodes = []
    current = [dates[0]]
    for i in range(1, len(dates)):
        if (dates[i] - dates[i - 1]).days > gap_calendar_days:
            episodes.append(current)
            current = [dates[i]]
        else:
            current.append(dates[i])
    episodes.append(current)
    return episodes


def build_increments_for_episodes(prices: pd.DataFrame, tau_list: list) -> dict:
    """
    Precompute the full increment panel once, for every tau needed by the
    joint screen. Pass the result into compute_quality_weights via
    precomputed_increments so episode conviction doesn't rebuild this
    per-row (the earlier, much slower version of this function did, which
    is why it was impractically slow on a full screen).
    """
    increments = {}
    for tau in tau_list:
        inc = pd.DataFrame(index=prices.index)
        for t in prices.columns:
            s = prices[t]
            inc[t] = np.log(s / s.shift(tau))
        increments[tau] = inc
    return increments


def _episode_conviction_for_row(row: pd.Series, increments: dict) -> float:
    """
    Recompute conviction for one joint-screen row using episode-level
    outcomes instead of raw n_joint. Returns 0.0 if the configuration
    has fewer than EPISODE_MIN_OBS_FOR_CONVICTION genuinely independent
    episodes -- a hard floor, not a discount, mirroring the lesson from
    the MIN_TRAIN_OBS fix: a relationship resting on 1-2 episodes is
    indistinguishable from luck and should not be sizeable at all.
    """
    direction = row["direction"]
    predictors = list(row["predictors"])
    tau_pasts = [int(t) for t in row["tau_pasts"]]
    q_xs = [float(q) for q in row["q_Xs"]]
    y = row["Y"]
    tau_f = int(row["tau_future"])
    q_y = float(row["q_Y"])

    if tau_f not in increments or any(tp not in increments for tp in tau_pasts):
        return 0.0

    train_dates = increments[tau_pasts[0]].index[increments[tau_pasts[0]].index <= TRAIN_CUTOFF]

    joint_mask = pd.Series(True, index=train_dates)
    for x, tau_p, q_x in zip(predictors, tau_pasts, q_xs):
        if x not in increments[tau_p].columns:
            return 0.0
        series = increments[tau_p][x].reindex(train_dates)
        train_series = increments[tau_p][x].loc[increments[tau_p].index <= TRAIN_CUTOFF]
        if direction == "bullish":
            thresh = train_series.quantile(q_x)
            joint_mask &= (series > thresh)
        else:
            thresh = train_series.quantile(round(1 - q_x, 10))
            joint_mask &= (series < thresh)

    firing_dates = joint_mask[joint_mask.fillna(False)].index
    if len(firing_dates) == 0:
        return 0.0

    episodes = _cluster_into_episodes(firing_dates, max(tau_pasts))

    # BUGFIX: outcome evaluation must use the target's FORWARD tau_f-day
    # return STARTING at the episode's anchor date, not the value stored
    # at increments[tau_f][y].get(anchor) -- that lookup returns the
    # TRAILING tau_f-day return ENDING at anchor, the wrong direction in
    # time entirely (it answers "what already happened by this date",
    # not "what happens next", which is the actual predictive question
    # the CPE framework exists to test, and is what joint_cpe_engine.py's
    # own joint_CPE / event_mask machinery correctly uses via
    # future_inc = increments[tau_f].shift(-tau_f)). This was found by
    # tracing a disagreement between this function's hit-rate (33%) and
    # joint_cpe_engine.py's own reported hit-rate (100%) for the
    # identical SPY<-[VIXM,VIXY] configuration -- both functions agreed
    # on firing dates and episode clustering, and disagreed only on
    # which value of SPY's return series counted as "the outcome".
    target_forward = increments[tau_f][y].shift(-tau_f)
    target_train_for_thresh = increments[tau_f][y].loc[increments[tau_f].index <= TRAIN_CUTOFF]
    target_thresh = target_train_for_thresh.quantile(q_y if direction == "bullish" else round(1 - q_y, 10))

    outcomes = []
    for ep in episodes:
        anchor = ep[-1]
        if anchor not in target_forward.index:
            continue
        val = target_forward.get(anchor, np.nan)
        if pd.isna(val):
            continue
        outcomes.append(bool(val > target_thresh) if direction == "bullish" else bool(val < target_thresh))

    n_episodes = len(outcomes)
    if n_episodes < EPISODE_MIN_OBS_FOR_CONVICTION:
        return 0.0

    hit_rate = float(np.mean(outcomes))
    agreement = max(0.0, 2 * hit_rate - 1)  # 0 at 50% hit rate (no better than chance), 1 at 100%
    return float(np.log(n_episodes) * agreement)


# ── QUALITY WEIGHT w(Pi) — spec A.4, MODIFIED to use episode conviction ──

def compute_quality_weights(joint_df: pd.DataFrame, prices: pd.DataFrame,
                             precomputed_increments: dict = None,
                             use_episode_conviction: bool = True) -> pd.Series:
    """
    w(Pi) = CPE * lift * conviction_term * h(Pi)

    conviction_term is, by default (use_episode_conviction=True), the
    episode-independence conviction score: log(n_episodes) * (2*hit_rate-1),
    zeroed out entirely below EPISODE_MIN_OBS_FOR_CONVICTION genuinely
    separated historical episodes. This REPLACES the original
    ln(n_joint) term, which counted overlapping daily observations and
    rewarded exactly the small-sample, single-episode configurations
    this change is meant to stop trusting.

    Set use_episode_conviction=False to recover the original
    ln(n_joint)-based formula for direct comparison.

    precomputed_increments: pass the output of
    build_increments_for_episodes(prices, tau_list) to avoid rebuilding
    the full increment panel on every call (expensive). If None, it is
    built once internally from the tau values actually present in
    joint_df.
    """
    h_vals = joint_df["predictors"].apply(lambda preds: compute_h_for_joint_row(list(preds), prices))

    if not use_episode_conviction:
        conviction = np.log(joint_df["n_joint"].clip(lower=1))
        return joint_df["joint_CPE"] * joint_df["lift"] * conviction * h_vals

    if precomputed_increments is None:
        needed_taus = sorted(set(
            int(t) for taus in joint_df["tau_pasts"] for t in taus
        ) | set(int(t) for t in joint_df["tau_future"]))
        precomputed_increments = build_increments_for_episodes(prices, needed_taus)

    conviction = joint_df.apply(
        lambda row: _episode_conviction_for_row(row, precomputed_increments), axis=1
    )
    return joint_df["joint_CPE"] * joint_df["lift"] * conviction * h_vals


# ── INCREMENT / THRESHOLD COMPUTATION (training-frozen) ─────────────────

TAU_LIST = [1, 5, 10, 21, 63, 126, 252, 300]
RATE_INDEX_TICKERS = {
    "^VIX","^VXN","^OVX","^GVZ","^EVZ","^VVIX","^SKEW",
    "^TNX","^TYX","^FVX","^IRX"
}


def build_increments_and_thresholds(prices: pd.DataFrame, q_grid: list):
    """
    Compute, for every (tau, ticker), the full daily increment series
    AND the training-period-only (<=2024-12-31) quantile thresholds.
    The increment series itself spans the full history (needed to
    evaluate live 2025 firing conditions); the THRESHOLDS are frozen
    using only data through TRAIN_CUTOFF, enforcing the train/test split
    (spec A.1).
    """
    all_tickers = list(prices.columns)
    rate_tickers = [t for t in all_tickers if t in RATE_INDEX_TICKERS]
    price_tickers = [t for t in all_tickers if t not in RATE_INDEX_TICKERS]

    increments = {}
    for tau in TAU_LIST:
        inc = pd.DataFrame(index=prices.index)
        for t in price_tickers:
            s = prices[t]
            inc[t] = np.log(s / s.shift(tau))
        for t in rate_tickers:
            s = prices[t]
            inc[t] = s - s.shift(tau)
        increments[tau] = inc

    full_q_grid = sorted(set(q_grid + [round(1 - q, 10) for q in q_grid]))
    thresholds = {}
    train_mask = prices.index <= TRAIN_CUTOFF
    for tau in TAU_LIST:
        train_inc = increments[tau].loc[train_mask]
        for q in full_q_grid:
            thresholds[(tau, q)] = train_inc.quantile(q, numeric_only=True).to_dict()

    return increments, thresholds


# ── DAILY FIRING CHECK ───────────────────────────────────────────────────

def configuration_fires_on_date(row, date, increments, thresholds) -> bool:
    """
    Check whether ALL predictors in a joint configuration row
    simultaneously clear their own (frozen, training-period) threshold
    on a given evaluation date. Mirrors cpe_signal_score.py's
    condition_fires() logic exactly, generalised from "latest date" to
    an arbitrary evaluation date (spec B.1).
    """
    direction = row["direction"]
    for x, tau_p, q_x in zip(row["predictors"], row["tau_pasts"], row["q_Xs"]):
        tau_p = int(tau_p)
        q_x = float(q_x)
        if x not in increments[tau_p].columns:
            return False
        if date not in increments[tau_p].index:
            return False
        curr = increments[tau_p].at[date, x]
        if pd.isna(curr):
            return False
        if direction == "bullish":
            thresh = thresholds.get((tau_p, q_x), {}).get(x, np.nan)
            if np.isnan(thresh) or not (curr > thresh):
                return False
        else:
            thresh = thresholds.get((tau_p, round(1 - q_x, 10)), {}).get(x, np.nan)
            if np.isnan(thresh) or not (curr < thresh):
                return False
    return True


# ── DAILY CLASS SCORE (STATIC MECHANISM) — spec B.2 ─────────────────────

def compute_daily_class_scores(sleeve_proxy: str, joint_for_target: pd.DataFrame,
                                weights: pd.Series, eval_dates: pd.DatetimeIndex,
                                increments, thresholds) -> pd.Series:
    """
    For one sleeve's proxy ticker, compute the combined daily tilt score
    S(class, t) for every evaluation date, using the horizon-weighted
    combination of per-horizon scores (spec A.5, B.2).
    """
    if joint_for_target.empty:
        return pd.Series(0.0, index=eval_dates)

    # Pre-fire matrix: rows = config index, cols = eval dates, bool fires
    fire_matrix = pd.DataFrame(
        index=joint_for_target.index, columns=eval_dates, dtype=bool
    )
    for idx, row in joint_for_target.iterrows():
        for d in eval_dates:
            fire_matrix.at[idx, d] = configuration_fires_on_date(row, d, increments, thresholds)

    bull_mask = (joint_for_target["direction"] == "bullish").values
    bear_mask = (joint_for_target["direction"] == "bearish").values
    w = weights.loc[joint_for_target.index].values

    total_bull_w = w[bull_mask].sum()
    total_bear_w = w[bear_mask].sum()
    total_w = total_bull_w + total_bear_w

    per_horizon_scores = {}
    for tau_f, hweight in HORIZON_WEIGHTS.items():
        tau_mask = (joint_for_target["tau_future"] == tau_f).values
        bull_tau = bull_mask & tau_mask
        bear_tau = bear_mask & tau_mask
        w_bull_tau_total = w[bull_tau].sum()
        w_bear_tau_total = w[bear_tau].sum()
        denom = w_bull_tau_total + w_bear_tau_total
        if denom <= 0:
            per_horizon_scores[tau_f] = pd.Series(0.0, index=eval_dates)
            continue
        fired_bull_w = fire_matrix.loc[bull_tau].mul(w[bull_tau], axis=0).sum(axis=0)
        fired_bear_w = fire_matrix.loc[bear_tau].mul(w[bear_tau], axis=0).sum(axis=0)
        per_horizon_scores[tau_f] = (fired_bull_w - fired_bear_w) / denom

    combined = pd.Series(0.0, index=eval_dates)
    for tau_f, hweight in HORIZON_WEIGHTS.items():
        combined = combined + hweight * per_horizon_scores[tau_f]

    return combined


def _static_tilt_delta(score: float) -> float:
    """Map a combined class score to a tilt delta in percentage points,
    per spec A.6's symmetric five-tier scheme."""
    abs_score = abs(score)
    sign = 1.0 if score > 0 else (-1.0 if score < 0 else 0.0)
    for thresh, delta in TILT_TIERS:
        if abs_score >= thresh:
            return sign * delta
    return 0.0


# ── WEIGHT CONSTRUCTION: CLIP + RENORMALISE — spec A.8 ──────────────────

def clip_and_renormalise(raw_weights: dict) -> dict:
    clipped = {k: min(max(v, WEIGHT_CLIP_MIN), WEIGHT_CLIP_MAX) for k, v in raw_weights.items()}
    total = sum(clipped.values())
    if total <= 0:
        return {k: 100.0 / len(clipped) for k in clipped}
    return {k: v * 100.0 / total for k, v in clipped.items()}


# ── PORTFOLIO SIMULATION ─────────────────────────────────────────────────

def simulate_portfolio(daily_weights: pd.DataFrame, prices: pd.DataFrame,
                        eval_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Given a DataFrame of daily target weights per sleeve (index = eval
    dates, columns = sleeve names, values = weight in percent, ALREADY
    LAGGED one day per spec A.7), simulate a $100,000 notional book and
    return the daily equity curve.
    """
    notional = 100_000.0
    equity = [notional]
    dates_out = [eval_dates[0]]

    sleeve_returns = pd.DataFrame(index=eval_dates, columns=SLEEVES.keys(), dtype=float)
    for sleeve, ticker in SLEEVES.items():
        # Forward-fill from the FULL price history, not just the eval
        # window, so the first eval date has a valid prior price to
        # compute a return against even if it happens to follow a gap.
        full_px = prices[ticker].ffill()
        px = full_px.reindex(eval_dates)
        sleeve_returns[sleeve] = px.pct_change()

    for i in range(1, len(eval_dates)):
        d = eval_dates[i]
        w = daily_weights.loc[d] / 100.0
        r = sleeve_returns.loc[d].fillna(0.0)
        port_ret = (w * r).sum()
        equity.append(equity[-1] * (1 + port_ret))
        dates_out.append(d)

    return pd.DataFrame({"date": dates_out, "equity": equity}).set_index("date")


def compute_performance_stats(equity_curve: pd.Series) -> dict:
    rets = equity_curve.pct_change().dropna()
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    ann_vol = rets.std() * np.sqrt(252) * 100
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else np.nan
    return {
        "total_return_pct": round(total_return, 2),
        "ann_vol_pct": round(ann_vol, 2),
        "sharpe": round(sharpe, 3),
    }
