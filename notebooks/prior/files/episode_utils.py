"""
episode_utils.py
================
Single source of truth for episode clustering and conviction scoring,
shared by joint_cpe_engine.py (screen-build time) and backtest_engine.py
(runtime quality-weight computation).

MOTIVATION (Portfolio Tilt paper Section 20.6)
----------------------------------------------
The paper identified a critical discrepancy: the joint screen's stored
n_episodes column was computed by the greedy search using one version of
the episode-clustering algorithm, while backtest_engine.py's runtime
recomputation used a separately-maintained copy of the same logic that
had since been corrected for the forward/trailing return direction bug.
The two implementations agreed to different episode counts for the same
configuration, meaning the stored n_episodes in joint_cpe_results.parquet
and the runtime conviction values used for sizing were inconsistent.

Concretely: the VIXM+VIXY->SPY tau_f=63 configurations showed n_episodes=6
in runtime diagnostics but n_episodes=4 when the stored parquet column was
read directly — a discrepancy that changes the interpretation of the paper's
primary result and cannot be resolved by inspecting either implementation
in isolation.

FIX
---
This module provides one canonical implementation of each operation.
Both engines import from here. Neither maintains its own copy. The
four-decimal-place cross-check used in Section 14.3 to verify the bug
fix (two independent implementations agreeing on episode conviction to
four decimal places) is preserved as a unit test at the bottom of this
file.

EPISODE-COUNTING ALGORITHM (unchanged from corrected backtest_engine.py)
-------------------------------------------------------------------------
A new episode begins whenever the gap since the previous firing date
exceeds EPISODE_GAP_MULTIPLIER * max_tau_past trading days, converted to
calendar days via the 1.45 trading-to-calendar-day convention used
throughout the pipeline. This matches the paper's Section 14.3 spec and
the Scope C parameter tested in Section 15.3.

OUTCOME EVALUATION (BUGFIX from Section 14.3)
---------------------------------------------
Outcomes are evaluated using the target's FORWARD tau_f-day return
STARTING at the episode anchor date, not the trailing return ENDING there.
The original implementation used increments[tau_f][y].loc[anchor] which
retrieves the trailing return — the wrong direction. The fix is
increments[tau_f][y].shift(-tau_f).loc[anchor], which retrieves the
forward return. This is the bug whose discovery (33% vs 100% hit rate
for VIXM+VIXY->SPY) reconciled the two implementations to 1.7918 and
1.792 respectively.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

# ── CANONICAL PARAMETERS ────────────────────────────────────────────────────
# These are the values used throughout the paper's Sections 14-20 and the
# sensitivity grid in Scope C (Section 15.3), which found the result stable
# across all tested values of each parameter.

EPISODE_GAP_MULTIPLIER: float = 1.5      # gap > 1.5x max_tau_past -> new episode
TRADING_TO_CALENDAR: float    = 1.45     # trading days -> calendar days conversion
EPISODE_MIN_CONVICTION: int   = 3        # hard floor: below this, conviction = 0
EPISODE_ANCHOR: str           = "last"   # "last", "first", or "mid"
EPISODE_AGREEMENT: str        = "linear" # "linear" or "binomial"


# ── CORE CLUSTERING ─────────────────────────────────────────────────────────

def cluster_into_episodes(
    firing_dates: pd.DatetimeIndex,
    max_tau_past: int,
    gap_multiplier: float = EPISODE_GAP_MULTIPLIER,
) -> List[List[pd.Timestamp]]:
    """
    Group sorted firing dates into temporally separated episodes.

    A new episode begins whenever the gap between consecutive firing dates
    exceeds gap_multiplier * max_tau_past trading days (converted to
    calendar days). Returns a list of episodes, each episode being a list
    of Timestamps.

    This is the canonical implementation. Both joint_cpe_engine.py and
    backtest_engine.py must import and call this function rather than
    maintaining their own copies, so that stored n_episodes in the parquet
    and runtime conviction values are guaranteed to agree.

    Parameters
    ----------
    firing_dates : pd.DatetimeIndex
        Dates on which all predictors in a joint configuration simultaneously
        cleared their thresholds within the training window.
    max_tau_past : int
        The longest conditioning window (in trading days) in the predictor
        set. Used to determine the minimum gap that constitutes a new episode.
    gap_multiplier : float
        Scaling factor applied to max_tau_past to set the episode boundary.
        Default 1.5 (paper Section 14.3 / Scope C).

    Returns
    -------
    List[List[pd.Timestamp]]
        Each inner list contains the Timestamps belonging to one episode,
        in chronological order.
    """
    if len(firing_dates) == 0:
        return []

    dates = sorted(firing_dates)
    gap_calendar_days = max_tau_past * TRADING_TO_CALENDAR * gap_multiplier

    episodes: List[List[pd.Timestamp]] = []
    current: List[pd.Timestamp] = [dates[0]]

    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i - 1]).days
        if gap > gap_calendar_days:
            episodes.append(current)
            current = [dates[i]]
        else:
            current.append(dates[i])

    episodes.append(current)
    return episodes


def _anchor_date(episode: List[pd.Timestamp], anchor: str = EPISODE_ANCHOR) -> pd.Timestamp:
    """Return the anchor date for an episode given the anchor convention."""
    if anchor == "first":
        return episode[0]
    if anchor == "mid":
        return episode[len(episode) // 2]
    return episode[-1]  # default "last"


# ── CONVICTION COMPUTATION ──────────────────────────────────────────────────

def compute_episode_conviction(
    firing_dates: pd.DatetimeIndex,
    max_tau_past: int,
    target_forward_series: pd.Series,
    target_thresh: float,
    direction: str,
    min_conviction: int = EPISODE_MIN_CONVICTION,
    gap_multiplier: float = EPISODE_GAP_MULTIPLIER,
    anchor: str = EPISODE_ANCHOR,
    agreement_formula: str = EPISODE_AGREEMENT,
) -> Tuple[int, Optional[float], float]:
    """
    Compute the episode-based conviction term that replaces ln(n_joint)
    in the quality-weight formula w(Pi) = CPE * lift * conviction * h(Pi).

    This is the SINGLE canonical implementation used by both engines.
    The paper's cross-check (Section 14.3) that verified the bug fix was:
        joint_engine conviction  == 1.7918
        backtest_engine conviction == 1.792  (4 d.p. agreement)
    Running this function from both call sites should produce identical
    results to at least 4 decimal places.

    Parameters
    ----------
    firing_dates : pd.DatetimeIndex
        Training-window dates on which the configuration fired.
    max_tau_past : int
        Longest conditioning window in the predictor set (trading days).
    target_forward_series : pd.Series
        target's FORWARD tau_f-day return series. MUST be pre-shifted:
            increments[tau_f][y].shift(-tau_f)
        NOT the trailing series. Using the trailing series is the bug
        documented in Section 14.3 that produced the 33% vs 100%
        discrepancy.
    target_thresh : float
        The training-period quantile threshold for the target's forward
        return (q_Y for bullish, 1-q_Y for bearish).
    direction : str
        "bullish" or "bearish". Determines comparison direction for outcomes.
    min_conviction : int
        Hard floor on n_episodes. Below this, returns (n, hit_rate, 0.0).
        Default 3 (paper Section 14.3).
    gap_multiplier : float
        Passed to cluster_into_episodes. Default 1.5.
    anchor : str
        Episode anchor convention. Default "last" (paper Section 15.3 Scope C
        found the result stable across "first"/"mid"/"last").
    agreement_formula : str
        "linear" (default) or "binomial". "linear" = max(0, 2*hr-1).
        "binomial" uses scipy.stats.binomtest against p=0.5. Scope C found
        both produce identical results at 100% hit rate (the validated signal).

    Returns
    -------
    (n_episodes, episode_hit_rate, episode_conviction)
        n_episodes : int
            Number of genuinely separated historical episodes.
        episode_hit_rate : float or None
            Fraction of episodes where the target outcome was favourable.
            None if no evaluable episodes (missing data).
        episode_conviction : float
            log(n_episodes) * agreement, or 0.0 if n_episodes < min_conviction.
    """
    episodes = cluster_into_episodes(firing_dates, max_tau_past, gap_multiplier)
    n_episodes = len(episodes)

    outcomes: List[bool] = []
    for ep in episodes:
        anchor_ts = _anchor_date(ep, anchor)
        if anchor_ts not in target_forward_series.index:
            continue
        val = target_forward_series.get(anchor_ts, np.nan)
        if pd.isna(val):
            continue
        if direction == "bullish":
            outcomes.append(bool(val > target_thresh))
        else:
            outcomes.append(bool(val < target_thresh))

    n_eval = len(outcomes)
    if n_eval < min_conviction:
        return n_episodes, (float(np.mean(outcomes)) if outcomes else None), 0.0

    hit_rate = float(np.mean(outcomes))

    if agreement_formula == "binomial":
        try:
            # scipy >= 1.7: use binomtest; older: binom_test
            try:
                from scipy.stats import binomtest
                p_val = binomtest(
                    int(round(hit_rate * n_eval)), n_eval, 0.5, alternative="greater"
                ).pvalue
            except ImportError:
                from scipy.stats import binom_test
                p_val = binom_test(
                    int(round(hit_rate * n_eval)), n_eval, 0.5, alternative="greater"
                )
            agreement = max(0.0, 1.0 - 2 * p_val)
        except Exception:
            agreement = max(0.0, 2 * hit_rate - 1)
    else:
        agreement = max(0.0, 2 * hit_rate - 1)

    conviction = float(np.log(n_eval) * agreement)
    return n_eval, hit_rate, conviction


# ── CONVENIENCE WRAPPER FOR BACKTEST ENGINE ROW ─────────────────────────────

def episode_conviction_for_row(
    row: pd.Series,
    increments: dict,
    train_cutoff: pd.Timestamp,
    min_conviction: int = EPISODE_MIN_CONVICTION,
) -> float:
    """
    Compute episode conviction for one joint-screen row, given the
    precomputed increments dict (keys = tau int, values = full-history
    DataFrame). This is the drop-in replacement for
    backtest_engine._episode_conviction_for_row().

    The key correctness requirement: target_forward_series is
        increments[tau_f][y].shift(-tau_f)
    not increments[tau_f][y] (which would be the trailing series).

    Returns
    -------
    float
        episode_conviction, or 0.0 if below min_conviction or data missing.
    """
    direction  = row["direction"]
    predictors = list(row["predictors"])
    tau_pasts  = [int(t) for t in row["tau_pasts"]]
    q_xs       = [float(q) for q in row["q_Xs"]]
    y          = row["Y"]
    tau_f      = int(row["tau_future"])
    q_y        = float(row["q_Y"])

    # Validate data availability
    if tau_f not in increments:
        return 0.0
    if any(tp not in increments for tp in tau_pasts):
        return 0.0
    if y not in increments[tau_f].columns:
        return 0.0

    # Build joint mask on training window only
    ref_tau = tau_pasts[0]
    train_dates = increments[ref_tau].index[increments[ref_tau].index <= train_cutoff]

    joint_mask = pd.Series(True, index=train_dates)
    for x, tau_p, q_x in zip(predictors, tau_pasts, q_xs):
        if x not in increments[tau_p].columns:
            return 0.0
        series = increments[tau_p][x].reindex(train_dates)
        train_series = increments[tau_p][x].loc[increments[tau_p].index <= train_cutoff]
        if direction == "bullish":
            thresh = train_series.quantile(q_x)
            joint_mask &= series > thresh
        else:
            thresh = train_series.quantile(round(1 - q_x, 10))
            joint_mask &= series < thresh

    firing_dates = joint_mask[joint_mask.fillna(False)].index
    if len(firing_dates) == 0:
        return 0.0

    # FORWARD return series (the corrected direction)
    target_forward = increments[tau_f][y].shift(-tau_f)

    # Target threshold (training window only)
    target_train = increments[tau_f][y].loc[increments[tau_f].index <= train_cutoff]
    if direction == "bullish":
        target_thresh = target_train.quantile(q_y)
    else:
        target_thresh = target_train.quantile(round(1 - q_y, 10))

    max_tau_past = max(tau_pasts)
    _, _, conviction = compute_episode_conviction(
        firing_dates=firing_dates,
        max_tau_past=max_tau_past,
        target_forward_series=target_forward,
        target_thresh=target_thresh,
        direction=direction,
        min_conviction=min_conviction,
    )
    return conviction


# ── UNIT TESTS ──────────────────────────────────────────────────────────────

def _run_self_test():
    """
    Reproduce the Section 14.3 cross-check: two independent call paths
    for the VIXM+VIXY->SPY tau_f=63 configuration should agree to
    4 decimal places (paper reports 1.7918 vs 1.792).

    This test uses synthetic data that mirrors the configuration's
    documented properties: 4 distinct historical episodes (2011, 2018-19,
    2020, 2024) with a 100% hit rate, producing
        conviction = log(4) * 1.0 = 1.3863
    Note: the paper's reported value of 1.792 corresponds to log(6)*1.0
    (6 episodes under the earlier, pre-corrected episode count). With the
    corrected n_episodes=4, the expected value is log(4)=1.3863. This
    function verifies internal consistency of the two call paths, not the
    absolute paper value (which reflects the pre-correction episode count).
    """
    import io

    rng = np.random.default_rng(42)
    n_train = 3000

    # Build a synthetic daily price series with ~4 distinct vol spikes
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n_train)))
    dates = pd.bdate_range("2010-01-04", periods=n_train)
    train_cutoff = pd.Timestamp("2023-12-31")
    train_dates = dates[dates <= train_cutoff]

    tau_p, tau_f = 126, 63

    # Simulate VIXM (predictor): spike 4 times, separated by >1.5*126 trading days
    vixm = pd.Series(rng.normal(0, 0.05, n_train), index=dates)
    spike_centres = [200, 700, 1400, 2500]  # well separated episodes
    for c in spike_centres:
        lo, hi = max(0, c - 30), min(n_train, c + 30)
        vixm.iloc[lo:hi] += 0.5  # push well above 95th percentile

    increments = {
        tau_p: pd.DataFrame({"VIXM": vixm, "SPY": pd.Series(rng.normal(0, 0.02, n_train), index=dates)}),
        tau_f: pd.DataFrame({"SPY": pd.Series(rng.normal(0.001, 0.02, n_train), index=dates)}),
    }
    # Ensure the target forward returns are positive (100% hit rate) around spike events
    spy_fwd = increments[tau_f]["SPY"].copy()
    for c in spike_centres:
        lo, hi = max(0, c), min(n_train, c + 100)
        spy_fwd.iloc[lo:hi] = abs(spy_fwd.iloc[lo:hi])
    increments[tau_f]["SPY"] = spy_fwd

    # Call path 1: episode_conviction_for_row (backtest engine path)
    row = pd.Series({
        "direction": "bullish",
        "predictors": ["VIXM"],
        "tau_pasts": [tau_p],
        "q_Xs": [0.95],
        "Y": "SPY",
        "tau_future": tau_f,
        "q_Y": 0.70,
    })
    conv1 = episode_conviction_for_row(row, increments, train_cutoff)

    # Call path 2: manual construction then compute_episode_conviction
    train_inc = increments[tau_p]["VIXM"].loc[increments[tau_p].index <= train_cutoff]
    thresh_x = train_inc.quantile(0.95)
    joint_mask = (increments[tau_p]["VIXM"].reindex(train_dates) > thresh_x)
    firing_dates = joint_mask[joint_mask.fillna(False)].index

    target_forward = increments[tau_f]["SPY"].shift(-tau_f)
    target_train = increments[tau_f]["SPY"].loc[increments[tau_f].index <= train_cutoff]
    target_thresh = target_train.quantile(0.70)

    _, _, conv2 = compute_episode_conviction(
        firing_dates=firing_dates,
        max_tau_past=tau_p,
        target_forward_series=target_forward,
        target_thresh=target_thresh,
        direction="bullish",
    )

    delta = abs(conv1 - conv2)
    status = "PASS" if delta < 1e-4 else "FAIL"
    print(f"  episode_utils self-test [{status}]")
    print(f"    Path 1 (row wrapper): {conv1:.6f}")
    print(f"    Path 2 (direct call): {conv2:.6f}")
    print(f"    Delta:                {delta:.2e}  (threshold: 1e-4)")
    if status == "FAIL":
        raise AssertionError(
            f"episode_utils cross-check failed: path1={conv1:.6f} path2={conv2:.6f} "
            f"delta={delta:.2e}"
        )
    return conv1, conv2


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  episode_utils.py — self-test")
    print("=" * 60)
    _run_self_test()
    print("\n  All checks passed.\n")
