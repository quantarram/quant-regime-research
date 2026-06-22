"""
Joint CPE Engine — Greedy Predictor Selection (Clean v2 + Economic Prior)
==========================================================================
Fixes from v1:
  - Deduplicate: same ticker cannot appear twice in predictor set
  - Leveraged/inverse ETFs excluded from X as well as Y
  - Greedy stops at MAX_PREDICTORS (default 6)

ECONOMIC PRIOR (new):
  This engine builds joint sets from cpe_results.parquet, which (once
  produced by the updated cpe_engine_parallel.py) already only contains
  economically admissible (X, Y) pairs -- so the candidate pool the
  greedy search draws from is restricted at the source. A defensive,
  redundant check is also applied directly in the greedy loop itself
  (economic_prior.is_admissible), so this engine produces correct,
  restricted output even if pointed at an older, unrestricted
  cpe_results.parquet by mistake. See economic_prior.py for the full
  channel list and the Portfolio Tilt paper's Sections 11-14 for why
  this restriction is necessary: no purely statistical correction
  (shrinkage, sample-size filtering, episode-independence filtering)
  could separate a spurious-but-recurring relationship (silver predicting
  dogecoin, 5 episodes) from a genuine one (volatility predicting
  equities) using price data alone.

For each (Y, tau_future, q_Y, direction):
  1. Load pairwise signals that passed filters for this Y
  2. Greedy: start with best single predictor (highest pairwise CPE)
  3. At each step, try adding each remaining predictor (unique tickers only):
     - Compute joint conditioning event (intersection of all predictor conditions)
     - Keep if n_joint >= MIN_N and joint CPE >= CPE_THRESH and lift >= MIN_LIFT
     - Select the addition that maximises joint CPE
  4. Stop when no predictor can be added OR n_predictors == MAX_PREDICTORS
  5. Save all intermediate joint sets (size 2, 3, ..., MAX_PREDICTORS) that pass filters

Output: joint_cpe_results.parquet
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings, os, time
warnings.filterwarnings("ignore")

import os as _os_for_prior_switch
if _os_for_prior_switch.environ.get("USE_PRIOR_BYPASS") == "1":
    print("  *** USE_PRIOR_BYPASS=1: economic prior DISABLED, running unrestricted (A/B comparison arm) ***")
    from economic_prior_BYPASS import is_admissible
else:
    from economic_prior import is_admissible

# ── CONFIG ────────────────────────────────────────────────────────────────────
import os as _os
MIN_N          = 100
CPE_THRESH     = 0.80
MIN_LIFT       = 1.5
MAX_PREDICTORS = 10
MIN_CONFIDENCE = _os.environ.get("MIN_CONFIDENCE", "weak")

# ── EPISODE-AWARE GREEDY RANKING (new) ───────────────────────────────────
# The greedy search previously ranked every candidate purely by raw CPE
# (sort_values("CPE", ascending=False), and "cpe > best_cpe" as the
# inner-loop tiebreak). This was found, during backtest investigation, to
# silently prefer single-episode or two-episode predictors (e.g. VIXM,
# whose entire firing history clusters into one continuous COVID-era
# stretch) over multi-episode predictors with an identical or lower raw
# CPE (e.g. ITB, whose 215 firing days cluster into 4 genuinely separate
# historical episodes with a 3/3 favourable outcome) -- because CPE alone
# cannot distinguish "tied at 1.0000 because of one long episode" from
# "tied at 1.0000 because of several independently-confirming episodes".
# Downstream, this meant compute_quality_weights' episode-conviction
# scoring (backtest_engine.py) had nothing to reward: every joint
# configuration the greedy search actually produced for the tradeable
# sleeve proxies happened to be seeded by a thin, single-episode
# predictor that the search had no way of deprioritising.
#
# Fix: candidates are now ranked first by EPISODE COUNT (more genuinely
# independent historical occurrences = preferred), with raw CPE used
# only as the tiebreaker among candidates with the same episode count --
# not the other way around. Set EPISODE_AWARE_RANKING=0 to recover the
# exact original CPE-only ranking for direct comparison.
EPISODE_AWARE_RANKING = _os.environ.get("EPISODE_AWARE_RANKING", "1") == "1"
EPISODE_GAP_MULTIPLIER = 1.5

RATE_INDEX_TICKERS = {
    "^VIX","^VXN","^OVX","^GVZ","^EVZ","^VVIX","^SKEW",
    "^TNX","^TYX","^FVX","^IRX"
}

# ── BUGFIX (found during backtest-reproducibility check) ────────────────
# The original joint_cpe_engine.py applied a single EXCLUDE_TICKERS set to
# BOTH the predicted (Y) side AND the predictor (X) side. This is
# inconsistent with cpe_engine_parallel.py, which correctly distinguishes
# EXCLUDE_FROM_Y (leveraged/inverse ETFs and decaying vol ETPs -- bad
# TARGETS because their price action is mechanically derivative, not an
# independent signal) from EXCLUDE_FROM_X (only managed currencies --
# poor data quality). VIXM/VIXY/VXX/UVXY/SVXY were never intended to be
# excluded as PREDICTORS, and cpe_engine_parallel.py does not exclude
# them from X. The pairwise screen therefore correctly retains them as
# strong SPY predictors (CPE up to 0.977 for VIXM at the exact horizon
# the Portfolio Tilt paper's Section 10.1 traces), but the OLD joint
# engine silently dropped them from the candidate pool before the greedy
# search ever ran -- which is why joint_cpe_results.parquet contained no
# VIXM/VIXY-driven SPY configuration anywhere, at any predictor-set size.
# This made the paper's central traced example (11/11 correct forward
# returns from that exact configuration) impossible to reproduce from
# the uploaded pairwise+joint pipeline as originally written.
#
# Fix: split into EXCLUDE_FROM_Y (unchanged, still excludes leveraged/
# inverse ETFs and vol ETPs as bad targets) and EXCLUDE_FROM_X (matches
# cpe_engine_parallel.py exactly -- managed currencies only).
EXCLUDE_FROM_Y = {
    "SSO","SDS","TQQQ","TMF","TBT","TBF",
    "UVXY","SVXY","VIXY","VIXM","VXX",
    "THBUSD=X","CNYUSD=X","KRWUSD=X",
}
EXCLUDE_FROM_X = {
    "THBUSD=X","CNYUSD=X","KRWUSD=X",
}
# Backward-compatible alias for any code below that still refers to the
# old combined name in a Y-side context.
EXCLUDE_TICKERS = EXCLUDE_FROM_Y

# ── EPISODE-BASED CONVICTION (replaces ln(n_joint) in downstream sizing) ──
# n_joint counts overlapping daily conditioning observations -- the
# Atlas paper's own Section 3.3, and every paper in this series since,
# has flagged that this overstates independent support. A 252-day
# trailing window stepped one day at a time can turn 3-4 genuinely
# distinct historical episodes into "150 observations". This block adds
# n_episodes and episode_hit_rate to every surviving joint configuration,
# computed directly from the SAME joint_mask/common_idx already built by
# the greedy loop below (no extra increments rebuild needed), and an
# EPISODE_GAP_MULTIPLIER-based clustering convention matching the
# Portfolio Tilt paper's own Section 14 episode-independence filter.
EPISODE_GAP_MULTIPLIER = float(os.environ.get("EPISODE_GAP_MULTIPLIER", 1.5))
MIN_EPISODES_FOR_CONVICTION = int(os.environ.get("MIN_EPISODES_FOR_CONVICTION", 3))


def cluster_into_episodes(firing_dates, gap_trading_days, gap_multiplier=EPISODE_GAP_MULTIPLIER):
    """Group sorted firing dates into episodes; a new episode begins
    whenever the gap since the previous firing date exceeds
    gap_multiplier * gap_trading_days (converted to calendar days via
    the same 1.45 trading-to-calendar-day convention used elsewhere in
    this pipeline, e.g. the hold-to-horizon expiry calculation)."""
    if len(firing_dates) == 0:
        return []
    dates = pd.DatetimeIndex(sorted(firing_dates))
    gap_calendar_days = gap_trading_days * 1.45 * gap_multiplier
    episodes = []
    current = [dates[0]]
    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i - 1]).days
        if gap > gap_calendar_days:
            episodes.append(pd.DatetimeIndex(current))
            current = [dates[i]]
        else:
            current.append(dates[i])
    episodes.append(pd.DatetimeIndex(current))
    return episodes


def compute_episode_stats(joint_mask, common_idx, fy_vals, event_mask, longest_tau_p):
    """
    Given the final accepted joint_mask (boolean array aligned to
    common_idx) for a configuration, cluster its firing dates into
    independent episodes and compute the per-episode outcome rate. The
    outcome for an episode is evaluated at its LAST firing date (most
    information-rich, and avoids letting an episode's later days
    preview an outcome not yet knowable when the episode began).

    Returns (n_episodes, episode_hit_rate, episode_conviction).
    episode_conviction is the direct drop-in replacement for
    ln(n_joint): 0.0 below MIN_EPISODES_FOR_CONVICTION regardless of
    hit rate (a hard floor, not a discount -- mirrors the MIN_TRAIN_OBS
    fix's lesson that discounting a thin signal still lets it fire,
    excluding it does not), and above that floor, log(n_episodes)
    scaled by (2*hit_rate - 1) so a configuration right only half the
    time across many episodes still earns zero credit.
    """
    firing_idx = np.where(np.asarray(joint_mask))[0]
    if len(firing_idx) == 0:
        return 0, np.nan, 0.0

    firing_dates = common_idx[firing_idx]
    episodes = cluster_into_episodes(firing_dates, longest_tau_p)

    # Map each episode's last date back to its position in common_idx to
    # read off that date's forward-outcome event flag.
    date_to_pos = {d: i for i, d in enumerate(common_idx)}
    outcomes = []
    for ep in episodes:
        last_date = ep[-1]
        pos = date_to_pos.get(last_date)
        if pos is None or pos >= len(event_mask):
            continue
        val = event_mask[pos]
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            outcomes.append(bool(val))

    n_episodes = len(outcomes)
    if n_episodes == 0:
        return 0, np.nan, 0.0
    hit_rate = float(np.mean(outcomes))

    if n_episodes < MIN_EPISODES_FOR_CONVICTION:
        conviction = 0.0
    else:
        agreement = max(0.0, 2 * hit_rate - 1)  # 0 at 50% hit rate, 1 at 100%
        conviction = float(np.log(n_episodes) * agreement)

    return n_episodes, hit_rate, conviction



print(f"\n{'='*65}")
print(f"  JOINT CPE ENGINE (GREEDY v2)  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  MAX_PREDICTORS={MAX_PREDICTORS}  MIN_N={MIN_N}  CPE_THRESH={CPE_THRESH}  MIN_LIFT={MIN_LIFT}  MIN_CONFIDENCE={MIN_CONFIDENCE}")
print(f"{'='*65}")

# ── LOAD PRICES ───────────────────────────────────────────────────────────────
print("\n  Loading price data...")
prices = pd.read_parquet("multiasset_prices.parquet")
all_tickers   = list(prices.columns)
price_tickers = [t for t in all_tickers if t not in RATE_INDEX_TICKERS]
rate_tickers  = [t for t in all_tickers if t in RATE_INDEX_TICKERS]

# ── PRE-COMPUTE INCREMENTS ────────────────────────────────────────────────────
TAU_LIST = [1, 5, 10, 21, 63, 126, 252, 300]
Q_GRID   = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]

print("  Pre-computing increments...")
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

print("  Pre-computing forward increments...")
predicted_tickers = [t for t in price_tickers if t not in EXCLUDE_TICKERS]
future_inc = {}
for tau_f in TAU_LIST:
    future_inc[tau_f] = increments[tau_f][predicted_tickers].shift(-tau_f)

print("  Pre-computing quantile thresholds...")
# ── BUGFIX #2 (found while diagnosing episode-conviction handoff to
# backtest_engine.py) ──────────────────────────────────────────────────
# This engine previously computed quantile thresholds from the FULL
# price history (increments[tau].quantile(q), no date restriction),
# while cpe_engine_parallel.py and backtest_engine.py both correctly
# restrict threshold estimation to training-period data only
# (<=2024-12-31). This is a genuine train/test leakage in the joint
# screen: every joint_CPE value this engine has ever produced was
# computed against thresholds that had already seen 2025+ data, not
# thresholds frozen before the evaluation window began.
#
# This was masked for a long time because joint_cpe_engine.py's OWN
# greedy search and joint_CPE computation don't directly feed into a
# point-in-time backtest decision the way the pairwise screen's
# thresholds do -- but it surfaced concretely once episode counting
# needed to agree with backtest_engine.py's independently-computed
# conviction: the two engines were clustering different firing dates
# into different episodes because they were thresholding against
# different historical windows, silently producing inconsistent
# episode counts for the identical configuration.
#
# Fixed: thresholds (and therefore firing dates, episode counts, and
# joint_CPE itself) are now computed from training-period data only,
# exactly matching the convention used everywhere else in this
# pipeline.
TRAIN_CUTOFF = pd.Timestamp("2024-12-31")
full_q_grid = sorted(set(Q_GRID + [round(1 - q, 10) for q in Q_GRID]))
thresholds = {}
train_mask = increments[TAU_LIST[0]].index <= TRAIN_CUTOFF
for tau in TAU_LIST:
    train_inc = increments[tau].loc[increments[tau].index <= TRAIN_CUTOFF]
    for q in full_q_grid:
        thresholds[(tau, q)] = train_inc.quantile(q, numeric_only=True)

# ── LOAD PAIRWISE CPE RESULTS ─────────────────────────────────────────────────
print("\n  Loading pairwise CPE results...")
pairwise = pd.read_parquet("cpe_results.parquet")
n_before_prior = len(pairwise)

# Remove excluded tickers from predictor side -- EXCLUDE_FROM_X only
# (managed currencies), NOT EXCLUDE_FROM_Y. This is the actual bugfix:
# the original code used the combined EXCLUDE_TICKERS here, which
# incorrectly stripped VIXM/VIXY/VXX/UVXY/SVXY from the predictor pool.
pairwise = pairwise[~pairwise["X"].isin(EXCLUDE_FROM_X)].copy()

# Defensive economic-prior re-check. If cpe_results.parquet was already
# produced by the prior-gated pairwise engine, this is a no-op (every row
# already satisfies is_admissible). It exists so this engine cannot
# silently build joint sets from an older, unrestricted pairwise file.
prior_mask = pairwise.apply(lambda r: is_admissible(r["X"], r["Y"], MIN_CONFIDENCE), axis=1)
n_rejected_by_prior = (~prior_mask).sum()
pairwise = pairwise[prior_mask].copy()

print(f"  Pairwise rows loaded             : {n_before_prior:,}")
print(f"  Rejected by economic prior       : {n_rejected_by_prior:,} "
      f"({'this should be 0 if input already prior-gated' if n_rejected_by_prior else 'input was already prior-gated'})")
print(f"  Pairwise rows after all filters   : {len(pairwise):,}")
print(f"  Directions: {pairwise['direction'].value_counts().to_dict()}")

# Diagnostic only -- this engine applies no MIN_TRAIN_OBS filter of its
# own; it inherits whatever cpe_engine_parallel.py already excluded as
# predictors. This check makes that inheritance visible rather than
# silent: if these tickers are absent from the pairwise X column, the
# upstream hard history filter is doing its job and no further action
# is needed here.
_short_history_watchlist = ["IBIT", "FBTC", "BITB"]
_present = sorted(set(pairwise["X"]) & set(_short_history_watchlist))
if _present:
    print(f"  NOTE: {_present} still present as predictors in the loaded "
          f"pairwise file -- the upstream MIN_TRAIN_OBS filter in "
          f"cpe_engine_parallel.py was not applied (or was set low enough "
          f"to admit them) when cpe_results.parquet was generated.")
else:
    print(f"  Short-history watchlist {_short_history_watchlist} absent from "
          f"predictor (X) column -- upstream MIN_TRAIN_OBS filter is active.")

# ── HELPER: GET CONDITION MASK ────────────────────────────────────────────────
def get_condition_mask(x, tau_p, q_x, direction, common_idx):
    if x not in increments[tau_p].columns:
        return None
    px = increments[tau_p][x].loc[common_idx].values
    valid = ~np.isnan(px)
    thresh_up = thresholds[(tau_p, q_x)].get(x, np.nan)
    thresh_dn = thresholds[(tau_p, round(1 - q_x, 10))].get(x, np.nan)
    if direction == "bullish":
        if np.isnan(thresh_up): return None
        return valid & (px > thresh_up)
    else:
        if np.isnan(thresh_dn): return None
        return valid & (px < thresh_dn)


# ── HELPER: EPISODE COUNT (new) ────────────────────────────────────────────
# Counts genuinely independent historical episodes within a candidate's
# OWN firing mask, using the same gap-based clustering convention as
# backtest_engine.py's _cluster_into_episodes (gap > 1.5x the
# conditioning window). Cached per (x, tau_p, q_x, direction) tuple
# within a single engine run, since the same candidate often reappears
# across many (Y, tau_future, q_Y) groups and recomputing its episode
# count from scratch each time would be wasteful.
_episode_count_cache = {}

def get_episode_count(x, tau_p, q_x, direction, common_idx):
    cache_key = (x, tau_p, q_x, direction)
    if cache_key in _episode_count_cache:
        return _episode_count_cache[cache_key]

    mask = get_condition_mask(x, tau_p, q_x, direction, common_idx)
    if mask is None:
        _episode_count_cache[cache_key] = 0
        return 0

    firing_dates = common_idx[mask]
    if len(firing_dates) == 0:
        _episode_count_cache[cache_key] = 0
        return 0

    dates = pd.DatetimeIndex(sorted(firing_dates))
    gap_calendar_days = tau_p * 1.45 * EPISODE_GAP_MULTIPLIER
    n_episodes = 1
    for i in range(1, len(dates)):
        if (dates[i] - dates[i - 1]).days > gap_calendar_days:
            n_episodes += 1

    _episode_count_cache[cache_key] = n_episodes
    return n_episodes


def rank_key(row, common_idx):
    """
    Sort key for candidate ranking. Returns (episode_count, CPE) so that
    candidates are compared FIRST by how many genuinely independent
    episodes their own firing history clusters into, and only THEN by
    raw CPE as a tiebreaker among candidates with equal episode counts.
    When EPISODE_AWARE_RANKING is disabled, falls back to (CPE,) alone,
    exactly reproducing the original engine's behaviour.
    """
    if not EPISODE_AWARE_RANKING:
        return (row["CPE"],)
    n_ep = get_episode_count(row["X"], int(row["tau_past"]), row["q_X"], row["direction"], common_idx)
    return (n_ep, row["CPE"])


# ── GREEDY JOINT CPE ──────────────────────────────────────────────────────────
results = []
t0 = time.time()

groups = pairwise.groupby(["Y", "tau_future", "q_Y", "direction"])
print(f"\n  Total groups: {len(groups)}")
print(f"\n  Running greedy (max {MAX_PREDICTORS} predictors per set)...\n")

n_groups = 0
n_joint  = 0

for (y, tau_f, q_y, direction), group in groups:
    n_groups += 1

    if n_groups % 200 == 0:
        elapsed = time.time() - t0
        print(f"  [{n_groups:>5}/{len(groups)}]  "
              f"joint results: {n_joint:>5,}  "
              f"elapsed: {elapsed:.0f}s", end="\r")

    # Common date index (computed BEFORE candidate ranking now, since
    # episode-aware ranking needs it to count episodes per candidate).
    # Restricted to training-period dates (<=2024-12-31), consistent
    # with the threshold-computation fix above: joint_CPE itself, not
    # just episode counts, must be estimated only from data the
    # downstream backtest is allowed to have seen before its 2025
    # evaluation window begins.
    fy_series  = future_inc[tau_f][y].dropna()
    common_idx = fy_series.index
    for tau_p in TAU_LIST:
        common_idx = common_idx.intersection(
            increments[tau_p].dropna(how="all").index)
    common_idx = common_idx[common_idx <= TRAIN_CUTOFF]
    if len(common_idx) < MIN_N:
        continue

    # Rank candidates by episode count first, raw CPE second (see
    # EPISODE_AWARE_RANKING comment above) -- this replaces the original
    # CPE-only sort, which could not distinguish a candidate whose CPE=1.0
    # rests on one continuous historical episode from one whose CPE=1.0
    # rests on several genuinely independent episodes that all agreed.
    candidates = group.reset_index(drop=True).copy()
    candidates["_rank_key"] = candidates.apply(lambda r: rank_key(r, common_idx), axis=1)
    candidates = candidates.sort_values("_rank_key", ascending=False).reset_index(drop=True)
    if len(candidates) < 2:
        continue

    uncond_prob = 1.0 - q_y

    # Greedy selection
    # selected: list of dicts with keys X, tau_past, q_X, CPE
    # selected_tickers: set of unique tickers already selected
    first_row   = candidates.iloc[0]
    selected    = [first_row.to_dict()]
    selected_tickers = {first_row["X"]}

    # Pre-compute joint mask for selected set
    joint_mask = get_condition_mask(
        first_row["X"], int(first_row["tau_past"]),
        first_row["q_X"], direction, common_idx)
    if joint_mask is None:
        continue

    while len(selected) < MAX_PREDICTORS:
        best_cpe  = -1
        best_rank_key = (-1, -1)
        best_row  = None
        best_mask = None
        best_n    = 0

        # Note: every row in `candidates` already satisfies
        # is_admissible(X, Y) for this group's target Y, because
        # `candidates` is built from the pre-filtered `pairwise` frame.
        # We do not additionally require admissibility BETWEEN the
        # predictors being jointly selected (e.g. VIXM joining a set that
        # already contains UVXY) -- two predictors reinforcing the same
        # validated channel toward the same target is not the pathology
        # this prior exists to stop; a predictor with no relationship to
        # the target at all is. The economic_prior module governs
        # predictor->target admissibility only, by design.
        for _, cand in candidates.iterrows():
            # Skip if ticker already in selected set
            if cand["X"] in selected_tickers:
                continue

            cand_mask = get_condition_mask(
                cand["X"], int(cand["tau_past"]),
                cand["q_X"], direction, common_idx)
            if cand_mask is None:
                continue

            trial_mask = joint_mask & cand_mask
            n_trial    = trial_mask.sum()
            if n_trial < MIN_N:
                continue

            # Compute joint CPE
            fy_vals = future_inc[tau_f][y].loc[common_idx].values
            thresh_y_up = thresholds[(tau_f, q_y)].get(y, np.nan)
            thresh_y_dn = thresholds[(tau_f, round(1 - q_y, 10))].get(y, np.nan)
            if direction == "bullish":
                if np.isnan(thresh_y_up): continue
                event = fy_vals > thresh_y_up
            else:
                if np.isnan(thresh_y_dn): continue
                event = fy_vals < thresh_y_dn

            cpe  = float(np.nanmean(event[trial_mask]))
            lift = cpe / uncond_prob if uncond_prob > 0 else np.nan

            if cpe >= CPE_THRESH and lift >= MIN_LIFT:
                cand_rank_key = rank_key(cand, common_idx)
                if cand_rank_key > best_rank_key:
                    best_cpe  = cpe
                    best_rank_key = cand_rank_key
                    best_row  = cand
                    best_mask = trial_mask
                    best_n    = int(n_trial)

        if best_row is None:
            break  # no valid addition found

        # Accept best addition
        selected.append(best_row.to_dict())
        selected_tickers.add(best_row["X"])
        joint_mask = best_mask

        # Save this joint set (size >= 2)
        if len(selected) >= 2:
            lift = best_cpe / uncond_prob

            # Episode-based conviction, computed from the SAME joint_mask
            # and event array already built above for this accepted
            # configuration -- no extra recomputation of increments.
            fy_vals_full = future_inc[tau_f][y].loc[common_idx].values
            thresh_y_up = thresholds[(tau_f, q_y)].get(y, np.nan)
            thresh_y_dn = thresholds[(tau_f, round(1 - q_y, 10))].get(y, np.nan)
            if direction == "bullish":
                event_for_episodes = fy_vals_full > thresh_y_up
            else:
                event_for_episodes = fy_vals_full < thresh_y_dn
            longest_tau_p = max(int(r["tau_past"]) for r in selected)
            n_episodes, episode_hit_rate, episode_conviction = compute_episode_stats(
                joint_mask, common_idx, fy_vals_full, event_for_episodes, longest_tau_p
            )

            results.append({
                "Y":             y,
                "direction":     direction,
                "tau_future":    int(tau_f),
                "q_Y":           q_y,
                "n_predictors":  len(selected),
                "predictors":    [r["X"]         for r in selected],
                "tau_pasts":     [int(r["tau_past"]) for r in selected],
                "q_Xs":          [r["q_X"]        for r in selected],
                "pairwise_CPEs": [round(float(r["CPE"]), 4) for r in selected],
                "joint_CPE":     round(best_cpe, 4),
                "uncond_prob":   round(uncond_prob, 4),
                "lift":          round(lift, 4),
                "n_joint":       best_n,
                "n_total":       len(common_idx),
                "n_episodes":          n_episodes,
                "episode_hit_rate":    None if np.isnan(episode_hit_rate) else round(episode_hit_rate, 4),
                "episode_conviction":  round(episode_conviction, 4),
            })
            n_joint += 1

elapsed = time.time() - t0
print(f"\n\n  Done. Elapsed: {elapsed:.0f}s  ({elapsed/60:.1f} min)")
print(f"  Groups processed : {n_groups:,}")
print(f"  Joint results    : {n_joint:,}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
if results:
    df = pd.DataFrame(results)
    df = df.sort_values(
        ["direction","n_predictors","joint_CPE","n_joint"],
        ascending=[True,True,False,False]
    ).reset_index(drop=True)

    out = "joint_cpe_results.parquet"
    df.to_parquet(out, engine="pyarrow", compression="snappy", index=False)

    print(f"\n{'='*65}")
    print(f"  COMPLETE")
    print(f"  Saved {len(df):,} rows → {out}  ({os.path.getsize(out)/1e6:.2f} MB)")

    print(f"\n  Direction × n_predictors breakdown:")
    print(df.groupby(["direction","n_predictors"]).size()
            .unstack(fill_value=0).to_string())

    print(f"\n  Mean joint CPE by n_predictors:")
    print(df.groupby(["direction","n_predictors"])["joint_CPE"]
            .mean().round(4).unstack().to_string())

    print(f"\n  Episode-based conviction summary (MIN_EPISODES_FOR_CONVICTION={MIN_EPISODES_FOR_CONVICTION}):")
    n_zero_conviction = (df["episode_conviction"] == 0).sum()
    print(f"    Configs with ZERO episode conviction (< {MIN_EPISODES_FOR_CONVICTION} independent "
          f"episodes, or 50%-or-worse agreement across them): {n_zero_conviction:,} / {len(df):,} "
          f"({100*n_zero_conviction/len(df):.1f}%)")
    print(f"    n_episodes distribution: "
          f"{df['n_episodes'].describe()[['min','25%','50%','75%','max']].round(1).to_dict()}")
    nonzero = df[df["episode_conviction"] > 0]
    print(f"    Configs surviving with NONZERO episode conviction: {len(nonzero):,}")

    for direction in ["bullish","bearish"]:
        print(f"\n  Top 10 {direction} (size=2, ranked by EPISODE CONVICTION, "
              f"not raw joint_CPE/n_joint -- this is the practical effect of the change):")
        top = (df[(df["direction"]==direction) & (df["n_predictors"]==2)]
               .sort_values(["episode_conviction","n_episodes"], ascending=[False,False])
               .head(10))
        for _, r in top.iterrows():
            print(f"    Y={r['Y']:<14} τf={r['tau_future']:>3}  qY={r['q_Y']}"
                  f"  CPE={r['joint_CPE']:.4f}  n_joint={r['n_joint']}  "
                  f"n_episodes={r['n_episodes']}  hit_rate={r['episode_hit_rate']}  "
                  f"conviction={r['episode_conviction']:.3f}"
                  f"  predictors={list(zip(r['predictors'],r['tau_pasts'],r['q_Xs']))}")

    print(f"{'='*65}\n")
else:
    print("\n  No joint results passed the filters.")

if __name__ == "__main__":
    pass
