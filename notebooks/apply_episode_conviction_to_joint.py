"""
apply_episode_conviction_to_joint.py
=======================================
Audits every row of joint_cpe_results.parquet against the episode-conviction
correction already built (and validated) in backtest_engine.py for the
pairwise/portfolio-tilt pipeline -- but never applied back to the joint
table itself, and not currently used by build_portfolio_dashboard.py at all
(jcpe is loaded there and never touched again).

Question: of the "top tier" high-predictor-count joint configurations (5-6
simultaneous instruments in their tails at once), how many actually rest on
>=3 genuinely independent historical episodes (the same hard floor already
used elsewhere in this codebase), versus being a single overlapping-window
episode counted as ~100+ nominal daily observations (n_joint)?

Reuses backtest_engine.py's _cluster_into_episodes and
build_increments_for_episodes unchanged. Does NOT reuse
_episode_conviction_for_row as-is, because that function restricts firing
dates and quantile thresholds to backtest_engine.py's TRAIN_CUTOFF
(2024-12-31), which is specific to that module's train/test forward-test
design. This script is a general audit of the raw table as originally
published, so it recomputes firing dates and quantile thresholds over the
FULL price history -- confirmed to match joint_cpe_engine.py's own
generation convention (full-sample increments[tau].quantile(q), not a
train-only split).

Usage:
    ../.venv/bin/python apply_episode_conviction_to_joint.py
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "files")  # backtest_engine.py lives in notebooks/files/
import backtest_engine as _be

EPISODE_MIN_OBS_FOR_CONVICTION = _be.EPISODE_MIN_OBS_FOR_CONVICTION  # 3, hard floor not a discount


def episode_diagnostics_for_row(row, increments) -> dict:
    direction = row["direction"]
    predictors = list(row["predictors"])
    tau_pasts = [int(t) for t in row["tau_pasts"]]
    q_xs = [float(q) for q in row["q_Xs"]]
    y = row["Y"]
    tau_f = int(row["tau_future"])
    q_y = float(row["q_Y"])

    if tau_f not in increments or any(tp not in increments for tp in tau_pasts):
        return dict(n_episodes=0, hit_rate=np.nan, conviction=0.0)

    all_dates = increments[tau_pasts[0]].index
    joint_mask = pd.Series(True, index=all_dates)
    for x, tau_p, q_x in zip(predictors, tau_pasts, q_xs):
        if x not in increments[tau_p].columns:
            return dict(n_episodes=0, hit_rate=np.nan, conviction=0.0)
        series = increments[tau_p][x].reindex(all_dates)
        if direction == "bullish":
            thresh = series.quantile(q_x)
            joint_mask &= (series > thresh)
        else:
            thresh = series.quantile(round(1 - q_x, 10))
            joint_mask &= (series < thresh)

    firing_dates = joint_mask[joint_mask.fillna(False)].index
    if len(firing_dates) == 0:
        return dict(n_episodes=0, hit_rate=np.nan, conviction=0.0)

    episodes = _be._cluster_into_episodes(firing_dates, max(tau_pasts))

    if y not in increments[tau_f].columns:
        return dict(n_episodes=0, hit_rate=np.nan, conviction=0.0)
    target_forward = increments[tau_f][y].shift(-tau_f)
    target_thresh = increments[tau_f][y].quantile(q_y if direction == "bullish" else round(1 - q_y, 10))

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
        return dict(n_episodes=n_episodes,
                     hit_rate=(float(np.mean(outcomes)) if outcomes else np.nan),
                     conviction=0.0)

    hit_rate = float(np.mean(outcomes))
    agreement = max(0.0, 2 * hit_rate - 1)
    conviction = float(np.log(n_episodes) * agreement)
    return dict(n_episodes=n_episodes, hit_rate=hit_rate, conviction=conviction)


def main():
    print("Loading price panel and joint CPE table...")
    prices = pd.read_parquet("multiasset_prices.parquet")
    joint = pd.read_parquet("joint_cpe_results.parquet")
    print(f"  Joint table: {len(joint)} rows")

    needed_taus = sorted(set(
        int(t) for taus in joint["tau_pasts"] for t in taus
    ) | set(int(t) for t in joint["tau_future"]))
    print(f"  Building increments for tau values: {needed_taus} ...")
    increments = _be.build_increments_for_episodes(prices, needed_taus)

    print(f"  Scoring {len(joint)} joint configurations against the episode-conviction "
          f"gate (min {EPISODE_MIN_OBS_FOR_CONVICTION} independent episodes)...")
    results = []
    for i, (idx, row) in enumerate(joint.iterrows()):
        results.append(episode_diagnostics_for_row(row, increments))
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(joint)} done")

    diag_df = pd.DataFrame(results, index=joint.index)
    out = joint.join(diag_df)
    out.to_parquet("joint_cpe_results_episode_corrected.parquet")

    print(f"\n{'='*78}")
    print("  SUMMARY: raw joint_CPE gate (>=0.80, lift>=1.5) vs episode-conviction gate")
    print(f"{'='*78}")
    raw_gated = out[(out["joint_CPE"] >= 0.80) & (out["lift"] >= 1.5)]
    survives = raw_gated[raw_gated["conviction"] > 0]
    print(f"  Rows passing raw gate (joint_CPE>=0.80, lift>=1.5): {len(raw_gated)}")
    print(f"  Of those, rows with >={EPISODE_MIN_OBS_FOR_CONVICTION} independent episodes "
          f"(conviction>0): {len(survives)}")
    print(f"  Survival rate: {len(survives)/len(raw_gated)*100:.1f}%")

    print(f"\n  Breakdown by n_predictors (raw-gated rows):")
    for k in sorted(raw_gated["n_predictors"].unique()):
        sub = raw_gated[raw_gated["n_predictors"] == k]
        surv = sub[sub["conviction"] > 0]
        print(f"    n_predictors={k:>2}: {len(sub):>5} raw-gated -> {len(surv):>5} survive "
              f"({len(surv)/len(sub)*100:5.1f}%), median n_episodes(raw)={sub['n_episodes'].median():.0f}")

    cols = ["Y", "direction", "tau_future", "n_predictors", "n_joint",
            "joint_CPE", "lift", "n_episodes", "hit_rate", "conviction"]

    print(f"\n  High-order (n_predictors>=5) raw-gated configs, best survivors by conviction:")
    high = raw_gated[raw_gated["n_predictors"] >= 5].sort_values("conviction", ascending=False)
    print(high[cols].head(10).to_string())

    print(f"\n  High-order (n_predictors>=5) configs that DON'T survive (conviction==0):")
    high_fail = raw_gated[(raw_gated["n_predictors"] >= 5) & (raw_gated["conviction"] == 0)]
    print(f"  {len(high_fail)} of {len(high[cols])} high-order raw-gated rows fail the episode floor")
    print(high_fail[cols].sort_values("joint_CPE", ascending=False).head(10).to_string())

    print(f"\nSaved full corrected table -> joint_cpe_results_episode_corrected.parquet")


if __name__ == "__main__":
    main()
