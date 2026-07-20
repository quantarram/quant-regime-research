"""
Chasing the credit_spread_regime build-up pattern found in
17_stress_episode_composite.py: a weak (sub-1-std) but real, roughly
monotonic rise from ~T-120 to ~T-10 before 6 of 7 identified stress
episodes. This asks the sharper, more useful question: is it the LEVEL of
credit_spread_regime that matters, or its TREND/velocity over the preceding
~4 months -- the financial analogue of "pressure falling rapidly" mattering
more than "pressure is low" in synoptic forecasting. And critically: is that
trend actually distinguishable from what happens in random, non-crisis
periods, or does a similar-looking rise happen by chance often enough that
it wouldn't be a usable warning signal?

Method (pure statistics, no ML):
1. For each of the 7 usable episodes AND a large sample of random windows,
   compute the OLS trend (slope) and net change of credit_spread_regime over
   the T-120..T-10 sub-window (where the composite build-up looked cleanest).
2. Compare the episode-window slopes against the random-window slope
   distribution with a Mann-Whitney U test (appropriate for n=7) -- is the
   pre-episode trend genuinely more extreme than chance, not just the
   composite average?
3. Show each episode's own trajectory individually -- is the build-up a
   broad, general pattern or driven by one or two episodes?
4. If discriminative: define a concrete threshold rule from the random-window
   distribution (e.g. 90th percentile of slopes) and backtest its historical
   hit rate against all 8 episodes and its false-alarm rate on the rest of
   history -- an honest precision/recall picture for this specific rule,
   before treating it as a usable early-warning signal.

Run: python 18_credit_spread_buildup_rule.py
Output: credit_spread_buildup_results.json
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import mannwhitneyu, linregress

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FWD_HORIZON = 21
CRISIS_THRESHOLD = -0.10
CLUSTER_GAP = 15
TREND_START, TREND_END = -120, -10   # the sub-window where the build-up looked cleanest
N_RANDOM = 500

print("=" * 60)
print("  CHASING THE credit_spread_regime BUILD-UP PATTERN")
print("=" * 60)

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
spy = prices["SPY"].dropna()
fwd_ret = np.log(spy.shift(-FWD_HORIZON) / spy)

macro = pd.read_parquet(os.path.join(OUT_DIR, "features_macro_interaction.parquet"))
macro["date"] = pd.to_datetime(macro["date"])
credit = macro[["date", "credit_spread_regime"]].drop_duplicates("date").set_index("date")["credit_spread_regime"]

vixy = prices["VIXY"].dropna()
vixm = prices["VIXM"].dropna()
common = vixy.index.intersection(vixm.index)
raw_ratio = np.log(vixm.reindex(common) / vixy.reindex(common))
vix_term_slope = ((raw_ratio - raw_ratio.rolling(200, min_periods=100).mean())
                   / raw_ratio.rolling(200, min_periods=100).std())

d = pd.concat([fwd_ret.rename("fwd_ret"), credit, vix_term_slope.rename("vix_term_slope")], axis=1).dropna()
d = d.reset_index().rename(columns={"index": "date"})
n = len(d)

flagged = d.index[d["fwd_ret"] <= CRISIS_THRESHOLD].tolist()
episodes = []
if flagged:
    cluster = [flagged[0]]
    for pos in flagged[1:]:
        if pos - cluster[-1] <= CLUSTER_GAP:
            cluster.append(pos)
        else:
            episodes.append(cluster[0])
            cluster = [pos]
    episodes.append(cluster[0])

PRE_WINDOW = 250
valid_episodes = [pos for pos in episodes if pos - PRE_WINDOW >= 0]
print(f"{len(valid_episodes)} usable episodes: {[str(d['date'].iloc[p].date()) for p in valid_episodes]}")


def window_trend(series_values, pos, start_off, end_off):
    seg = series_values[pos + start_off: pos + end_off + 1]
    x = np.arange(len(seg))
    slope, intercept, r, p, se = linregress(x, seg)
    net_change = seg[-1] - seg[0]
    return slope, net_change


credit_vals = d["credit_spread_regime"].values

print(f"\n--- Individual episode trajectories, credit_spread_regime trend over T{TREND_START}..T{TREND_END} ---")
episode_slopes, episode_changes = [], []
for pos in valid_episodes:
    slope, net_change = window_trend(credit_vals, pos, TREND_START, TREND_END)
    episode_slopes.append(slope)
    episode_changes.append(net_change)
    val_start = credit_vals[pos + TREND_START]
    val_end = credit_vals[pos + TREND_END]
    print(f"  {d['date'].iloc[pos].date()}: value at T{TREND_START}={val_start:.3f}, at T{TREND_END}={val_end:.3f}, "
          f"net_change={net_change:+.3f}, slope={slope:+.5f}/day")

rng = np.random.default_rng(0)
episode_set = set(episodes)
candidate_positions = [p for p in range(abs(TREND_START), n - 1)
                        if p + TREND_END < n and not any(abs(p - ep) < 250 for ep in episode_set)]
random_positions = rng.choice(candidate_positions, size=min(N_RANDOM, len(candidate_positions)), replace=False)

random_slopes, random_changes = [], []
for pos in random_positions:
    slope, net_change = window_trend(credit_vals, pos, TREND_START, TREND_END)
    random_slopes.append(slope)
    random_changes.append(net_change)

episode_slopes, episode_changes = np.array(episode_slopes), np.array(episode_changes)
random_slopes, random_changes = np.array(random_slopes), np.array(random_changes)

print(f"\n--- Comparison: episode slopes vs {len(random_slopes)} random-window slopes ---")
print(f"  Episode slopes: mean={episode_slopes.mean():.5f}, median={np.median(episode_slopes):.5f}")
print(f"  Random slopes:  mean={random_slopes.mean():.5f}, median={np.median(random_slopes):.5f}, "
      f"std={random_slopes.std():.5f}")
u_stat, u_pval = mannwhitneyu(episode_slopes, random_slopes, alternative="greater")
print(f"  Mann-Whitney U test (episode slopes > random slopes): U={u_stat:.1f}, p={u_pval:.4f}")

pctile_rank = [float((random_slopes < s).mean()) for s in episode_slopes]
print(f"  Each episode's slope, as a percentile of the random-window slope distribution:")
for pos, pr in zip(valid_episodes, pctile_rank):
    print(f"    {d['date'].iloc[pos].date()}: percentile={pr:.1%}")

# ── Threshold rule + historical backtest ──────────────────────────────────
print("\n--- Threshold rule backtest ---")
for pctile in [75, 90]:
    threshold = np.percentile(random_slopes, pctile)
    print(f"\n  Rule: credit_spread_regime's {abs(TREND_START)-abs(TREND_END)}-day trailing slope > "
          f"{threshold:.5f} (the {pctile}th percentile of random-window slopes)")
    hits = 0
    for pos in valid_episodes:
        slope, _ = window_trend(credit_vals, pos, TREND_START, TREND_END)
        flagged_rule = slope > threshold
        hits += int(flagged_rule)
        print(f"    {d['date'].iloc[pos].date()}: slope={slope:.5f} -> {'FLAGGED' if flagged_rule else 'missed'}")
    hit_rate = hits / len(valid_episodes)

    # false-alarm rate: rolling slope computed on EVERY day in history, what fraction of ALL days
    # (excluding episode windows) would have been flagged
    all_slopes = []
    for pos in range(abs(TREND_START), n - abs(TREND_END)):
        if any(abs(pos - ep) < 250 for ep in episode_set):
            continue
        seg = credit_vals[pos + TREND_START: pos + TREND_END + 1]
        x = np.arange(len(seg))
        s, *_ = linregress(x, seg)
        all_slopes.append(s)
    all_slopes = np.array(all_slopes)
    false_alarm_rate = float((all_slopes > threshold).mean())
    print(f"    Hit rate: {hits}/{len(valid_episodes)} = {hit_rate:.1%}. "
          f"False-alarm rate across all non-episode history: {false_alarm_rate:.1%} of days")

results = {
    "episodes": [str(d["date"].iloc[p].date()) for p in valid_episodes],
    "episode_slopes": episode_slopes.tolist(),
    "episode_changes": episode_changes.tolist(),
    "random_slope_mean": float(random_slopes.mean()),
    "random_slope_std": float(random_slopes.std()),
    "mannwhitney_p": float(u_pval),
    "episode_percentiles": pctile_rank,
}
out_path = os.path.join(OUT_DIR, "credit_spread_buildup_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved results to {out_path}")
print("\nDone.")
