"""
Event-study / composite analysis: the structural-mechanism step this
program has been missing. Everything so far (Granger causality, quantile
regression) asks "does this variable have a stable statistical relationship
with the target across all days" -- a blanket regression coefficient. This
asks a different, more physically-grounded question, the one that actually
mattered for the precipitation work: what does the ACTUAL TRAJECTORY of the
validated fields look like in the run-up to real, known stress episodes --
is there a genuine structural signature (a joint threshold, a lead-lag
sequence between the two fields) analogous to "convergence intensifying
while moisture builds, in that order, over the preceding hours" -- rather
than treating every day as an interchangeable statistical observation.

Step 1: objectively detect SPY stress episodes (NOT hand-picked) -- any day
whose forward 21-trading-day return is <= -10%, clustered into episodes
(consecutive flagged days within 15 trading days of each other collapse to
one episode, onset = earliest flagged date).

Step 2: for each episode, extract credit_spread_regime and vix_term_slope
from T-60 to T+20 trading days relative to onset, build the cross-episode
composite (average) trajectory.

Step 3: compare the composite against a matched sample of random non-episode
windows -- does the pre-episode signature genuinely stand out, or is it
indistinguishable from typical day-to-day fluctuation?

Step 4: lead-lag structure -- for each episode, find the first day (relative
to onset) each field crosses +1 std; is there a consistent ordering between
the two fields across episodes (one reliably leads the other), the
structural-sequence question that matters for an actual early-warning use.

Run: python 17_stress_episode_composite.py
Output: stress_episode_composite_results.json, stress_episode_composite_plot.png
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FWD_HORIZON = 21
CRISIS_THRESHOLD = -0.10
CLUSTER_GAP = 15
PRE_WINDOW = 250
POST_WINDOW = 20

print("=" * 60)
print("  STRESS-EPISODE COMPOSITE ANALYSIS")
print("=" * 60)

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
spy = prices["SPY"].dropna()
fwd_ret = np.log(spy.shift(-FWD_HORIZON) / spy)

macro = pd.read_parquet(os.path.join(OUT_DIR, "features_macro_interaction.parquet"))
macro["date"] = pd.to_datetime(macro["date"])
credit_spread_regime = macro[["date", "credit_spread_regime"]].drop_duplicates("date").set_index("date")["credit_spread_regime"]

vixy = prices["VIXY"].dropna()
vixm = prices["VIXM"].dropna()
common = vixy.index.intersection(vixm.index)
raw_ratio = np.log(vixm.reindex(common) / vixy.reindex(common))
vix_term_slope = ((raw_ratio - raw_ratio.rolling(200, min_periods=100).mean())
                   / raw_ratio.rolling(200, min_periods=100).std())

d = pd.concat([fwd_ret.rename("fwd_ret"), credit_spread_regime, vix_term_slope.rename("vix_term_slope")], axis=1).dropna()
d = d.reset_index().rename(columns={"index": "date"})
idx = d["date"].values
n = len(d)
print(f"Dataset: n={n}, {d['date'].min()} .. {d['date'].max()}")

# ── Step 1: objectively detect episodes ──────────────────────────────────
flagged = d.index[d["fwd_ret"] <= CRISIS_THRESHOLD].tolist()
print(f"\nFlagged days (fwd 21d return <= {CRISIS_THRESHOLD:.0%}): {len(flagged)}")

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

print(f"Clustered into {len(episodes)} distinct episodes (onset = earliest flagged date in each cluster):")
for pos in episodes:
    print(f"  onset: {d['date'].iloc[pos].date()}  (SPY forward-21d return from here: {d['fwd_ret'].iloc[pos]:.1%})")

# ── Step 2: extract trajectories, build composite ────────────────────────
rel_days = list(range(-PRE_WINDOW, POST_WINDOW + 1))
traj_credit, traj_vix = [], []
valid_episodes = []
for pos in episodes:
    if pos - PRE_WINDOW < 0 or pos + POST_WINDOW >= n:
        print(f"  (skipping episode at {d['date'].iloc[pos].date()}, insufficient window)")
        continue
    traj_credit.append(d["credit_spread_regime"].iloc[pos - PRE_WINDOW: pos + POST_WINDOW + 1].values)
    traj_vix.append(d["vix_term_slope"].iloc[pos - PRE_WINDOW: pos + POST_WINDOW + 1].values)
    valid_episodes.append(d["date"].iloc[pos])

traj_credit = np.array(traj_credit)
traj_vix = np.array(traj_vix)
composite_credit = traj_credit.mean(axis=0)
composite_vix = traj_vix.mean(axis=0)
print(f"\n{len(valid_episodes)} episodes with full window available: {[str(e.date()) for e in valid_episodes]}")

# ── Step 3: compare against random non-episode windows ───────────────────
rng = np.random.default_rng(0)
episode_positions = set(episodes)
candidate_positions = [p for p in range(PRE_WINDOW, n - POST_WINDOW)
                        if not any(abs(p - ep) < PRE_WINDOW + POST_WINDOW for ep in episode_positions)]
n_random = 200
random_positions = rng.choice(candidate_positions, size=min(n_random, len(candidate_positions)), replace=False)
rand_credit = np.array([d["credit_spread_regime"].iloc[p - PRE_WINDOW: p + POST_WINDOW + 1].values for p in random_positions])
rand_vix = np.array([d["vix_term_slope"].iloc[p - PRE_WINDOW: p + POST_WINDOW + 1].values for p in random_positions])
rand_composite_credit = rand_credit.mean(axis=0)
rand_composite_vix = rand_vix.mean(axis=0)
rand_std_credit = rand_credit.std(axis=0)
rand_std_vix = rand_vix.std(axis=0)

print("\n--- Composite at key relative days (episode average vs random-window average +/- 1 std) ---")
for rd in [-250, -200, -150, -120, -100, -80, -60, -40, -20, -10, -5, -1, 0, 5, 10]:
    i = rel_days.index(rd)
    z_credit = (composite_credit[i] - rand_composite_credit[i]) / rand_std_credit[i]
    z_vix = (composite_vix[i] - rand_composite_vix[i]) / rand_std_vix[i]
    print(f"  T{rd:+4d}: credit_spread_regime episode={composite_credit[i]:.3f} vs random={rand_composite_credit[i]:.3f} "
          f"(z={z_credit:+.2f})  |  vix_term_slope episode={composite_vix[i]:.3f} vs random={rand_composite_vix[i]:.3f} (z={z_vix:+.2f})")

# ── Step 4: lead-lag -- first crossing of +1 std, per episode ────────────
print("\n--- Lead-lag: first relative day each field crosses +1 std, per episode ---")
lead_lag_rows = []
for k, ep_date in enumerate(valid_episodes):
    credit_cross = next((rel_days[i] for i in range(len(rel_days)) if traj_credit[k, i] > 1.0), None)
    vix_cross = next((rel_days[i] for i in range(len(rel_days)) if traj_vix[k, i] > 1.0), None)
    print(f"  {ep_date.date()}: credit_spread_regime crosses +1std at T{credit_cross if credit_cross is not None else 'never'}, "
          f"vix_term_slope crosses +1std at T{vix_cross if vix_cross is not None else 'never'}")
    lead_lag_rows.append({"episode": str(ep_date.date()), "credit_cross_day": credit_cross, "vix_cross_day": vix_cross})

results = {
    "episodes": [str(e.date()) for e in valid_episodes],
    "composite_credit": composite_credit.tolist(),
    "composite_vix": composite_vix.tolist(),
    "random_composite_credit": rand_composite_credit.tolist(),
    "random_composite_vix": rand_composite_vix.tolist(),
    "rel_days": rel_days,
    "lead_lag": lead_lag_rows,
}
out_path = os.path.join(OUT_DIR, "stress_episode_composite_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved results to {out_path}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(rel_days, composite_credit, color="tab:red", lw=2, label="Episode composite")
    axes[0].plot(rel_days, rand_composite_credit, color="gray", lw=1, ls="--", label="Random-window composite")
    axes[0].fill_between(rel_days, rand_composite_credit - rand_std_credit, rand_composite_credit + rand_std_credit,
                          color="gray", alpha=0.15)
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_title("credit_spread_regime: composite trajectory around stress-episode onset (T=0)")
    axes[0].legend(fontsize=8)

    axes[1].plot(rel_days, composite_vix, color="tab:blue", lw=2, label="Episode composite")
    axes[1].plot(rel_days, rand_composite_vix, color="gray", lw=1, ls="--", label="Random-window composite")
    axes[1].fill_between(rel_days, rand_composite_vix - rand_std_vix, rand_composite_vix + rand_std_vix,
                          color="gray", alpha=0.15)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_title("vix_term_slope: composite trajectory around stress-episode onset (T=0)")
    axes[1].set_xlabel("Trading days relative to episode onset")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "stress_episode_composite_plot.png"), dpi=120)
    print(f"Saved plot to {os.path.join(OUT_DIR, 'stress_episode_composite_plot.png')}")
except Exception as e:
    print(f"Plot failed (non-fatal): {e}")

print("\nDone.")
