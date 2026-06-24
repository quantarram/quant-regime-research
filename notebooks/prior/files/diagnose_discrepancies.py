"""
Diagnose the episode_conviction discrepancies between stored parquet values
and episode_utils.py recomputation.
"""
import pandas as pd, numpy as np, sys
sys.path.insert(0, '.')

from backtest_engine import TRAIN_CUTOFF, build_increments_for_episodes
from episode_utils import (
    episode_conviction_for_row, cluster_into_episodes,
    compute_episode_conviction, EPISODE_GAP_MULTIPLIER, EPISODE_MIN_CONVICTION
)

joint  = pd.read_parquet('joint_cpe_results.parquet')
prices = pd.read_parquet('multiasset_prices.parquet')

needed = sorted(set(int(t) for taus in joint['tau_pasts'] for t in taus)
              | set(int(t) for t in joint['tau_future']))
inc = build_increments_for_episodes(prices, needed)

recomp = joint.apply(lambda r: episode_conviction_for_row(r, inc, TRAIN_CUTOFF), axis=1)
stored = joint['episode_conviction'].fillna(0.0)
delta  = (recomp - stored).abs()

bad = joint[delta > 1e-4].copy()
bad['delta']           = delta[delta > 1e-4]
bad['conv_stored']     = stored[delta > 1e-4]
bad['conv_recomputed'] = recomp[delta > 1e-4]

# For each discrepant row, check n_episodes stored vs recomputed
def recompute_n_episodes(row):
    direction  = row['direction']
    predictors = list(row['predictors'])
    tau_pasts  = [int(t) for t in row['tau_pasts']]
    q_xs       = [float(q) for q in row['q_Xs']]
    tau_f      = int(row['tau_future'])

    ref_tau    = tau_pasts[0]
    if ref_tau not in inc:
        return -1
    train_dates = inc[ref_tau].index[inc[ref_tau].index <= TRAIN_CUTOFF]
    joint_mask  = pd.Series(True, index=train_dates)

    for x, tau_p, q_x in zip(predictors, tau_pasts, q_xs):
        if x not in inc[tau_p].columns:
            return -1
        series = inc[tau_p][x].reindex(train_dates)
        train_s = inc[tau_p][x].loc[inc[tau_p].index <= TRAIN_CUTOFF]
        if direction == 'bullish':
            joint_mask &= series > train_s.quantile(q_x)
        else:
            joint_mask &= series < train_s.quantile(round(1 - q_x, 10))

    firing = joint_mask[joint_mask.fillna(False)].index
    if len(firing) == 0:
        return 0
    eps = cluster_into_episodes(firing, max(tau_pasts))
    return len(eps)

print("Diagnosing discrepant rows...\n")
bad['n_episodes_stored']    = bad['n_episodes']
bad['n_episodes_recomputed'] = bad.apply(recompute_n_episodes, axis=1)
bad['episode_count_changed'] = bad['n_episodes_stored'] != bad['n_episodes_recomputed']

print("=== DISCREPANCY SUMMARY ===")
print(f"Total discrepant rows: {len(bad)}")
print(f"Rows where n_episodes ALSO changed: {bad['episode_count_changed'].sum()}")
print(f"Rows where n_episodes same but conviction differs: {(~bad['episode_count_changed']).sum()}")
print()

# Show cases where n_episodes changed
if bad['episode_count_changed'].sum() > 0:
    print("=== ROWS WHERE n_episodes CHANGED ===")
    cols = ['Y','direction','tau_future','n_episodes_stored','n_episodes_recomputed',
            'conv_stored','conv_recomputed','delta']
    print(bad[bad['episode_count_changed']][cols].to_string(index=False))
    print()

# Show cases where n_episodes is the same but conviction still differs
same_ep = bad[~bad['episode_count_changed']].head(10)
if len(same_ep) > 0:
    print("=== ROWS WHERE n_episodes SAME BUT CONVICTION DIFFERS (first 10) ===")
    cols = ['Y','direction','tau_future','q_Y','n_episodes_stored','n_episodes_recomputed',
            'conv_stored','conv_recomputed','delta','predictors']
    print(same_ep[cols].to_string(index=False))
    print()
    # For the first such row, check the hit_rate
    row = same_ep.iloc[0]
    print(f"Deep-dive: Y={row['Y']} dir={row['direction']} tau_f={row['tau_future']}")
    print(f"  stored  conv={row['conv_stored']:.6f}  n_ep={row['n_episodes_stored']}")
    print(f"  recomp  conv={row['conv_recomputed']:.6f}")
    # stored conv = log(n_ep) * agreement -> agreement = conv / log(n_ep)
    n = int(row['n_episodes_stored'])
    if n > 0 and np.log(n) > 0:
        stored_agreement = float(row['conv_stored']) / np.log(n)
        recomp_agreement = float(row['conv_recomputed']) / np.log(n) if np.log(n) > 0 else 0
        print(f"  implied stored agreement : {stored_agreement:.4f}")
        print(f"  implied recomp agreement : {recomp_agreement:.4f}")
        print(f"  -> hit_rate stored  ~ {(stored_agreement + 1)/2:.4f}")
        print(f"  -> hit_rate recomp  ~ {(recomp_agreement + 1)/2:.4f}")
    print()

# Check: does the stored episode_hit_rate column match what we recompute?
if 'episode_hit_rate' in bad.columns:
    print("=== STORED HIT RATE vs IMPLIED FROM CONVICTION ===")
    for _, row in same_ep.head(5).iterrows():
        n = int(row['n_episodes_stored'])
        stored_conv = float(row['conv_stored'])
        if n >= EPISODE_MIN_CONVICTION and np.log(n) > 0:
            implied_hr = (stored_conv / np.log(n) + 1) / 2
            print(f"  Y={row['Y']:<14} stored hit_rate={row['episode_hit_rate']}  "
                  f"implied_from_conv={implied_hr:.4f}  "
                  f"stored_conv={stored_conv:.4f}  n_ep={n}")

print("\n=== DELTA DISTRIBUTION ===")
print(bad['delta'].describe().round(4))
print()
# Are most deltas log(n) values? That would indicate n_episodes difference
log_vals = [np.log(i) for i in range(1, 15)]
print("Common log(n) values for n=1..14:", [round(v, 4) for v in log_vals])
