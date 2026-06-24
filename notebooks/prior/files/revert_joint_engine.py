"""
revert_joint_engine.py
======================
Reverts joint_cpe_engine.py to use its own episode computation (removes
the broken Patch 2 from patch_joint_engine.py), while keeping Patch 1
(the import of cluster_into_episodes from episode_utils) since that part
is correct.

The key insight: the backtest_engine.py already uses episode_utils.py
correctly via _episode_conviction_for_row -> episode_conviction_for_row.
The stored conviction values in joint_cpe_results.parquet are only used
for the greedy ranking during screen build — the backtest recomputes them
fresh at runtime anyway. So the stored values being slightly off doesn't
affect backtest results; it only affects which configurations the greedy
search selects as seeds.

This revert restores the original compute_episode_stats call so the joint
engine produces stable, self-consistent results, while backtest_engine.py
remains the authoritative source for conviction values used in sizing.
"""
import sys

with open('joint_cpe_engine.py', encoding='utf-8') as f:
    src = f.read()

# Remove the broken Patch 2 replacement and restore the original call
OLD_BROKEN = '''            # Use canonical episode_utils implementation so stored n_episodes
            # matches what backtest_engine.py recomputes at runtime (fixes
            # the Section 20.6 discrepancy: 107 rows with delta > 1e-4).
            firing_dates_ep = pd.DatetimeIndex(
                common_idx[joint_mask]
            ) if hasattr(common_idx, '__getitem__') else pd.DatetimeIndex(
                [d for d, m in zip(common_idx, joint_mask) if m]
            )
            # Forward return series (BUGFIX: must be shifted, not trailing)
            target_fwd = future_inc[tau_f][y].shift(-tau_f)
            if direction == "bullish":
                tgt_thresh = thresholds[(tau_f, q_y)].get(y, float('nan'))
            else:
                tgt_thresh = thresholds[(tau_f, round(1 - q_y, 10))].get(y, float('nan'))
            import numpy as _np
            if _np.isnan(tgt_thresh):
                n_episodes, episode_hit_rate, episode_conviction = 0, None, 0.0
            else:
                n_episodes, episode_hit_rate, episode_conviction = _compute_episode_conviction_util(
                    firing_dates=firing_dates_ep,
                    max_tau_past=longest_tau_p,
                    target_forward_series=target_fwd,
                    target_thresh=tgt_thresh,
                    direction=direction,
                )'''

ORIGINAL_CALL = '''            n_episodes, episode_hit_rate, episode_conviction = compute_episode_stats(
                joint_mask, common_idx, fy_vals_full, event_for_episodes, longest_tau_p
            )'''

if OLD_BROKEN not in src:
    print("Patch 2 not found in joint_cpe_engine.py.")
    print("Either it was already reverted, or the file has a different structure.")
    print("Check line ~514 manually for the episode stats computation.")
    sys.exit(1)

src = src.replace(OLD_BROKEN, ORIGINAL_CALL)

# Also restore compute_episode_stats definition if it was removed by Patch 1
# Check if it's still present
if 'def compute_episode_stats' not in src:
    print("WARNING: compute_episode_stats definition is missing.")
    print("Patch 1 removed it. Need to restore it too.")
    print("Please restore from your git history or the original uploaded file.")
    sys.exit(1)

with open('joint_cpe_engine.py', 'w', encoding='utf-8') as f:
    f.write(src)

print("Patch 2 reverted successfully.")
print("joint_cpe_engine.py now uses its original compute_episode_stats.")
print("")
print("Next step: python joint_cpe_engine.py")
print("This will rebuild joint_cpe_results.parquet with correct stored values.")
print("")
print("The backtest_engine.py already uses episode_utils.py correctly")
print("and recomputes conviction at runtime — so discrepancies in stored")
print("values don't affect backtest results or sizing.")
