"""
restore_joint_engine.py
=======================
Restores joint_cpe_engine.py to a clean working state:
  - Keeps Patch 1's import of cluster_into_episodes from episode_utils
    (correct and harmless -- it's just importing the same algorithm)
  - Restores the original compute_episode_stats function body
    (which was removed by Patch 1)
  - Removes the broken Patch 2 replacement (if still present)
  - Ensures the isnan None-safety fix is applied

Run from your project directory.
"""
import sys

COMPUTE_EPISODE_STATS = '''def compute_episode_stats(joint_mask, common_idx, fy_vals, event_mask, longest_tau_p):
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
        agreement = max(0.0, 2 * hit_rate - 1)
        conviction = float(np.log(n_episodes) * agreement)

    return n_episodes, hit_rate, conviction

'''

ORIGINAL_EPISODE_CALL = '''            n_episodes, episode_hit_rate, episode_conviction = compute_episode_stats(
                joint_mask, common_idx, fy_vals_full, event_for_episodes, longest_tau_p
            )'''

BROKEN_PATCH2 = '''            # Use canonical episode_utils implementation so stored n_episodes
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

ISNAN_BROKEN = '"episode_hit_rate":    None if np.isnan(episode_hit_rate) else round(episode_hit_rate, 4),'
ISNAN_FIXED  = '"episode_hit_rate":    None if (episode_hit_rate is None or np.isnan(episode_hit_rate)) else round(episode_hit_rate, 4),'

# ── ANCHOR: insert compute_episode_stats just before get_condition_mask ────
ANCHOR = 'def get_condition_mask('

with open('joint_cpe_engine.py', encoding='utf-8') as f:
    src = f.read()

changes = []

# Step 1: remove broken Patch 2 if present, restore original call
if BROKEN_PATCH2 in src:
    src = src.replace(BROKEN_PATCH2, ORIGINAL_EPISODE_CALL)
    changes.append("Removed broken Patch 2, restored original compute_episode_stats call")
elif ORIGINAL_EPISODE_CALL in src:
    changes.append("Original compute_episode_stats call already present (Patch 2 not found)")
else:
    print("ERROR: neither the broken Patch 2 nor the original call was found.")
    print("Check joint_cpe_engine.py around the results.append block manually.")
    sys.exit(1)

# Step 2: restore compute_episode_stats definition if missing
if 'def compute_episode_stats' not in src:
    if ANCHOR not in src:
        print("ERROR: anchor 'def get_condition_mask(' not found.")
        print("Cannot insert compute_episode_stats automatically.")
        sys.exit(1)
    insert_pos = src.index(ANCHOR)
    src = src[:insert_pos] + COMPUTE_EPISODE_STATS + '\n' + src[insert_pos:]
    changes.append("Restored compute_episode_stats definition")
else:
    changes.append("compute_episode_stats already present")

# Step 3: apply None-safe isnan fix
if ISNAN_BROKEN in src:
    src = src.replace(ISNAN_BROKEN, ISNAN_FIXED)
    changes.append("Applied None-safe isnan fix")
elif ISNAN_FIXED in src:
    changes.append("None-safe isnan fix already applied")
else:
    changes.append("WARNING: isnan line not found in expected form - check manually")

# Step 4: verify compute_episode_stats is now callable (references cluster_into_episodes)
if 'def compute_episode_stats' in src and 'cluster_into_episodes' in src:
    changes.append("compute_episode_stats references cluster_into_episodes: OK")

with open('joint_cpe_engine.py', 'w', encoding='utf-8') as f:
    f.write(src)

print("Restore complete. Changes applied:")
for c in changes:
    print(f"  - {c}")

# Quick syntax check
import ast
try:
    ast.parse(src)
    print("\nSyntax check: OK")
except SyntaxError as e:
    print(f"\nSyntax check: FAILED — {e}")
    print("Do not run joint_cpe_engine.py until this is resolved.")
    sys.exit(1)

print("\nNext steps:")
print("  python joint_cpe_engine.py")
print("  python diagnose_discrepancies.py  # expect ~107 rows, not 220")
