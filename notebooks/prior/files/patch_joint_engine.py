"""
patch_joint_engine.py
=====================
Prints the exact lines to change in joint_cpe_engine.py to make it import
from episode_utils.py instead of using its own episode implementation.
Run this to see the diff, then apply it manually (or use the patched
copy this script writes to joint_cpe_engine_patched.py).

Usage:
    python patch_joint_engine.py                        # print diff only
    python patch_joint_engine.py --apply                # write patched copy
    python patch_joint_engine.py --apply --inplace      # overwrite original
"""
import argparse, re, sys, os

parser = argparse.ArgumentParser()
parser.add_argument("--apply",   action="store_true")
parser.add_argument("--inplace", action="store_true")
parser.add_argument("--src",     default="joint_cpe_engine.py")
args = parser.parse_args()

with open(args.src, encoding='utf-8') as f:
    src = f.read()

# ── PATCH 1: replace local cluster_into_episodes and compute_episode_stats
#             with imports from episode_utils ─────────────────────────────────

OLD_CLUSTER_FUNC = '''def cluster_into_episodes(firing_dates, gap_trading_days, gap_multiplier=EPISODE_GAP_MULTIPLIER):'''
NEW_IMPORT_BLOCK = '''# ── EPISODE UTILS (shared canonical implementation, fixes Section 20.6 discrepancy)
from episode_utils import (
    cluster_into_episodes,
    compute_episode_conviction as _compute_episode_conviction_util,
    EPISODE_MIN_CONVICTION as MIN_EPISODES_FOR_CONVICTION,
)


def cluster_into_episodes(firing_dates, gap_trading_days, gap_multiplier=EPISODE_GAP_MULTIPLIER):
    # Thin wrapper retained so any call sites in this file still work.
    from episode_utils import cluster_into_episodes as _c
    return _c(firing_dates, gap_trading_days, gap_multiplier)'''

# ── PATCH 2: replace the compute_episode_stats call inside the results.append
#             block with a call to episode_utils.compute_episode_conviction ──────

OLD_EPISODE_CALL = '''            n_episodes, episode_hit_rate, episode_conviction = compute_episode_stats(
                joint_mask, common_idx, fy_vals_full, event_for_episodes, longest_tau_p
            )'''

NEW_EPISODE_CALL = '''            # Use canonical episode_utils implementation so stored n_episodes
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

# Apply patches
patched = src

found1 = OLD_CLUSTER_FUNC in patched
found2 = OLD_EPISODE_CALL in patched

print(f"Patch 1 (replace cluster_into_episodes def): {'FOUND' if found1 else 'NOT FOUND - check manually'}")
print(f"Patch 2 (replace compute_episode_stats call): {'FOUND' if found2 else 'NOT FOUND - check manually'}")

if not found1 or not found2:
    print("\nOne or both patches could not be located automatically.")
    print("The joint_cpe_engine.py may have a different structure.")
    print("Apply manually using the diff below as a guide.")
    print("\n--- PATCH 1: find the def cluster_into_episodes(...) line and the")
    print("    entire function block below it, and the compute_episode_stats function,")
    print("    and replace the two function definitions with:")
    print(NEW_IMPORT_BLOCK)
    print("\n--- PATCH 2: find compute_episode_stats( call in results.append block")
    print("    and replace with episode_utils call.")
    sys.exit(0)

if found1:
    # Find the start of the cluster_into_episodes function and insert import before it
    idx = patched.index(OLD_CLUSTER_FUNC)
    # Walk back to find the start of any docstring/comment block above the function
    block_start = patched.rfind('\n\n', 0, idx) + 2
    # Find the end of compute_episode_stats function (next def at same indentation)
    after_cluster = patched.index('\n\ndef ', idx)
    after_episode_stats_start = patched.index('\n\ndef compute_episode_stats', idx)
    after_episode_stats = patched.index('\n\n\n', after_episode_stats_start)

    # Replace the two functions with the import block
    patched = (
        patched[:block_start]
        + NEW_IMPORT_BLOCK + "\n\n"
        + patched[after_episode_stats + 3:]
    )

if found2:
    patched = patched.replace(OLD_EPISODE_CALL, NEW_EPISODE_CALL)

if args.apply:
    out = args.src if args.inplace else args.src.replace('.py', '_patched.py')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(patched)
    print(f"\nPatched file written to: {out}")
    print("Verify it looks correct, then run: python joint_cpe_engine.py")
else:
    print("\n--apply not set. Run with --apply to write the patched file.")
    print("Or apply the two patches manually to joint_cpe_engine.py.")
