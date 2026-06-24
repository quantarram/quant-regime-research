"""
fix_compute_episode_stats.py
============================
Fixes the episode-counting discrepancy between joint_cpe_engine.py and
backtest_engine.py by replacing compute_episode_stats with a call to
episode_utils.compute_episode_conviction using the same inputs.

ROOT CAUSE
----------
The two engines produce different episode counts because they operate on
different date sets:

  joint_cpe_engine.py (compute_episode_stats):
    - Uses common_idx = intersection of future_inc[tau_f][y].dropna()
      with ALL increments[tau_p].dropna() for ALL taus in TAU_LIST
    - This is a strict intersection requiring all tau windows to be
      non-null simultaneously — a smaller date set

  backtest_engine.py (episode_conviction_for_row via episode_utils):
    - Uses train_dates = increments[tau_pasts[0]].index[<= TRAIN_CUTOFF]
    - Only requires the first predictor's tau to be non-null
    - A larger date set, so firing dates near the tail may be included
      in one but not the other

The fix: replace compute_episode_stats with a call to
episode_utils.compute_episode_conviction, which is the same function
the backtest engine uses, passing it:
  - firing_dates derived from joint_mask & common_idx (same as before)
  - target_forward_series = future_inc[tau_f][y] (already forward-shifted)
  - target_thresh from thresholds (already training-period frozen)
  - direction, longest_tau_p (unchanged)

This makes the joint engine and backtest engine use the SAME function
with equivalent inputs, eliminating the discrepancy.

The key insight: future_inc[tau_f][y] is already
    increments[tau_f][y].shift(-tau_f)
so it IS the forward return series. Passing it directly to
compute_episode_conviction (which expects a forward-shifted series)
is correct — no additional shifting needed.
"""
import sys, ast

with open('joint_cpe_engine.py', encoding='utf-8') as f:
    src = f.read()

# ── Verify episode_utils is already imported (from Patch 1) ──────────────
if 'from episode_utils import' not in src:
    print("ERROR: episode_utils import not found.")
    print("Run patch_joint_engine.py --apply --inplace first.")
    sys.exit(1)

# ── Extend the episode_utils import to include compute_episode_conviction ─
OLD_IMPORT = '''# ── EPISODE UTILS (shared canonical implementation, fixes Section 20.6 discrepancy)
from episode_utils import (
    cluster_into_episodes,
    compute_episode_conviction as _compute_episode_conviction_util,
    EPISODE_MIN_CONVICTION as MIN_EPISODES_FOR_CONVICTION,
)'''

if OLD_IMPORT not in src:
    # Try the simpler form that may exist after partial patching
    ALT_IMPORT = 'from episode_utils import ('
    if ALT_IMPORT not in src:
        print("ERROR: Could not locate episode_utils import block.")
        print("Check joint_cpe_engine.py imports manually.")
        sys.exit(1)
    # Find and extend whatever import block exists
    idx = src.index(ALT_IMPORT)
    end = src.index(')', idx) + 1
    existing = src[idx:end]
    if 'compute_episode_conviction' not in existing:
        new_import = existing.rstrip(')') + '\n    compute_episode_conviction as _ep_conv,\n)'
        src = src[:idx] + new_import + src[end:]
        print("Extended existing episode_utils import with compute_episode_conviction")
    else:
        print("compute_episode_conviction already imported")
        # Use whatever alias it was given
else:
    print("Full import block found with _compute_episode_conviction_util alias")

# ── Replace the compute_episode_stats call in the greedy results block ─────
OLD_CALL = '''            n_episodes, episode_hit_rate, episode_conviction = compute_episode_stats(
                joint_mask, common_idx, fy_vals_full, event_for_episodes, longest_tau_p
            )'''

NEW_CALL = '''            # Use episode_utils.compute_episode_conviction so this matches
            # exactly what backtest_engine.py computes at runtime, eliminating
            # the 107-row discrepancy found in Phase 1 (fix_compute_episode_stats.py).
            #
            # Key inputs:
            #   firing_dates  = dates in common_idx where joint_mask is True
            #   target_forward_series = future_inc[tau_f][y], which is already
            #     increments[tau_f][y].shift(-tau_f) — the forward return.
            #     Passing this directly is correct; do NOT shift again.
            #   target_thresh = training-period quantile threshold for Y
            _firing_dates = pd.DatetimeIndex(common_idx[np.asarray(joint_mask, dtype=bool)])
            _target_fwd   = future_inc[tau_f][y]   # already shift(-tau_f), correct direction
            if direction == "bullish":
                _target_thresh = thresholds[(tau_f, q_y)].get(y, float('nan'))
            else:
                _target_thresh = thresholds[(tau_f, round(1 - q_y, 10))].get(y, float('nan'))
            if np.isnan(_target_thresh):
                n_episodes, episode_hit_rate, episode_conviction = 0, None, 0.0
            else:
                _alias = globals().get('_compute_episode_conviction_util') or globals().get('_ep_conv')
                n_episodes, episode_hit_rate, episode_conviction = _alias(
                    firing_dates=_firing_dates,
                    max_tau_past=longest_tau_p,
                    target_forward_series=_target_fwd,
                    target_thresh=_target_thresh,
                    direction=direction,
                )'''

if OLD_CALL not in src:
    print("ERROR: Original compute_episode_stats call not found.")
    print("The greedy loop may have been modified. Check line ~534 manually.")
    sys.exit(1)

src = src.replace(OLD_CALL, NEW_CALL, 1)
print("Replaced compute_episode_stats call with episode_utils.compute_episode_conviction")

# ── Syntax check ──────────────────────────────────────────────────────────
try:
    ast.parse(src)
    print("Syntax check: OK")
except SyntaxError as e:
    print(f"Syntax check: FAILED — {e}")
    sys.exit(1)

# ── Write output ──────────────────────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--inplace', action='store_true')
args = parser.parse_args()

out = 'joint_cpe_engine.py' if args.inplace else 'joint_cpe_engine_fixed.py'
with open(out, 'w', encoding='utf-8') as f:
    f.write(src)
print(f"\nFixed file written to: {out}")
if not args.inplace:
    print("Review it, then run with --inplace to overwrite.")
print("\nNext: python joint_cpe_engine.py && python diagnose_discrepancies.py")
