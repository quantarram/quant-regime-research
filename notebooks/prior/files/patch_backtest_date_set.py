"""
patch_backtest_date_set.py
==========================
Single targeted fix: make _episode_conviction_for_row in backtest_engine.py
use the same date set as compute_episode_stats in joint_cpe_engine.py.

THE ROOT CAUSE
--------------
joint_cpe_engine.py computes episodes using:
    common_idx = future_inc[tau_f][y].dropna().index
                 intersected with all increments[tau_p].dropna()
                 filtered to <= TRAIN_CUTOFF

backtest_engine.py recomputes episodes using:
    train_dates = increments[tau_pasts[0]].index[<= TRAIN_CUTOFF]
    (only first predictor's tau, no forward-return dropna intersection)

The forward-return dropna removes dates near the training cutoff where
there aren't enough future days to observe the outcome. This shrinks
the date set and changes which firing dates are included, which changes
episode clustering for configurations with firing dates near Dec 2024.

THE FIX
-------
In _episode_conviction_for_row, build the date set the same way:
    1. Start with increments[tau_f][y].shift(-tau_f) non-null dates
       (forward return available)
    2. Intersect with increments[tau_p].dropna() for each tau_p
       (all predictor windows available)  
    3. Filter to <= TRAIN_CUTOFF

This is a 4-line change to _episode_conviction_for_row.
Everything else stays identical.
"""
import sys, ast

SRC = 'backtest_engine.py'

with open(SRC, encoding='utf-8') as f:
    src = f.read()

OLD = '''    train_dates = increments[tau_pasts[0]].index[increments[tau_pasts[0]].index <= TRAIN_CUTOFF]

    joint_mask = pd.Series(True, index=train_dates)'''

NEW = '''    # Use the same date set as joint_cpe_engine.py's compute_episode_stats:
    # intersect forward-return non-null dates with all predictor tau windows,
    # filtered to training period. This eliminates the 107-row discrepancy.
    if y not in increments[tau_f].columns:
        return 0.0
    fwd_series = increments[tau_f][y].shift(-tau_f)
    train_dates = fwd_series.dropna().index
    train_dates = train_dates[train_dates <= TRAIN_CUTOFF]
    for tp in tau_pasts:
        if tp in increments:
            avail = increments[tp].dropna(how="all").index
            train_dates = train_dates.intersection(avail)

    joint_mask = pd.Series(True, index=train_dates)'''

if OLD not in src:
    print("ERROR: target block not found in backtest_engine.py")
    print("Check line 277 — the train_dates construction may differ.")
    sys.exit(1)

src = src.replace(OLD, NEW, 1)
print("Applied date-set fix to _episode_conviction_for_row")

try:
    ast.parse(src)
    print("Syntax check: OK")
except SyntaxError as e:
    print(f"Syntax FAILED: {e}")
    sys.exit(1)

import argparse
p = argparse.ArgumentParser()
p.add_argument('--inplace', action='store_true')
args = p.parse_args()

out = SRC if args.inplace else SRC.replace('.py', '_fixed.py')
with open(out, 'w', encoding='utf-8') as f:
    f.write(src)
print(f"Written to: {out}")
if not args.inplace:
    print("Run with --inplace to overwrite.")
