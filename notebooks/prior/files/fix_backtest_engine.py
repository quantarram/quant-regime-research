"""
fix_backtest_engine.py
======================
Fixes backtest_engine.py to use the stored episode_conviction column
from joint_cpe_results.parquet instead of recomputing it at runtime.

This eliminates the discrepancy between stored and recomputed conviction
values, making the backtest deterministic and consistent with whatever
the joint engine computed at screen-build time.

The key insight: the joint engine already stores episode_conviction
correctly in the parquet. The backtest engine ignores it and recomputes
from scratch using a different date set (increments[ref_tau].index vs
the joint engine's common_idx intersection), producing different episode
counts for 107 rows. Reading the stored value directly is both faster
and correct.

A fallback to recomputation is kept for rows where episode_conviction
is missing or NaN (e.g. old parquets without the column).
"""
import sys, ast

with open('backtest_engine.py', encoding='utf-8') as f:
    src = f.read()

OLD_CONVICTION = '''    conviction = joint_df.apply(
        lambda row: _episode_conviction_for_row(row, precomputed_increments), axis=1
    )
    return joint_df["joint_CPE"] * joint_df["lift"] * conviction * h_vals'''

NEW_CONVICTION = '''    # Use stored episode_conviction from the parquet if available.
    # This is the value computed by joint_cpe_engine.py using common_idx
    # (the exact date set used for CPE estimation). Recomputing at runtime
    # uses a different date set and produces 107 discrepant rows (Phase 1
    # finding). Reading the stored value is both faster and consistent.
    if "episode_conviction" in joint_df.columns:
        stored = joint_df["episode_conviction"].fillna(0.0)
        # Fallback: recompute only for rows where stored value is missing
        missing = joint_df["episode_conviction"].isna()
        if missing.any():
            recomputed = joint_df[missing].apply(
                lambda row: _episode_conviction_for_row(row, precomputed_increments), axis=1
            )
            conviction = stored.copy()
            conviction[missing] = recomputed
        else:
            conviction = stored
    else:
        # No stored column (old parquet format) — fall back to recomputation
        conviction = joint_df.apply(
            lambda row: _episode_conviction_for_row(row, precomputed_increments), axis=1
        )
    return joint_df["joint_CPE"] * joint_df["lift"] * conviction * h_vals'''

if OLD_CONVICTION not in src:
    print("ERROR: Target block not found in backtest_engine.py.")
    print("Check lines 383-386 manually.")
    sys.exit(1)

src = src.replace(OLD_CONVICTION, NEW_CONVICTION, 1)
print("Replaced runtime recomputation with stored episode_conviction lookup")

try:
    ast.parse(src)
    print("Syntax check: OK")
except SyntaxError as e:
    print(f"Syntax check: FAILED — {e}")
    sys.exit(1)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--inplace', action='store_true')
args = parser.parse_args()

out = 'backtest_engine.py' if args.inplace else 'backtest_engine_fixed.py'
with open(out, 'w', encoding='utf-8') as f:
    f.write(src)
print(f"\nFixed file written to: {out}")
if not args.inplace:
    print("Review it, then run with --inplace to overwrite.")
print("\nNext steps:")
print("  python fix_backtest_engine.py --inplace")
print("  python diagnose_discrepancies.py   # expect 0 discrepancies")
print("  python run_backtest.py --joint joint_cpe_results.parquet --sleeves base")
