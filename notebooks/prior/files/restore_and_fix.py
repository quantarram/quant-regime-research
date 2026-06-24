"""
restore_and_fix.py
==================
Restores joint_cpe_engine.py and backtest_engine.py to their original
uploaded state, then applies the single targeted fix:

  backtest_engine.py _episode_conviction_for_row:
    Use the same date set as joint_cpe_engine.py's compute_episode_stats
    (intersect forward-return non-null dates with all predictor tau windows).

No other changes. All other improvements (episode_utils, detrend_utils,
vol_targeting) remain as separate optional modules — nothing is baked
into the core engines except this one date-set alignment.

Usage:
    python restore_and_fix.py --originals /path/to/uploads
    python restore_and_fix.py --originals /path/to/uploads --apply
"""
import argparse, shutil, sys, ast, os

parser = argparse.ArgumentParser()
parser.add_argument('--originals', default='/mnt/user-data/uploads',
                    help='Directory containing original uploaded files')
parser.add_argument('--apply', action='store_true',
                    help='Actually write files (dry-run by default)')
args = parser.parse_args()

ORIG = args.originals
CWD  = os.getcwd()

FILES_TO_RESTORE = [
    'joint_cpe_engine.py',
    'backtest_engine.py',
    'cpe_engine_parallel.py',
]

print(f"\nOriginals from : {ORIG}")
print(f"Restoring to   : {CWD}")
print(f"Dry-run        : {not args.apply}\n")

# ── STEP 1: restore originals ────────────────────────────────────────────────
for fname in FILES_TO_RESTORE:
    src_path = os.path.join(ORIG, fname)
    dst_path = os.path.join(CWD, fname)
    if not os.path.exists(src_path):
        print(f"  SKIP {fname} (not found in {ORIG})")
        continue
    if args.apply:
        shutil.copy2(src_path, dst_path)
        print(f"  RESTORED {fname}")
    else:
        print(f"  WOULD restore {fname}")

# ── STEP 2: apply the single date-set fix to backtest_engine.py ─────────────
be_path = os.path.join(CWD, 'backtest_engine.py')
with open(be_path if args.apply else os.path.join(ORIG, 'backtest_engine.py'),
          encoding='utf-8') as f:
    src = f.read()

OLD = '''    train_dates = increments[tau_pasts[0]].index[increments[tau_pasts[0]].index <= TRAIN_CUTOFF]

    joint_mask = pd.Series(True, index=train_dates)'''

NEW = '''    # Use the same date set as joint_cpe_engine.py's compute_episode_stats:
    # intersect forward-return non-null dates with all predictor tau windows,
    # filtered to training period. This makes episode counts identical
    # between screen-build and backtest runtime.
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
    print("\nERROR: target block not found in backtest_engine.py")
    print("The original file may differ from what was uploaded.")
    sys.exit(1)

patched = src.replace(OLD, NEW, 1)

try:
    ast.parse(patched)
    print("\n  Date-set fix: syntax OK")
except SyntaxError as e:
    print(f"\n  Date-set fix: SYNTAX FAILED — {e}")
    sys.exit(1)

if args.apply:
    with open(be_path, 'w', encoding='utf-8') as f:
        f.write(patched)
    print(f"  APPLIED date-set fix to {be_path}")
else:
    print(f"  WOULD apply date-set fix to backtest_engine.py")

print(f"\n{'DONE' if args.apply else 'DRY RUN COMPLETE'}.")
if not args.apply:
    print("Run with --apply to actually restore and patch the files.")
else:
    print("\nNext steps:")
    print("  python joint_cpe_engine.py          # rebuild screen")
    print("  python diagnose_discrepancies.py    # verify 0 discrepancies")
    print("  python run_backtest.py --joint joint_cpe_results.parquet --sleeves base")
