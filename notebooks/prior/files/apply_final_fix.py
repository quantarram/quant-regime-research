"""
apply_final_fix.py
==================
Reads the ORIGINAL uploaded files as source of truth.
Writes two corrected files to the target directory:

  joint_cpe_engine.py  -- exact original (no patches)
  backtest_engine.py   -- original + single date-set fix

Run:
    python apply_final_fix.py --target /path/to/project
"""
import argparse, shutil, os, ast

ORIG_DIR = "/mnt/user-data/uploads"
parser = argparse.ArgumentParser()
parser.add_argument("--target", required=True)
args = parser.parse_args()

TARGET = args.target
os.makedirs(TARGET, exist_ok=True)

# ── 1. joint_cpe_engine.py: copy original exactly ─────────────────────────
src_joint = os.path.join(ORIG_DIR, "joint_cpe_engine.py")
dst_joint = os.path.join(TARGET, "joint_cpe_engine.py")
shutil.copy2(src_joint, dst_joint)
with open(dst_joint, encoding="utf-8") as f:
    jce = f.read()
assert "def compute_episode_stats" in jce
assert "_alias" not in jce
assert "episode_utils" not in jce
print(f"joint_cpe_engine.py: copied original [OK]")

# ── 2. backtest_engine.py: original + date-set fix ────────────────────────
src_be = os.path.join(ORIG_DIR, "backtest_engine.py")
with open(src_be, encoding="utf-8") as f:
    be = f.read()

# Verify we have the exact original line before patching
OLD = "    train_dates = increments[tau_pasts[0]].index[increments[tau_pasts[0]].index <= TRAIN_CUTOFF]\n\n    joint_mask = pd.Series(True, index=train_dates)"
assert OLD in be, "Original train_dates line not found in uploaded backtest_engine.py"

NEW = """    # Same date set as joint_cpe_engine.py compute_episode_stats:
    # intersect forward-return non-null dates with predictor tau windows.
    if y not in increments[tau_f].columns:
        return 0.0
    fwd_series = increments[tau_f][y].shift(-tau_f)
    train_dates = fwd_series.dropna().index
    train_dates = train_dates[train_dates <= TRAIN_CUTOFF]
    for tp in tau_pasts:
        if tp in increments:
            train_dates = train_dates.intersection(
                increments[tp].dropna(how="all").index)

    joint_mask = pd.Series(True, index=train_dates)"""

be_fixed = be.replace(OLD, NEW, 1)
assert "fwd_series" in be_fixed
assert OLD not in be_fixed

ast.parse(be_fixed)  # syntax check

dst_be = os.path.join(TARGET, "backtest_engine.py")
with open(dst_be, "w", encoding="utf-8") as f:
    f.write(be_fixed)
print(f"backtest_engine.py: original + date-set fix written [OK]")
print(f"\nBoth files written to: {TARGET}")
print("Next: python joint_cpe_engine.py && python diagnose_discrepancies.py && python run_backtest.py --joint joint_cpe_results.parquet --sleeves base")
