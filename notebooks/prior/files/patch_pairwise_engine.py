"""
patch_pairwise_engine.py
========================
Patches cpe_engine_parallel.py to apply detrending to VIX-complex ETP
increments before threshold computation. This is the minimal integration
needed to run the Phase 2 detrending improvement through the full screen.

What it does:
  - Adds 'from detrend_utils import integrate_detrending' after the
    existing imports
  - Adds a single integrate_detrending() call after the increment loop
    and before the threshold computation loop
  - Adds a note explaining what TRAIN_CUTOFF is used (same as the
    threshold computation's own training restriction)

What it does NOT change:
  - The worker function (_compute_cpe_for_y) — it receives the already-
    detrended increments dict via the shared memory pool, so no change
    needed there
  - The threshold computation loop — it now automatically computes
    thresholds from detrended increments for the decay instruments
  - economic_prior.py — detrending doesn't change which pairs are admissible
  - joint_cpe_engine.py — it reads cpe_results.parquet, which will
    contain the detrended-threshold-gated pairwise results

Run:
    python patch_pairwise_engine.py          # preview only
    python patch_pairwise_engine.py --apply  # write patched copy
    python patch_pairwise_engine.py --apply --inplace  # overwrite original
"""
import argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument("--apply",   action="store_true")
parser.add_argument("--inplace", action="store_true")
parser.add_argument("--src",     default="cpe_engine_parallel.py")
args = parser.parse_args()

with open(args.src, encoding='utf-8') as f:
    src = f.read()

# ── PATCH 1: add detrend_utils import after existing imports ─────────────────
IMPORT_ANCHOR = "import os as _os_for_prior_switch"
DETREND_IMPORT = (
    "from detrend_utils import integrate_detrending, DECAY_INSTRUMENTS\n"
)

# ── PATCH 2: call integrate_detrending after increment loop, before thresholds
# Anchor on the print statement that immediately precedes threshold computation
THRESHOLD_ANCHOR = '    print(f"\\n  Pre-computing quantile thresholds...")'

DETREND_CALL = '''    # ── DETRENDING FOR VIX-COMPLEX ETPs (Phase 2 improvement) ─────────────
    # Apply EWMA-trend detrending to UVXY, VIXY, VXX, VIXM increments so
    # that their structural roll-decay does not dominate extreme-quantile
    # thresholds. Detrending uses per-instrument halflives calibrated to
    # each product's decay rate (detrend_utils.INSTRUMENT_HALFLIVES).
    # This call modifies increments in-place before threshold computation,
    # so all downstream threshold dicts automatically use detrended values.
    # The training cutoff restriction for thresholds (lines below) applies
    # identically to detrended increments -- no additional change needed.
    print(f"  Applying detrending to {sorted(DECAY_INSTRUMENTS & set(prices.columns))}...")
    integrate_detrending(
        increments, prices, all_taus,
        rate_index_tickers=set(RATE_INDEX_TICKERS),
    )
    print(f"  Detrending complete.")

'''

# Check anchors exist
found_import = IMPORT_ANCHOR in src
found_threshold = THRESHOLD_ANCHOR in src

print(f"Patch 1 (add detrend_utils import): {'FOUND' if found_import else 'NOT FOUND'}")
print(f"Patch 2 (add integrate_detrending call): {'FOUND' if found_threshold else 'NOT FOUND'}")

# Check detrend_utils not already imported
already_imported = "from detrend_utils import" in src
if already_imported:
    print("detrend_utils already imported — Patch 1 will be skipped.")

if not found_import or not found_threshold:
    print("\nOne or both anchors not found. Check cpe_engine_parallel.py structure.")
    sys.exit(1)

patched = src

# Apply Patch 1: insert import before the anchor line
if not already_imported:
    patched = patched.replace(
        IMPORT_ANCHOR,
        DETREND_IMPORT + IMPORT_ANCHOR,
        1
    )

# Apply Patch 2: insert detrend call before threshold computation
patched = patched.replace(
    THRESHOLD_ANCHOR,
    DETREND_CALL + THRESHOLD_ANCHOR,
    1
)

# Verify
import ast
try:
    ast.parse(patched)
    print("Syntax check: OK")
except SyntaxError as e:
    print(f"Syntax check: FAILED — {e}")
    sys.exit(1)

assert "integrate_detrending" in patched, "integrate_detrending call missing"
assert "from detrend_utils import" in patched, "detrend_utils import missing"
print("Patch structure: OK")

if args.apply:
    out = args.src if args.inplace else args.src.replace('.py', '_detrend.py')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(patched)
    print(f"\nPatched file written to: {out}")
    if not args.inplace:
        print(f"Review it, then run with --inplace to overwrite the original.")
else:
    print("\n--apply not set. Run with --apply to write the patched file.")
