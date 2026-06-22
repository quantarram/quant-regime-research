# CPE Pipeline — Economic Prior + Bugfix + Backtest Rebuild

This is the full, updated pipeline incorporating everything from this
conversation: the economic prior restricting which predictor->target
pairs are even candidates, a literature review refining that prior's
confidence levels, a real bug found and fixed in the original joint
engine, and an independently-written backtest engine built from the
Portfolio Tilt paper's prose spec (not from its original code, which no
longer exists).

## What changed and why (read this first)

1. **Economic prior** (`economic_prior.py`): every (predictor, target)
   pair must now clear a pre-specified, data-blind admissibility check
   before any CPE is computed for it. This directly addresses the
   Portfolio Tilt paper's own finding (Sections 11-14) that no purely
   statistical correction -- shrinkage, sample-size filtering,
   episode-independence filtering -- could separate a spurious-but-
   recurring relationship (their example: silver predicting dogecoin
   across 5 historical episodes) from a genuine one (volatility
   predicting equities). See `literature_review_tier2.md` for a
   citation-backed review of the borderline channels, with several
   downgraded or removed based on what the published literature
   actually supports at this screen's specific horizon and instrument
   resolution (a recurring finding: real macro relationships, like
   "Dr. Copper" or IG credit spreads predicting equities, are
   well-documented at 6-18 month/YoY horizons, not at this screen's
   1-300 trading-day, single-ETF resolution).

2. **A real bug, found and fixed**: the original `joint_cpe_engine.py`
   excluded VIXM/VIXY/VXX/UVXY/SVXY from the PREDICTOR side, not just
   the target side, via a single combined `EXCLUDE_TICKERS` set. This
   meant the joint screen could never contain the exact volatility-
   complex-driven SPY configuration the Portfolio Tilt paper traces in
   its Section 10.1 as its one validated, accurate signal -- the
   pairwise screen had it (VIXM as SPY's single strongest predictor,
   CPE 0.977), but the joint engine silently dropped it before the
   greedy search ever ran. Fixed by splitting into `EXCLUDE_FROM_Y`
   (still excludes these as bad TARGETS -- they're mechanically
   derivative/decaying) and `EXCLUDE_FROM_X` (only managed currencies,
   matching `cpe_engine_parallel.py`'s own convention). Verified: after
   the fix, `Y=SPY, predictors=[VIXM,VIXY], tau_future=63, CPE=0.98`
   appears in the joint screen, closely matching what the paper
   describes.

3. **Backtest engine, independently rebuilt**: the code that produced
   the Portfolio Tilt paper's reported numbers (7.65% return, Sharpe
   0.560, etc.) no longer exists -- it was run once in a separate
   session and not saved. `backtest_engine.py` + `run_backtest.py` are
   a FRESH implementation written only from the paper's prose
   methodology (extracted into `SPEC.md`), not a recovery of the
   original code. When run, this rebuild's numbers diverged
   substantially from the paper's reported figures, including in ways
   that don't reconcile even arithmetically (the paper's own stated
   no-tilt benchmark return is inconsistent with its own stated neutral
   weights and its own stated individual sleeve returns -- see the
   note at the bottom of this file). **Treat the original paper's
   specific numbers as an unverified qualitative guide only, not a
   target to reproduce.** This rebuild's own numbers are independently
   checkable from the code here.

## Pipeline order

```
multiasset_prices.parquet, multiasset_metadata.parquet   (your existing data files)
         |
         v
cpe_engine_parallel.py        -->  cpe_results.parquet
  (pairwise screen, economic-prior-gated)
         |
         v
joint_cpe_engine.py           -->  joint_cpe_results.parquet
  (greedy joint screen, economic-prior-gated + bugfixed)
         |
         v
cpe_signal_score.py           -->  cpe_signal_scores.parquet / .csv
  (current-day signal snapshot -- unmodified from your original)
         |
         v
run_backtest.py               -->  backtest_result_*.csv
  (full 2025 walk-forward simulation, both static-tilt and
   hold-to-horizon mechanisms, plus benchmarks)
```

## How to run

```bash
# 1. Pairwise screen (takes a while -- full universe is ~13M combinations
#    after prior-gating, vs ~176M unrestricted)
export N_WORKERS=8 MIN_N=100 CPE_THRESH=0.80 MIN_LIFT=1.5
# Optional: raise the confidence floor to exclude literature-review-
# downgraded channels entirely (see economic_prior.py CONFIDENCE_OVERRIDES)
export MIN_CONFIDENCE=weak   # or: caveat, standard, high
python cpe_engine_parallel.py

# 2. Joint screen (reads cpe_results.parquet from step 1)
python joint_cpe_engine.py

# 3. Current signal snapshot (reads joint_cpe_results.parquet from step 2)
python cpe_signal_score.py

# 4. Backtest (reads multiasset_prices.parquet + a joint screen parquet)
python run_backtest.py --joint joint_cpe_results.parquet --label "My run"
# Add --skip-randomisation to skip the (slower) 1000-rep randomisation test
```

## MIN_CONFIDENCE: what each level does

The literature review (`literature_review_tier2.md`) found that several
channels have real published support but only at the WRONG horizon or
instrument type for what this screen tests (e.g. "Dr. Copper" is
genuinely documented at 6-18 month/YoY horizons, not 1-300 trading
days). Rather than silently delete these, they're tagged with a
confidence tier you can filter on:

- `weak` (default) -- every economically admissible channel, same as
  having no confidence filter at all
- `caveat` -- excludes only channels with a documented, named failure
  mode (e.g. the dollar-smile channel, which broke down specifically in
  April 2025)
- `standard` -- also excludes channels with a horizon/instrument
  mismatch against their supporting literature (Dr. Copper, IG credit
  spreads predicting equities directly)
- `high` -- only the one channel with actual out-of-sample validation
  in the Portfolio Tilt paper itself (volatility spike -> equity
  reversal), not just literature support

Recommend running the full pipeline at `standard` for a more
conservative result, and comparing against `weak` to see how much the
literature-review downgrades actually matter for your specific universe
and dates.

## The arithmetic inconsistency, for the record

Independently confirmed during this rebuild: this backtest's individual
2025 sleeve returns match the Portfolio Tilt paper's own Table 2 almost
exactly (SPY 18.01% vs paper's 18.01%; Gold 62.68% vs 62.68%; Bonds
3.96% vs 3.96%; FX -5.79% vs -5.79%). But a weighted average of those
SAME returns, using the paper's OWN stated neutral weights (Equities
30.87%, Gold 24.29%, Bonds 4.74%, Crypto 30.10%, FX 10.00%), gives
~18%, not the paper's reported no-tilt benchmark of 7.60%. No
alternative weighting tried reconciles this. This is disclosed rather
than silently worked around -- it's not resolved, and it's a specific,
checkable reason (beyond the unavailable original code) to treat the
paper's reported figures as unverified.

## Files in this package

- `economic_prior.py` -- the core addition: sub-class taxonomy,
  admissible-channel whitelist, confidence tiers, duplicate-instrument
  exclusion
- `cpe_engine_parallel.py` -- pairwise CPE screen, prior-gated
- `joint_cpe_engine.py` -- greedy joint CPE screen, prior-gated AND
  bugfixed (EXCLUDE_FROM_X vs EXCLUDE_FROM_Y)
- `cpe_signal_score.py` -- current-day signal snapshot (unmodified)
- `backtest_engine.py` / `run_backtest.py` -- independently-rebuilt
  walk-forward backtest, static-tilt and hold-to-horizon mechanisms
- `SPEC.md` -- the prose-to-code spec extraction the backtest is built
  from, including every place the paper's description was ambiguous and
  what choice was made
- `literature_review_tier2.md` -- citation-backed review of the 19
  borderline economic-prior channels
