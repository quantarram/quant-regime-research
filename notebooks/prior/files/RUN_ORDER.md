# Run Order — Diagnosing the Prior's Effect

You already have a working pipeline from last time. Replace
`cpe_engine_parallel.py`, `joint_cpe_engine.py`, and `run_backtest.py`
with the versions here (same filenames, drop-in), and add the two new
files (`economic_prior_BYPASS.py`, `check_history_weights.py`) to the
same folder as everything else.

Nothing about your existing `multiasset_prices.parquet` /
`multiasset_metadata.parquet` changes -- keep using the same ones.

## What's new in this batch

- `cpe_engine_parallel.py` / `joint_cpe_engine.py`: now check a
  `USE_PRIOR_BYPASS` environment variable. Unset (or `0`) = normal
  behaviour, exactly as before. Set to `1` = ignore the economic prior
  entirely and admit every pair, for the unrestricted comparison arm.
- `economic_prior_BYPASS.py`: the null prior the above switches to.
  You won't run this directly.
- `run_backtest.py`: now accepts a repeatable `--exclude-sleeve` flag
  to drop a sleeve (e.g. Crypto) from the book entirely, with the
  remaining sleeves' neutral weights renormalised to 100%.
- `check_history_weights.py`: new standalone script, no changes to
  anything else needed. Shows how much the history-weight ramp
  discounts short-history predictors (IBIT/FBTC/BITB) and how many
  surviving joint configs actually involve them.

## Run order

### Step 1 — Unrestricted comparison arm (answers: "did the prior help at all?")

Run in a **separate folder** (or rename outputs immediately after) so
you don't overwrite your prior-gated `cpe_results.parquet` /
`joint_cpe_results.parquet` from last time.

```bash
mkdir unrestricted_run && cd unrestricted_run
cp ../economic_prior.py ../economic_prior_BYPASS.py ../cpe_engine_parallel.py ../joint_cpe_engine.py ../run_backtest.py ../backtest_engine.py .
cp ../multiasset_prices.parquet ../multiasset_metadata.parquet .

export N_WORKERS=8 MIN_N=100 CPE_THRESH=0.80 MIN_LIFT=1.5
export USE_PRIOR_BYPASS=1
python3 cpe_engine_parallel.py
python3 joint_cpe_engine.py

python3 run_backtest.py --joint joint_cpe_results.parquet --label "UNRESTRICTED (no prior)"
```

Compare this SUMMARY table and randomisation-test `pct_exceeding`
directly against your prior-gated run from before. This is the actual
A/B test. If `pct_exceeding` is meaningfully lower (further from "looks
random") with the prior than without it, that's evidence the prior
helped. If they're similar, it didn't move the needle on statistical
significance, regardless of what the raw return numbers show.

### Step 2 — Re-run your normal (prior-gated) pipeline at `standard` confidence

Back in your original folder (`USE_PRIOR_BYPASS` unset or `0`):

```bash
export USE_PRIOR_BYPASS=0
export MIN_CONFIDENCE=standard
python3 cpe_engine_parallel.py
python3 joint_cpe_engine.py
python3 run_backtest.py --joint joint_cpe_results.parquet --label "standard confidence"
```

This excludes the literature-review-downgraded channels (dollar smile,
Dr. Copper, etc. -- see `literature_review_tier2.md`). Compare against
your `weak`-confidence run from before.

### Step 3 — Is Crypto carrying the result? (no new screen needed)

Re-use whichever `joint_cpe_results.parquet` you already have (the
`weak`-confidence prior-gated one from your first run is fine):

```bash
python3 run_backtest.py --joint joint_cpe_results.parquet --label "no crypto" --exclude-sleeve Crypto
```

Compare this SUMMARY table against your original 5-sleeve run. If
total return and Sharpe both drop sharply with Crypto removed, the
result was largely crypto-driven. If they hold up, the lift is coming
from elsewhere (most likely Equities, the one validated channel).

### Step 4 — How much is the history-weight ramp actually doing?

```bash
python3 check_history_weights.py --joint joint_cpe_results.parquet
```

No prerequisites beyond having a `joint_cpe_results.parquet` and
`multiasset_prices.parquet` in the folder. Shows the exact discount
applied to IBIT/FBTC/BITB and how many surviving configs still depend
on them despite the discount.

## Reading the results together

Run these roughly in order 1 → 2 → 3 → 4. Step 1 is the one that
actually answers your original question ("is it better because of the
prior"). Steps 2-4 explain *why* the result looks the way it does,
regardless of which way Step 1 comes out.
