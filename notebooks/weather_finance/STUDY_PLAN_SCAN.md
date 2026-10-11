# Weather-impact predictors vs the whole instrument universe: study plan (written before any result is computed)

**Question.** Across every liquid instrument in the multi-asset universe, does any weather-impact tail (extreme heating or cooling demand,
by region and persistence) carry a conditional exceedance probability that survives from a formation period into a later test period? This
is the CPE logic applied to weather, with 30 years of ERA5 history so that formation and test are each long.

## Predictors (daily, weather through the previous calendar day is public by the next morning; each is in-season only)
Daily mean temperature at the 34 GFS metros (ERA5, 1996-2025, `data/metro_hist`), population-weighted within a region, degree-days base 65 F,
anomaly against the **formation-period only** day-of-year normals (1996-2012, 15-day smoothing), summed over the trailing 7 or 28 days.
| id | region | quantity | season (month of the trading day) |
|---|---|---|---|
| HDD7_nat, HDD28_nat | all 34 metros | heating degree-day anomaly | Nov-Mar |
| HDD7_cold, HDD28_cold | COLD list (STUDY_PLAN_GAS_REGIONAL.md) | heating | Nov-Mar |
| CDD7_nat, CDD28_nat | all 34 | cooling | Jun-Sep |
| CDD7_south, CDD28_south | SOUTH list | cooling | Jun-Sep |
| CDD7_tx | Dallas, Houston, Austin, San Antonio | cooling | Jun-Sep |
| CDD7_mw | Chicago, Minneapolis, Kansas City, St. Louis, Indianapolis, Columbus, Detroit | cooling (heat on the corn belt) | Jun-Aug |
Orientation: higher = more demand or more stress.

## Targets
Every instrument in `multiasset_prices_live_history.parquet` with at least 1,000 daily observations before 2013 (so formation has substance).
Forward log return from the close of the signal day over h = 1, 5, 21, 63 trading days.

## Event and statistic (the series' conditional exceedance)
Signal tail: predictor in its formation-period top 20% (or bottom 20%) among in-season formation days. Target event: instrument's h-day return
above its formation-period 80th percentile (or below the 20th), computed on in-season days. P(event | tail) is reported with its unconditional
rate in the same period (the base rate); four directional configurations per predictor-instrument-horizon.
**Formation 1996-2012, test 2013-2025**; every threshold from formation. A configuration is *admitted* if in formation it has at least 5
separated episodes (tail runs at least 10 trading days apart) and P(event | tail) >= 0.30 (base 0.20).

## What is reported
1. Number of configurations tested, number admitted in formation.
2. For the admitted: the distribution of test P(event | tail) minus the test base rate (mean, median, share above 0), by predictor, by instrument group
   and by horizon. If admission were pure selection from noise, this centres on zero; a real weather channel centres above it.
3. The ten admitted configurations with the largest formation-to-test persistence, with n and episodes, as examples only.
Overlapping windows are fine; no resampling, shuffling or significance language: the test period is the test.

## Limits stated up front
Realised (trailing) weather is public and priced in; the forecast version of the same idea was tested separately for gas on 2021-2026.
Instruments differ in seasonality; the target thresholds are in-season. Continuous front-month futures have roll jumps. Thousands of
configurations means formation winners will exist by selection alone, which is why admitted configurations are judged on the test period only.
