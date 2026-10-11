# Crop weather -> grain futures: study plan (fixed before any result is looked at)

**Question.** Weather moves crop condition; condition is released every Monday after the close (USDA NASS Crop Progress, 4:00 pm ET).
Where in that chain does price react, and is any of it left to trade? Three links, each tested separately:

1. **Contemporaneous:** does the week's weather stress move the week's futures return (the "weather market")?
2. **Release:** does the Tuesday return (first session after the Monday release) load on the *surprise* (rating change not explained by
   that week's weather), on the *weather-explained* part (already knowable Sunday night), or on neither?
3. **Forward:** does weather stress, known at the end of the week, predict the next 5 / 21 trading-day return (conditional exceedance,
   the framework of this series)?

## Data
- Weekly weather drivers for the ten corn/soy states, ERA5 hourly 2015-2025 from the dashboard's cache (three points per state,
  definitions as the live model: heat-stress degree-days above 30 C, days above 35 C, afternoon VPD, rain; Monday-Sunday weeks).
- NASS weekly good+excellent, ten-state average of the week-on-week change (`dge`), ISO weeks 22-38.
- Weather-explained part `p` = leave-one-year-out ridge prediction of `dge` from that week's observed weather (RIDGE 20, as the live
  model); surprise `e = dge - p`. Before use, the replication must reproduce the live model's LOYO MAE (corn 1.076, soy 1.030).
- Prices: ZC=F and ZS=F daily closes (continuous front month: roll days contain calendar-spread jumps, so rank correlations and tail
  frequencies are reported next to means).

## Definitions
- Weather stress index `S` (higher = worse for the crop) = standardised sum of kdd, hot days and VPD, minus rain, using formation-period
  means and standard deviations; also `p` (negative = condition falls).
- Release return `R_rel`: log(close of the first session after the Monday release / close of the Monday). Week return `R_wk`: Friday to Friday.
  Forward returns `R_5`, `R_21`: from the Monday close (the release is after the close, so the release reaction is inside `R_5`).
- Split: formation 2015-2020, test 2021-2025 (and 2026 as live check, not in the split). Thresholds, standard deviations and tail cut-offs come
  from formation only. Weeks in June-August carry most of the weather premium; ISO 22-38 is the whole season and is the headline.

## Reported
Spearman and Pearson correlations, mean return by tercile of each variable with n, P(return in its formation top 20% | S in its formation top
20%) and the mirror for the bottom tail, by period. Overlapping windows are fine; point estimates with sample sizes; no significance language.

## Known limits (stated up front)
Eleven seasons of about 17 weeks each; roll jumps in continuous futures; the ten-state average is a proxy for the national rating;
observed weather is used (weather is public in real time, so for links 1 and 3 the signal is known by the Monday close).

## Addendum (written before any price result was computed)
The stress index `S` is built from each driver's anomaly against its formation-period mean for the same ISO week (not the raw level), then
divided by the formation-period standard deviation of that anomaly, so that the seasonal rise of heat in July and the seasonal drift of grain
prices into harvest cannot produce a spurious relationship. `S = (z_kdd + z_hot + z_vpd - z_rain) / 4`. Release-reaction session: the first
trading day after the first trading day on or after the Monday following the week's end (covers Monday holidays).
