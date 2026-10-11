# Natural gas, second study: regional degree-days and absolute moves (written before any result is computed)

Motivation (user, 2026-10-11): a national population-weighted degree-day sum blurs where gas is actually burned. Winter gas demand is
heating in the cold-weather metros; summer gas demand is cooling-driven power burn in the South. Direction also mattered in the first study
(the summer sign came out opposite to the pre-specified one), so this study also tests *magnitude*, which does not need an orientation.

## Regions (fixed here)
- COLD (heating): New York, Chicago, Washington, Philadelphia, Boston, Detroit, Minneapolis, Denver, Baltimore, St. Louis, Pittsburgh,
  Cincinnati, Kansas City, Columbus, Indianapolis, Cleveland. Used for HDD, November-March.
- SOUTH (cooling / power burn): Dallas, Houston, Miami, Atlanta, Phoenix, Tampa, Orlando, Charlotte, San Antonio, Austin, Las Vegas,
  Nashville. Used for CDD, June-September.
Weights: the same approximate metro populations, renormalised within the region. Everything else exactly as STUDY_PLAN.md: GFS 00z, days
t+1..t+7, base 65 F, anomaly vs ERA5 1996-2025 (regional normals, 15-day smoothing), revision vs previous cycle, formation 2021-03..2023-12,
test 2024-01..2026-10 run once, thresholds from formation only.

## Reported
1. Directional: same tables as the first study (correlation with same-day, next-day and 5-day returns; top-tail exceedance; the long/short rule).
2. Magnitude: Spearman correlation of |signal| (|revision|, |anomaly|) with |same-day return| and |next-day return|, and the mean absolute
   return on days when |signal| is in its formation-period top 20% against all other days, by season and period.
No significance language; point estimates with n.
