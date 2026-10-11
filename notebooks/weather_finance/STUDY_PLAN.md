# Weather forecasts -> natural gas: study plan (fixed before any result is looked at)

**Question.** After the forecast is public, do (a) the *day-to-day revision* and (b) the *anomaly* of the GFS 00z population-weighted
degree-day forecast predict natural-gas futures returns?

**Why this first.** Gas is the cleanest weather-to-price link: heating and cooling demand move with temperature, and the market
trades *forecast changes*, not levels. If a free, public model's revisions carry nothing after settlement, that is a result.

## Data
- GFS 00z 2 m temperature, 34 large US metros, 6-hourly to 10 days, 2021-03-23 to present (`gfs_archive.py`, NOAA open data on AWS).
  Population-weighted (approximate metro populations), local-day mean, base 65 F. Same construction as published HDD/CDD.
- Climatology for anomalies: ERA5 daily mean temperature at the same metros, 1996-2025 (30 years), by day of year, smoothed 15 days.
- NG=F front-month futures (daily settle) and UNG; context: XLU. Cycle on date t is known ~03:30 UTC; NG settles ~19:30 UTC,
  so the trade is entered at settle(t) and the target is settle(t) -> settle(t+h).

## Signals (definitions fixed here)
- **Anomaly A(t):** forecast degree-days summed over valid days t+1..t+7, minus climatology for those days. HDD in the heating
  season (Nov-Mar), CDD in the cooling season (Jun-Sep). Shoulder months (Apr-May, Oct) are excluded: little weather-gas link.
- **Revision R(t):** the same sum over days t+1..t+7 from today's cycle minus yesterday's cycle (same valid days, leads 1-7 vs 2-8).
- Orientation: more degree-days than before / than normal is bullish gas in the matching season.

## Split and evaluation
- Formation 2021-03 to 2023-12 (all normalisation and quantiles taken only from here); test 2024-01 to present, untouched
  until the definitions above are final. No parameter is tuned on the test period.
- **Conditional exceedance (the CPE logic):** P(NG next-day return above its formation-period 80th percentile | signal in its
  formation-period top 20%) against the unconditional 20%, and the mirror for the bottom tail; horizons 1 and 5 days.
- **Simple rule:** in season, long gas when the signal is above +0.5 formation-period standard deviations, short below -0.5, flat
  otherwise; position size 1 (no leverage). Reported: mean return per active day, annualised mean/vol, max drawdown, by season and
  year, gross and with a cost per side, against buy-and-hold NG and against flat, plus exposure and correlation with NG.
- Overlapping windows are fine; results are point estimates with sample sizes, no significance testing.

## Known limits (stated up front)
- About five winters and five summers. Front-month continuous NG=F has roll gaps. The market also watches ECMWF, GFS, private
  models and consensus, so one public model's revision may already be in the price by settle.
