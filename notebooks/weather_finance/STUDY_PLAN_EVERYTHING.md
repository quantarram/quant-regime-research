# Test everything: a walk-forward search for weather-based decisions (written before any result is computed)

**Mandate (user, 2026-10-11).** Test everything possible; only then will we know whether a real edge exists. Because a search this wide
produces formation winners by selection alone, the unit of evidence is not a configuration but the **walk-forward meta-strategy**: the
out-of-sample P&L of the whole process "each year, admit rules using only data before that year, trade the admitted set that year".

## Universe
**Signals** (all oriented a priori: more weather-driven demand or stress = long; no sign chosen from data), computed daily, in season, standardised
with data available at the time (expanding window; no future information):
- *Observed* (ERA5 daily temperature at the 34 metros, 1996-2025, trailing through the previous day): regions NAT, COLD, SOUTH, TX, MW, NE (New York,
  Boston, Philadelphia, Washington, Baltimore, Pittsburgh), WEST (Los Angeles, San Francisco, San Diego, Sacramento, Seattle, Portland, Phoenix,
  Las Vegas, Denver); quantity HDD (Nov-Mar) or CDD (Jun-Sep); windows 3, 7, 14, 28 days; level anomaly and weather momentum (anomaly now minus
  anomaly one window earlier). Corn-belt stress index (2015-2025).
- *Forecast* (GFS 00z archive 2021-03-23 to 2026-10-10, true forecasts): the same regions; anomaly, revision (versus previous cycle), and forecast
  minus observed-proxy change; lead windows days 1-3, 4-7 and 1-7.
- *Calendar* (the climatology twin): month-of-year long/short rules and the seasonal-ramp signal from day-of-year normals; no weather information.
**Targets**: energy, power and gas utilities, independent power producers, solar, gas producers and midstream, refiners, fertilizer and ag
equities, rails, weather-exposed retail, insurers, plus futures (NG, CL, HO, RB, ZC, ZS, ZW, ZL, ZM, KE, KC, SB, CC, CT, OJ, LE, HE) and the
existing universe's commodity and sector ETFs. Daily adjusted closes.
**Rules**: threshold long/short at +-0.5 sd, long-only, short-only, and market-neutral against SPY; holds of 1, 5 and 21 trading days (overlapping
holds averaged). Cost 5 bp per side on every position change. Entry at the close of the signal day, earned from the next day.

## Walk-forward protocol
For each test year Y, using only data before Y (at least three years): a configuration is *admitted* if its prior-years net mean return per active
day is positive, its beta-adjusted mean-difference alpha is positive, it has at least 60 active days, and it beats buy-and-hold of the same
instrument on the same days. Admitted configurations are ranked by prior-years mean per active day divided by its standard error, and the top 50
are traded in year Y with equal weight (configuration P&L streams equal-weighted). Three universes, each its own walk-forward:
(1) long-history futures and ETFs with observed-weather signals, test years 2008-2025; (2) equities and newer instruments with observed-weather
signals, test years 2019-2025; (3) forecast signals, test years 2024-2026 (warm-up 2021-2023).

## Baselines and reporting (per walk-forward)
Meta-strategy annualised mean, volatility, Sharpe, maximum drawdown, correlation with SPY and mean-difference alpha next to beta, yearly returns,
and number of admitted configurations per year. Against: (A) flat; (B) the equal-weighted buy-and-hold of the universe; (C) the same walk-forward
run on the calendar-only signals (climatology twin); (D) the equal-weighted average of *all* configurations (what admission adds).
Also reported: family-level tables (which signal families, rules, instrument groups hold up out of sample). Point estimates and sample sizes, no
significance language, no shuffling or resampling.

## What counts as an edge
The weather meta-strategy beats the calendar-only twin (C) and (D) after costs in the walk-forward, with beta-adjusted alpha above zero, in more
than half of the test years and without one year or one day carrying the result. Anything else is reported as "no edge". Any survivor is then
frozen and scored only on live data from 2026-10-11.

## Addendum (written before any result was computed)
1. Buy-and-hold criterion: applied only to the long-only rule (mean per active day must exceed the instrument's own mean on the same days). For
   long/short, short-only and market-neutral rules it is not meaningful (a short rule in a falling market beats it trivially), so those are admitted
   on positive net mean and positive beta-adjusted mean-difference alpha against SPY.
2. Active days include the days on which a position is closed, so round-trip costs are inside the mean per active day.
3. Expanding standardisation: each signal's standard deviation is computed from in-season values strictly before the day, and observed-weather
   normals for year Y from years before Y (at least five). The forecast universe needs two prior years, not three.
4. Instruments in the long-history universe: first price on or before 2002-12-31; the equities universe uses all instruments from 2016 on.
5. The climatology twin is run through the identical walk-forward, with calendar signals only (month-of-year long or short, and the seasonal-ramp
   signal from normals). 'Weather' meta-strategies use weather signals only.
