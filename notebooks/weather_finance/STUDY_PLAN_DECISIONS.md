# From weather forecasts to decisions that make money: study plan (written before any result is computed)

**Goal (user, 2026-10-11).** The Weather Impacts dashboard should end in decisions: what to buy and sell, and when, from the forecasts. The
test of every rule is money: it must beat (A) the same decisions with no forecast and (B) the same decisions fed climatology, after costs,
in a period that was not used to choose it. If no rule survives, the dashboard says "no actionable edge" for that instrument.

## Why the instrument set matters
Liquid futures (gas, grains) already tested null on public weather (Paper 17, Sections 5-6): the market watches the same models. The
instruments most mechanically tied to the quantities the dashboard forecasts (power demand, wind and solar output, gas burn) are in
equities that nobody prices from one model run: Texas/Northeast power producers, utilities, solar and gas producers.

## Instruments (fixed here)
Power producers and nuclear: VST, NRG, CEG. Regulated utilities: NEE, DUK, SO, D, AEP, XEL, plus XLU. Solar: FSLR, ENPH, TAN. Gas
producers: EQT, AR (plus NG=F, UNG for continuity). Market control: SPY. Daily adjusted closes, 2016 to date (as available).

## Signals (true forecasts, GFS 00z archive 2021-03-23 to 2026-10-10; formation 2021-03..2023-12, test 2024-01..2026-10, run once)
Population-weighted degree-day anomaly forecasts over days t+1..t+7 and their revision vs the previous cycle, exactly as the gas studies, for:
national (34 metros); Texas (Dallas, Houston, Austin, San Antonio) cooling, Jun-Sep; Northeast and Midwest (COLD list) heating, Nov-Mar;
southern cooling, Jun-Sep. Standardised with formation-period standard deviations only.

## Decision rules (a fixed small family; no tuning on the test period)
R1: long when signal > +0.5 sd, short when < -0.5 sd, flat otherwise, entered at the close of the cycle date, held h in {1, 5} trading days.
R2: long-only (flat instead of short).
R3: market-neutral: the R1 position in the instrument against the opposite position in SPY of equal dollars (removes market beta).
Costs 5 bp per side on every position change.

## Baselines (reported next to every rule)
(A) no forecast: flat, and buy-and-hold of the same instrument over the same days.
(B) climatology: the same rule driven by the climatological expected change in weather-driven demand over the next seven days (the seasonal
ramp, with no forecast information); for the anomaly signal above the climatological version is identically zero, so (B) is computed on a
second signal family, forecast change in weather-driven demand versus the last seven observed days, whose climatology-fed version is the pure
seasonal-timing rule. Both families are reported.

## Reported (per rule, formation and test, with n and exposure)
Mean return per active day (bp), annualised mean and volatility on all days, hit rate, maximum drawdown, correlation with SPY and
mean-difference alpha next to beta (an apparent alpha with beta far from zero is leverage, not skill), and the difference against (A) and (B).
Overlapping windows are fine; point estimates with sample sizes; no significance language; no shuffling or resampling.

## What counts as a result
A rule is *admitted* only if in the test period it beats both baselines after costs with beta-adjusted mean-difference alpha above zero, and the
formation-period direction agrees. Admitted rules are the only ones the dashboard will ever display as actions; the rest display "no edge".

## Addendum (written before any result was computed): admission is decided on formation only
Selecting rules by their test-period results would contaminate the test. A rule is *admitted* on **formation-period results alone**: net mean
return per active day above zero after costs, beta-adjusted mean-difference alpha above zero, a mean per active day above buy-and-hold of the same
instrument on the same days, for the climatology-fed family also above the climatology-fed rule, and at least 60 active days. The test period
then reports how the admitted rules did (mean, median, share positive, share beating each baseline), next to the same statistics for all rules.
A rule is shown on the dashboard as an action only if it was admitted in formation and is positive net of costs in the test; the live ledger
from 2026-10-11 is the clean out-of-sample record. Signal orientation is fixed a priori (more weather-driven demand or stress: long) for every
instrument; no orientation is chosen from data. Second signal family (climatology-comparable): forecast seven-day sum for t+1..t+7 minus the
observed seven-day sum for t-7..t-1, the observed days proxied by the previous cycle's lead-1 forecast (a one-day-ahead forecast); the
climatology-fed version replaces both sums with day-of-year normals.
