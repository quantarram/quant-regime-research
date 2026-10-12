# CPE on irradiance extremes and JEPX prices (2026-10-12)

Script `jepx_cpe_scan.py`, table `jepx_cpe_scan.csv`, figure `jepx_cpe_figure.png`. The CPE definition and the program's gates, applied directly (not the full engine with its economic-prior registry).

Design, frozen before looking at JEPX. X: mean daily ERA5 irradiance z-score over the past p = 7, 14, 30, 60 days (z against the calendar-month mean and sd of 1996-2015, so the extreme thresholds come from before JEPX exists); both tails at 2%, 5%, 10%; Tokyo irradiance for Tokyo and the system price, Fukuoka for Kyushu. Y: next f = 7, 14, 30 days, (a) log change of the mean price against the previous f days, (b) mean premium over the system price; upper and lower 10% tails frozen on the formation window 2016-04..2020-12; test 2021-01..2026-09. Base rate matched to the same calendar months. 720 conditions.

Results
- Gate (CPE >= 0.80, lift >= 1.5, n >= 100 days): 0 of 720 pass on formation. 0 survivors.
- Looser, lift-only admission on formation (lift >= 1.5, n >= 100, >= 3 episodes): 73 conditions. In the test window their mean lift is 1.09 (median 1.09); 15% keep lift >= 1.5; 55% are above base.
- Average excess over the season-matched base in the test window (points): mechanism direction (dull -> price up, bright -> price down) -1.3 over 198 conditions; opposite direction -0.6. Dull states in the mechanism direction average -4.7.
- Only 7 conditions have lift >= 1.5 in both windows. Four run against the mechanism (dull -> Kyushu price down, bright -> Tokyo/system price up), and the others are lift 1.5-2 on 14-30 day windows with base rates of 4-15%.
- The one faint mechanism-consistent pocket is the Kyushu premium over the system price after bright spells (test excess +4.6 points over 30 conditions), but it is -1.7 on formation, the test base rate itself is about 0.5 because the Kyushu discount deepened after 2020, and it is not confirmed.

The 20 August 2026 state: over the past 30 and 60 days Tokyo irradiance stood at the 99th percentile of the 1996-2015 distribution, and Kyushu at the 100th (30-day state +0.88 sd against a bright-10% cut of +0.34). That is a bright extreme and in the mechanism direction implies a lower September price. The matching conditions (Tokyo, bright 10%, p = 30, f = 30) had formation lift 2.0 (CPE 0.25 against base 0.13) and test lift 0.93 (0.04 against 0.05). The outcome was the opposite: the dullest September since 1996 and a price of 23.8 against 20.1 in August.
Earlier note: the 80th percentile quoted in the first reply was for 1-19 August only.

Limits: ten years of prices; the windows overlap and conditions are highly correlated, so 720 rows are far fewer independent findings; no seasonal irradiance forecast; ERA5 grid cells.

## Midday-only rerun (slots 21-30, 10:00-15:00 JST; `python jepx_cpe_scan.py midday`, outputs `*_midday.*`)
Same 720 conditions, same frozen thresholds, targets built from the mean of the ten midday slots.
- Gate: 0 of 720 pass; 0 survivors. Lift-only admission: 82 conditions; test lift mean 1.04, 17% keep lift >= 1.5.
- Test excess over season-matched base: mechanism direction -4.1 points (198 conditions), opposite direction +4.2. The sign is reversed: dull spells are followed by LOWER midday prices, bright spells by HIGHER ones. Fourteen conditions hold lift >= 1.5 in both windows, all short (p = 7), with 30-49 episodes each, e.g. Kyushu dull 7 days -> price falls over the next 7 days: CPE 0.59 against base 0.29 in test.
- Diagnosis (`midday_contemporaneous_check.py`): the weather effect is real but contemporaneous. Same-week midday price against same-week irradiance anomaly: Kyushu -0.34, Tokyo -0.10. The 'move' target measures the next f days against the previous f days, and with p = f the previous days are the dull or bright days themselves, so the effect reverses the following week (correlation of this week's irradiance anomaly with next week's change +0.31 Kyushu, +0.19 Tokyo). Measured against the week before the weather (next week vs previous week) it is +0.09 for both: nothing left to forecast.
- Reading: irradiance moves the midday price in the same week, mainly in Kyushu; it does not predict the price a week or a month later, which is why nothing passes a forward-looking gate. The 20 August 2026 conclusion is unchanged.

## Change-in-irradiance rerun (`python jepx_cpe_scan.py diff [midday]`, outputs `*_diff*.*`)
X as the engine builds it for climate series: the change over the past p days, i.e. the mean anomaly of the last p days minus the mean anomaly of the p days before (thresholds from the 1996-2015 distribution of that change; "dull" = large fall, "bright" = large rise). Everything else unchanged.
- Full day: 0 of 720 pass the gate; 63 pass lift-only admission, test lift mean 1.24 (29% keep >= 1.5); mechanism direction -2.4 points, opposite -0.4.
- Midday: 0 of 720 pass; 60 admitted, test lift mean 1.12; mechanism direction -4.8, opposite +1.8; the 13 conditions with lift >= 1.5 in both windows are short (p = 7-14) and again run against the mechanism, with 15-70 episodes each.
- Same reading as the level version: the short-window reversal comes from the price-move target comparing the next f days with the previous f days (see the midday section), not from a forecastable signal.

## Would foreknowledge of a month's irradiance make money? (`oracle_money.py`, `oracle_money_months.csv`, `oracle_money_figure.png`)
Perfect-foresight upper bound with only irradiance and JEPX: for each month M from 2018-05 to 2026-09 (89 months), reference R = mean price over days 1-19 of the previous month, outcome O = mean price over M, surprise = (O - R) minus the average (O - R) for the same calendar month in earlier years. Signal = ERA5 irradiance anomaly of M in monthly sd against 1996-2015; long 1 kWh if dull (bottom 10% cut), short if bright (top 10% cut), flat otherwise.
- The sign is right but the strength is small. Slope of the surprise on the anomaly (yen/kWh per sd, excluding Jan 2021 as the reference): Kyushu midday -0.83, Tokyo midday -0.22, system -0.04. Spearman correlations: Kyushu midday -0.22, Tokyo -0.01, system +0.07. Standard deviation of the surprise: 4.8, 6.7, 5.6 yen/kWh.
- Trades (excluding Feb 2021): Tokyo midday 30 trades, 50% hit rate, mean -0.39 yen/kWh (t -0.5). Kyushu midday 23 trades, 61% hit rate, mean +1.11 (t 1.6; Kyushu dull months 4 of 7 right, bright 10 of 16). With Jan 2021 included, mean P&L looks positive (+1.5 to +4.3) only because of a single month (a winter price spike, not weather).
- The ten most extreme months in Tokyo midday go the right way 4 times out of 10; in Kyushu 6 of 10.
- September 2026 itself (Tokyo irradiance -3.6 monthly sd, the lowest in the data): long would have gained +7.9 yen/kWh midday (+5.2 all-day) against the 1-19 August level after the seasonal drift. The other three dull Tokyo months lost in midday (Jan 2020, Jul 2020, Mar 2025).
- Reading: knowing a record-dull month in advance does not reliably make money in Tokyo and is a marginal, unproven edge in Kyushu midday; ten years of data.
