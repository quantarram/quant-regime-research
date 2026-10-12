# Does solar irradiance help trade JEPX? A short note, and a pointer to Kyushu

*Arun Ramanathan, October 2026. Prompted by a question from Antony Stace on the Paper 17 post: standing on 20 August 2026, what would the model have said about September in the Tokyo area?*

## The question

Rare extremes in irradiance are the kind of event the CPE framework looks for. The practical question is whether such an extreme, if known in advance, would have made money in the Japanese day-ahead power market (JEPX). The test uses only two data sets: ERA5 irradiance and JEPX day-ahead prices (April 2016 to September 2026).

## What was done

1. **A CPE scan.** 720 conditions: irradiance extremes over 7 to 60 days, as levels and as changes, against Tokyo, Kyushu and system price moves and premiums, on full-day and midday (10:00 to 15:00) prices. Thresholds were frozen before the test period. No condition passes the framework's gates, and the short-horizon patterns that did survive run against the physical mechanism: a same-week effect reverses the following week.
2. **A perfect-foresight trade.** For each month from May 2019 to September 2026, assume the month's actual irradiance is known on the 20th of the month before. Go long if the month is in the dullest tenth of 1996 to 2015, short if it is in the brightest tenth, against the average price of the first 19 days of the previous month, after removing the usual seasonal drift.
3. **A check that this is even possible.** ERA5 describes what happened. Irradiance in the first 19 days of a month explains about 0.3% of the next month's anomaly, and the resulting forecast never reaches an extreme. For September 2026, 20 August pointed to an ordinary month (+0.03 sd) and the outcome was −3.6 sd, the dullest in the record. So step 2 is a ceiling, not something a forecaster could have run.

## What was found

- **Tokyo:** nothing, even with perfect foresight. September 2026 was a win, the other dull months were not, and the result depends on the sub-period.
- **Kyushu, midday:** a weak effect with the expected sign (dull months, higher prices), present in both halves of the sample and in most variations, but not conclusive on ten years.
- **Mechanism:** irradiance moves the midday price on the day itself. That is already in the day-ahead price when it happens, which is why it does not forecast anything a week or a month ahead.
- **The size of the dullness matters, on the day.** In the CPE form (a dull day against the usual top-decile midday price, thresholds frozen on 2016 to 2020, scored on 2021 to 2026) the probability rises with rarity in every area. In Kyushu a dull day (dullest 10% of days) brings a top-decile midday price about 70% of the time against a 24% base, and the dullest 2% about the same. In Tokyo the lift grows from 3 to nearly 5 as days get rarer, but only outside the summer months, when cooler weather offsets lost solar. In Kyushu the result holds with temperature held near normal. This is a measure of risk on the day, not a forecast.

## Why Kyushu is the place to look

Kyushu has a lot of solar and limited transmission out, so midday prices can sink to the exchange's floor on bright days. That makes the relationship non-linear, which a straight-line fit and a three-month window blur. An earlier exploratory study (`../kyushu/RESULTS.md`) found a larger effect in 2022 to 2026 than before, consistent with more installed solar.

## Next steps

1. Model the response by size of dullness and year, and the floor explicitly: the probability that midday price falls to the floor given irradiance and demand, by year, rather than a linear slope.
2. Let the sensitivity change with installed solar capacity.
3. Use station irradiance (JMA) in place of ERA5 grid cells, and the other solar-heavy areas (Kyushu neighbours, Shikoku, Chugoku) as a check on whether Kyushu is special.
4. Bring in forward or futures prices, which are what a trader would deal against. They are not in this data set.
5. Use seasonal forecast hindcasts of irradiance if they can be obtained, to replace the perfect-foresight ceiling with a real forecast.

## Limits

Ten years of prices, with 2021 to 2022 dominated by fuel-price shocks; ERA5 grid cells rather than stations; spot prices instead of forwards; no costs. Everything above is in `RESULTS.md` in this folder, with the code.
