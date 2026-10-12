# Does solar irradiance help trade JEPX? A short note, and a pointer to Kyushu

*Arun Ramanathan, October 2026. Prompted by a question from Antony Stace on the Paper 17 post: standing on 20 August 2026, what would the model have said about September in the Tokyo area?*

## The question

Rare extremes in irradiance are the kind of event the CPE framework looks for. The practical question is whether such an extreme, if known in advance, would have made money in the Japanese day-ahead power market (JEPX). The test uses only two data sets: ERA5 irradiance and JEPX day-ahead prices (April 2016 to September 2026).

## What was done

1. **A CPE scan.** 720 conditions: irradiance extremes over 7 to 60 days, as levels and as changes, against Tokyo, Kyushu and system price moves and premiums, on full-day and midday (10:00 to 15:00) prices. Thresholds were frozen before the test period. No condition passes the framework's gates, and the short-horizon patterns that did survive run against the physical mechanism: a same-week effect reverses the following week.
2. **A perfect-foresight trade.** For each month from May 2019 to September 2026, assume the month's actual irradiance is known on the 20th of the month before. Go long if the month is in the dullest tenth of 1996 to 2015, short if it is in the brightest tenth, against the average price of the first 19 days of the previous month, after removing the usual seasonal drift.
3. **How to read it.** ERA5 describes what actually happened, so step 2 is a ceiling: no real forecast of a month's irradiance made on the 20th can do better than knowing it exactly.

## What was found

- **Strip out the non-solar part of the price and the picture is cleaner.** Using the ratio of the midday to the night-time price (night has no solar, so fuel, demand and drift cancel), the Tokyo history as of 20 August 2026 points the right way (70% to 77% of dull windows rise above the seasonal drift, where the raw price said 22%), and the Kyushu effect shows up in 2016 to 2020 as well as 2021 to 2026. What detrending cannot remove is that the sensitivity of the midday price to sunlight roughly tripled between the two periods.
- **Antony's question, answered on that logic.** Standing on 20 August 2026 and perfectly knowing September's irradiance (a dullness rarer than 99% of 30-day windows since 1996), the model's history of Tokyo windows that dull, all nine from 2016 to 2021, gave no support for a higher price: none of them put the price in its top decile and on average they came in about 2 yen/kWh below the seasonal drift. September 2026 came in about 3 yen/kWh above it. The history predates the years in which the price response to irradiance grew, and it holds no dull window of that rarity since 2021, so the model had no comparable evidence.
- **With perfect foresight of the coming week, a dull week pays in Kyushu and nowhere else.** For the dullest tenth of 7-day windows, the Kyushu midday price rose into its top decile about half the time against a quarter normally, and a long position earned about 3 yen/kWh per week with a 90% hit rate, in both halves of 2021 to 2026. The rarer the window, the higher the odds. Tokyo and the system price show nothing stable, and windows of 30 days are too few to test. In 2016 to 2020 the Kyushu effect was small.
- **This is a ceiling.** ERA5 is what actually happened; no real forecast does better. The entry price is a trailing average of spot prices, not a forward price, so what is left after the market uses the same forecasts is not known.
- **It is also a same-day effect.** On the day itself, dull days raise the midday price, and the effect grows with rarity (Kyushu even with temperature held normal; Tokyo outside summer). It is gone from the next working day on, and the day-ahead price already uses the day's forecast.

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
