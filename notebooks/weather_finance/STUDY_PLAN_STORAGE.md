# Natural gas storage and weather: study plan (written before any result is computed)

**Data.** EIA weekly Lower-48 working gas in storage (bulk file, 2010-2026), week ending Friday, released the following Thursday 10:30 ET. NG=F daily
closes. Weekly weather: ERA5 population-weighted national HDD and CDD, Saturday-Friday weeks matching the storage week, 2010-2025.
**Definitions.** Change = storage(w) - storage(w-1). Norm(w) = average change in the same ISO week of the prior five years. Surprise vs norm
D = change - norm. Weather-implied change E = expanding-window (refit yearly, years before the test year) regression of the change on HDD, CDD,
their squares and two seasonal harmonics; unexplained part u = change - E; weather-explained deviation W = E - norm (known before the release).
Level state L = (storage - five-year average level) / five-year average level.
**Tests.** (1) Thursday release reaction: log(close Thursday / close Wednesday) against D, u and W, by period (formation 2010-2017, test 2018-2025). If
the market has priced the weather, it loads on u and not on W. (2) A rule entered at the Wednesday close: long when W is below its formation 20th
percentile (a colder-than-normal week implies a smaller injection), short when above the 80th, flat otherwise, held to Thursday's close and for five
trading days; costs 5 bp per side; against buy-and-hold on the same days. Thursdays inside the continuous-contract roll window (last four trading days
of a month) are also excluded in a robustness run. (3) State conditioning: when L is in its formation top quintile (deficit) or bottom quintile
(surplus), the next 5 and 21 trading-day NG return in tail probability and mean, formation versus test; and, 2021-2026 with true GFS forecasts, the
winter forecast-anomaly rule (long above +0.5 sd) conditioned on deficit versus on no state (formation 2021-2023, test 2024-2026).
Point estimates with n; no significance language.
