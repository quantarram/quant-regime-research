# ECMWF wind, solar and demand forecasts as decision signals: study plan (written before any result is computed)

**Signals.** From historical IFS 00z cycles (2024-06-01 to 2026-05-31, every third day, leads 0-9 days; the 2024 archive has no steps beyond 240 h), the dashboard's own
demand, wind and solar maps give per region (ERCO, PJM, CISO, NYIS, ISNE, MISO) the forecast and the climatology of daily demand, wind and solar generation.
Net load = demand - wind - solar. Signals (percent of the region's climatological demand, oriented so that higher = tighter power system = long):
net-load anomaly, demand anomaly, minus the wind anomaly and minus the solar anomaly, averaged over lead windows days 1-3, 4-7 and 1-7; also the national sum of
net-load anomalies weighted by regional demand. Standardised with the expanding standard deviation of earlier values.
**Caveat stated up front.** The wind and solar maps were fitted on 2023-2025 generation, so for 2024-2025 cycles the weather-to-generation map is in-sample; the
price series are not, and nothing is fitted to prices.
**Decisions.** The rule family of STUDY_PLAN_EVERYTHING.md (long/short, long-only, short-only, market-neutral; holds 1, 5 and 21 trading days; 5 bp per side) on the 122
instruments. Formation: cycles through 2025-05-31; test: 2025-06-01 to 2026-05-31, scored once. Admission on formation only (positive net mean per active day, positive
beta-adjusted alpha, at least 30 active days, and for long-only above buy-and-hold). Reported: the test performance of the admitted set, its share positive and beating
buy-and-hold, and the same statistics for all configurations, plus the leading examples. Sample is one year of test: indicative only.
