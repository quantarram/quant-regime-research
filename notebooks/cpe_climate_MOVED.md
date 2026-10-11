# CPE climate signal ledger: moved

The frozen climate-state signal table (43 rows, frozen 2026-10-10, sha 6933fe6d64), its event log and its live scoring moved on 2026-10-11 to the Climate and Shock Signals dashboard,
which runs them in one daily pipeline together with the shock and policy monitor:

- Code: https://github.com/quantarram/weather-impacts-dashboard (`weatherdash/cpe_climate.py`, `data/cpe_climate/`)
- Dashboard: https://quantarram.github.io/weather-impacts-dashboard/

The table file is byte-identical to the one that was here (sha unchanged) and the watchlist output of the old and new code was checked identical before the move. The history of the old files is in this repository's git log.
