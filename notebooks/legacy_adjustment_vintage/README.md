# CPE monitor rows: recomputed from the current price history

`monitor_history_as_logged.csv` is monitor_history.csv exactly as the daily runs wrote it up to 2026-11-03.

Yahoo applies each ex-dividend adjustment to its history a few days late (bond ETFs pay monthly). Until it does, the stored
bars are not total-return consistent across the ex-date, and 21-day CPE windows that straddle it understate bond returns by
the dividend (about 0.3-0.4%). On 2026-11-04 this made the 21-day realised rate read 31.3% (as-logged value for the same
date once Yahoo had caught up: 40.5%). The first restatement that day (commit 1968dd95) recomputed the rows while the lag was
in place and so baked the transient value in; it was corrected on 2026-11-11.

Check: for 105 equity, bond and commodity ETFs the stored daily returns since 2026-06-17 match raw Close plus Dividends from
Yahoo to within 0.1% on every bar.

`restate_cpe_monitor.py` recomputes every CPE row from the current history (events entered on or before each date).
monitor_history.py now does the same on every daily run (`restate_cpe_rows`), so earlier rows settle to their final value and
only the newest few rows can be transient. Non-CPE rows are never touched.
