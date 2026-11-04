# CPE monitor rows restated 2026-11-04

`monitor_history_as_logged.csv` is monitor_history.csv exactly as the daily runs wrote it up to 2026-11-03.

Why it was restated: Yahoo applies a dividend adjustment to its history with a delay. Until the 2026-11-04 run the stored
bond-ETF bars up to 2026-09-30 and those from 2026-10-01 were on different adjustment vintages (TLT 0.4%, EDV 1.4%, JNK 0.6%),
which lifted 21-day bond outcomes. That run refreshed the bars (83 per bond ETF, 2026-06-03 to 2026-09-30) and the series is
consistent again. The CPE metrics for every logged date were then recomputed from it with `restate_cpe_monitor.py`
(events entered on or before each date, current prices). Only rows with dashboard == cpe changed; all other rows are identical.

Largest effect: 21-day realised rate on 2026-10-29, 40.1% as logged, 30.8% restated (live base 49.8% as logged).
63-day and 126-day rates move by 1 point or less.
