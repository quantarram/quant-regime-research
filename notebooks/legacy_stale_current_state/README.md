# Restatement of gold / silver / platinum calls, 2026-10-29

Files here are the state before the restatement: `gold_predictions_as_logged.csv`, `metals_predictions_as_logged.csv`, `monitor_history_as_logged.csv` and the two dashboard pages as published on 2026-10-29.

Method: `run_one.sh DATE` runs both builders offline with `DASH_ASOF=DATE` (bars strictly before DATE from the live price history) and `extract_state.py` reads the resulting page the way `log_predictions.py` does; `restate_gold_metals.py` rewrites composite, verdict, direction, recovery %, the change columns and percentile for every logged row and recomputes `prediction_correct` from the unchanged `actual_return_pct`. Paths inside the scripts point at the scratch directory used on 2026-10-29.

Second pass, same day: the replay was rerun after the drawdown rule was unified across the gold and metals pages and gold's log/simple comparison was fixed (see PANEL_CONVENTIONS.md), so the files in the repository root are the second restatement; the *_as_logged files are still the original log.
