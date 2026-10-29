# Portfolio and IBKR records before the 2026-10-29 restatement

`portfolio_predictions_as_logged.csv`, `ibkr_paper_ledger_as_logged.csv` and `monitor_history_before_portfolio_restatement.csv` are the files as they stood before portfolio tilts logged 2026-06-18..2026-10-13 were replaced by a point-in-time replay with the current CPE tables (see PANEL_CONVENTIONS.md in the parent folder). Method: `build_portfolio_dashboard.py` with `DASH_ASOF=DATE` for every logged date, outcomes re-scored by `log_predictions.py --resolve`, ledger rebuilt by `replay_ibkr_ledger.py`.
