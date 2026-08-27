#!/bin/bash
cd "$(dirname "$0")"

echo ""
echo "============================================================"
echo "  CPE DASHBOARD DAILY RUN"
echo "  $(date)"
echo "============================================================"

source /c/Users/NEA-ARN/Documents/soft/anaconda3/Scripts/activate quant_env

echo ""
echo "[1/8] Building gold dashboard..."
python build_gold_dashboard.py || { echo "ERROR: build_gold_dashboard.py failed."; exit 1; }

echo ""
echo "[2/8] Building portfolio dashboard..."
python build_portfolio_dashboard.py || { echo "ERROR: build_portfolio_dashboard.py failed."; exit 1; }

echo ""
echo "[3/8] Building precious metals dashboard..."
python build_metals_dashboard.py || { echo "ERROR: build_metals_dashboard.py failed."; exit 1; }

echo ""
echo "[4/8] Building predictor dashboard (22-instrument price forecasts)..."
python build_predictor_dashboard.py || { echo "ERROR: build_predictor_dashboard.py failed."; exit 1; }

echo ""
echo "[5/8] Logging predictions and resolving expired..."
python log_predictions.py --both || { echo "ERROR: log_predictions.py failed."; exit 1; }

echo ""
echo "[6/8] Updating IBKR paper ledger (simulated, bullish-only)..."
python ibkr_paper_ledger.py || { echo "ERROR: ibkr_paper_ledger.py failed."; exit 1; }

echo ""
echo "[7/9] Building football (Singapore Pools) checklist..."
python football_betting/daily_dashboard.py || { echo "ERROR: football_betting/daily_dashboard.py failed."; exit 1; }
echo "NOTE: this only refreshes the local HTML/log -- ask Claude to also republish the live"
echo "dashboard artifact (it can't self-publish from a plain script)."

echo ""
echo "[8/9] Committing to git..."
git add gold_predictions.csv portfolio_predictions.csv metals_predictions.csv gold_dashboard.html portfolio_dashboard.html precious_metals_dashboard.html predictor_dashboard.html ibkr_paper_ledger.csv football_betting/output/dashboard.html football_betting/output/dashboard_qualifying.json football_betting/output/qualifying_log.csv
git commit -m "CPE daily update $(date +%Y-%m-%d)"

echo ""
echo "[9/9] Pushing to GitHub..."
git push

echo ""
echo "============================================================"
echo "  Done."
echo "============================================================"
