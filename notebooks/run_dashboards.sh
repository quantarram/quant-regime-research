#!/bin/bash
cd "$(dirname "$0")"

echo ""
echo "============================================================"
echo "  CPE DASHBOARD DAILY RUN"
echo "  $(date)"
echo "============================================================"

source /c/Users/NEA-ARN/Documents/soft/anaconda3/Scripts/activate quant_env

echo ""
echo "[1/7] Building gold dashboard..."
python build_gold_dashboard.py || { echo "ERROR: build_gold_dashboard.py failed."; exit 1; }

echo ""
echo "[2/7] Building portfolio dashboard..."
python build_portfolio_dashboard.py || { echo "ERROR: build_portfolio_dashboard.py failed."; exit 1; }

echo ""
echo "[3/7] Building precious metals dashboard..."
python build_metals_dashboard.py || { echo "ERROR: build_metals_dashboard.py failed."; exit 1; }

echo ""
echo "[4/7] Logging predictions and resolving expired..."
python log_predictions.py --both || { echo "ERROR: log_predictions.py failed."; exit 1; }

echo ""
echo "[5/7] Updating IBKR paper ledger (simulated, bullish-only)..."
python ibkr_paper_ledger.py || { echo "ERROR: ibkr_paper_ledger.py failed."; exit 1; }

echo ""
echo "[6/7] Committing to git..."
git add gold_predictions.csv portfolio_predictions.csv metals_predictions.csv gold_dashboard.html portfolio_dashboard.html precious_metals_dashboard.html ibkr_paper_ledger.csv
git commit -m "CPE daily update $(date +%Y-%m-%d)"

echo ""
echo "[7/7] Pushing to GitHub..."
git push

echo ""
echo "============================================================"
echo "  Done."
echo "============================================================"
