#!/bin/bash
cd "$(dirname "$0")"

echo ""
echo "============================================================"
echo "  CPE DASHBOARD DAILY RUN"
echo "  $(date)"
echo "============================================================"

source /c/Users/NEA-ARN/Documents/soft/anaconda3/Scripts/activate quant_env

echo ""
echo "[1/5] Building gold dashboard..."
python build_gold_dashboard.py || { echo "ERROR: build_gold_dashboard.py failed."; exit 1; }

echo ""
echo "[2/5] Building portfolio dashboard..."
python build_portfolio_dashboard.py || { echo "ERROR: build_portfolio_dashboard.py failed."; exit 1; }

echo ""
echo "[3/5] Logging predictions and resolving expired..."
python log_predictions.py --both || { echo "ERROR: log_predictions.py failed."; exit 1; }

echo ""
echo "[4/5] Committing to git..."
git add gold_predictions.csv portfolio_predictions.csv gold_dashboard.html portfolio_dashboard.html
git commit -m "CPE daily update $(date +%Y-%m-%d)"

echo ""
echo "[5/5] Pushing to GitHub..."
git push

echo ""
echo "============================================================"
echo "  Done."
echo "============================================================"
