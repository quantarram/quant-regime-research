@echo off
REM ============================================================
REM  CPE DASHBOARD DAILY RUN
REM  Dr. Arun Ramanathan
REM
REM  Run this file every morning. It:
REM    1. Builds the gold dashboard
REM    2. Builds the portfolio dashboard
REM    3. Logs today's predictions (reads from freshly built HTMLs)
REM    4. Resolves any expired predictions
REM    5. Commits and pushes CSVs and HTMLs to GitHub
REM
REM  NEVER run build_gold_dashboard.py or build_portfolio_dashboard.py
REM  manually after this script — the logger reads from the HTML
REM  and any rebuild will cause divergence.
REM ============================================================

cd /d "%~dp0"

echo.
echo ============================================================
echo   CPE DASHBOARD DAILY RUN
echo   %date% %time%
echo ============================================================

echo.
echo [1/5] Building gold dashboard...
call conda activate quant_env
python build_gold_dashboard.py
if errorlevel 1 (
    echo ERROR: build_gold_dashboard.py failed. Aborting.
    pause
    exit /b 1
)

echo.
echo [2/5] Building portfolio dashboard...
python build_portfolio_dashboard.py
if errorlevel 1 (
    echo ERROR: build_portfolio_dashboard.py failed. Aborting.
    pause
    exit /b 1
)

echo.
echo [3/5] Logging predictions and resolving expired...
python log_predictions.py --both
if errorlevel 1 (
    echo ERROR: log_predictions.py failed. Aborting.
    pause
    exit /b 1
)

echo.
echo [4/5] Committing to git...
git add gold_predictions.csv portfolio_predictions.csv gold_dashboard.html portfolio_dashboard.html
git commit -m "CPE daily update %date%"
if errorlevel 1 (
    echo NOTE: Nothing new to commit, or git error.
)

echo.
echo [5/5] Pushing to GitHub...
git push
if errorlevel 1 (
    echo ERROR: git push failed. Check your connection.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Done. All dashboards updated and predictions logged.
echo ============================================================
echo.
pause
