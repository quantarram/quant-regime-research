# Broad-stock test of the football form edge / home PPG (2026-11-27)

Universe: current S&P 500 constituents (Wikipedia list), adjusted daily prices from yfinance (`dl.py`), each stock playing only from the day it joined the index. This is SURVIVOR-BIASED (names that were dropped or went bankrupt are absent), which flatters any "weak form then recovered" result.

- `search_stocks.py`: 864 fixture-list definitions (see ../sb_match_search.py), picks = strongest form on both statistics. Chosen on 2008-14, holdout 2015+. Result: 0 cells with discovery t>=2; mean holdout t -1.1; 392 cells t<=-2 in discovery, 384 of them negative in holdout (strong form does WORSE than the same stock's own average).
- `search_stocks_mirror.py`: same grid, picks = weakest form on both. 103 cells t>=2 in discovery, 100 positive in holdout (mean t +1.05, hit lift +2.2 pts, +0.3% per match).
- `econ.py`: equal-weight portfolios of the mirror picks vs the average eligible stock. Discovery 2008-14: +3% to +31% a year excess (crisis rebound plus survivorship). Holdout 2015-26: -3% to +3% a year gross, negative after 20 bps per match. No tradeable edge.

Scripts expect the data in /private/tmp/stocks (constituents.csv, stock_prices.parquet) and the module path of ../; adjust before rerunning.
