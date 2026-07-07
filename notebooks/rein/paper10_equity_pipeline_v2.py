"""
Paper 10 -- Reinsurer Equity Data Pipeline (LOCAL RUN ONLY)
================================================================
RUN ON YOUR LOCAL MACHINE -- Yahoo Finance, Nasdaq, and Stooq all block
automated fetching from this sandbox (robots.txt / anti-bot measures).
This mirrors the pattern from paper9_split_zone_pipeline.py.

pip install yfinance --break-system-packages

Tickers chosen for direct hurricane/cat-risk exposure:
  RNR   - RenaissanceRe Holdings: pure-play catastrophe reinsurer, the
          cleanest single-name test case (largest ILS/retro book)
  EG    - Everest Group: major P&C reinsurer, meaningful cat exposure
  ACGL  - Arch Capital Group: diversified re/insurer, cat-exposed
  SREN.SW - Swiss Re (SIX Swiss Exchange)
  MUV2.DE - Munich Re (Deutsche Boerse / XETRA)

Also pulling PGR (Progressive) and TRV (Travelers) as a NON-cat-exposed
control group (personal auto / commercial lines, minimal hurricane
exposure) -- if ENSO predicts RNR/EG but not PGR/TRV, that's a much
stronger, more specific result than testing cat-reinsurers alone.
"""
import yfinance as yf
import pandas as pd
import os

TICKERS = {
    'RNR': 'RenaissanceRe (pure-play cat reinsurer)',
    'EG': 'Everest Group',
    'ACGL': 'Arch Capital',
    'SREN.SW': 'Swiss Re',
    'MUV2.DE': 'Munich Re',
    'PGR': 'Progressive (control - minimal cat exposure)',
    'TRV': 'Travelers (control - some cat exposure but diversified)',
    'SPY': 'S&P 500 ETF (market benchmark for CAR event study)',
}

START = '1995-01-01'
END = '2026-07-05'

if __name__ == "__main__":
    os.makedirs('paper10_workdir', exist_ok=True)
    all_data = {}
    for ticker, desc in TICKERS.items():
        print(f"Fetching {ticker} ({desc})...")
        try:
            df = yf.download(ticker, start=START, end=END, progress=False)
            if len(df) == 0:
                print(f"  WARNING: no data returned for {ticker}")
                continue
            close = df['Close'].squeeze()
            close.name = ticker
            all_data[ticker] = close
            print(f"  OK: {close.index.min().date()} to {close.index.max().date()}, {len(close)} days")
        except Exception as e:
            print(f"  FAILED: {e}")

    combined = pd.DataFrame(all_data)
    combined.to_parquet('paper10_workdir/reinsurer_equity_prices.parquet')
    print(f"\nSaved combined price data -> paper10_workdir/reinsurer_equity_prices.parquet")
    print(combined.tail())
