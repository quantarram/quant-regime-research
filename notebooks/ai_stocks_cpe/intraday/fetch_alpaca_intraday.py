"""
fetch_alpaca_intraday.py
========================================
Pulls real 1-minute SIP bars from Alpaca's free Basic market-data tier
for NVDA, AMD, and SPY (market benchmark), caching to local parquet files,
extended incrementally on each run (same pattern as
files/cpe_engine_intraday_btc.py's fetch_symbol_1m for crypto).

Requires environment variables (never hardcoded, never committed):
    ALPACA_API_KEY_ID
    ALPACA_API_SECRET_KEY

Free-tier notes (verified 2026-10-01):
  - Full consolidated SIP historical bars are available for free as long
    as the data is >15 minutes old -- this script only ever requests
    data well in the past, so that's not a constraint here.
  - 200 API calls/minute limit.
  - feed=sip must be set explicitly; the default IEX feed only reflects
    trades executed on IEX and produces gappy, incomplete bars.

Run: python fetch_alpaca_intraday.py
"""
import os
import time
import requests
import pandas as pd

API_KEY = os.environ.get("ALPACA_API_KEY_ID")
API_SECRET = os.environ.get("ALPACA_API_SECRET_KEY")
if not API_KEY or not API_SECRET:
    raise SystemExit("Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY environment variables first.")

BASE_URL = "https://data.alpaca.markets/v2/stocks"
HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}

SYMBOLS = ["NVDA", "AMD", "SPY"]
START = "2023-10-01T00:00:00Z"
END = "2026-09-30T23:59:00Z"
LIMIT = 10000
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def fetch_symbol_1m(symbol: str, start: str, end: str) -> pd.DataFrame:
    cache_path = os.path.join(DATA_DIR, f"intraday_{symbol.lower()}_1m.parquet")
    all_bars = []
    page_token = None
    n_requests = 0
    while True:
        params = {"timeframe": "1Min", "start": start, "end": end, "feed": "sip", "limit": LIMIT}
        if page_token:
            params["page_token"] = page_token
        r = requests.get(f"{BASE_URL}/{symbol}/bars", headers=HEADERS, params=params, timeout=30)
        n_requests += 1
        if r.status_code != 200:
            print(f"  ERROR {r.status_code}: {r.text[:300]}")
            break
        data = r.json()
        bars = data.get("bars", [])
        all_bars.extend(bars)
        page_token = data.get("next_page_token")
        print(f"  [{symbol}] request {n_requests}: +{len(bars)} bars (total so far: {len(all_bars)})", end="\r")
        if not page_token:
            break
        if n_requests % 150 == 0:  # stay well under 200 calls/min
            time.sleep(60)

    print()
    if not all_bars:
        return pd.DataFrame()
    df = pd.DataFrame(all_bars)
    df["t"] = pd.to_datetime(df["t"])
    df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low",
                             "c": "close", "v": "volume", "n": "n_trades", "vw": "vwap"})
    df = df.set_index("timestamp").sort_index()
    df.to_parquet(cache_path)
    print(f"  Saved {cache_path}: {len(df)} bars, {df.index.min()} to {df.index.max()}")
    return df


if __name__ == "__main__":
    for sym in SYMBOLS:
        print(f"Fetching {sym} 1-minute SIP bars, {START} to {END}...")
        fetch_symbol_1m(sym, START, END)
