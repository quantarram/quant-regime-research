"""
Live data refresh for the predictor dashboard. Mirrors
`../build_gold_dashboard.py`'s exact top-up pattern for the main price
series (in-memory only, never writes `multiasset_prices.parquet` back to
disk -- that file stays exactly as the research pipeline built it), plus an
incremental refresh for `sector_proxy_cache.parquet` (IYR/VOX), which -- per
the approved plan -- is a deliberate exception that DOES persist to disk,
since that cache exists specifically to avoid repeated full re-downloads.

Tickers needed: 20 of the 22 target instruments (all but IYR, VOX) +
HYG, LQD (credit) + VIXY, VIXM (vix term) + ^VIX, TLT (context features,
orig-panel group only).
"""
import datetime as dt
import os

import numpy as np
import pandas as pd
import yfinance as yf

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

MAIN_FETCH_TICKERS = [
    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLU", "XLI", "XLB", "XLY", "XLP", "XLV",
    "AAPL", "MSFT", "JPM", "XOM", "GLD", "EURUSD=X", "DIA", "VTI",
    "HYG", "LQD", "VIXY", "VIXM", "^VIX", "TLT",
]
PROXY_FETCH_TICKERS = ["IYR", "VOX"]


def refresh_main_prices(period="400d"):
    """In-memory-only top-up of multiasset_prices.parquet, following
    build_gold_dashboard.py's pattern: strip any partial 'today' session
    row, append new dates past the parquet's own max date, fill NaN gaps in
    existing columns, never write back to disk."""
    prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
    parquet_max_date = prices.index.max()

    try:
        raw = yf.download(MAIN_FETCH_TICKERS, period=period, auto_adjust=True, progress=False)["Close"]
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        if len(raw.index) and raw.index.max().date() >= dt.date.today():
            raw = raw[raw.index.date < dt.date.today()]

        for col in raw.columns:
            if col in prices.columns:
                new = raw[[col]].loc[raw.index > prices.index.max()]
                if not new.empty:
                    prices = pd.concat([prices, new])
                fresh_col = raw[[col]].reindex(prices.index)
                prices[col] = prices[col].fillna(fresh_col[col])
            else:
                prices = prices.join(raw[[col]], how="left", rsuffix="_new")
                prices[col] = raw[col].reindex(prices.index)
        prices = prices.sort_index().loc[~prices.index.duplicated(keep="last")]
        latest_date = prices.index.max()
        age_days = (dt.date.today() - latest_date.date()).days
        stale = age_days > 4
        if stale:
            print(f"  WARNING: latest data is {age_days} days old ({latest_date.date()}). "
                  f"Yahoo may be lagging or markets closed.")
        return prices, {"parquet_max_date": parquet_max_date, "latest_date": latest_date,
                         "age_days": age_days, "stale": stale, "fetch_ok": True}
    except Exception as e:
        print(f"  yfinance error refreshing main prices: {e}")
        print(f"  Falling back to parquet-only data (as of {parquet_max_date.date()}).")
        return prices, {"parquet_max_date": parquet_max_date, "latest_date": parquet_max_date,
                         "age_days": (dt.date.today() - parquet_max_date.date()).days,
                         "stale": True, "fetch_ok": False}


def refresh_sector_proxy_cache(period="60d"):
    """IYR/VOX -- deliberate exception, DOES persist the top-up back to
    sector_proxy_cache.parquet (a designated-refreshable cache, not a
    validated research artifact)."""
    cache_path = os.path.join(OUT_DIR, "sector_proxy_cache.parquet")
    proxy = pd.read_parquet(cache_path)
    cache_max_date = proxy.index.max()

    try:
        raw = yf.download(PROXY_FETCH_TICKERS, period=period, auto_adjust=True, progress=False)["Close"]
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        if len(raw.index) and raw.index.max().date() >= dt.date.today():
            raw = raw[raw.index.date < dt.date.today()]

        for col in PROXY_FETCH_TICKERS:
            if col not in raw.columns:
                continue
            new = raw[[col]].loc[raw.index > proxy.index.max()]
            if not new.empty:
                proxy = pd.concat([proxy, new])
            fresh_col = raw[[col]].reindex(proxy.index)
            proxy[col] = proxy[col].fillna(fresh_col[col])
        proxy = proxy.sort_index().loc[~proxy.index.duplicated(keep="last")]

        if proxy.index.max() > cache_max_date:
            tmp_path = cache_path + ".tmp"
            proxy.to_parquet(tmp_path)
            os.replace(tmp_path, cache_path)  # atomic write -- see live_features.py's _atomic_to_parquet
            print(f"  sector_proxy_cache.parquet refreshed: {cache_max_date.date()} -> {proxy.index.max().date()}")
        return proxy, {"cache_max_date": cache_max_date, "latest_date": proxy.index.max(), "fetch_ok": True}
    except Exception as e:
        print(f"  yfinance error refreshing sector proxy cache: {e}")
        print(f"  Falling back to cached data (as of {cache_max_date.date()}).")
        return proxy, {"cache_max_date": cache_max_date, "latest_date": cache_max_date, "fetch_ok": False}


def market_status_banner(latest_date):
    """Same closed-market heuristic as build_gold_dashboard.py."""
    days_old = (dt.datetime.now() - latest_date.to_pydatetime().replace(tzinfo=None)).days
    sgt_hour = (dt.datetime.utcnow().hour + 8) % 24
    weekday = dt.datetime.now().weekday()  # 0=Mon, 6=Sun
    if days_old >= 1 and (weekday >= 5 or sgt_hour < 21):
        return (f"Markets are currently closed. Showing last available close: {latest_date.date()}. "
                f"Fresh prices are available after the next trading session opens.")
    return None
