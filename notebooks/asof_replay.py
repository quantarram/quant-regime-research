"""Point-in-time replay support for the gold and metals dashboard builders.

With DASH_ASOF=YYYY-MM-DD set, a builder reads its "fresh" prices from multiasset_prices_live_history.parquet,
keeping only bars dated strictly before that day (what a morning run on that day could see), instead of downloading
from Yahoo, and does not write the live-history file. The stored history is already FX-aligned, so it is not re-aligned.
"""
import os
import pandas as pd

ASOF = os.environ.get("DASH_ASOF")


def raw_asof(columns, live_path="multiasset_prices_live_history.parquet"):
    live = pd.read_parquet(live_path)
    live = live[live.index < pd.Timestamp(ASOF)]
    return live[[c for c in columns if c in live.columns]].dropna(how="all")


def prune_unreported(stored, raw, since=None, min_cover=0.95):
    """Inside the span a fresh download covers, drop stored bars that the download no longer reports (non-FX tickers only).
    Yahoo can date a Sunday-evening futures session or a US-holiday session as its own bar and later fold it into the next trading day;
    keeping the earlier bar adds a row to that ticker's own-bar count. FX columns are skipped (stored FX is realigned by one bar).
    `since` is the last date of the frozen panel. A ticker is only pruned when the download covers at least `min_cover` of its stored bars in the span, so a thin response never wipes history."""
    stored = stored.copy()
    lo, hi = raw.index.min(), raw.index.max()
    if since is not None:
        lo = max(lo, pd.Timestamp(since) + pd.Timedelta(days=1))   # bars before the frozen panel's end are the panel's own
    win = (stored.index >= lo) & (stored.index <= hi)
    dropped = {}
    for c in raw.columns:
        if c not in stored.columns or c.endswith("=X"):
            continue
        s = stored.loc[win, c].dropna()
        if s.empty:
            continue
        r = raw[c].reindex(s.index)
        if r.notna().mean() < min_cover:
            continue
        bad = s.index[r.isna()]
        if len(bad):
            stored.loc[bad, c] = float("nan")
            dropped[c] = [d.strftime("%Y-%m-%d") for d in bad]
    return stored, dropped
