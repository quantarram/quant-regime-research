"""
Incrementally-refreshed, disk-persisted feature caches for the live
predictor pipeline. Seeds once from the existing validated research caches
(`features_daily_panel.parquet`, `features_new_tickers_baseline_cache.parquet`)
and appends only genuinely new dates on each run -- the expensive 512-day
rolling multifractal computation is never redone for the full history.

Two z-score groups, matching the research pipeline exactly:
  - "orig": SPY,QQQ,IWM,XLK,XLF,XLE,AAPL,MSFT,JPM,XOM,GLD,EURUSD=X (12 live
    targets) + ^VIX + TLT (context sources, not targets themselves).
    Judgment call, disclosed: the historical panel's z-score group also
    included BTC-USD (dropped from the final 22 targets, and its price data
    is stale/not fetched live) -- live mode z-scores across 14 tickers
    instead of the historical 15. A minor, disclosed distributional
    difference, not worth a 15th live ticker fetch just for exact fidelity.
  - "new": XLI,XLB,XLY,XLP,XLU,XLV,DIA,VTI,IYR,VOX (10 live targets, no ctx,
    no self_ref_score) -- matches the historical group exactly.
"""
import os

import numpy as np
import pandas as pd

import feature_lib as fl

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(OUT_DIR)

ORIG_CACHE = os.path.join(OUT_DIR, "live_features_orig_panel.parquet")
NEW_CACHE = os.path.join(OUT_DIR, "live_features_new_panel.parquet")
SEED_ORIG = os.path.join(OUT_DIR, "features_daily_panel.parquet")
SEED_NEW = os.path.join(OUT_DIR, "features_new_tickers_baseline_cache.parquet")

ORIG_ZSCORE_GROUP = fl.ORIG_GROUP_TICKERS + fl.CONTEXT_TICKERS  # 12 targets + ^VIX + TLT = 14
NEW_ZSCORE_GROUP = fl.NEW_GROUP_TICKERS  # 10 targets, exact match to historical


def _atomic_to_parquet(df, path):
    """Write to a temp file in the same directory, then atomically rename
    over the target. Prevents a killed/crashed process from ever leaving a
    partially-written or truncated cache file on disk -- os.replace is a
    single filesystem-level rename, not a byte-by-byte write, so there is
    no window where `path` exists in a half-written state. This was added
    after repeatedly finding live_features_orig_panel.parquet reverted to a
    smaller, pre-fix row count between runs during development (rapid
    overlapping kill/retry cycles while debugging, not realistic single-
    shot usage) -- whatever the exact cause, atomic writes make that class
    of corruption structurally impossible going forward."""
    tmp_path = path + ".tmp"
    df.to_parquet(tmp_path)
    os.replace(tmp_path, path)


def _seed_or_load(cache_path, seed_path, group_tickers, needs_zscore=False):
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
    else:
        seed = pd.read_parquet(seed_path)
        df = seed[seed["ticker"].isin(group_tickers)].copy()
        if needs_zscore:
            # The "new" ticker group's seed cache (unlike "orig") is saved
            # WITHOUT z-score columns -- confirmed directly (KeyError on
            # gap_tau21_q4_z during the first live run). Z-score it now,
            # matching 38_fss_selection_holdout_split.py's own construction.
            df = fl.zscore_group_df(df, cols=fl.ZSCORE_COLS)
        _atomic_to_parquet(df, cache_path)
        print(f"  Seeded {os.path.basename(cache_path)} from {os.path.basename(seed_path)}: "
              f"{len(df)} rows, max date {df['date'].max().date()}")
    return df


def _missing_dates(cache_df, price_index, group_tickers):
    """Trading dates present in the live-refreshed price index, after the
    cache's current max date, up to and including the latest available
    date -- these are the dates that need fresh feature rows computed.

    Uses the MINIMUM per-ticker max date, not the cache's global max date.
    Real bug caught and fixed: if even one ticker in the group is fresher
    than the others (e.g. IYR/VOX, refreshed independently in an earlier
    session run, sitting at a later date than XLI/XLP/XLU/XLV's rows in the
    same cache), the global max silently masked staleness in every other
    ticker -- confirmed directly: XLI/XLP/XLU/XLV's live predictions were
    dated over a month stale (2026-06-02) while every other ticker showed
    2026-07-17, because the group-wide check saw SOME row at 07-17 and
    concluded nothing needed backfilling. Every ticker in the group must be
    backfilled up to the same common frontier for cross-sectional
    z-scoring to be valid in the first place."""
    if len(cache_df) == 0:
        cache_max = pd.Timestamp("1970-01-01")
    else:
        per_ticker_max = cache_df.groupby("ticker")["date"].max()
        if any(t not in per_ticker_max.index for t in group_tickers):
            cache_max = pd.Timestamp("1970-01-01")
        else:
            cache_max = per_ticker_max.reindex(group_tickers).min()
    return [d for d in price_index if d > cache_max]


def _get_proxy_or_main_series(ticker, prices, prices_proxy):
    return prices_proxy[ticker] if ticker in ("IYR", "VOX") else prices[ticker]


def refresh_orig_panel(prices, pair_df):
    """prices: live-refreshed main price DataFrame (from live_data.refresh_main_prices).
    pair_df: cpe_results.parquet, for self_ref_score."""
    all_tickers = fl.ORIG_GROUP_TICKERS + fl.CONTEXT_TICKERS
    # Real bug caught and fixed: seeding only ORIG_GROUP_TICKERS (12) while
    # _missing_dates checks against all_tickers (14, including ^VIX/TLT
    # context) meant ^VIX/TLT were never in the cache's ticker column --
    # per_ticker_max.reindex(all_tickers) had NaN for both, tripping
    # _missing_dates' "ticker not in cache" fallback and treating the ENTIRE
    # ~30-year history as missing on every run, not just genuinely new
    # dates. Confirmed directly: this produced 14,465 "missing" dates
    # instead of the real ~23, which is why the live pipeline appeared to
    # hang for 5+ minutes -- it was silently recomputing the full historical
    # panel from scratch. Fixed by seeding (and therefore caching) all 14
    # tickers, matching what _missing_dates actually checks against.
    cache = _seed_or_load(ORIG_CACHE, SEED_ORIG, all_tickers)
    common_index = None
    for t in all_tickers:
        idx = prices[t].dropna().index
        common_index = idx if common_index is None else common_index.union(idx)
    missing = _missing_dates(cache, common_index, all_tickers)
    if not missing:
        print("  live_features_orig_panel.parquet already current, no new dates to compute.")
        return cache

    new_rows = []
    for d in missing:
        raw_by_ticker = {}
        for t in all_tickers:
            series = prices[t].dropna()
            series = series[series.index <= d]
            self_ref_rows = fl.get_self_ref_rows(pair_df, t) if t not in fl.CONTEXT_TICKERS else None
            feat = fl.compute_instrument_features_latest(t, series, self_ref_rows)
            if feat is not None:
                raw_by_ticker[t] = feat
        if len(raw_by_ticker) < len(all_tickers):
            continue  # not enough history yet for every ticker on this date -- skip, will catch up later

        zscored = fl.zscore_group(raw_by_ticker, cols=fl.ZSCORE_COLS)
        vix_feat, tlt_feat = raw_by_ticker.get("^VIX"), raw_by_ticker.get("TLT")
        for t in fl.ORIG_GROUP_TICKERS:
            row = {"ticker": t, "date": d, **zscored[t]}
            if vix_feat is not None:
                row.update(fl.ctx_columns(vix_feat, "^VIX"))
            if tlt_feat is not None:
                row.update(fl.ctx_columns(tlt_feat, "TLT"))
            new_rows.append(row)
        # Also persist ^VIX/TLT's own rows (no ctx_* of themselves, they ARE
        # the context source) -- required so _missing_dates can correctly
        # track their staleness too, not just the 12 target tickers'.
        for t in fl.CONTEXT_TICKERS:
            new_rows.append({"ticker": t, "date": d, **zscored[t]})

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        cache = pd.concat([cache, new_df], ignore_index=True)
        cache = cache.drop_duplicates(subset=["ticker", "date"], keep="last")
        _atomic_to_parquet(cache, ORIG_CACHE)
        print(f"  live_features_orig_panel.parquet: appended {len(new_df)} rows "
              f"({len(missing)} dates x {len(all_tickers)} tickers incl. context), now {len(cache)} total")
    return cache


def refresh_new_panel(prices, prices_proxy):
    cache = _seed_or_load(NEW_CACHE, SEED_NEW, fl.NEW_GROUP_TICKERS, needs_zscore=True)
    common_index = None
    for t in fl.NEW_GROUP_TICKERS:
        idx = _get_proxy_or_main_series(t, prices, prices_proxy).dropna().index
        common_index = idx if common_index is None else common_index.union(idx)
    missing = _missing_dates(cache, common_index, fl.NEW_GROUP_TICKERS)
    if not missing:
        print("  live_features_new_panel.parquet already current, no new dates to compute.")
        return cache

    new_rows = []
    for d in missing:
        raw_by_ticker = {}
        for t in fl.NEW_GROUP_TICKERS:
            series = _get_proxy_or_main_series(t, prices, prices_proxy).dropna()
            series = series[series.index <= d]
            feat = fl.compute_instrument_features_latest(t, series, self_ref_rows=None)
            if feat is not None:
                raw_by_ticker[t] = feat
        if len(raw_by_ticker) < len(fl.NEW_GROUP_TICKERS):
            continue

        zscored = fl.zscore_group(raw_by_ticker, cols=fl.ZSCORE_COLS)
        for t in fl.NEW_GROUP_TICKERS:
            new_rows.append({"ticker": t, "date": d, **zscored[t]})

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        cache = pd.concat([cache, new_df], ignore_index=True)
        cache = cache.drop_duplicates(subset=["ticker", "date"], keep="last")
        _atomic_to_parquet(cache, NEW_CACHE)
        print(f"  live_features_new_panel.parquet: appended {len(new_df)} rows "
              f"({len(missing)} dates x {len(fl.NEW_GROUP_TICKERS)} tickers), now {len(cache)} total")
    return cache
