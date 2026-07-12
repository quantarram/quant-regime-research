"""
Intraday CPE Feasibility Test — BTC / ETH, 1-minute bars
=========================================================
Every daily-horizon CPE paper (1-8) has found reliable exceedance structure
only at multi-month-to-annual horizons; the README's own Limitations section
lists "no intraday data" as an open gap. This script asks the same question
cpe_engine_parallel.py asks — P(future exceedance | past exceedance) beats a
CPE >= 0.80 / lift >= 1.5x / n >= 100 bar — but at short horizons (1 to 240
one-minute bars, i.e. 1min to 4hr) on real BTC/ETH tick data, instead of the
161-instrument daily universe.

Data: Binance public klines REST endpoint (no auth), 1-minute BTCUSDT and
ETHUSDT bars, cached locally and extended incrementally on each run.

Deliberate simplifications vs. cpe_engine_parallel.py (small feasibility
universe, not the full screen):
  - No economic_prior admissibility gate — with only {BTC, ETH} there's no
    spurious-pair risk that gate exists to filter.
  - No MIN_TRAIN_OBS predictor-history floor — both series have identical,
    ample history over the fetch window by construction.
  - Self-referential pairs (X == Y) are INCLUDED, deliberately: at these
    horizons "does BTC's own past return predict its own future return" is
    the momentum/reversal question the pm_btc_th reference baseline is
    itself built on, and is at least as interesting here as the cross-asset
    BTC<->ETH channel.

Usage:
    python cpe_engine_intraday_btc.py
Output:
    intraday_btc_1m.parquet, intraday_eth_1m.parquet   (cached raw bars)
    intraday_cpe_results.parquet                        (surviving signals, if any)
"""

import os
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
# BTC/ETH were the original pair; SOL and BNB added as a second liquid-pair
# check (both large-cap, long Binance history) before treating the negative
# 180-day BTC/ETH result as final.
SYMBOLS       = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "BNB": "BNBUSDT"}
INTERVAL      = "1m"
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", 365))

TAU_PAST    = [1, 5, 15, 30, 60, 240]     # bars = minutes: 1m .. 4hr
TAU_FUTURE  = [1, 5, 15, 30, 60, 240]
Q_GRID      = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]

CPE_THRESH  = float(os.environ.get("CPE_THRESH", 0.80))
MIN_SAMPLE  = int(os.environ.get("MIN_N", 100))
MIN_LIFT    = float(os.environ.get("MIN_LIFT", 1.5))

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(os.path.dirname(BASE_DIR), "data")
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


# ── DATA FETCH (cached, incremental) ─────────────────────────────────────────
def _fetch_klines_range(symbol, start_ms, end_ms):
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        resp = requests.get(
            BINANCE_KLINES,
            params={
                "symbol": symbol, "interval": INTERVAL,
                "startTime": cursor, "endTime": end_ms, "limit": 1000,
            },
            timeout=15,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last_close = batch[-1][6]
        if last_close <= cursor:
            break
        cursor = last_close + 1
        if len(batch) < 1000:
            break
        time.sleep(0.15)  # stay well under Binance's public rate limit
    return rows


def _fetch_and_merge(symbol_label, symbol, cached, start_ms, end_ms, tag):
    """Fetch [start_ms, end_ms) and merge into `cached`. Returns updated df."""
    print(f"    {symbol_label}: {tag} {symbol} {INTERVAL} bars "
          f"from {datetime.fromtimestamp(start_ms/1000, tz=timezone.utc)} "
          f"to {datetime.fromtimestamp(end_ms/1000, tz=timezone.utc)}...")
    raw = _fetch_klines_range(symbol, start_ms, end_ms)
    if not raw:
        print(f"    {symbol_label}: no bars returned for this range")
        return cached
    new_df = pd.DataFrame(raw, columns=KLINE_COLS)
    new_df = new_df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]
    for c in ["open", "high", "low", "close", "volume"]:
        new_df[c] = new_df[c].astype(float)
    combined = pd.concat([cached, new_df], ignore_index=True) if not cached.empty else new_df
    combined = combined.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
    print(f"    {symbol_label}: {len(new_df):,} bars fetched, {len(combined):,} total cached")
    return combined


def fetch_symbol_1m(symbol_label, symbol, lookback_days):
    """Fetch/extend the cached 1m bar history for `symbol` to cover the last
    `lookback_days`. Handles both directions of extension: backfilling older
    history (when lookback_days grows past what's cached) and forward-filling
    new bars since the cache was last updated -- a pure forward-only cache
    would silently miss backfill when the lookback window is widened."""
    cache_path = os.path.join(DATA_DIR, f"intraday_{symbol_label.lower()}_1m.parquet")
    os.makedirs(DATA_DIR, exist_ok=True)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - lookback_days * 24 * 60 * 60 * 1000

    cached = pd.read_parquet(cache_path) if os.path.exists(cache_path) else pd.DataFrame()

    if cached.empty:
        cached = _fetch_and_merge(symbol_label, symbol, cached, start_ms, now_ms, "fetching")
    else:
        cached_min = int(cached["open_time"].min())
        cached_max_close = int(cached["close_time"].max())

        if start_ms < cached_min - 60_000:
            cached = _fetch_and_merge(symbol_label, symbol, cached, start_ms, cached_min - 1, "backfilling")

        if cached_max_close < now_ms - 60_000:
            cached = _fetch_and_merge(symbol_label, symbol, cached, cached_max_close + 1, now_ms, "forward-filling")

        if start_ms >= cached_min - 60_000 and cached_max_close >= now_ms - 60_000:
            print(f"    {symbol_label}: cache already current ({len(cached):,} bars)")

    cached.to_parquet(cache_path, engine="pyarrow", index=False)
    cached = cached[cached["open_time"] >= start_ms].reset_index(drop=True)
    cached["dt"] = pd.to_datetime(cached["open_time"], unit="ms", utc=True)
    return cached.set_index("dt")["close"]


# ── CPE COMPUTATION ───────────────────────────────────────────────────────────
def compute_intraday_cpe(prices: pd.DataFrame):
    tickers = list(prices.columns)
    all_taus = sorted(set(TAU_PAST + TAU_FUTURE))

    print(f"\n  Pre-computing {len(all_taus)} tau increments over {len(prices):,} bars...")
    increments = {}
    for tau in all_taus:
        inc = pd.DataFrame(index=prices.index)
        for t in tickers:
            s = prices[t]
            inc[t] = np.log(s / s.shift(tau))
        increments[tau] = inc

    future_inc = {tau_f: increments[tau_f][tickers].shift(-tau_f) for tau_f in TAU_FUTURE}

    full_q_grid = sorted(set(Q_GRID + [round(1 - q, 10) for q in Q_GRID]))
    thresholds = {}
    for tau in all_taus:
        for q in full_q_grid:
            thresholds[(tau, q)] = increments[tau].quantile(q, numeric_only=True).to_dict()

    results = []
    combos_tested = 0

    for y in tickers:
        for tau_f in TAU_FUTURE:
            fy = future_inc[tau_f][y]
            for tau_p in TAU_PAST:
                px_all = increments[tau_p]
                common_idx = fy.dropna().index.intersection(px_all.dropna(how="all").index)
                if len(common_idx) < MIN_SAMPLE:
                    continue
                fy_vals = fy.loc[common_idx].values
                px_aligned = px_all.loc[common_idx]

                for q_y in Q_GRID:
                    thresh_y_up = thresholds[(tau_f, q_y)].get(y, np.nan)
                    thresh_y_dn = thresholds[(tau_f, round(1 - q_y, 10))].get(y, np.nan)
                    if np.isnan(thresh_y_up) or np.isnan(thresh_y_dn):
                        continue
                    uncond = 1.0 - q_y
                    event_bull = fy_vals > thresh_y_up
                    event_bear = fy_vals < thresh_y_dn

                    for x in tickers:  # includes X == Y (self/momentum test)
                        px_vals = px_aligned[x].values
                        valid_mask = ~np.isnan(px_vals)
                        if valid_mask.sum() < MIN_SAMPLE:
                            continue

                        for q_x in Q_GRID:
                            thresh_x_up = thresholds[(tau_p, q_x)].get(x, np.nan)
                            thresh_x_dn = thresholds[(tau_p, round(1 - q_x, 10))].get(x, np.nan)
                            if np.isnan(thresh_x_up) or np.isnan(thresh_x_dn):
                                continue
                            combos_tested += 2

                            cond_bull = valid_mask & (px_vals > thresh_x_up)
                            n_bull = cond_bull.sum()
                            if n_bull >= MIN_SAMPLE:
                                cpe_bull = event_bull[cond_bull].mean()
                                lift_bull = cpe_bull / uncond if uncond > 0 else np.nan
                                if cpe_bull >= CPE_THRESH and lift_bull >= MIN_LIFT:
                                    results.append((y, x, tau_p, tau_f, q_x, q_y,
                                                     round(float(cpe_bull), 4), round(float(uncond), 4),
                                                     round(float(lift_bull), 4), int(n_bull), len(common_idx),
                                                     "bullish", x == y))

                            cond_bear = valid_mask & (px_vals < thresh_x_dn)
                            n_bear = cond_bear.sum()
                            if n_bear >= MIN_SAMPLE:
                                cpe_bear = event_bear[cond_bear].mean()
                                lift_bear = cpe_bear / uncond if uncond > 0 else np.nan
                                if cpe_bear >= CPE_THRESH and lift_bear >= MIN_LIFT:
                                    results.append((y, x, tau_p, tau_f, q_x, q_y,
                                                     round(float(cpe_bear), 4), round(float(uncond), 4),
                                                     round(float(lift_bear), 4), int(n_bear), len(common_idx),
                                                     "bearish", x == y))

    cols = ["Y", "X", "tau_past", "tau_future", "q_X", "q_Y", "CPE", "uncond_prob",
            "lift", "n_condition", "n_total", "direction", "is_self_referential"]
    df = pd.DataFrame(results, columns=cols)
    return df, combos_tested


def main():
    print(f"\n{'='*65}")
    print(f"  INTRADAY CPE FEASIBILITY TEST ({'/'.join(SYMBOLS)}, 1-min bars)  |  "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")
    print(f"  Lookback         : {LOOKBACK_DAYS} days")
    print(f"  tau_past (bars)  : {TAU_PAST}")
    print(f"  tau_future (bars): {TAU_FUTURE}")
    print(f"  Filter           : CPE >= {CPE_THRESH} AND lift >= {MIN_LIFT} AND n >= {MIN_SAMPLE}")

    print(f"\n  Fetching / updating cached bars...")
    prices = {}
    for label, symbol in SYMBOLS.items():
        prices[label] = fetch_symbol_1m(label, symbol, LOOKBACK_DAYS)

    df_prices = pd.DataFrame(prices).dropna(how="any")
    print(f"\n  Aligned price panel: {df_prices.shape[0]:,} bars x {df_prices.shape[1]} tickers "
          f"({df_prices.index.min()} -> {df_prices.index.max()})")

    df_results, combos_tested = compute_intraday_cpe(df_prices)

    out_path = os.path.join(BASE_DIR, "intraday_cpe_results.parquet")
    df_results.to_parquet(out_path, engine="pyarrow", index=False)

    print(f"\n{'='*65}")
    print(f"  COMPLETE")
    print(f"  Combinations tested (post threshold pre-filter): ~{combos_tested:,}")
    print(f"  Rows surviving all 3 gates                      : {len(df_results):,}")
    print(f"  Saved -> {out_path}")

    if df_results.empty:
        print(f"\n  NO CONFIGURATION CLEARED CPE>={CPE_THRESH}, lift>={MIN_LIFT}, n>={MIN_SAMPLE}")
        print(f"  at any tested horizon (1-240 min). This is a genuine negative result,")
        print(f"  consistent with the existing papers' finding that exceedance structure")
        print(f"  is a slow-tail phenomenon, not a microstructure one -- at least for")
        print(f"  BTC/ETH at these horizons over this {LOOKBACK_DAYS}-day window.")
    else:
        print(f"\n  Direction breakdown:")
        print(df_results.groupby("direction")["CPE"].describe().round(3).to_string())
        print(f"\n  Self-referential (momentum/reversal) vs cross-asset:")
        print(df_results.groupby("is_self_referential")["CPE"].describe().round(3).to_string())
        print(f"\n  Top 15 by CPE:")
        print(df_results.nlargest(15, "CPE")[
            ["Y", "X", "tau_past", "tau_future", "q_X", "q_Y", "CPE", "uncond_prob", "lift", "n_condition"]
        ].to_string(index=False))
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
