"""
Wave-2 predictor panel (intraday-only): Polymarket-implied-probability vs.
Binance-spot cross-field coupling -- feasibility check.

Two genuinely different fields describing the same event: BTC spot price
(btc_last, from Binance) and Polymarket's market-implied probability that
BTC is up over the current 5-minute window (up_mid) -- not a proxy, an
actual market-implied-expectation field. Data source: the pm_btc_th
accumulate_loop.sh supervisor (launchd-managed, running continuously since
2026-07-11), consolidated at /Users/arrams/pm_btc_th/runs/master_ticks.parquet
-- ~517k 1Hz ticks over ~6 days as of this run, resampled to 1-minute bars.

6 days is still short relative to the ~1-year daily-refreshed BTC/ETH/SOL/BNB
intraday track elsewhere in this program, so this is treated as a feasibility
check (does any coupling structure show up at all, tested against a
circular-shift null) rather than a walk-forward-validated model -- consistent
with how the original intraday BTC/ETH feasibility test was framed.

Run: python 09_polymarket_cross_field.py
Output: polymarket_feasibility_results.json
"""
import pandas as pd
import numpy as np
import json
import os

MASTER_TICKS = "/Users/arrams/pm_btc_th/runs/master_ticks.parquet"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LAGS_MIN = [0, 1, 5, 15, 60]
N_NULL = 200

print("=" * 60)
print("  POLYMARKET-VS-SPOT CROSS-FIELD FEASIBILITY CHECK")
print("=" * 60)

raw = pd.read_parquet(MASTER_TICKS)
raw["ts_dt"] = pd.to_datetime(raw["ts"], unit="s")
raw = raw.sort_values("ts_dt").drop_duplicates("ts_dt")
print(f"Raw ticks: {len(raw)}, span {raw['ts_dt'].min()} .. {raw['ts_dt'].max()}")

bars = raw.set_index("ts_dt")[["btc_last", "up_mid"]].resample("1min").last().ffill(limit=5).dropna()
print(f"1-min bars: {len(bars)}, span {bars.index.min()} .. {bars.index.max()}")

btc_abs_incr = bars["btc_last"].diff().abs().values
up_abs_incr = bars["up_mid"].diff().abs().values
n = len(bars)


def cross_moment_lag(fx, fy, lag, q):
    """BTC-leads-Polymarket (lag>0: fx at t, fy at t+lag) normalized cross q-moment,
    same construction as 04_cross_field_features.py's cross_moment()."""
    if lag == 0:
        a, b = fx, fy
    else:
        a, b = fx[:-lag], fy[lag:]
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 100:
        return np.nan
    a_h, b_h = a ** (q / 2), b ** (q / 2)
    cross = np.mean(a_h * b_h)
    norm = np.sqrt(np.mean(a ** q) * np.mean(b ** q))
    return float(cross / norm) if norm > 0 and np.isfinite(norm) else np.nan


results = {"n_bars": int(n), "span_start": str(bars.index.min()), "span_end": str(bars.index.max()),
           "lags_min": {}}

rng = np.random.default_rng(0)
for lag in LAGS_MIN:
    for q in [2.0, 4.0]:
        real = cross_moment_lag(btc_abs_incr, up_abs_incr, lag, q)
        null_vals = []
        for _ in range(N_NULL):
            shift = rng.integers(200, n - 200)
            up_shuffled = np.roll(up_abs_incr, shift)
            null_vals.append(cross_moment_lag(btc_abs_incr, up_shuffled, lag, q))
        null_vals = np.array([v for v in null_vals if np.isfinite(v)])
        pctile = float((null_vals < real).mean()) if len(null_vals) and np.isfinite(real) else np.nan
        key = f"lag{lag}_q{int(q)}"
        results["lags_min"][key] = {
            "real_coupling": real,
            "null_mean": float(null_vals.mean()) if len(null_vals) else np.nan,
            "null_std": float(null_vals.std()) if len(null_vals) else np.nan,
            "percentile_vs_null": pctile,
        }
        print(f"  lag={lag}min q={int(q)}: real={real:.4f}, null_mean={results['lags_min'][key]['null_mean']:.4f} "
              f"+/- {results['lags_min'][key]['null_std']:.4f}, percentile={pctile:.3f}")

out_path = os.path.join(OUT_DIR, "polymarket_feasibility_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved results to {out_path}")
print("\nDone.")
