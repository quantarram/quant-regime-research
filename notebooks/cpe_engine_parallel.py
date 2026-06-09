"""
Pairwise CPE Engine — Parallelised, Bullish & Bearish (Clean Universe v2)
==========================================================================
Excluded from Y (predicted):
  - Leveraged/inverse ETFs : SSO, SDS, TQQQ, TMF, TBT, TBF, UVXY, SVXY, VIXY, VIXM, VXX
  - Managed currencies     : THBUSD=X, CNYUSD=X, KRWUSD=X

Excluded from X (predictor):
  - Managed currencies     : THBUSD=X, CNYUSD=X, KRWUSD=X

Usage:
    export N_WORKERS=15
    export MIN_N=100
    export CPE_THRESH=0.80
    export MIN_LIFT=1.5
    python cpe_engine_parallel.py
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from multiprocessing import Pool, cpu_count
from datetime import datetime
import warnings, os, time, glob
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TAU_PAST    = [1, 5, 10, 21, 63, 126, 252, 300]
TAU_FUTURE  = [1, 5, 10, 21, 63, 126, 252, 300]
Q_GRID      = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]

N_WORKERS   = int(os.environ.get("N_WORKERS", max(1, cpu_count() - 1)))
CPE_THRESH  = float(os.environ.get("CPE_THRESH", 0.80))
MIN_SAMPLE  = int(os.environ.get("MIN_N", 100))
MIN_LIFT    = float(os.environ.get("MIN_LIFT", 1.5))

# ── EXCLUSIONS ────────────────────────────────────────────────────────────────
RATE_INDEX_TICKERS = {
    "^VIX","^VXN","^OVX","^GVZ","^EVZ","^VVIX","^SKEW",
    "^TNX","^TYX","^FVX","^IRX"
}

EXCLUDE_FROM_Y = {
    # Leveraged/inverse ETFs — mechanically implied signals
    "SSO","SDS","TQQQ",
    "TMF","TBT","TBF",
    "UVXY","SVXY","VIXY","VIXM","VXX",
    # Managed currencies — poor data quality / spurious signals
    "THBUSD=X","CNYUSD=X","KRWUSD=X",
}

EXCLUDE_FROM_X = {
    # Managed currencies — poor data quality / spurious signals
    "THBUSD=X","CNYUSD=X","KRWUSD=X",
}

TMP_DIR = "cpe_tmp_chunks"

# ── WORKER GLOBALS ────────────────────────────────────────────────────────────
_increments        = None
_future_inc        = None
_thresholds        = None
_predictor_tickers = None
_config            = None

def _init_worker(increments, future_inc, thresholds, predictor_tickers, config):
    global _increments, _future_inc, _thresholds, _predictor_tickers, _config
    _increments        = increments
    _future_inc        = future_inc
    _thresholds        = thresholds
    _predictor_tickers = predictor_tickers
    _config            = config


def _compute_cpe_for_y(args):
    y, chunk_id = args
    results = []

    cpe_thresh = _config["cpe_thresh"]
    min_n      = _config["min_n"]
    min_lift   = _config["min_lift"]
    tau_past_list   = _config["tau_past"]
    tau_future_list = _config["tau_future"]
    q_grid     = _config["q_grid"]

    for tau_f in tau_future_list:
        fy = _future_inc[tau_f][y]

        for tau_p in tau_past_list:
            px_all = _increments[tau_p]

            common_idx = fy.dropna().index.intersection(
                px_all.dropna(how="all").index
            )
            if len(common_idx) < min_n:
                continue

            fy_vals    = fy.loc[common_idx].values
            px_aligned = px_all.loc[common_idx]

            for q_y in q_grid:
                thresh_y_up = _thresholds[(tau_f, q_y)].get(y, np.nan)
                thresh_y_dn = _thresholds[(tau_f, round(1 - q_y, 10))].get(y, np.nan)

                if np.isnan(thresh_y_up) or np.isnan(thresh_y_dn):
                    continue

                uncond = 1.0 - q_y

                event_bull = fy_vals > thresh_y_up
                event_bear = fy_vals < thresh_y_dn

                for x in _predictor_tickers:
                    px_vals    = px_aligned[x].values
                    valid_mask = ~np.isnan(px_vals)
                    if valid_mask.sum() < min_n:
                        continue

                    for q_x in q_grid:
                        thresh_x_up = _thresholds[(tau_p, q_x)].get(x, np.nan)
                        thresh_x_dn = _thresholds[(tau_p, round(1 - q_x, 10))].get(x, np.nan)

                        if np.isnan(thresh_x_up) or np.isnan(thresh_x_dn):
                            continue

                        # ── BULLISH ───────────────────────────────────────
                        cond_bull = valid_mask & (px_vals > thresh_x_up)
                        n_bull    = cond_bull.sum()
                        if n_bull >= min_n:
                            cpe_bull  = event_bull[cond_bull].mean()
                            lift_bull = cpe_bull / uncond if uncond > 0 else np.nan
                            if cpe_bull >= cpe_thresh and lift_bull >= min_lift:
                                results.append((
                                    y, x, tau_p, tau_f, q_x, q_y,
                                    round(float(cpe_bull), 4),
                                    round(float(uncond), 4),
                                    round(float(lift_bull), 4),
                                    int(n_bull), len(common_idx),
                                    "bullish"
                                ))

                        # ── BEARISH ───────────────────────────────────────
                        cond_bear = valid_mask & (px_vals < thresh_x_dn)
                        n_bear    = cond_bear.sum()
                        if n_bear >= min_n:
                            cpe_bear  = event_bear[cond_bear].mean()
                            lift_bear = cpe_bear / uncond if uncond > 0 else np.nan
                            if cpe_bear >= cpe_thresh and lift_bear >= min_lift:
                                results.append((
                                    y, x, tau_p, tau_f, q_x, q_y,
                                    round(float(cpe_bear), 4),
                                    round(float(uncond), 4),
                                    round(float(lift_bear), 4),
                                    int(n_bear), len(common_idx),
                                    "bearish"
                                ))

    if results:
        os.makedirs(TMP_DIR, exist_ok=True)
        cols = ["Y","X","tau_past","tau_future","q_X","q_Y",
                "CPE","uncond_prob","lift","n_condition","n_total","direction"]
        df_chunk = pd.DataFrame(results, columns=cols)
        chunk_path = os.path.join(TMP_DIR, f"chunk_{chunk_id:05d}.parquet")
        df_chunk.to_parquet(chunk_path, engine="pyarrow",
                            compression="snappy", index=False)
        return len(results)
    return 0


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*65}")
    print(f"  CPE ENGINE (CLEAN UNIVERSE v2)  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Workers : {N_WORKERS}")
    print(f"{'='*65}")

    prices = pd.read_parquet("multiasset_prices.parquet")

    all_tickers   = list(prices.columns)
    price_tickers = [t for t in all_tickers if t not in RATE_INDEX_TICKERS]
    rate_tickers  = [t for t in all_tickers if t in RATE_INDEX_TICKERS]

    predicted_tickers = [t for t in price_tickers if t not in EXCLUDE_FROM_Y]
    predictor_tickers = [t for t in all_tickers   if t not in EXCLUDE_FROM_X]

    print(f"  All tickers          : {len(all_tickers)}")
    print(f"  Excluded from Y      : {sorted(EXCLUDE_FROM_Y)}")
    print(f"  Excluded from X      : {sorted(EXCLUDE_FROM_X)}")
    print(f"  Predicted Y          : {len(predicted_tickers)}")
    print(f"  Predictors X         : {len(predictor_tickers)}")

    print(f"\n  Pre-computing increments...")
    all_taus = sorted(set(TAU_PAST + TAU_FUTURE))
    increments = {}
    for tau in all_taus:
        inc = pd.DataFrame(index=prices.index)
        for t in price_tickers:
            s = prices[t]
            inc[t] = np.log(s / s.shift(tau))
        for t in rate_tickers:
            s = prices[t]
            inc[t] = s - s.shift(tau)
        increments[tau] = inc
        print(f"    tau={tau:>3}  shape={inc.shape}")

    print(f"\n  Pre-computing forward increments for Y...")
    future_inc = {}
    for tau_f in TAU_FUTURE:
        fi = increments[tau_f][predicted_tickers].shift(-tau_f)
        future_inc[tau_f] = fi
        print(f"    tau_f={tau_f:>3}  non-null rows: {fi.dropna(how='all').shape[0]}")

    full_q_grid = sorted(set(Q_GRID + [round(1 - q, 10) for q in Q_GRID]))
    print(f"\n  Pre-computing quantile thresholds...")
    thresholds = {}
    for tau in all_taus:
        for q in full_q_grid:
            thresholds[(tau, q)] = increments[tau].quantile(
                q, numeric_only=True).to_dict()
    print(f"  Done. {len(thresholds)} (tau, q) threshold dicts.")

    config = dict(
        cpe_thresh=CPE_THRESH, min_n=MIN_SAMPLE, min_lift=MIN_LIFT,
        tau_past=TAU_PAST, tau_future=TAU_FUTURE, q_grid=Q_GRID,
    )

    if os.path.exists(TMP_DIR):
        for f in glob.glob(os.path.join(TMP_DIR, "*.parquet")):
            os.remove(f)

    tasks = [(y, i) for i, y in enumerate(predicted_tickers)]

    total_combos = (len(predicted_tickers) * len(predictor_tickers) *
                    len(TAU_PAST) * len(TAU_FUTURE) * len(Q_GRID) ** 2 * 2)
    print(f"\n  Total combinations : {total_combos:,}")
    print(f"  Tasks (one per Y)  : {len(tasks)}")
    print(f"  Filter             : CPE >= {CPE_THRESH} AND lift >= {MIN_LIFT} AND n >= {MIN_SAMPLE}")
    print(f"\n  Running...\n")

    t0 = time.time()
    n_kept_total = 0

    initargs = (increments, future_inc, thresholds, predictor_tickers, config)

    with Pool(processes=N_WORKERS, initializer=_init_worker, initargs=initargs) as pool:
        for i, n_kept in enumerate(pool.imap_unordered(_compute_cpe_for_y, tasks)):
            n_kept_total += n_kept
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed if elapsed > 0 else 0
            eta     = (len(tasks) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:>3}/{len(tasks)}]  kept: {n_kept_total:>8,}  "
                  f"rate: {rate:.1f} Y/s  ETA: {eta/60:.1f} min", end="\r")

    print(f"\n\n  All workers done. Elapsed: {(time.time()-t0)/60:.1f} min")

    print(f"\n  Concatenating chunk files...")
    chunk_files = sorted(glob.glob(os.path.join(TMP_DIR, "*.parquet")))
    print(f"  Found {len(chunk_files)} chunk files")

    if chunk_files:
        dfs    = [pd.read_parquet(f) for f in chunk_files]
        df_all = pd.concat(dfs, ignore_index=True)
        df_all = df_all.sort_values(
            ["direction","Y","tau_future","tau_past","q_Y","q_X","CPE"],
            ascending=[True,True,True,True,True,True,False]
        ).reset_index(drop=True)

        out_path = "cpe_results.parquet"
        df_all.to_parquet(out_path, engine="pyarrow",
                          compression="snappy", index=False)

        elapsed_total = time.time() - t0
        print(f"\n{'='*65}")
        print(f"  COMPLETE")
        print(f"  Total rows kept   : {len(df_all):,}")
        print(f"  Total elapsed     : {elapsed_total/60:.1f} min")
        print(f"  Saved → {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")
        print(f"\n  Direction breakdown:")
        print(df_all.groupby("direction")["CPE"].describe().round(3).to_string())
        print(f"\n  Top CPE bullish:")
        print(df_all[df_all["direction"]=="bullish"].nlargest(10,"CPE")[
            ["Y","X","tau_past","tau_future","q_X","q_Y","CPE","uncond_prob","lift","n_condition"]
        ].to_string(index=False))
        print(f"\n  Top CPE bearish:")
        print(df_all[df_all["direction"]=="bearish"].nlargest(10,"CPE")[
            ["Y","X","tau_past","tau_future","q_X","q_Y","CPE","uncond_prob","lift","n_condition"]
        ].to_string(index=False))
        print(f"{'='*65}\n")

        for f in chunk_files:
            os.remove(f)
        os.rmdir(TMP_DIR)
    else:
        print("  No results passed the filter thresholds.")


if __name__ == "__main__":
    main()
