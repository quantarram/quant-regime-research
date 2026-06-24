"""
scope_oos_2015_2025.py
=======================
Proper single-screen OOS test of the CPE framework.

Design (agreed spec):
  - Training:   single frozen screen from all data ≤ 2014-12-31
  - Evaluation: 2015-01-01 through 2025-12-31 (11 years, ~2,750 trading days)
  - Mechanism:  hold-to-horizon, episode-conviction sizing, capped weights
  - Test:       single randomisation test across the ENTIRE 11-year period,
                all hold events pooled — maximum statistical power
  - 2025 is part of the OOS period, not a separate confirmation year

Why this design is better than year-by-year:
  - VIXM/VIXY have 1,005+ training observations by 2014-12-31, enough for
    the episode-conviction filter to have genuine historical content
  - Pooling holds across 11 years eliminates the low-power problem:
    3–4 holds per year × 11 years = potentially 30–40+ total hold events,
    giving a tight null distribution and a meaningful significance test
  - Single training cutoff removes annual re-estimation choices that
    complicated interpretation of the year-by-year OOS results

Steps:
  1. Run full pairwise CPE screen on data ≤ 2014-12-31
     → cpe_results_train2014_final.parquet
  2. Run full joint greedy screen
     → joint_cpe_results_train2014_final.parquet
  3. Run hold-to-horizon backtest over 2015-01-01 to 2025-12-31
  4. Run single randomisation test across the full 11-year period
  5. Report performance year-by-year AND for the full period

Use --skip-pairwise / --skip-joint to reuse existing parquets.

Usage:
    python scope_oos_2015_2025.py
    python scope_oos_2015_2025.py --skip-pairwise --skip-joint
    python scope_oos_2015_2025.py --skip-randomisation
    N_WORKERS=8 python scope_oos_2015_2025.py
"""

import argparse
import sys
import os
import time
import warnings
import glob
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

# ── Imports ────────────────────────────────────────────────────────────────
try:
    import backtest_engine as _be
    from backtest_engine import (
        BASE_SLEEVES, compute_neutral_weights,
        build_increments_and_thresholds, compute_quality_weights,
        clip_and_renormalise, simulate_portfolio, compute_performance_stats,
        configuration_fires_on_date, HORIZON_WEIGHTS,
    )
except ImportError as e:
    sys.exit(f"ERROR: Cannot import backtest_engine.py\n  {e}")

try:
    from run_backtest import (
        run_no_tilt_benchmark, run_buy_and_hold,
        load_and_filter_joint,
    )
except ImportError as e:
    sys.exit(f"ERROR: Cannot import run_backtest.py\n  {e}")

try:
    from economic_prior import is_admissible, admissible_predictors_for
except ImportError:
    try:
        from economic_prior_BYPASS import is_admissible, admissible_predictors_for
        print("  WARNING: using economic_prior_BYPASS (unrestricted)")
    except ImportError:
        sys.exit("ERROR: Neither economic_prior.py nor economic_prior_BYPASS.py found")

# ── Fixed parameters ──────────────────────────────────────────────────────
TRAIN_CUTOFF    = pd.Timestamp("2014-12-31")
EVAL_START      = pd.Timestamp("2015-01-01")
EVAL_END        = pd.Timestamp("2025-12-31")

TAU_PAST_LIST   = [5, 21, 63, 126, 252]
TAU_FUTURE_LIST = [21, 63, 126, 252]
Q_GRID          = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]
CPE_THRESH      = 0.80
MIN_LIFT        = 1.5
MIN_N           = 100
MIN_TRAIN_OBS   = 500
MAX_PREDICTORS  = 6
MIN_CONFIDENCE  = "weak"
EPISODE_GAP_MULTIPLIER        = 1.5
MIN_EPISODES_FOR_CONVICTION   = 3

CRYPTO_CAP_PCT  = 15.0
GOLD_CAP_PCT    = 20.0
WEIGHT_CAPS     = {"Equities": None, "Gold": GOLD_CAP_PCT,
                   "Bonds": None, "Crypto": CRYPTO_CAP_PCT, "FX": None}

RATE_INDEX_TICKERS = {
    "^VIX","^VXN","^OVX","^GVZ","^EVZ","^VVIX","^SKEW",
    "^TNX","^TYX","^FVX","^IRX"
}
EXCLUDE_FROM_Y = {
    "SSO","SDS","TQQQ","TMF","TBT","TBF",
    "UVXY","SVXY","VIXY","VIXM","VXX",
    "THBUSD=X","CNYUSD=X","KRWUSD=X",
}
EXCLUDE_FROM_X = {"THBUSD=X","CNYUSD=X","KRWUSD=X"}

PAIRWISE_OUT = "cpe_results_train2014_final.parquet"
JOINT_OUT    = "joint_cpe_results_train2014_final.parquet"
TMP_DIR      = "cpe_oos_final_tmp"


# ── Helpers ────────────────────────────────────────────────────────────────

def apply_weight_cap(weights, caps):
    capped, surplus = {}, 0.0
    for sleeve, w in weights.items():
        cap = caps.get(sleeve)
        if cap is not None and w > cap:
            surplus += w - cap
            capped[sleeve] = cap
        else:
            capped[sleeve] = w
    if surplus <= 0:
        return capped
    uncapped = [s for s, w in weights.items()
                if caps.get(s) is None or w < (caps.get(s) or float("inf"))]
    uncapped_total = sum(capped[s] for s in uncapped)
    if uncapped_total <= 0:
        per = surplus / len(capped)
        return {s: v + per for s, v in capped.items()}
    for s in uncapped:
        capped[s] += surplus * (capped[s] / uncapped_total)
    return capped


# ── STEP 1: Pairwise screen ────────────────────────────────────────────────

_w_inc = _w_finc = _w_thresh = _w_preds = _w_cfg = None

def _init_pw(inc, finc, thresh, preds, cfg):
    global _w_inc, _w_finc, _w_thresh, _w_preds, _w_cfg
    _w_inc, _w_finc, _w_thresh, _w_preds, _w_cfg = inc, finc, thresh, preds, cfg


def _pw_worker(args):
    y, chunk_id = args
    results = []
    cpe_thresh  = _w_cfg["cpe_thresh"]
    min_n       = _w_cfg["min_n"]
    min_lift    = _w_cfg["min_lift"]
    tau_p_list  = _w_cfg["tau_past"]
    tau_f_list  = _w_cfg["tau_future"]
    q_grid      = _w_cfg["q_grid"]
    min_conf    = _w_cfg["min_confidence"]
    tmp_dir     = _w_cfg["tmp_dir"]
    train_cutoff = pd.Timestamp(_w_cfg["train_cutoff"])

    y_preds = [x for x in _w_preds if is_admissible(x, y, min_conf)]
    if not y_preds:
        return 0

    full_q = sorted(set(q_grid + [round(1 - q, 10) for q in q_grid]))

    for tau_f in tau_f_list:
        if y not in _w_finc.get(tau_f, {}).columns if hasattr(_w_finc.get(tau_f, {}), 'columns') else y not in _w_finc.get(tau_f, pd.DataFrame()).columns:
            continue
        fy = _w_finc[tau_f][y]

        for tau_p in tau_p_list:
            px_all = _w_inc[tau_p]
            common_idx = (fy.dropna().index
                           .intersection(px_all.dropna(how="all").index)
                           .intersection(px_all.index[px_all.index <= train_cutoff]))
            if len(common_idx) < min_n:
                continue

            fy_vals    = fy.loc[common_idx].values
            px_aligned = px_all.loc[common_idx]

            for q_y in q_grid:
                thresh_y_up = _w_thresh.get((tau_f, q_y), {}).get(y, np.nan)
                thresh_y_dn = _w_thresh.get((tau_f, round(1 - q_y, 10)), {}).get(y, np.nan)
                if np.isnan(thresh_y_up) or np.isnan(thresh_y_dn):
                    continue

                uncond_up = float(np.nanmean(fy_vals > thresh_y_up))
                uncond_dn = float(np.nanmean(fy_vals < thresh_y_dn))

                for x in y_preds:
                    if x == y or x not in px_aligned.columns:
                        continue
                    px = px_aligned[x].values

                    for q_x in q_grid:
                        # Bullish
                        thr_xu = _w_thresh.get((tau_p, q_x), {}).get(x, np.nan)
                        if not np.isnan(thr_xu):
                            mask = px > thr_xu
                            n_c  = int(mask.sum())
                            if n_c >= min_n:
                                cpe  = float(np.nanmean((fy_vals > thresh_y_up)[mask]))
                                lift = cpe / uncond_up if uncond_up > 0 else np.nan
                                if cpe >= cpe_thresh and not np.isnan(lift) and lift >= min_lift:
                                    results.append({
                                        "Y": y, "X": x,
                                        "tau_past": tau_p, "tau_future": tau_f,
                                        "q_X": q_x, "q_Y": q_y,
                                        "direction": "bullish",
                                        "CPE": round(cpe, 4),
                                        "uncond_prob": round(uncond_up, 4),
                                        "lift": round(lift, 4),
                                        "n_condition": n_c,
                                    })
                        # Bearish
                        thr_xd = _w_thresh.get((tau_p, round(1 - q_x, 10)), {}).get(x, np.nan)
                        if not np.isnan(thr_xd):
                            mask = px < thr_xd
                            n_c  = int(mask.sum())
                            if n_c >= min_n:
                                cpe  = float(np.nanmean((fy_vals < thresh_y_dn)[mask]))
                                lift = cpe / uncond_dn if uncond_dn > 0 else np.nan
                                if cpe >= cpe_thresh and not np.isnan(lift) and lift >= min_lift:
                                    results.append({
                                        "Y": y, "X": x,
                                        "tau_past": tau_p, "tau_future": tau_f,
                                        "q_X": q_x, "q_Y": q_y,
                                        "direction": "bearish",
                                        "CPE": round(cpe, 4),
                                        "uncond_prob": round(uncond_dn, 4),
                                        "lift": round(lift, 4),
                                        "n_condition": n_c,
                                    })

    if results:
        path = os.path.join(tmp_dir, f"chunk_{chunk_id:04d}.parquet")
        pd.DataFrame(results).to_parquet(path, engine="pyarrow",
                                          compression="snappy", index=False)
    return len(results)


def run_pairwise(prices, n_workers):
    print(f"\n  STEP 1: Pairwise CPE screen (cutoff {TRAIN_CUTOFF.date()})...")

    all_tickers   = list(prices.columns)
    price_tickers = [t for t in all_tickers if t not in RATE_INDEX_TICKERS]
    rate_tickers  = [t for t in all_tickers if t in RATE_INDEX_TICKERS]
    predicted     = [t for t in price_tickers if t not in EXCLUDE_FROM_Y]
    predictors    = [t for t in all_tickers   if t not in EXCLUDE_FROM_X]

    # Hard history floor on predictors
    train_obs = {t: int((prices[t].dropna().index <= TRAIN_CUTOFF).sum())
                 for t in predictors}
    predictors = [t for t in predictors if train_obs.get(t, 0) >= MIN_TRAIN_OBS]

    short_excluded = [t for t in all_tickers
                      if t not in EXCLUDE_FROM_X
                      and train_obs.get(t, 0) < MIN_TRAIN_OBS
                      and train_obs.get(t, 0) > 0]
    if short_excluded:
        print(f"    Excluded for <{MIN_TRAIN_OBS} train obs: {short_excluded}")

    print(f"    Predicted Y: {len(predicted)}  "
          f"Predictors X: {len(predictors)}  Workers: {n_workers}")

    # Build training-restricted increments
    all_taus = sorted(set(TAU_PAST_LIST + TAU_FUTURE_LIST))
    increments = {}
    train_idx = prices.index[prices.index <= TRAIN_CUTOFF]
    for tau in all_taus:
        inc = pd.DataFrame(index=train_idx)
        for t in price_tickers:
            s = prices[t]
            inc[t] = np.log(s / s.shift(tau)).reindex(train_idx)
        for t in rate_tickers:
            s = prices[t]
            inc[t] = (s - s.shift(tau)).reindex(train_idx)
        increments[tau] = inc

    future_inc = {}
    for tau_f in TAU_FUTURE_LIST:
        fi = {}
        for t in predicted:
            if t in prices.columns:
                s = prices[t].reindex(train_idx)
                fi[t] = np.log(s / s.shift(tau_f)).shift(-tau_f)
        future_inc[tau_f] = pd.DataFrame(fi)

    full_q = sorted(set(Q_GRID + [round(1 - q, 10) for q in Q_GRID]))
    thresholds = {}
    for tau in all_taus:
        for q in full_q:
            thresholds[(tau, q)] = increments[tau].quantile(
                q, numeric_only=True).to_dict()

    config = dict(
        cpe_thresh=CPE_THRESH, min_n=MIN_N, min_lift=MIN_LIFT,
        tau_past=TAU_PAST_LIST, tau_future=TAU_FUTURE_LIST,
        q_grid=Q_GRID, min_confidence=MIN_CONFIDENCE,
        tmp_dir=TMP_DIR, train_cutoff=str(TRAIN_CUTOFF),
    )

    os.makedirs(TMP_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(TMP_DIR, "*.parquet")):
        os.remove(f)

    tasks = [(y, i) for i, y in enumerate(predicted)]
    t0 = time.time()
    n_kept = 0

    with Pool(n_workers, initializer=_init_pw,
              initargs=(increments, future_inc, thresholds, predictors, config)) as pool:
        for i, n in enumerate(pool.imap_unordered(_pw_worker, tasks)):
            n_kept += n
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta  = (len(tasks) - i - 1) / rate if rate > 0 else 0
            print(f"    [{i+1:>3}/{len(tasks)}]  kept={n_kept:>7,}  "
                  f"rate={rate:.1f} Y/s  ETA={eta/60:.1f}m", end="\r")

    print()
    elapsed = time.time() - t0
    print(f"    Done in {elapsed/60:.1f} min.  {n_kept:,} rows kept.")

    chunks = sorted(glob.glob(os.path.join(TMP_DIR, "*.parquet")))
    if not chunks:
        print("    WARNING: no rows survived filters.")
        return pd.DataFrame()

    df = pd.concat([pd.read_parquet(f) for f in chunks], ignore_index=True)
    df = df.sort_values(["direction","Y","tau_future","tau_past","q_Y","q_X","CPE"],
                         ascending=[True]*6+[False]).reset_index(drop=True)
    df.to_parquet(PAIRWISE_OUT, engine="pyarrow", compression="snappy", index=False)
    for f in chunks: os.remove(f)
    try: os.rmdir(TMP_DIR)
    except OSError: pass

    print(f"    Saved {len(df):,} rows → {PAIRWISE_OUT}")
    return df


# ── STEP 2: Joint screen ───────────────────────────────────────────────────

def cluster_episodes(firing_dates, gap_td):
    if len(firing_dates) == 0:
        return []
    dates = pd.DatetimeIndex(sorted(firing_dates))
    gap_cal = gap_td * 1.45 * EPISODE_GAP_MULTIPLIER
    eps, cur = [], [dates[0]]
    for i in range(1, len(dates)):
        if (dates[i] - dates[i-1]).days > gap_cal:
            eps.append(pd.DatetimeIndex(cur))
            cur = [dates[i]]
        else:
            cur.append(dates[i])
    eps.append(pd.DatetimeIndex(cur))
    return eps


def episode_conviction(joint_mask, common_idx, fy_vals, event_mask, longest_tau_p):
    firing = common_idx[joint_mask]
    if len(firing) == 0:
        return 0, np.nan, 0.0
    eps = cluster_episodes(firing, longest_tau_p)
    outcomes = []
    for ep in eps:
        pos = common_idx.get_loc(ep[-1])
        if pos < len(fy_vals):
            outcomes.append(bool(event_mask[pos]))
    n = len(outcomes)
    if n < MIN_EPISODES_FOR_CONVICTION:
        return n, np.nan, 0.0
    hr = float(np.mean(outcomes))
    return n, hr, float(np.log(n) * max(0.0, 2 * hr - 1))


def run_joint(pairwise_df, prices):
    if pairwise_df.empty:
        print("    Joint screen skipped: empty pairwise input.")
        return pd.DataFrame()

    print(f"\n  STEP 2: Joint screen ({len(pairwise_df):,} pairwise rows)...")

    all_tickers   = list(prices.columns)
    price_tickers = [t for t in all_tickers if t not in RATE_INDEX_TICKERS]
    rate_tickers  = [t for t in all_tickers if t in RATE_INDEX_TICKERS]
    train_idx     = prices.index[prices.index <= TRAIN_CUTOFF]

    all_taus = sorted(set(TAU_PAST_LIST + TAU_FUTURE_LIST))
    t0 = time.time()
    increments = {}
    for tau in all_taus:
        inc = pd.DataFrame(index=train_idx)
        for t in price_tickers:
            s = prices[t]
            inc[t] = np.log(s / s.shift(tau)).reindex(train_idx)
        for t in rate_tickers:
            s = prices[t]
            inc[t] = (s - s.shift(tau)).reindex(train_idx)
        increments[tau] = inc

    future_inc = {}
    for tau_f in TAU_FUTURE_LIST:
        fi = {}
        for t in pairwise_df["Y"].unique():
            if t in prices.columns:
                s = prices[t].reindex(train_idx)
                fi[t] = np.log(s / s.shift(tau_f)).shift(-tau_f)
        future_inc[tau_f] = pd.DataFrame(fi)

    full_q = sorted(set(Q_GRID + [round(1 - q, 10) for q in Q_GRID]))
    thresholds = {}
    for tau in all_taus:
        for q in full_q:
            thresholds[(tau, q)] = increments[tau].quantile(
                q, numeric_only=True).to_dict()

    def get_mask(x, tau_p, q_x, direction, common_idx):
        if tau_p not in increments or x not in increments[tau_p].columns:
            return None
        series = increments[tau_p][x].reindex(common_idx)
        if direction == "bullish":
            thresh = thresholds.get((tau_p, q_x), {}).get(x, np.nan)
            if np.isnan(thresh): return None
            return (series > thresh).values
        else:
            thresh = thresholds.get((tau_p, round(1 - q_x, 10)), {}).get(x, np.nan)
            if np.isnan(thresh): return None
            return (series < thresh).values

    def rank_key(row, common_idx):
        mask = get_mask(row["X"], int(row["tau_past"]), row["q_X"],
                        row.get("direction","bullish"), common_idx)
        if mask is None: return (0, 0.0)
        eps = cluster_episodes(common_idx[mask], int(row["tau_past"]))
        return (len(eps), float(row["CPE"]))

    results = []
    groups  = pairwise_df.groupby(["Y","tau_future","q_Y","direction"])
    n_groups = len(groups)

    for gi, ((y, tau_f, q_y, direction), grp) in enumerate(groups):
        if gi % 200 == 0:
            print(f"    Group {gi:>4}/{n_groups}  results: {len(results):,}", end="\r")

        if tau_f not in future_inc or y not in future_inc[tau_f].columns:
            continue

        fy       = future_inc[tau_f][y]
        common_idx = fy.dropna().index
        if len(common_idx) < MIN_N:
            continue

        fy_vals = fy.loc[common_idx].values
        thresh_up = thresholds.get((tau_f, q_y), {}).get(y, np.nan)
        thresh_dn = thresholds.get((tau_f, round(1 - q_y, 10)), {}).get(y, np.nan)

        if direction == "bullish":
            if np.isnan(thresh_up): continue
            event    = fy_vals > thresh_up
            uncond_p = float(np.nanmean(event))
        else:
            if np.isnan(thresh_dn): continue
            event    = fy_vals < thresh_dn
            uncond_p = float(np.nanmean(event))

        if uncond_p <= 0:
            continue

        # Sort candidates by episode-aware rank
        candidates = grp.copy()
        candidates["_rk"] = candidates.apply(
            lambda r: rank_key(r, common_idx), axis=1)
        candidates = candidates.sort_values("_rk", ascending=False)

        selected, sel_tickers = [], set()
        joint_mask = None

        for _, seed_row in candidates.iterrows():
            m = get_mask(seed_row["X"], int(seed_row["tau_past"]),
                         seed_row["q_X"], direction, common_idx)
            if m is None or m.sum() < MIN_N: continue
            seed_cpe = float(np.nanmean(event[m]))
            if seed_cpe < CPE_THRESH: continue
            selected   = [seed_row.to_dict()]
            sel_tickers = {seed_row["X"]}
            joint_mask  = m
            break

        if joint_mask is None:
            continue

        while len(selected) < MAX_PREDICTORS:
            best_rk, best_cpe = (-1,-1), -1
            best_row = best_mask = None
            best_n   = 0

            for _, cand in candidates.iterrows():
                if cand["X"] in sel_tickers: continue
                cm = get_mask(cand["X"], int(cand["tau_past"]),
                              cand["q_X"], direction, common_idx)
                if cm is None: continue
                trial = joint_mask & cm
                n_t   = int(trial.sum())
                if n_t < MIN_N: continue
                cpe  = float(np.nanmean(event[trial]))
                lift = cpe / uncond_p if uncond_p > 0 else np.nan
                if cpe >= CPE_THRESH and not np.isnan(lift) and lift >= MIN_LIFT:
                    rk = rank_key(cand, common_idx)
                    if rk > best_rk:
                        best_rk, best_cpe = rk, cpe
                        best_row, best_mask, best_n = cand, trial, n_t

            if best_row is None: break
            selected.append(best_row.to_dict())
            sel_tickers.add(best_row["X"])
            joint_mask = best_mask

            if len(selected) >= 2:
                lift = best_cpe / uncond_p
                longest_tp = max(int(r["tau_past"]) for r in selected)
                n_ep, hr, conv = episode_conviction(
                    joint_mask, common_idx, fy_vals, event, longest_tp)
                results.append({
                    "Y": y, "direction": direction,
                    "tau_future": int(tau_f), "q_Y": q_y,
                    "n_predictors": len(selected),
                    "predictors":  [r["X"]           for r in selected],
                    "tau_pasts":   [int(r["tau_past"]) for r in selected],
                    "q_Xs":        [r["q_X"]          for r in selected],
                    "joint_CPE":   round(best_cpe, 4),
                    "uncond_prob": round(uncond_p, 4),
                    "lift":        round(lift, 4),
                    "n_joint":     best_n,
                    "n_episodes":  n_ep,
                    "episode_hit_rate": None if np.isnan(hr) else round(hr, 4),
                    "episode_conviction": round(conv, 4),
                })

    print()
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.0f}s.  {len(results):,} joint configs.")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df.to_parquet(JOINT_OUT, engine="pyarrow", compression="snappy", index=False)
    print(f"    Saved → {JOINT_OUT}")
    return df


# ── STEP 3: Multi-year hold-to-horizon backtest ────────────────────────────

def run_hth_multiyear(joint, prices, eval_dates, neutral_weights):
    """
    Hold-to-horizon across the full 2015–2025 evaluation period.
    Same logic as run_backtest.py's run_hold_to_horizon but operating
    over a multi-year window with the training cutoff frozen at 2014-12-31.
    """
    _be.TRAIN_CUTOFF = TRAIN_CUTOFF
    _be.EVAL_START   = EVAL_START
    _be.EVAL_END     = EVAL_END
    _be.SLEEVES.clear()
    _be.SLEEVES.update(BASE_SLEEVES)
    _be.NEUTRAL_WEIGHTS.clear()
    _be.NEUTRAL_WEIGHTS.update(neutral_weights)

    Q_GRID_BT = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]
    increments, thresholds = build_increments_and_thresholds(prices, Q_GRID_BT)

    weights = compute_quality_weights(joint, prices,
                                       precomputed_increments=increments)

    sleeve_proxies   = set(BASE_SLEEVES.values())
    tradeable_mask   = joint["Y"].isin(sleeve_proxies)
    tradeable_weights = weights[tradeable_mask]

    if len(tradeable_weights) == 0 or tradeable_weights.max() <= 0:
        w95 = 1.0
    else:
        w95 = float(np.percentile(tradeable_weights, 95))
        if w95 <= 0:
            w95 = tradeable_weights.max()

    active_holds    = {s: [] for s in BASE_SLEEVES}
    raw_weight_df   = pd.DataFrame(
        {s: neutral_weights[s] for s in BASE_SLEEVES},
        index=eval_dates, dtype=float
    )
    hold_events     = []   # for randomisation test
    fire_cache      = {}

    # Pre-compute firing for all tradeable configs
    for sleeve, proxy in BASE_SLEEVES.items():
        sub = joint[(joint["Y"] == proxy) &
                    tradeable_mask.reindex(joint.index, fill_value=False)]
        print(f"  [{sleeve:<10}] {len(sub)} configs across "
              f"{len(eval_dates)} days...", end=" ", flush=True)
        t0 = time.time()
        for idx, row in sub.iterrows():
            fires = {}
            prev  = False
            for d in eval_dates:
                today = configuration_fires_on_date(row, d, increments, thresholds)
                fires[d] = (today, today and not prev)
                prev = today
            fire_cache[idx] = (row, fires)
        print(f"{time.time()-t0:.0f}s")

    # Build daily weight path and collect hold events
    n_eval = len(eval_dates)
    for sleeve, proxy in BASE_SLEEVES.items():
        rel = [idx for idx, (row, _) in fire_cache.items()
               if row["Y"] == proxy]
        for d in eval_dates:
            active_holds[sleeve] = [
                h for h in active_holds[sleeve] if h["expiry"] > d
            ]
            for idx in rel:
                row, fires = fire_cache[idx]
                fired_today, newly = fires[d]
                if newly:
                    tau_f    = int(row["tau_future"])
                    expiry   = d + pd.Timedelta(days=int(tau_f * 1.45))
                    sign     = 1.0 if row["direction"] == "bullish" else -1.0
                    w_idx    = weights.loc[idx] if idx in weights.index else 0.0
                    conviction = min(w_idx / w95, 1.0) if w95 > 0 else 0.0
                    tilt     = sign * conviction * 100.0
                    active_holds[sleeve].append({"expiry": expiry, "tilt": tilt})
                    hold_events.append({
                        "sleeve":       sleeve,
                        "entry_date":   d,
                        "duration_days": int(tau_f * 1.45),
                        "tilt":         tilt,
                        "tau_future":   tau_f,
                        "Y":            proxy,
                        "predictors":   list(row["predictors"]),
                        "joint_CPE":    float(row["joint_CPE"]),
                        "episode_conviction": float(
                            row.get("episode_conviction", np.nan)),
                    })
            if active_holds[sleeve]:
                tilt_today = max(active_holds[sleeve],
                                 key=lambda h: abs(h["tilt"]))["tilt"]
            else:
                tilt_today = 0.0
            raw_weight_df.at[d, sleeve] = neutral_weights[sleeve] + tilt_today

    clipped = pd.DataFrame(
        [clip_and_renormalise(raw_weight_df.loc[d].to_dict())
         for d in eval_dates],
        index=eval_dates
    )
    lagged = clipped.shift(1)
    lagged.iloc[0] = pd.Series(neutral_weights)

    equity = simulate_portfolio(lagged, prices, eval_dates)
    stats  = compute_performance_stats(equity["equity"])

    return {
        "equity_curve": equity,
        "stats":        stats,
        "hold_events":  hold_events,
        "increments":   increments,
        "thresholds":   thresholds,
        "w95":          w95,
        "weights":      weights,
    }


# ── STEP 4: Multi-year randomisation test ─────────────────────────────────

def randomisation_test_multiyear(hold_events, prices, eval_dates,
                                  neutral_weights, actual_sharpe,
                                  n_reps=1000, seed=42):
    """
    Single randomisation test across the full 11-year evaluation period.
    Shuffles each hold event's entry date uniformly across ALL eval_dates,
    preserving duration and magnitude exactly.
    """
    if not hold_events:
        return {"note": "No hold events — test not possible"}

    n_eval      = len(eval_dates)
    eval_list   = list(eval_dates)
    rng         = np.random.default_rng(seed)
    null_sharpes = []

    print(f"  Randomisation test: {len(hold_events)} hold events, "
          f"{n_eval} eval days, {n_reps} reps...")
    t0 = time.time()

    for rep in range(n_reps):
        raw = pd.DataFrame(
            {s: neutral_weights[s] for s in BASE_SLEEVES},
            index=eval_dates, dtype=float
        )
        active = {s: [] for s in BASE_SLEEVES}

        # Randomly shift each hold's entry to any eval date
        shuffled = []
        for ev in hold_events:
            new_idx = int(rng.integers(0, n_eval))
            shuffled.append({**ev, "entry_date": eval_list[new_idx]})

        for di, d in enumerate(eval_dates):
            for s in BASE_SLEEVES:
                active[s] = [h for h in active[s] if h["expiry"] > d]
            for ev in shuffled:
                if ev["entry_date"] == d:
                    s      = ev["sleeve"]
                    expiry = d + pd.Timedelta(days=ev["duration_days"])
                    active[s].append({"expiry": expiry, "tilt": ev["tilt"]})
            for s in BASE_SLEEVES:
                if active[s]:
                    tilt = max(active[s], key=lambda h: abs(h["tilt"]))["tilt"]
                else:
                    tilt = 0.0
                raw.at[d, s] = neutral_weights[s] + tilt

        clipped = pd.DataFrame(
            [clip_and_renormalise(raw.loc[d].to_dict()) for d in eval_dates],
            index=eval_dates
        )
        lagged = clipped.shift(1)
        lagged.iloc[0] = pd.Series(neutral_weights)
        eq = simulate_portfolio(lagged, prices, eval_dates)
        sh = compute_performance_stats(eq["equity"])["sharpe"]
        if not np.isnan(sh):
            null_sharpes.append(sh)

        if (rep + 1) % 100 == 0:
            print(f"    {rep+1}/{n_reps} reps  "
                  f"elapsed {time.time()-t0:.0f}s", end="\r")

    print()
    null = np.array(null_sharpes)
    pct  = float((null >= actual_sharpe).mean() * 100)

    return {
        "actual_sharpe": actual_sharpe,
        "null_mean":     float(null.mean()),
        "null_std":      float(null.std()),
        "pct_exceeding": round(pct, 1),
        "n_reps":        len(null),
        "n_holds":       len(hold_events),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Single-screen OOS: train ≤2014, evaluate 2015-2025"
    )
    parser.add_argument("--prices",             default="multiasset_prices.parquet")
    parser.add_argument("--skip-pairwise",      action="store_true")
    parser.add_argument("--skip-joint",         action="store_true")
    parser.add_argument("--skip-randomisation", action="store_true")
    parser.add_argument("--n-reps",   type=int, default=1000)
    parser.add_argument("--n-workers",type=int,
                        default=max(1, cpu_count() - 1))
    parser.add_argument("--output",             default="oos_2015_2025_results.csv")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  SINGLE-SCREEN OOS WALK-FORWARD")
    print(f"  Training cutoff:  {TRAIN_CUTOFF.date()}")
    print(f"  Evaluation:       {EVAL_START.date()} → {EVAL_END.date()} "
          f"(11 years, single continuous period)")
    print(f"  Universe:         All 161 instruments, economic prior")
    print(f"  Weights:          Capped (Crypto≤{CRYPTO_CAP_PCT}%, "
          f"Gold≤{GOLD_CAP_PCT}%)")
    print(f"  Workers:          {args.n_workers}")
    print(f"{'='*72}")

    prices = pd.read_parquet(args.prices)
    print(f"\n  Price history: {prices.index.min().date()} to "
          f"{prices.index.max().date()}")
    print(f"  Instruments:   {prices.shape[1]}")

    # VIXM/VIXY status at training cutoff
    for t in ["VIXM", "VIXY"]:
        if t in prices.columns:
            n = int((prices[t].dropna().index <= TRAIN_CUTOFF).sum())
            print(f"  {t} training obs at {TRAIN_CUTOFF.date()}: {n}")

    # ── Step 1: Pairwise ──────────────────────────────────────────────────
    if args.skip_pairwise and os.path.exists(PAIRWISE_OUT):
        print(f"\n  Step 1 SKIPPED (reusing {PAIRWISE_OUT})")
        pairwise_df = pd.read_parquet(PAIRWISE_OUT)
        print(f"    Loaded {len(pairwise_df):,} pairwise rows")
    else:
        t0 = time.time()
        pairwise_df = run_pairwise(prices, args.n_workers)
        print(f"  Step 1 complete in {(time.time()-t0)/60:.1f} min")

    if pairwise_df.empty:
        sys.exit("No pairwise results — cannot continue")

    # ── Step 2: Joint ─────────────────────────────────────────────────────
    if args.skip_joint and os.path.exists(JOINT_OUT):
        print(f"\n  Step 2 SKIPPED (reusing {JOINT_OUT})")
        joint_df = pd.read_parquet(JOINT_OUT)
        print(f"    Loaded {len(joint_df):,} joint configs")
    else:
        t0 = time.time()
        joint_df = run_joint(pairwise_df, prices)
        print(f"  Step 2 complete in {(time.time()-t0):.0f}s")

    if joint_df.empty:
        sys.exit("No joint configs — cannot continue")

    joint_filtered = joint_df[joint_df["n_predictors"] <= MAX_PREDICTORS].copy()
    print(f"\n  Joint configs (n_pred ≤ {MAX_PREDICTORS}): {len(joint_filtered)}")

    # Screen diagnostics
    spy_rows = joint_filtered[joint_filtered["Y"] == "SPY"]
    vv_rows  = spy_rows[spy_rows["predictors"].apply(
        lambda p: {"VIXM","VIXY"}.issubset(set(p))
    )]
    print(f"  SPY configs:          {len(spy_rows)}")
    print(f"  VIXM+VIXY→SPY:        {len(vv_rows)}")
    if len(vv_rows) > 0 and "episode_conviction" in vv_rows.columns:
        n_active = int((vv_rows["episode_conviction"] > 0).sum())
        print(f"  VIXM+VIXY active:     {n_active} "
              f"(episode_conviction > 0)")
        for _, r in vv_rows.iterrows():
            print(f"    tau_f={r['tau_future']}  "
                  f"CPE={r['joint_CPE']:.3f}  "
                  f"conv={r['episode_conviction']:.4f}  "
                  f"n_ep={r['n_episodes']}")

    # ── Neutral weights ───────────────────────────────────────────────────
    _be.TRAIN_CUTOFF = TRAIN_CUTOFF
    raw_weights    = compute_neutral_weights(BASE_SLEEVES, prices)
    capped_weights = apply_weight_cap(raw_weights, WEIGHT_CAPS)
    print(f"\n  Neutral weights (capped):")
    for k, v in capped_weights.items():
        raw = raw_weights[k]
        flag = f"  (raw {raw:.1f}%, capped)" if abs(v - raw) > 0.1 else ""
        print(f"    {k:<12}: {v:.2f}%{flag}")

    # ── Evaluation dates (all trading days 2015–2025) ────────────────────
    mask       = (prices.index >= EVAL_START) & (prices.index <= EVAL_END)
    spy_valid  = prices["SPY"].notna()
    eval_dates = prices.index[mask & spy_valid]
    print(f"\n  Evaluation dates: {eval_dates[0].date()} to "
          f"{eval_dates[-1].date()} ({len(eval_dates)} trading days)")

    # ── Step 3: HTH backtest ─────────────────────────────────────────────
    print(f"\n  STEP 3: Hold-to-horizon backtest (2015–2025)...")
    t0  = time.time()
    hth = run_hth_multiyear(joint_filtered, prices, eval_dates, capped_weights)
    print(f"  Done in {time.time()-t0:.0f}s")
    print(f"  HTH result: ret={hth['stats']['total_return_pct']}%  "
          f"Sharpe={hth['stats']['sharpe']}  "
          f"vol={hth['stats']['ann_vol_pct']}%")
    print(f"  Total hold events: {len(hth['hold_events'])}")

    # Hold event breakdown
    if hth["hold_events"]:
        hdf = pd.DataFrame(hth["hold_events"])
        print(f"\n  Hold events by sleeve:")
        for s, grp in hdf.groupby("sleeve"):
            print(f"    {s:<12}: {len(grp)} holds")
        print(f"\n  Hold events by year:")
        hdf["year"] = hdf["entry_date"].dt.year
        for yr, grp in hdf.groupby("year"):
            print(f"    {yr}: {len(grp)} holds")

    # Benchmarks
    _be.NEUTRAL_WEIGHTS.clear()
    _be.NEUTRAL_WEIGHTS.update(capped_weights)
    bench  = run_no_tilt_benchmark(prices, eval_dates)
    spy_bh = run_buy_and_hold(prices, "SPY", eval_dates)
    print(f"\n  No-tilt benchmark:  "
          f"ret={bench['stats']['total_return_pct']}%  "
          f"Sharpe={bench['stats']['sharpe']}")
    print(f"  SPY buy-and-hold:   "
          f"ret={spy_bh['stats']['total_return_pct']}%  "
          f"Sharpe={spy_bh['stats']['sharpe']}")

    # ── Step 4: Randomisation test ────────────────────────────────────────
    rtest = None
    if not args.skip_randomisation:
        if len(hth["hold_events"]) >= 3:
            print(f"\n  STEP 4: Randomisation test across full 2015–2025 period...")
            t0    = time.time()
            rtest = randomisation_test_multiyear(
                hth["hold_events"], prices, eval_dates, capped_weights,
                actual_sharpe=hth["stats"]["sharpe"],
                n_reps=args.n_reps,
            )
            print(f"  Done in {(time.time()-t0)/60:.1f} min")
            print(f"\n  Actual Sharpe:   {rtest['actual_sharpe']}")
            print(f"  Null mean:       {rtest['null_mean']:.3f}")
            print(f"  Null std:        {rtest['null_std']:.3f}")
            print(f"  Pct exceeding:   {rtest['pct_exceeding']}%")
            print(f"  N holds pooled:  {rtest['n_holds']}")
            print(f"  N reps:          {rtest['n_reps']}")
            if rtest["pct_exceeding"] <= 5.0:
                print(f"\n  *** SIGNIFICANT at 5% level ***")
            elif rtest["pct_exceeding"] <= 10.0:
                print(f"\n  *** SIGNIFICANT at 10% level ***")
            else:
                print(f"\n  Not significant at 10% level")
        else:
            print(f"\n  Randomisation test skipped: "
                  f"{len(hth['hold_events'])} hold events (need ≥ 3)")

    # ── Year-by-year breakdown ────────────────────────────────────────────
    print(f"\n\n{'='*72}")
    print(f"  YEAR-BY-YEAR PERFORMANCE (single screen, 2014-12-31 cutoff)")
    print(f"{'='*72}")
    print(f"\n  {'Year':>6}  {'HTH ret%':>9}  {'HTH Sh':>7}  "
          f"{'Bench ret%':>11}  {'Bench Sh':>9}  "
          f"{'SPY ret%':>9}  {'Holds':>6}")
    print(f"  {'─'*72}")

    yearly_rows = []
    for yr in range(2015, 2026):
        yr_mask    = (prices.index >= pd.Timestamp(f"{yr}-01-01")) & \
                     (prices.index <= pd.Timestamp(f"{yr}-12-31"))
        yr_spy     = prices["SPY"].notna()
        yr_dates   = prices.index[yr_mask & yr_spy]
        if len(yr_dates) < 50:
            continue

        # Slice equity curves
        hth_eq  = hth["equity_curve"]["equity"]
        bench_eq = bench["equity_curve"]["equity"]
        spy_eq   = spy_bh["equity_curve"]

        # Normalise to start-of-year value for annual return
        hth_yr  = hth_eq.reindex(yr_dates)
        bench_yr = bench_eq.reindex(yr_dates)
        spy_yr   = spy_eq.reindex(yr_dates)

        def yr_ret(s):
            s = s.dropna()
            if len(s) < 2: return np.nan
            return round((s.iloc[-1] / s.iloc[0] - 1) * 100, 2)

        def yr_sh(s):
            s = s.dropna()
            if len(s) < 2: return np.nan
            r = s.pct_change().dropna()
            return round((r.mean() / r.std()) * np.sqrt(252), 3) if r.std() > 0 else np.nan

        hth_ret  = yr_ret(hth_yr)
        bench_ret = yr_ret(bench_yr)
        spy_ret   = yr_ret(spy_yr)
        hth_sh   = yr_sh(hth_yr)
        bench_sh  = yr_sh(bench_yr)

        # Count holds in this year
        n_holds_yr = sum(
            1 for ev in hth["hold_events"]
            if ev["entry_date"].year == yr
        ) if hth["hold_events"] else 0

        print(f"  {yr:>6}  {hth_ret:>9.2f}%  {hth_sh:>7.3f}  "
              f"{bench_ret:>11.2f}%  {bench_sh:>9.3f}  "
              f"{spy_ret:>9.2f}%  {n_holds_yr:>6}")

        yearly_rows.append({
            "year": yr, "hth_ret_pct": hth_ret, "hth_sharpe": hth_sh,
            "bench_ret_pct": bench_ret, "bench_sharpe": bench_sh,
            "spy_ret_pct": spy_ret, "holds_yr": n_holds_yr,
        })

    # ── Full-period summary ───────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  FULL PERIOD SUMMARY (2015–2025, ~2,750 trading days)")
    print(f"{'='*72}")
    print(f"\n  Strategy (HTH):     "
          f"ret={hth['stats']['total_return_pct']}%  "
          f"Sharpe={hth['stats']['sharpe']}  "
          f"vol={hth['stats']['ann_vol_pct']}%")
    print(f"  No-tilt benchmark:  "
          f"ret={bench['stats']['total_return_pct']}%  "
          f"Sharpe={bench['stats']['sharpe']}")
    print(f"  SPY buy-and-hold:   "
          f"ret={spy_bh['stats']['total_return_pct']}%  "
          f"Sharpe={spy_bh['stats']['sharpe']}")
    print(f"  Total hold events:  {len(hth['hold_events'])}")

    if rtest:
        print(f"\n  RANDOMISATION TEST (single test, full 2015–2025 period):")
        print(f"    Actual Sharpe:   {rtest['actual_sharpe']}")
        print(f"    Null mean:       {rtest['null_mean']:.3f}")
        print(f"    Null std:        {rtest['null_std']:.3f}")
        print(f"    Pct exceeding:   {rtest['pct_exceeding']}%  "
              f"({rtest['n_holds']} hold events pooled, {rtest['n_reps']} reps)")

        excess = hth["stats"]["sharpe"] - bench["stats"]["sharpe"]
        print(f"\n  Sharpe excess over benchmark: {excess:+.3f}")

        if rtest["pct_exceeding"] <= 5.0:
            verdict = ("SIGNIFICANT at 5% — the CPE framework's timing of "
                       "hold entries adds demonstrable value over the full "
                       "2015–2025 OOS period under a screen trained solely "
                       "on pre-2015 data.")
        elif rtest["pct_exceeding"] <= 10.0:
            verdict = ("SIGNIFICANT at 10% — borderline. The framework shows "
                       "evidence of timing skill across the 11-year OOS period "
                       "but does not clear the 5% threshold used throughout "
                       "this paper series.")
        else:
            verdict = ("NOT SIGNIFICANT — the CPE framework trained on "
                       f"pre-2015 data does not demonstrate timing skill "
                       f"at the 10% level across the full 2015–2025 OOS "
                       f"period. The 2025 paper result may reflect a screen "
                       f"trained on data through 2024 being better calibrated "
                       f"to that specific market environment.")
        print(f"\n  VERDICT: {verdict}")

    # ── Save ──────────────────────────────────────────────────────────────
    summary = {
        "train_cutoff": str(TRAIN_CUTOFF.date()),
        "eval_start":   str(EVAL_START.date()),
        "eval_end":     str(EVAL_END.date()),
        "eval_days":    len(eval_dates),
        "joint_configs": len(joint_filtered),
        "vv_spy_configs": len(vv_rows),
        "total_holds":   len(hth["hold_events"]),
        "hth_ret_pct":   hth["stats"]["total_return_pct"],
        "hth_sharpe":    hth["stats"]["sharpe"],
        "hth_vol_pct":   hth["stats"]["ann_vol_pct"],
        "bench_ret_pct": bench["stats"]["total_return_pct"],
        "bench_sharpe":  bench["stats"]["sharpe"],
        "spy_ret_pct":   spy_bh["stats"]["total_return_pct"],
        "spy_sharpe":    spy_bh["stats"]["sharpe"],
        "pct_exceeding": rtest["pct_exceeding"] if rtest else "skipped",
        "rand_null_mean": rtest["null_mean"] if rtest else np.nan,
        "rand_null_std":  rtest["null_std"] if rtest else np.nan,
        "rand_n_holds":   rtest["n_holds"] if rtest else 0,
        "rand_n_reps":    rtest["n_reps"] if rtest else 0,
    }
    pd.DataFrame([summary]).to_csv(args.output, index=False)
    pd.DataFrame(yearly_rows).to_csv(
        args.output.replace(".csv", "_yearly.csv"), index=False)
    if hth["hold_events"]:
        hdf = pd.DataFrame(hth["hold_events"])
        hdf["entry_date"] = hdf["entry_date"].astype(str)
        hdf.to_csv(args.output.replace(".csv", "_holds.csv"), index=False)

    print(f"\n  Saved: {args.output}")
    print(f"  Saved: {args.output.replace('.csv','_yearly.csv')}")
    print(f"  Saved: {args.output.replace('.csv','_holds.csv')}")
    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    main()
