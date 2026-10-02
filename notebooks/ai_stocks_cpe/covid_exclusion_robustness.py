"""
covid_exclusion_robustness.py
========================================
Robustness check on the daily CPE regime comparison (ai_cpe_regime_comparison.py):
the pre-ChatGPT window (2019-02-07 to 2022-11-29) contains the COVID crash
(Feb-Apr 2020), a single, extreme, crisis-driven cluster of tail
co-movement that could be doing most of the work in the pre>post decline
reported in the main result. This re-runs the identical scan with the
COVID crash months excluded from the pre-window entirely (not just
down-weighted), to see whether the decline survives removing the single
biggest confound.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import json

from ai_cpe_regime_comparison import build_increments, TAU_PAST, TAU_FUTURE, Q_GRID, CPE_THRESH, MIN_LIFT, MIN_N

COVID_START = pd.Timestamp("2020-02-15")
COVID_END = pd.Timestamp("2020-04-30")


def run_regime_scan_excluding(prices: pd.DataFrame, window_start, window_end, exclude_start, exclude_end, label: str) -> pd.DataFrame:
    all_taus = sorted(set(TAU_PAST + TAU_FUTURE))
    increments = build_increments(prices, all_taus)
    future_inc = {tau_f: increments[tau_f].shift(-tau_f) for tau_f in TAU_FUTURE}
    full_q_grid = sorted(set(Q_GRID + [round(1 - q, 10) for q in Q_GRID]))

    def regime_mask(idx):
        m = (idx >= window_start) & (idx <= window_end)
        if exclude_start is not None:
            m = m & ~((idx >= exclude_start) & (idx <= exclude_end))
        return m

    thresholds = {}
    for tau in all_taus:
        mask = regime_mask(increments[tau].index)
        regime_inc = increments[tau].loc[mask]
        for q in full_q_grid:
            thresholds[(tau, q)] = regime_inc.quantile(q, numeric_only=True).to_dict()

    tickers = list(prices.columns)
    results = []
    for y in tickers:
        for tau_f in TAU_FUTURE:
            fy_full = future_inc[tau_f][y]
            mask_y = regime_mask(fy_full.index)
            fy = fy_full.loc[mask_y].dropna()
            if len(fy) < MIN_N:
                continue
            for tau_p in TAU_PAST:
                px_all = increments[tau_p]
                common_idx = fy.index.intersection(px_all.dropna(how="all").index)
                if len(common_idx) < MIN_N:
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
                    for x in tickers:
                        px_vals = px_aligned[x].values
                        valid = ~np.isnan(px_vals)
                        if valid.sum() < MIN_N:
                            continue
                        for q_x in Q_GRID:
                            thresh_x_up = thresholds[(tau_p, q_x)].get(x, np.nan)
                            thresh_x_dn = thresholds[(tau_p, round(1 - q_x, 10))].get(x, np.nan)
                            if np.isnan(thresh_x_up) or np.isnan(thresh_x_dn):
                                continue
                            cond_bull = valid & (px_vals > thresh_x_up)
                            if cond_bull.sum() >= MIN_N:
                                cpe = event_bull[cond_bull].mean()
                                lift = cpe / uncond if uncond > 0 else np.nan
                                if cpe >= CPE_THRESH and lift >= MIN_LIFT:
                                    results.append((y, x, tau_p, tau_f, q_x, q_y, round(float(cpe),4), round(float(lift),4), int(cond_bull.sum()), "bullish"))
                            cond_bear = valid & (px_vals < thresh_x_dn)
                            if cond_bear.sum() >= MIN_N:
                                cpe = event_bear[cond_bear].mean()
                                lift = cpe / uncond if uncond > 0 else np.nan
                                if cpe >= CPE_THRESH and lift >= MIN_LIFT:
                                    results.append((y, x, tau_p, tau_f, q_x, q_y, round(float(cpe),4), round(float(lift),4), int(cond_bear.sum()), "bearish"))

    cols = ["Y","X","tau_past","tau_future","q_X","q_Y","CPE","lift","n_condition","direction"]
    df = pd.DataFrame(results, columns=cols)
    print(f"[{label}] {len(df)} surviving configs")
    return df


if __name__ == "__main__":
    prices = pd.read_parquet("aiq_universe_prices.parquet")
    pre_start, pre_end = pd.Timestamp("2019-02-07"), pd.Timestamp("2022-11-29")

    print("Re-running PRE-ChatGPT window with COVID crash (2020-02-15 to 2020-04-30) excluded...")
    pre_excl = run_regime_scan_excluding(prices, pre_start, pre_end, COVID_START, COVID_END, "PRE-ChatGPT, COVID excluded")
    pre_excl_tail = pre_excl[pre_excl["q_Y"] >= 0.80]
    print(f"Tail-only (q_Y>=0.80): {len(pre_excl_tail)} configs "
          f"(bull={int((pre_excl_tail.direction=='bullish').sum())}, bear={int((pre_excl_tail.direction=='bearish').sum())})")

    post = pd.read_parquet("ai_cpe_post_chatgpt_results.parquet")
    post_tail = post[post["q_Y"] >= 0.80]

    print(f"\nComparison:")
    print(f"  PRE (COVID excluded), tail-only:  {len(pre_excl_tail)}")
    print(f"  POST (unchanged), tail-only:      {len(post_tail)}")

    json.dump({
        "pre_covid_excluded_tail_configs": len(pre_excl_tail),
        "pre_covid_excluded_bull": int((pre_excl_tail.direction=="bullish").sum()),
        "pre_covid_excluded_bear": int((pre_excl_tail.direction=="bearish").sum()),
        "post_tail_configs": len(post_tail),
        "post_bull": int((post_tail.direction=="bullish").sum()),
        "post_bear": int((post_tail.direction=="bearish").sum()),
    }, open("covid_exclusion_robustness_results.json", "w"), indent=2)
