"""
ai_cpe_regime_comparison.py
========================================
Applies this program's exact core CPE methodology (cpe_engine_parallel.py:
same tau grid, same quantile grid, same CPE>=0.80/lift>=1.5/n>=100 gates)
to a frozen, dated AI-stock universe, comparing pre- vs. post-ChatGPT
regimes.

Universe: 52 US-listed tickers from Global X's AIQ ETF (tracking the
Indxx AI & Big Data Index), real holdings as of 2022-12-01 (sourced from
an archived Wayback Machine snapshot of the fund's own holdings page --
not a hand-picked "today's AI winners" list). 7 of the original 59
US-listed names (ABMD, SPLK, SMAR, CCCS, INFA, EXAI, VRNT) were acquired
or delisted since 2022 and are excluded -- a real fact about this
frozen universe, not silently dropped.

Regime split: 2022-11-30, ChatGPT's public launch date, chosen because
it is the actual event this program's own framing points to, not a date
picked to flatter either window.

Design: quantile thresholds AND the CPE/lift gate are computed SEPARATELY
within each regime's own increment distribution (not one pooled
distribution) -- this isolates "did the conditional structure change,"
not just "did the raw price level change." Trailing lookback windows
are still built from the continuous price series (a stock doesn't reset
its own history at a regime boundary), but every quantile threshold and
every evaluated exceedance event belongs entirely to one regime or the
other.

Every reported number is a real point estimate; no significance
threshold beyond this program's own standing CPE/lift/n gates.

Run: python ai_cpe_regime_comparison.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import json

REGIME_SPLIT = pd.Timestamp("2022-11-30")  # ChatGPT public launch

TAU_PAST   = [1, 5, 10, 21, 63, 126, 252, 300]
TAU_FUTURE = [1, 5, 10, 21, 63, 126, 252, 300]
Q_GRID     = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]
CPE_THRESH = 0.80
MIN_LIFT   = 1.5
MIN_N      = 100


def build_increments(prices: pd.DataFrame, taus: list) -> dict:
    out = {}
    for tau in taus:
        inc = pd.DataFrame(index=prices.index)
        for t in prices.columns:
            s = prices[t]
            inc[t] = np.log(s / s.shift(tau))
        out[tau] = inc
    return out


def run_regime_scan(prices: pd.DataFrame, window_start, window_end, label: str) -> pd.DataFrame:
    """Runs the exact cpe_engine_parallel.py logic, restricted to dates
    within [window_start, window_end] for both threshold computation and
    the evaluated exceedance events. Trailing increments may reach back
    before window_start (real prior history), but every date SCORED
    belongs to this regime."""
    all_taus = sorted(set(TAU_PAST + TAU_FUTURE))
    increments = build_increments(prices, all_taus)

    future_inc = {}
    for tau_f in TAU_FUTURE:
        future_inc[tau_f] = increments[tau_f].shift(-tau_f)

    full_q_grid = sorted(set(Q_GRID + [round(1 - q, 10) for q in Q_GRID]))

    # regime-scoped thresholds: quantiles computed ONLY from this
    # regime's own increment values (not the full pooled history)
    thresholds = {}
    for tau in all_taus:
        mask = (increments[tau].index >= window_start) & (increments[tau].index <= window_end)
        regime_inc = increments[tau].loc[mask]
        for q in full_q_grid:
            thresholds[(tau, q)] = regime_inc.quantile(q, numeric_only=True).to_dict()

    tickers = list(prices.columns)
    results = []
    for y in tickers:
        for tau_f in TAU_FUTURE:
            fy_full = future_inc[tau_f][y]
            mask_y = (fy_full.index >= window_start) & (fy_full.index <= window_end)
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
                            n_bull = cond_bull.sum()
                            if n_bull >= MIN_N:
                                cpe_bull = event_bull[cond_bull].mean()
                                lift_bull = cpe_bull / uncond if uncond > 0 else np.nan
                                if cpe_bull >= CPE_THRESH and lift_bull >= MIN_LIFT:
                                    results.append((y, x, tau_p, tau_f, q_x, q_y,
                                                     round(float(cpe_bull), 4), round(float(uncond), 4),
                                                     round(float(lift_bull), 4), int(n_bull), "bullish"))

                            cond_bear = valid & (px_vals < thresh_x_dn)
                            n_bear = cond_bear.sum()
                            if n_bear >= MIN_N:
                                cpe_bear = event_bear[cond_bear].mean()
                                lift_bear = cpe_bear / uncond if uncond > 0 else np.nan
                                if cpe_bear >= CPE_THRESH and lift_bear >= MIN_LIFT:
                                    results.append((y, x, tau_p, tau_f, q_x, q_y,
                                                     round(float(cpe_bear), 4), round(float(uncond), 4),
                                                     round(float(lift_bear), 4), int(n_bear), "bearish"))

    cols = ["Y", "X", "tau_past", "tau_future", "q_X", "q_Y", "CPE", "uncond_prob", "lift", "n_condition", "direction"]
    df = pd.DataFrame(results, columns=cols)
    print(f"[{label}] window {window_start.date()} to {window_end.date()}: {len(df)} surviving configurations")
    return df


if __name__ == "__main__":
    prices = pd.read_parquet("aiq_universe_prices.parquet")
    print(f"Universe: {prices.shape[1]} tickers, {prices.index.min().date()} to {prices.index.max().date()}")

    avg_corr = prices.pct_change().corr().values
    avg_corr = avg_corr[np.triu_indices_from(avg_corr, k=1)]
    print(f"Average pairwise daily-return correlation across the basket: {np.nanmean(avg_corr):.3f} "
          f"(vs ~0.3-0.4 typical for the core CPE framework's 161-instrument, multi-asset-class universe)")

    post_start = REGIME_SPLIT
    post_end = prices.index.max()
    n_post_days = int(((prices.index >= post_start) & (prices.index <= post_end)).sum())

    # Length-matched pre-window: the same number of trading days
    # immediately preceding the split, not all available history back to
    # 1962. A 60-year pre-window mixes in the dot-com crash, 2008 GFC,
    # and decades unrelated to any "AI regime" question, which would
    # make a raw pre-vs-post count comparison meaningless.
    pre_end = REGIME_SPLIT - pd.Timedelta(days=1)
    all_dates_before_split = prices.index[prices.index <= pre_end]
    pre_start = all_dates_before_split[-n_post_days]
    print(f"Length-matched pre-window: {n_post_days} trading days, "
          f"{pre_start.date()} to {pre_end.date()} (post-window is {n_post_days} days, "
          f"{post_start.date()} to {post_end.date()})")

    pre_df = run_regime_scan(prices, pre_start, pre_end, "PRE-ChatGPT")
    post_df = run_regime_scan(prices, post_start, post_end, "POST-ChatGPT")

    pre_df.to_parquet("ai_cpe_pre_chatgpt_results.parquet")
    post_df.to_parquet("ai_cpe_post_chatgpt_results.parquet")

    def self_ref_share(df):
        if len(df) == 0:
            return np.nan
        return float((df["Y"] == df["X"]).mean())

    summary = {
        "universe_size": prices.shape[1],
        "regime_split": str(REGIME_SPLIT.date()),
        "avg_pairwise_correlation": float(np.nanmean(avg_corr)),
        "pre_chatgpt": {
            "window": [str(pre_start.date()), str(pre_end.date())],
            "n_trading_days": int(((prices.index >= pre_start) & (prices.index <= pre_end)).sum()),
            "n_surviving_configs": len(pre_df),
            "n_bullish": int((pre_df["direction"] == "bullish").sum()) if len(pre_df) else 0,
            "n_bearish": int((pre_df["direction"] == "bearish").sum()) if len(pre_df) else 0,
            "self_referential_share": self_ref_share(pre_df),
            "n_unique_Y": int(pre_df["Y"].nunique()) if len(pre_df) else 0,
            "n_unique_X": int(pre_df["X"].nunique()) if len(pre_df) else 0,
        },
        "post_chatgpt": {
            "window": [str(post_start.date()), str(post_end.date())],
            "n_trading_days": int(((prices.index >= post_start) & (prices.index <= post_end)).sum()),
            "n_surviving_configs": len(post_df),
            "n_bullish": int((post_df["direction"] == "bullish").sum()) if len(post_df) else 0,
            "n_bearish": int((post_df["direction"] == "bearish").sum()) if len(post_df) else 0,
            "self_referential_share": self_ref_share(post_df),
            "n_unique_Y": int(post_df["Y"].nunique()) if len(post_df) else 0,
            "n_unique_X": int(post_df["X"].nunique()) if len(post_df) else 0,
        },
    }
    json.dump(summary, open("ai_cpe_regime_comparison_summary.json", "w"), indent=2)
    print("\n" + json.dumps(summary, indent=2))
