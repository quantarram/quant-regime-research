"""
alpha_generated_plot.py
=========================
Plots real alpha generated (%/yr, or per-instrument alpha vs. each
instrument's own buy-and-hold) for every strategy tested this session,
including the two published controls and the final combined strategy.
Every individual real observation (a real year, a real historical block,
a real instrument) is shown as its own point, not collapsed into a
single number -- so the reader sees the actual spread, not just a
central tendency. No error bars, no confidence intervals, no p-values:
literal real point estimates only, per the standing methodology note in
the README's Limitations section.

Run: python alpha_generated_plot.py
Output: alpha_generated_plot.png
"""
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def per_instrument_alphas(json_path: str, alpha_field: str) -> list:
    d = json.load(open(json_path))
    out = []
    for tkr, row in d.items():
        if isinstance(row, dict) and alpha_field in row and row[alpha_field] is not None:
            a = row[alpha_field]
            if isinstance(a, (int, float)) and np.isfinite(a):
                out.append(a)
    return out


def kelly_mean_diff_alphas() -> list:
    """Real mean-difference alpha (not the stored Jensen's-alpha field), all
    22 instruments in kelly_strategy_returns.parquet -- matches this paper's
    own standing convention (Section 4.4: never Jensen's alpha)."""
    strat = pd.read_parquet("predictor_v1/kelly_strategy_returns.parquet")
    prices = pd.read_parquet("multiasset_prices.parquet")
    out = []
    for tkr in strat.columns:
        net = strat[tkr].dropna()
        if len(net) < 100 or tkr not in prices.columns:
            continue
        bh = prices[tkr].reindex(net.index).pct_change().fillna(0.0)
        out.append(float((net.mean() - bh.mean()) * 252 * 100))
    return out


def master_model_mean_diff_alphas() -> list:
    """Real mean-difference alpha for the master-model directional
    strategy, all 22 instruments, replicating 44_alpha_test_own_benchmark.py's
    own position construction but comparing means directly instead of via
    Jensen's-alpha regression."""
    decisions = json.load(open("predictor_v1/master_model_final_decision.json"))
    oos_all = pd.read_parquet("predictor_v1/oos_predictions_all.parquet")
    prices = pd.read_parquet("multiasset_prices.parquet")
    prices_proxy = pd.read_parquet("predictor_v1/sector_proxy_cache.parquet")
    proxy_tickers = ("IYR", "VOX")
    holdout_start = pd.Timestamp("2022-01-01")
    cost_bps = 5
    out = []
    for tkr in decisions:
        dec = decisions[tkr]
        horizon, winner = dec["horizon"], dec["price_based_winner"]
        sub = oos_all[(oos_all["ticker"] == tkr) & (oos_all["horizon"] == horizon)]
        variants = sub["variant"].unique().tolist()
        if not variants:
            continue
        if winner == "climatology":
            src = sub[sub["variant"] == "both"] if "both" in variants else sub[sub["variant"] == variants[0]]
            pred_col = "clim_q0.5"
        else:
            src = sub[sub["variant"] == winner]
            pred_col = "q0.5"
        src = src[src["date"] >= holdout_start].sort_values("date")
        if len(src) < 50:
            continue
        series = (prices_proxy[tkr] if tkr in proxy_tickers else prices[tkr]).dropna()
        daily_ret = series.pct_change().dropna()
        position = pd.Series((src[pred_col].values > 0).astype(float), index=src["date"].values)
        position = position[~position.index.duplicated(keep="last")]
        idx = daily_ret.index[(daily_ret.index >= position.index.min()) & (daily_ret.index <= position.index.max())]
        if len(idx) < 50:
            continue
        pos_daily = position.reindex(idx).ffill().fillna(0.0)
        ret = daily_ret.reindex(idx)
        applied = pos_daily.shift(1).fillna(0.0)
        gross = applied * ret
        turnover = applied.diff().abs().fillna(0.0)
        net = gross - turnover * (cost_bps / 10000.0)
        out.append(float((net.mean() - ret.mean()) * 252 * 100))
    return out


if __name__ == "__main__":
    series = {}

    # -- CPE portfolio-tilt engine, 4 real years each --
    series["CPE hold-to-horizon\n(4 real years)"] = [0.16, -1.53, 1.73, 23.87]
    series["CPE static tilt\n(4 real years)"] = [0.13, 0.93, -0.86, 0.56]
    series["CPE breadth-40, floor=3\n(4 real years)"] = [-0.19, -0.00, -0.22, 0.84]
    series["CPE breadth-40, floor=2\n(4 real years)"] = [0.21, 0.03, 1.35, 1.03]

    # -- Paper 12 Kelly-sized / master model, per instrument, 2022+ holdout --
    # real mean-difference, not the originally-published Jensen's-alpha figures
    # (Section 4.4 excludes Jensen's alpha from this paper's own analysis)
    series["Kelly-sized, mean-diff\n(22 instruments)"] = kelly_mean_diff_alphas()
    series["Master model, mean-diff\n(22 instruments)"] = master_model_mean_diff_alphas()
    rl = json.load(open("predictor_v1/81_rl_sizing_results.json"))
    series["RL sizing\n(12 instruments)"] = [r["fresh"]["alpha_annualized_pct"] for r in rl.values() if r.get("fresh")]

    # -- SPY momentum, XSMOM: 5 real non-overlapping historical blocks --
    spy_blocks = json.load(open("spy_momentum_subperiod_results.json"))["blocks"]
    series["SPY momentum vs. own buy-hold\n(5 real blocks)"] = [b["excess_over_buyhold_pct"] for b in spy_blocks]
    xsmom_blocks = json.load(open("xsmom_evt_subperiod_results.json"))["blocks"]
    series["XSMOM vs. passive\n(5 real blocks)"] = [b["excess_pct"] for b in xsmom_blocks]

    # -- TSMOM: 5 calendar blocks (already computed) --
    series["TSMOM vs. passive, calendar blocks\n(5 real blocks)"] = [-0.99, -5.08, -0.96, -4.97, -5.29]

    # -- Combined strategy: alpha of each variant over the unfiltered baseline, 5 real blocks --
    combo = json.load(open("combined_strategy_results.json"))["blocks"]
    series["Pocket-filtered TSMOM vs. baseline\n(5 real blocks)"] = [b["pocket_pct"] - b["baseline_pct"] for b in combo]
    series["Combined (pocket+CPE) vs. baseline\n(5 real blocks)"] = [b["combined_pct"] - b["baseline_pct"] for b in combo]

    # sort by median
    order = sorted(series.items(), key=lambda kv: np.median(kv[1]))
    labels = [k for k, v in order]
    medians = [np.median(v) for k, v in order]

    fig, ax = plt.subplots(figsize=(12, 9))
    y = np.arange(len(order))
    colors = ["#2f8a4e" if m > 0 else "#B0492F" for m in medians]

    for i, (name, vals) in enumerate(order):
        ax.scatter(vals, [i] * len(vals), color="#9AA1AD", s=28, zorder=3, alpha=0.85,
                   edgecolors="white", linewidths=0.5)
    ax.scatter(medians, y, color=colors, s=110, zorder=4, marker="D", label="Median")
    ax.axvline(0, color="black", lw=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Real alpha vs. correct benchmark (%/yr) -- each grey dot is one real year, block, or instrument")
    ax.set_title("Alpha generated, every strategy tested this session\n"
                  "Diamonds = median; grey dots = every individual real observation, not collapsed -- no significance test applied",
                  fontsize=11)
    ax.set_xlim(-20, 27)
    fig.tight_layout()
    fig.savefig("alpha_generated_plot.png", dpi=140)
    plt.close(fig)
    print("Saved: alpha_generated_plot.png")
    for name, vals in order:
        print(f"  {name.replace(chr(10), ' '):<55} median={np.median(vals):+7.2f}%  n={len(vals)}  range=[{min(vals):+.2f}, {max(vals):+.2f}]")
