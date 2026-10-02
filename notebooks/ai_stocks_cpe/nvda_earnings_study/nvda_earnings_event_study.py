"""
nvda_earnings_event_study.py
========================================
Event study on NVDA's quarterly earnings reports, modeled on Paper 10's
hurricane-landfall/reinsurer-equity design (real, dated events; a market
benchmark; a control group to check whether the reaction is event-specific
or just broad co-movement) -- with one deliberate change from Paper 10's
original construction: no OLS market-model (regression alpha/beta) and no
t-tests. Every abnormal-return figure here is a real mean-difference
(actual NVDA return minus the benchmark's own real return over the
identical window), matching the standard this program settled on from
Paper 16 onward (TSMOM Section 4.4: OLS/Jensen's-alpha-style regression
statistics are not used to evaluate this program's own findings, at any
sample size).

Events: 71 real, dated NVDA quarterly earnings reports (2006-2026),
sourced directly from Nasdaq/Yahoo Finance's own earnings-date record
(get_earnings_dates), not hand-selected. NVDA reports after market close,
so the "reaction day" is the next trading day, not the report date itself.

Controls:
  - SPY: broad-market excess return (is this bigger than the market's own
    move that day?)
  - AMD, SOXX: sector-peer excess return (is this NVDA-specific news, or
    did the whole semiconductor sector move together that day?)

Run: python nvda_earnings_event_study.py
"""
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HORIZONS = {"1-day": 1, "1-week": 5, "1-month": 21}
BENCHMARKS = ["SPY", "AMD", "SOXX"]


def next_trading_day(idx: pd.DatetimeIndex, report_date: pd.Timestamp) -> pd.Timestamp:
    """NVDA reports after market close -- the reaction shows up starting
    the next trading day, not the report date itself."""
    future = idx[idx > report_date]
    return future[0] if len(future) else None


def cum_return(price: pd.Series, start: pd.Timestamp, days: int):
    idx = price.index
    pos = idx.searchsorted(start)
    if pos + days >= len(idx) or pos >= len(idx):
        return None
    return float(price.iloc[pos + days] / price.iloc[pos] - 1) * 100


if __name__ == "__main__":
    prices = pd.read_parquet("nvda_event_study_prices.parquet")
    events = json.load(open("nvda_earnings_events.json"))
    idx = prices.index

    rows = []
    for ev in events:
        report_date = pd.Timestamp(ev["report_date"])
        reaction_start = next_trading_day(idx, report_date)
        if reaction_start is None:
            continue
        row = {"report_date": ev["report_date"], "reaction_start": str(reaction_start.date()),
               "surprise_pct": ev["surprise_pct"]}
        skip = False
        for h_label, h_days in HORIZONS.items():
            nvda_ret = cum_return(prices["NVDA"], reaction_start, h_days)
            if nvda_ret is None:
                skip = True
                break
            row[f"nvda_{h_label}"] = nvda_ret
            for b in BENCHMARKS:
                b_ret = cum_return(prices[b], reaction_start, h_days)
                row[f"{b}_{h_label}"] = b_ret
                row[f"excess_vs_{b}_{h_label}"] = (nvda_ret - b_ret) if b_ret is not None else None
        if not skip:
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"{len(df)} of {len(events)} events have a full 1-month forward window "
          f"(most recent events excluded if too close to today)")
    df.to_parquet("nvda_earnings_event_results.parquet")

    df["report_date_dt"] = pd.to_datetime(df["report_date"])
    # NaN > 0 is False in pandas, which would silently count a missing
    # surprise_pct (one real event, 2015-05-07, has no reported surprise%)
    # as a "miss" -- exclude it from both buckets instead.
    df["beat"] = df["surprise_pct"] > 0
    df.loc[df["surprise_pct"].isna(), "beat"] = None
    df["ai_era"] = df["report_date_dt"] >= pd.Timestamp("2023-01-01")

    summary = {}
    for h_label in HORIZONS:
        summary[h_label] = {}
        for b in BENCHMARKS:
            col = f"excess_vs_{b}_{h_label}"
            vals = df[col].dropna()
            summary[h_label][b] = {
                "n": int(len(vals)), "mean_pct": float(vals.mean()), "median_pct": float(vals.median()),
                "share_positive": float((vals > 0).mean()),
                "mean_pre_ai_era": float(df.loc[~df["ai_era"], col].dropna().mean()),
                "mean_ai_era": float(df.loc[df["ai_era"], col].dropna().mean()),
                "mean_on_beat": float(df.loc[df["beat"] == True, col].dropna().mean()),
                "mean_on_miss": float(df.loc[df["beat"] == False, col].dropna().mean()),
                "n_beat": int((df["beat"] == True).sum()),
                "n_miss": int((df["beat"] == False).sum()),
            }
    json.dump(summary, open("nvda_earnings_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))

    # ---- Figure 1: excess return vs. each benchmark, by horizon, real points ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=False)
    for ax, h_label in zip(axes, HORIZONS):
        for i, b in enumerate(BENCHMARKS):
            col = f"excess_vs_{b}_{h_label}"
            vals = df[col].dropna()
            x = np.full(len(vals), i) + np.random.uniform(-0.08, 0.08, len(vals))
            ax.scatter(x, vals, alpha=0.55, s=28, color=["#4a7a9d", "#c96a3a", "#2E7D4F"][i])
            ax.scatter([i], [vals.mean()], color="black", marker="_", s=400, linewidths=2, zorder=5)
        ax.axhline(0, color="#444", lw=0.8)
        ax.set_xticks(range(len(BENCHMARKS)))
        ax.set_xticklabels(BENCHMARKS)
        ax.set_title(f"{h_label} excess return\n(black bar = mean)")
        ax.set_ylabel("NVDA return minus benchmark's own return (%)")
    fig.suptitle("NVDA earnings reactions, 2006-2026: real excess return per event, vs. market and sector peers", y=1.02)
    fig.tight_layout()
    fig.savefig("nvda_earnings_excess_return_plot.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: pre-AI-era vs AI-era (2023+), vs SPY, 1-month horizon ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    col = "excess_vs_SPY_1-month"
    pre = df.loc[~df["ai_era"], col].dropna()
    post = df.loc[df["ai_era"], col].dropna()
    ax.scatter(np.zeros(len(pre)) + np.random.uniform(-0.08, 0.08, len(pre)), pre, alpha=0.6, s=32, color="#9AA1AD", label=f"Pre-2023 (n={len(pre)})")
    ax.scatter(np.ones(len(post)) + np.random.uniform(-0.08, 0.08, len(post)), post, alpha=0.7, s=40, color="#2E7D4F", label=f"2023-2026, \"AI era\" (n={len(post)})")
    ax.scatter([0], [pre.mean()], color="black", marker="_", s=500, linewidths=2.5, zorder=5)
    ax.scatter([1], [post.mean()], color="black", marker="_", s=500, linewidths=2.5, zorder=5)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pre-2023", "2023-2026"])
    ax.set_ylabel("1-month excess return vs. SPY (%)")
    ax.set_title("NVDA's 1-month earnings-reaction excess return, real points\nPre-2023 vs. the 2023-2026 AI-earnings era")
    ax.legend()
    fig.tight_layout()
    fig.savefig("nvda_earnings_regime_split_plot.png", dpi=140)
    plt.close(fig)

    print("\nSaved: nvda_earnings_excess_return_plot.png, nvda_earnings_regime_split_plot.png")
