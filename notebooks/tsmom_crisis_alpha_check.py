"""
tsmom_crisis_alpha_check.py
============================
Follow-up to tsmom_benchmark.py: TSMOM's entire full-sample alpha traced
back to one visible feature of its equity curve -- it barely dips in 2008
while the passive benchmark craters, then never gives that relative gain
back. That's the textbook "crisis alpha" / trend-following-as-convexity
story (Hurst, Ooi & Pedersen, "A Century of Evidence on Trend-Following
Investing", AQR): trend strategies are claimed to behave like a long
volatility position, profiting specifically during large, sustained
directional moves (most visibly of the down variety), not as a steady,
uniform edge across all conditions. This script tests that claim directly
against our own replication rather than assuming it: does TSMOM's edge
concentrate in identifiable crisis/drawdown episodes, or is it spread
evenly across the whole sample (in which case "crisis alpha" would be the
wrong description of what's actually happening here)?

Method: reuse tsmom_benchmark.py's exact TSMOM and passive return series
(same universe, same unfitted parameters, no changes). Define "crisis"
using the PASSIVE benchmark's own drawdown from its running peak --
crisis = drawdown <= -10% (a standard, round, undisclosed-tuning
threshold, checked for robustness at -5%/-15% too, not cherry-picked).
Cluster consecutive crisis days into discrete episodes (matching the
episode-clustering convention already used elsewhere in this program for
CPE), report each episode's passive max-drawdown against TSMOM's own
cumulative return over the identical window, and split full-sample
annualized performance and return correlation into crisis-day vs.
calm-day subsets.

No significance/randomisation-test games -- descriptive decomposition of
a result already found, reported honestly whichever way it comes out.

Run: python tsmom_crisis_alpha_check.py
Output: tsmom_crisis_episodes.json, tsmom_crisis_alpha_check.png
"""
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tsmom_benchmark as _tm

CRISIS_THRESHOLD = -0.10
ROBUSTNESS_THRESHOLDS = [-0.05, -0.10, -0.15]
MIN_EPISODE_DAYS = 15


def build_series():
    prices = pd.read_parquet("multiasset_prices.parquet")
    n_bad = int((prices <= 0).sum().sum())
    if n_bad:
        prices = prices.mask(prices <= 0)
    daily_index = prices.index

    first_valid = min(prices[t].dropna().index.min() for t in _tm.UNIVERSE)
    start = first_valid + pd.Timedelta(days=int(1.5 * (_tm.LOOKBACK_DAYS + _tm.VOL_WINDOW_DAYS)))
    rebal_dates = _tm.month_end_dates(daily_index[daily_index >= start])
    weight_df = _tm.build_monthly_signals(prices, rebal_dates)
    trading_index = daily_index[daily_index >= rebal_dates[0]]

    tsmom_ret = _tm.simulate(weight_df, prices, trading_index, equal_weight_passive=False)
    passive_ret = _tm.simulate(weight_df, prices, trading_index, equal_weight_passive=True)
    return tsmom_ret, passive_ret


def cluster_episodes(mask: pd.Series, min_days: int) -> list:
    dates = mask.index[mask.values]
    if len(dates) == 0:
        return []
    episodes, current = [], [dates[0]]
    for d in dates[1:]:
        if (d - current[-1]).days > 10:  # allow small gaps (a few calm days inside a crisis) to stay one episode
            if len(current) >= min_days:
                episodes.append((current[0], current[-1]))
            current = [d]
        else:
            current.append(d)
    if len(current) >= min_days:
        episodes.append((current[0], current[-1]))
    return episodes


def regime_stats(tsmom_ret, passive_ret, mask, label):
    t, p = tsmom_ret[mask].dropna(), passive_ret[mask].dropna()
    common = t.index.intersection(p.index)
    t, p = t.reindex(common), p.reindex(common)
    n = len(common)
    if n < 10:
        return {"label": label, "n_days": n}
    ann_ret_t = float(t.mean() * 252 * 100)
    ann_ret_p = float(p.mean() * 252 * 100)
    sharpe_t = float(t.mean() / t.std() * np.sqrt(252)) if t.std() > 0 else float("nan")
    sharpe_p = float(p.mean() / p.std() * np.sqrt(252)) if p.std() > 0 else float("nan")
    corr = float(np.corrcoef(t.values, p.values)[0, 1])
    return {
        "label": label, "n_days": n,
        "tsmom_ann_ret_pct": ann_ret_t, "passive_ann_ret_pct": ann_ret_p,
        "tsmom_sharpe": sharpe_t, "passive_sharpe": sharpe_p,
        "corr_tsmom_passive": corr,
    }


if __name__ == "__main__":
    print("Rebuilding TSMOM and passive return series (same spec as tsmom_benchmark.py)...")
    tsmom_ret, passive_ret = build_series()

    passive_equity = np.exp(passive_ret.fillna(0)).cumprod()
    running_peak = passive_equity.cummax()
    drawdown = passive_equity / running_peak - 1.0

    print(f"\nCrisis definition: passive benchmark drawdown <= {CRISIS_THRESHOLD:.0%} from its own running peak")
    crisis_mask = drawdown <= CRISIS_THRESHOLD
    calm_mask = ~crisis_mask
    print(f"  {crisis_mask.sum()} crisis days ({crisis_mask.mean()*100:.1f}% of sample), "
          f"{calm_mask.sum()} calm days")

    print("\n--- Full-sample split: crisis days vs calm days ---")
    crisis_stats = regime_stats(tsmom_ret, passive_ret, crisis_mask, "crisis")
    calm_stats = regime_stats(tsmom_ret, passive_ret, calm_mask, "calm")
    for s in (crisis_stats, calm_stats):
        print(f"  [{s['label']:>6}] n={s['n_days']:>5}  "
              f"TSMOM ann_ret={s.get('tsmom_ann_ret_pct', float('nan')):+7.2f}%  Sharpe={s.get('tsmom_sharpe', float('nan')):+.2f}  |  "
              f"Passive ann_ret={s.get('passive_ann_ret_pct', float('nan')):+7.2f}%  Sharpe={s.get('passive_sharpe', float('nan')):+.2f}  |  "
              f"corr={s.get('corr_tsmom_passive', float('nan')):+.2f}")

    print("\n--- Robustness: same split at alternate crisis thresholds ---")
    robustness = {}
    for thresh in ROBUSTNESS_THRESHOLDS:
        cm = drawdown <= thresh
        cs = regime_stats(tsmom_ret, passive_ret, cm, f"crisis@{thresh:.0%}")
        ks = regime_stats(tsmom_ret, passive_ret, ~cm, f"calm@{thresh:.0%}")
        robustness[f"{thresh:.0%}"] = {"crisis": cs, "calm": ks}
        print(f"  threshold={thresh:>5.0%}: crisis n={cs['n_days']:>5} TSMOM ann_ret={cs.get('tsmom_ann_ret_pct', float('nan')):+7.2f}% "
              f"vs calm n={ks['n_days']:>5} TSMOM ann_ret={ks.get('tsmom_ann_ret_pct', float('nan')):+7.2f}%")

    episodes = cluster_episodes(crisis_mask, MIN_EPISODE_DAYS)
    print(f"\n--- {len(episodes)} discrete crisis episodes (>= {MIN_EPISODE_DAYS} days, drawdown <= {CRISIS_THRESHOLD:.0%}) ---")
    episode_results = []
    for start, end in episodes:
        window = (tsmom_ret.index >= start) & (tsmom_ret.index <= end)
        t_cum = float((np.exp(tsmom_ret[window].fillna(0)).prod() - 1) * 100)
        p_cum = float((np.exp(passive_ret[window].fillna(0)).prod() - 1) * 100)
        p_dd = float(drawdown[window].min() * 100)
        episode_results.append({
            "start": str(start.date()), "end": str(end.date()), "n_days": int(window.sum()),
            "passive_max_drawdown_pct": p_dd, "passive_cum_ret_pct": p_cum, "tsmom_cum_ret_pct": t_cum,
        })
        print(f"  {start.date()} to {end.date()} ({window.sum():>4}d): passive max DD={p_dd:+6.1f}%  "
              f"passive ret={p_cum:+6.1f}%  TSMOM ret={t_cum:+6.1f}%  "
              f"{'<- TSMOM WON' if t_cum > p_cum else ''}")

    n_tsmom_won = sum(1 for e in episode_results if e["tsmom_cum_ret_pct"] > e["passive_cum_ret_pct"])
    print(f"\nTSMOM beat passive in {n_tsmom_won}/{len(episode_results)} identified crisis episodes")

    with open("tsmom_crisis_episodes.json", "w") as f:
        json.dump({
            "crisis_threshold": CRISIS_THRESHOLD, "full_sample_split": {"crisis": crisis_stats, "calm": calm_stats},
            "robustness": robustness, "episodes": episode_results, "n_tsmom_won": n_tsmom_won,
        }, f, indent=2, default=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, height_ratios=[2, 1])
    tsmom_eq = np.exp(tsmom_ret.fillna(0)).cumprod() * 100_000
    passive_eq = passive_equity * 100_000
    ax1.plot(passive_eq.index, passive_eq.values, color="#9AA1AD", lw=1.1, label="Passive equal-weight")
    ax1.plot(tsmom_eq.index, tsmom_eq.values, color="#2E6DA4", lw=1.1, label="TSMOM")
    for start, end in episodes:
        ax1.axvspan(start, end, color="#B0492F", alpha=0.12)
    ax1.set_yscale("log")
    ax1.set_ylabel("Equity ($100k notional, log)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title(f"Is TSMOM's edge crisis alpha? {len(episodes)} drawdown episodes shaded (passive DD <= {CRISIS_THRESHOLD:.0%})\n"
                  f"TSMOM beat passive in {n_tsmom_won}/{len(episode_results)} of them")

    ax2.fill_between(drawdown.index, drawdown.values * 100, 0, color="#B0492F", alpha=0.3)
    ax2.set_ylabel("Passive drawdown (%)")
    ax2.axhline(CRISIS_THRESHOLD * 100, color="black", lw=0.7, linestyle="--")
    fig.tight_layout()
    fig.savefig("tsmom_crisis_alpha_check.png", dpi=140)
    plt.close(fig)
    print("\nSaved: tsmom_crisis_episodes.json, tsmom_crisis_alpha_check.png")
