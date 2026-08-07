"""
pooled_credit_duration_stress_search.py
==========================================
Broader, properly-pooled version of credit_stress_channel_search.py, built
directly from the user's correction: requiring the SAME narrow ticker pair
(HYG+TLT) to cross the same threshold 3+ times is not how real extreme
value analysis characterizes a rare event class -- EVT pools ANALOGOUS
extreme events (comparable regional floods, not repeats of the same
river) to characterize a tail distribution. This pools across the full
available credit-risk complex (HYG, JNK: high-yield; LQD, VCLT:
investment-grade; EMB: emerging-market) and duration complex (TLT, TLH,
ZROZ, EDV: long duration; IEF: mid duration) -- 5x5 = 25 credit/duration
pairs -- and clusters the UNION of all their joint-bearish firing dates
into genuinely distinct real episodes BY DATE, not by ticker-pair, so
near-duplicate instruments (HYG vs JNK, TLT vs TLH) tracking the same
underlying market event don't get double-counted as separate episodes.

HYPOTHESIS (unchanged from the narrower search, restated for the pooled
version): when SOME credit proxy and SOME duration proxy are jointly in
their own bearish tail, does broad equity subsequently show below-median
returns? Pre-specified quantile thresholds: q=0.10 (primary, broad enough
to let pooling do its job) and q=0.05 (stricter check), fixed before
looking at how many distinct episodes either one produces.

Uses direct price-level computation for every forward return this time,
not a shift-based trick -- the exact bug that inverted the previous
search's headline finding is checked against directly here.

Run (from notebooks/files/):
    python pooled_credit_duration_stress_search.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

CREDIT_PROXIES = ["HYG", "JNK", "LQD", "VCLT", "EMB"]
DURATION_PROXIES = ["TLT", "TLH", "ZROZ", "EDV", "IEF"]
TARGETS = ["SPY", "QQQ", "IWM", "DIA"]
TAU_PAST_GRID = [63, 126, 252]
TAU_FUTURE_GRID = [63, 126, 252]
Q_GRID = [0.10, 0.05]
EPISODE_GAP_MULTIPLIER = 1.5
MIN_EPISODE_GAP_DAYS = 45  # floor on the clustering gap so short tau_past doesn't over-split one real event


def cluster_episodes(dates: pd.DatetimeIndex, gap_days: int) -> list:
    if len(dates) == 0:
        return []
    dates = pd.DatetimeIndex(sorted(set(dates)))
    gap_calendar = max(gap_days * 1.45 * EPISODE_GAP_MULTIPLIER, MIN_EPISODE_GAP_DAYS)
    episodes, cur = [], [dates[0]]
    for d in dates[1:]:
        if (d - cur[-1]).days > gap_calendar:
            episodes.append(cur)
            cur = [d]
        else:
            cur.append(d)
    episodes.append(cur)
    return episodes


def forward_return_direct(prices: pd.DataFrame, ticker: str, anchor: pd.Timestamp, tau_f: int):
    """Direct price-level forward return -- no shift(-n) tricks, the exact
    thing that produced a sign error in the previous search."""
    pos = prices.index.get_indexer([anchor])[0]
    if pos + tau_f >= len(prices.index):
        return None, None
    future_date = prices.index[pos + tau_f]
    p0, p1 = prices.loc[anchor, ticker], prices.loc[future_date, ticker]
    if pd.isna(p0) or pd.isna(p1) or p0 <= 0:
        return None, None
    return float(np.log(p1 / p0)), future_date


if __name__ == "__main__":
    prices = pd.read_parquet("../multiasset_prices.parquet")
    all_tickers = list(set(CREDIT_PROXIES + DURATION_PROXIES + TARGETS))

    for tau_p in TAU_PAST_GRID:
        for q in Q_GRID:
            inc = {t: np.log(prices[t] / prices[t].shift(tau_p)) for t in CREDIT_PROXIES + DURATION_PROXIES}
            thresh = {t: inc[t].quantile(q) for t in CREDIT_PROXIES + DURATION_PROXIES}

            union_dates = set()
            n_pairs_firing = 0
            for c in CREDIT_PROXIES:
                for d in DURATION_PROXIES:
                    joint = (inc[c] < thresh[c]) & (inc[d] < thresh[d])
                    firing = joint[joint.fillna(False)].index
                    if len(firing) > 0:
                        n_pairs_firing += 1
                    union_dates.update(firing)

            episodes = cluster_episodes(pd.DatetimeIndex(sorted(union_dates)), tau_p)
            print(f"\n{'='*100}\ntau_past={tau_p}d, q={q} (bearish tail): "
                  f"{n_pairs_firing}/25 credit-duration pairs fired at least once, "
                  f"{len(union_dates)} total pooled firing days -> {len(episodes)} genuinely distinct real episodes\n{'='*100}")

            for i, ep in enumerate(episodes):
                anchor = max(ep)
                years = sorted(set(d.year for d in ep))
                print(f"\n  Episode {i+1}: {min(ep).date()} to {max(ep).date()} ({len(ep)} pooled firing days, years={years})")
                for tau_f in TAU_FUTURE_GRID:
                    row = []
                    for target_t in TARGETS:
                        fwd, future_date = forward_return_direct(prices, target_t, anchor, tau_f)
                        if fwd is None:
                            continue
                        med = np.log(prices[target_t] / prices[target_t].shift(tau_f)).median()
                        direction = "BELOW median (bearish-confirms)" if fwd < med else "ABOVE median (bullish-contradicts)"
                        row.append(f"{target_t}: {fwd:+.3f} vs med {med:+.3f} [{direction}]")
                    print(f"    tau_future={tau_f}d: " + "  |  ".join(row))

            # Pooled, aggregate episode-level hit rate across all distinct episodes, all targets, all horizons
            all_outcomes = []
            for ep in episodes:
                anchor = max(ep)
                for tau_f in TAU_FUTURE_GRID:
                    for target_t in TARGETS:
                        fwd, _ = forward_return_direct(prices, target_t, anchor, tau_f)
                        if fwd is None:
                            continue
                        med = np.log(prices[target_t] / prices[target_t].shift(tau_f)).median()
                        all_outcomes.append(fwd < med)
            if all_outcomes:
                print(f"\n  POOLED episode-level hit rate (bearish-confirms), {len(episodes)} episodes x targets x horizons, "
                      f"n={len(all_outcomes)}: {np.mean(all_outcomes):.3f}")
