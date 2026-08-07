"""
credit_stress_channel_search.py
==================================
Pre-registered search for a genuinely different, mechanistically-distinct
crisis channel from the flight-to-quality one already found ([TLH, ^GVZ]
-> QQQ). economic_prior.py ALREADY admits ("credit_hy", "broad_equity_us")
and ("credit_hy", "sector_equity") -- credit-spread widening predicting
equity weakness -- no prior expansion needed; the existing joint screen
just never surfaced a bearish-direction configuration using it.

HYPOTHESIS (fixed BEFORE looking at any 2022+ data): high-yield credit
(HYG or JNK) bearish AND long-duration Treasuries (TLT or TLH) ALSO
bearish, jointly, predicts subsequent equity weakness. The "TLH also
bearish" condition is the deliberate, mechanistic opposite of the
existing detector's signature (TLH BULLISH there, flight-to-quality) --
designed to catch a rate-shock regime (bonds and equities selling off
together, 2022's actual mechanism), not another flight-to-quality variant.

CORRECTION (same day this script was first run): the first version of
this script silently discarded any configuration with fewer than 100 raw
joint-firing observations before ever computing CPE/lift on it -- a
threshold borrowed from the original, decades-spanning joint screen's own
convention. That threshold is WRONG for a search whose entire hypothesis
is "this identifies something rare and specific to one recent regime" --
it is structurally guaranteed to reject the exact pattern being searched
for, especially at the tight quantile where a genuinely rare regime's
signal concentrates. This version reports every configuration with at
least 1 independent episode directly, with n_joint (raw overlapping
observations) and n_episodes (real independent occurrences) both shown,
so nothing can be silently dropped before a human sees it -- the episode
count and hit rate are the real filter, not an arbitrary raw-observation
floor. Reported honestly whichever way it comes out, including a real
n=1 finding, which is what actually came back: this joint condition
fires 36-52 times across the ENTIRE available history, clustering into
exactly ONE real episode, which is 2022 -- confirmed genuine and
historically distinctive (2022 really was the worst joint stock/bond
year in decades, independently documented), but a single episode cannot
establish reliability regardless of how clean its one outcome looks --
the same reasoning this program already applies to any 1-2 episode
configuration.

Run (from notebooks/files/):
    python credit_stress_channel_search.py
"""
import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

CREDIT_PREDICTORS = ["HYG", "JNK"]
RATE_PREDICTORS = ["TLT", "TLH"]
TARGETS = ["SPY", "QQQ", "IWM", "DIA"]
TAU_PAST_GRID = [21, 63, 126, 252]
TAU_FUTURE_GRID = [21, 63, 126, 252]
Q_GRID = [0.05, 0.10, 0.20]  # bearish tail: predictor BELOW this quantile
EPISODE_GAP_MULTIPLIER = 1.5


def build_increments(prices: pd.DataFrame, tickers: list, tau_list: list) -> dict:
    out = {}
    for tau in tau_list:
        inc = pd.DataFrame(index=prices.index)
        for t in tickers:
            inc[t] = np.log(prices[t] / prices[t].shift(tau))
        out[tau] = inc
    return out


def cluster_episodes(dates: pd.DatetimeIndex, gap_days: int) -> list:
    if len(dates) == 0:
        return []
    dates = pd.DatetimeIndex(sorted(dates))
    gap_calendar = gap_days * 1.45 * EPISODE_GAP_MULTIPLIER
    episodes, cur = [], [dates[0]]
    for d in dates[1:]:
        if (d - cur[-1]).days > gap_calendar:
            episodes.append(cur)
            cur = [d]
        else:
            cur.append(d)
    episodes.append(cur)
    return episodes


if __name__ == "__main__":
    prices = pd.read_parquet("../multiasset_prices.parquet")
    all_tickers = list(set(CREDIT_PREDICTORS + RATE_PREDICTORS + TARGETS))
    needed_taus = sorted(set(TAU_PAST_GRID + TAU_FUTURE_GRID))
    increments = build_increments(prices, all_tickers, needed_taus)

    results = []
    total_configs = len(CREDIT_PREDICTORS) * len(RATE_PREDICTORS) * len(TARGETS) * len(TAU_PAST_GRID) * len(TAU_FUTURE_GRID) * len(Q_GRID)
    print(f"Searching {total_configs} configs (bearish credit AND bearish rates, pre-specified) -- "
          f"NO raw-observation pre-filter this time, every config with >=1 real episode is reported\n")

    for credit_t, rate_t, target_t, tau_p, tau_f, q in itertools.product(
            CREDIT_PREDICTORS, RATE_PREDICTORS, TARGETS, TAU_PAST_GRID, TAU_FUTURE_GRID, Q_GRID):
        credit_inc = increments[tau_p][credit_t]
        rate_inc = increments[tau_p][rate_t]
        credit_thresh = credit_inc.quantile(q)
        rate_thresh = rate_inc.quantile(q)
        joint_mask = (credit_inc < credit_thresh) & (rate_inc < rate_thresh)
        firing_dates = joint_mask[joint_mask.fillna(False)].index
        n_joint = len(firing_dates)
        if n_joint == 0:
            continue

        target_fwd = increments[tau_f][target_t].shift(-tau_f)
        target_thresh = increments[tau_f][target_t].quantile(0.5)
        joint_dates_valid = firing_dates.intersection(target_fwd.dropna().index)
        if len(joint_dates_valid) == 0:
            continue
        hits = (target_fwd.loc[joint_dates_valid] < target_thresh)
        cpe = float(hits.mean())
        uncond = float((target_fwd.dropna() < target_thresh).mean())
        lift = cpe / uncond if uncond > 0 else np.nan

        episodes = cluster_episodes(firing_dates, tau_p)
        ep_outcomes, ep_years = [], []
        for ep in episodes:
            anchor = ep[-1]
            if anchor in target_fwd.index and pd.notna(target_fwd.get(anchor)):
                ep_outcomes.append(bool(target_fwd[anchor] < target_thresh))
                ep_years.append(sorted(set(d.year for d in ep)))
        n_episodes = len(ep_outcomes)
        ep_hit_rate = float(np.mean(ep_outcomes)) if ep_outcomes else float("nan")
        results.append({
            "credit": credit_t, "rate": rate_t, "target": target_t, "tau_past": tau_p, "tau_future": tau_f,
            "q": q, "n_joint": n_joint, "cpe": cpe, "lift": lift,
            "n_episodes": n_episodes, "episode_hit_rate": ep_hit_rate,
            "episode_years": str(ep_years), "reliable_n>=3": n_episodes >= 3,
        })

    df = pd.DataFrame(results)
    print(f"Total configs with >=1 real firing episode: {len(df)}")
    print(f"  n_episodes==1: {(df['n_episodes']==1).sum()}   n_episodes==2: {(df['n_episodes']==2).sum()}   "
          f"n_episodes>=3 (reliable by this program's own floor): {(df['n_episodes']>=3).sum()}\n")

    reliable = df[df["reliable_n>=3"]]
    if len(reliable):
        print("=== Configs clearing the >=3-episode reliability floor ===")
        print(reliable.sort_values("lift", ascending=False).to_string(index=False))
    else:
        print("=== ZERO configs clear the >=3-episode reliability floor -- confirmed, not assumed ===")

    print("\n=== The n=1 finding driving this whole search (tau_past>=126, q=0.05, all targets/horizons) ===")
    n1 = df[(df["n_episodes"] == 1) & (df["tau_past"] >= 126) & (df["q"] == 0.05)]
    print(n1[["credit", "rate", "target", "tau_past", "tau_future", "n_joint", "episode_hit_rate", "episode_years"]].to_string(index=False))

    df.to_csv("credit_stress_channel_search_results.csv", index=False)
    print("\nSaved: credit_stress_channel_search_results.csv (full, unfiltered)")
