"""
Joint CPE on football's "actual events" predictors -- does combining them
(intersection of tail conditions, same greedy-joint idea as
files/joint_cpe_engine.py) push past the 80% bar that none of them reached
individually? Restricted to the 3 non-market predictors on purpose: this
answers "can real historical results alone, combined, produce a genuine
>=80%-with-lift signal, with no reference to the bookmaker's own price."
market_prob_home is tried as a 4th, optional addition at the end for context.
"""
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).parent / "output"
CPE_THRESH = 0.80
MIN_LIFT = 1.5
MIN_N = 100
Q_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95]

ACTUAL_EVENT_PREDICTORS = ["ppg_diff", "home_ppg_home_only", "away_ppg_away_only"]
ALL_PREDICTORS = ACTUAL_EVENT_PREDICTORS + ["market_prob_home"]


def joint_search(feat, w, predictors, max_size):
    sub = feat[feat["form_window"] == w].dropna(subset=predictors)
    home_win = sub["outcome_home_win"].values
    uncond = home_win.mean()
    vals = {p: sub[p].values for p in predictors}
    thresh = {p: {q: np.quantile(vals[p], q) for q in Q_GRID} for p in predictors}

    results = []
    for size in range(2, max_size + 1):
        for combo in itertools.combinations(predictors, size):
            for qs in itertools.product(Q_GRID, repeat=size):
                mask = np.ones(len(sub), dtype=bool)
                for p, q in zip(combo, qs):
                    mask &= vals[p] > thresh[p][q]
                n = mask.sum()
                if n < MIN_N:
                    continue
                cpe = home_win[mask].mean()
                lift = cpe / uncond if uncond > 0 else np.nan
                results.append(dict(predictors=combo, quantiles=qs, n=n, CPE=cpe, lift=lift,
                                     passes=(cpe >= CPE_THRESH and lift >= MIN_LIFT)))
    return pd.DataFrame(results)


def main():
    feat = pd.read_parquet(OUT_DIR / "football_cpe_features.parquet")

    for w in [5, 10]:
        print(f"\n{'='*70}\n  Joint CPE, actual-events-only predictors, form_window={w}\n{'='*70}")
        res = joint_search(feat, w, ACTUAL_EVENT_PREDICTORS, max_size=3)
        res = res.sort_values("CPE", ascending=False)
        print(f"  Combos tested: {len(res)}   Passing all 3 gates: {res['passes'].sum()}")
        print(f"  Best 5 by CPE (any n>={MIN_N}):")
        print(res.head(5).to_string(index=False))
        if res["passes"].any():
            print(f"\n  PASSING combos:")
            print(res[res["passes"]].sort_values("n", ascending=False).to_string(index=False))

        print(f"\n  Now adding market_prob_home as a 4th optional predictor, for context:")
        res_with_mkt = joint_search(feat, w, ALL_PREDICTORS, max_size=4)
        res_with_mkt = res_with_mkt.sort_values("CPE", ascending=False)
        print(f"  Combos tested: {len(res_with_mkt)}   Passing all 3 gates: {res_with_mkt['passes'].sum()}")
        print(f"  Best 5 by CPE:")
        print(res_with_mkt.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
