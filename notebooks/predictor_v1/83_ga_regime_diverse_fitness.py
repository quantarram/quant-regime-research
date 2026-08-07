"""
83_ga_regime_diverse_fitness.py
==================================
Direct follow-up to 82_ga_decision_layer_search.py's real result: a
1.2M-policy genetic-algorithm search found a policy with genuine,
sustained training-period learning (fitness climbed cleanly -3% -> +51%/
yr across 1,500 generations, no plateau) that then failed decisively on
the 2022+ holdout (positive alpha on only 1/12 instruments, pooled
-10.86%/yr) -- textbook overfitting to the single pre-2022 training
window, which included no equity bear market and no rate-hiking cycle
comparable to 2022 itself.

The diagnosis named a concrete improvement, not just the failure: fitness
was computed on ONE pooled training period, so nothing in the search was
ever penalized for finding a policy that only works in a low-rate,
mostly-rising regime. This script implements that improvement directly:
REGIME-DIVERSE FITNESS. The pre-2022 training window is split into 4 real,
disjoint, chronological sub-periods (2015-2016.75, 2016.75-2018.5,
2018.5-2020.25, 2020.25-2022), each individual's exposure-normalized P&L
is computed SEPARATELY on each sub-period, and the final fitness used for
selection is the MINIMUM across the four -- a policy only scores well if
it holds up in every one of the four real sub-regimes, not just on
average. This is the same walk-forward-across-multiple-real-windows
discipline already used for every other strategy in this paper, applied
to the fitness function itself.

Reuses 82's own data pipeline (load_tau_pocket_features, load_forecast_
series, load_cpe_series, build_state_and_returns, simulate_population_
batched) unchanged -- only the fitness aggregation and the GA loop built
on top of it are new, so any difference in the result is attributable to
the fitness change itself, not a different feature set or simulation.

Same population/generation budget as 82 (800 x 1500 = 1.2M policies) for
a fair, like-for-like comparison. Same real walk-forward discipline: the
2022+ holdout is touched exactly once, at the end, with the single best-
evolved policy.

Run (from notebooks/predictor_v1/):
    python 83_ga_regime_diverse_fitness.py
Output: 83_ga_results.json, 83_ga_fitness_history.png,
        83_ga_decision_layer_equity.png
"""
import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
_ga82 = import_module("82_ga_decision_layer_search")

INSTRUMENTS = _ga82.INSTRUMENTS
HOLDOUT_START = _ga82.HOLDOUT_START
TRAIN_START = _ga82.TRAIN_START
N_FEATURES = _ga82.N_FEATURES
COST_BPS = _ga82.COST_BPS
MAX_LEV = _ga82.MAX_LEV

REPO_DIR = _ga82.REPO_DIR
OUT_DIR = _ga82.OUT_DIR

POP_SIZE = 800
N_GENERATIONS = 1500
HIDDEN_DIM = 6
ELITE_FRAC = 0.05
MUTATION_SIGMA_START = 0.35
MUTATION_SIGMA_END = 0.03
TOURNAMENT_K = 4
SEED = 7
N_SUBPERIODS = 4

sharpe = _ga82.sharpe
simulate_population_batched = _ga82.simulate_population_batched


def subperiod_fitness(state, fwd_ret, inst_ids, n_instruments, W1, b1, W2, b2, boundaries):
    net_ret, positions = simulate_population_batched(state, fwd_ret, inst_ids, n_instruments, W1, b1, W2, b2, boundaries)
    avg_exposure = np.abs(positions).mean(axis=1) + 1e-6
    return (net_ret.mean(axis=1) / avg_exposure) * 252 * 100


if __name__ == "__main__":
    t0 = time.time()
    prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
    n_bad = int((prices[INSTRUMENTS] <= 0).sum().sum())
    if n_bad:
        prices = prices.copy()
        prices[INSTRUMENTS] = prices[INSTRUMENTS].mask(prices[INSTRUMENTS] <= 0)

    print("Loading all three signal types (identical pipeline to 82_ga_decision_layer_search.py)...")
    pocket_features = _ga82.load_tau_pocket_features()
    forecasts = _ga82.load_forecast_series(prices)
    cpe_series = _ga82.load_cpe_series(prices)

    full = _ga82.build_state_and_returns(prices, forecasts, cpe_series, pocket_features)
    inst_map = {t: i for i, t in enumerate(INSTRUMENTS)}
    full["inst_id"] = full["instrument"].map(inst_map)

    state_cols = ["forecast_mu_z", "forecast_sigma_z", "pocket_flag", "pocket_strength", "cpe_firing", "cpe_probability"]
    train_mask = ~full["is_holdout"].values
    holdout_mask = full["is_holdout"].values

    # 4 real, disjoint, chronological sub-periods spanning the pre-2022 training window
    sub_edges = pd.date_range(TRAIN_START, HOLDOUT_START, periods=N_SUBPERIODS + 1)
    print(f"\nRegime-diverse training sub-periods:")
    for i in range(N_SUBPERIODS):
        print(f"  Sub-period {i+1}: {sub_edges[i].date()} to {sub_edges[i+1].date()}")

    subperiods = []
    train_full = full.loc[train_mask].copy()
    for i in range(N_SUBPERIODS):
        lo, hi = sub_edges[i], sub_edges[i + 1]
        m = (train_full.index >= lo) & (train_full.index < hi if i < N_SUBPERIODS - 1 else train_full.index <= hi)
        sub = train_full.loc[m]
        s_state = sub[state_cols].values.astype(np.float32)
        s_fwd = sub["fwd_ret"].values.astype(np.float32)
        s_inst = sub["inst_id"].values.astype(np.int32)
        s_bound = np.zeros(len(s_inst), dtype=bool)
        if len(s_inst) > 0:
            s_bound[0] = True
            s_bound[1:] = s_inst[1:] != s_inst[:-1]
        subperiods.append((s_state, s_fwd, s_inst, s_bound))
        print(f"    -> {len(sub)} rows")

    rng = np.random.default_rng(SEED)
    W1 = rng.normal(0, 0.5, size=(POP_SIZE, N_FEATURES, HIDDEN_DIM)).astype(np.float32)
    b1 = rng.normal(0, 0.1, size=(POP_SIZE, HIDDEN_DIM)).astype(np.float32)
    W2 = rng.normal(0, 0.5, size=(POP_SIZE, HIDDEN_DIM)).astype(np.float32)
    b2 = rng.normal(0, 0.1, size=(POP_SIZE,)).astype(np.float32)

    best_fitness_history, mean_fitness_history = [], []
    worst_subperiod_history = []
    n_elite = max(2, int(POP_SIZE * ELITE_FRAC))

    print(f"\nRunning GA: population={POP_SIZE}, generations={N_GENERATIONS}, "
          f"{N_SUBPERIODS} sub-periods/individual/generation "
          f"({POP_SIZE * N_GENERATIONS:,} total distinct policies evaluated)...")
    for gen in range(N_GENERATIONS):
        sub_fitnesses = np.stack([
            subperiod_fitness(s_state, s_fwd, s_inst, len(INSTRUMENTS), W1, b1, W2, b2, s_bound)
            for (s_state, s_fwd, s_inst, s_bound) in subperiods
        ], axis=0)  # (N_SUBPERIODS, pop)
        fitness = sub_fitnesses.min(axis=0)  # worst-sub-period fitness: must hold up everywhere
        order = np.argsort(-fitness)
        best_fitness_history.append(float(fitness[order[0]]))
        mean_fitness_history.append(float(fitness.mean()))
        worst_subperiod_history.append(int(np.argmin(sub_fitnesses[:, order[0]])))

        if gen % 100 == 0 or gen == N_GENERATIONS - 1:
            elapsed = time.time() - t0
            per_sub_best = sub_fitnesses[:, order[0]]
            print(f"  gen {gen:>5}  best_fitness(worst-subperiod P&L %/yr)={fitness[order[0]]:+.2f}%  "
                  f"mean={fitness.mean():+.2f}%  per-sub={['%.1f' % x for x in per_sub_best]}  elapsed={elapsed:.1f}s")

        elite_idx = order[:n_elite]
        mut_sigma = MUTATION_SIGMA_START + (MUTATION_SIGMA_END - MUTATION_SIGMA_START) * (gen / N_GENERATIONS)

        new_W1, new_b1, new_W2, new_b2 = np.empty_like(W1), np.empty_like(b1), np.empty_like(W2), np.empty_like(b2)
        new_W1[:n_elite], new_b1[:n_elite] = W1[elite_idx], b1[elite_idx]
        new_W2[:n_elite], new_b2[:n_elite] = W2[elite_idx], b2[elite_idx]

        for i in range(n_elite, POP_SIZE):
            pa = elite_idx[rng.integers(0, len(elite_idx))] if rng.random() < 0.5 else order[np.min(rng.integers(0, POP_SIZE, TOURNAMENT_K))]
            pb = elite_idx[rng.integers(0, len(elite_idx))] if rng.random() < 0.5 else order[np.min(rng.integers(0, POP_SIZE, TOURNAMENT_K))]
            alpha = rng.random()
            new_W1[i] = alpha * W1[pa] + (1 - alpha) * W1[pb] + rng.normal(0, mut_sigma, size=W1.shape[1:])
            new_b1[i] = alpha * b1[pa] + (1 - alpha) * b1[pb] + rng.normal(0, mut_sigma, size=b1.shape[1:])
            new_W2[i] = alpha * W2[pa] + (1 - alpha) * W2[pb] + rng.normal(0, mut_sigma, size=W2.shape[1:])
            new_b2[i] = alpha * b2[pa] + (1 - alpha) * b2[pb] + rng.normal(0, mut_sigma)

        W1, b1, W2, b2 = new_W1, new_b1, new_W2, new_b2

    final_sub_fitnesses = np.stack([
        subperiod_fitness(s_state, s_fwd, s_inst, len(INSTRUMENTS), W1, b1, W2, b2, s_bound)
        for (s_state, s_fwd, s_inst, s_bound) in subperiods
    ], axis=0)
    final_fitness = final_sub_fitnesses.min(axis=0)
    best_i = int(np.argmax(final_fitness))
    print(f"\nBest evolved policy (worst-subperiod train P&L): {final_fitness[best_i]:+.2f}%/yr "
          f"(per-subperiod: {[round(float(x),2) for x in final_sub_fitnesses[:, best_i]]}), "
          f"total elapsed {time.time()-t0:.1f}s, {POP_SIZE * N_GENERATIONS:,} policies evaluated")

    best_W1, best_b1, best_W2, best_b2 = W1[best_i:best_i+1], b1[best_i:best_i+1], W2[best_i:best_i+1], b2[best_i:best_i+1]

    holdout_state = full.loc[holdout_mask, state_cols].values.astype(np.float32)
    holdout_fwd = full.loc[holdout_mask, "fwd_ret"].values.astype(np.float32)
    holdout_inst = full.loc[holdout_mask, "inst_id"].values.astype(np.int32)
    holdout_boundaries = np.zeros(len(holdout_inst), dtype=bool)
    holdout_boundaries[0] = True
    holdout_boundaries[1:] = holdout_inst[1:] != holdout_inst[:-1]
    holdout_net = simulate_population_batched(holdout_state, holdout_fwd, holdout_inst, len(INSTRUMENTS), best_W1, best_b1, best_W2, best_b2, holdout_boundaries)[0][0]

    holdout_df = full.loc[holdout_mask].copy()
    holdout_df["strategy_ret"] = holdout_net

    print(f"\n{'='*90}\nBest evolved policy (regime-diverse fitness), OOS 2022+ holdout, real point estimates only\n{'='*90}")
    results = {}
    for t in INSTRUMENTS:
        sub = holdout_df[holdout_df["instrument"] == t]
        if len(sub) < 30:
            continue
        strat = sub["strategy_ret"].values
        bh = sub["fwd_ret"].values
        alpha = float((strat.mean() - bh.mean()) * 252 * 100)
        sh = sharpe(strat)
        bh_sh = sharpe(bh)
        results[t] = {"alpha_vs_own_pct": alpha, "strategy_sharpe": sh, "buy_hold_sharpe": bh_sh, "n_days": int(len(sub))}
        print(f"  {t:<10} alpha vs own buy-hold={alpha:+7.2f}%/yr   strategy Sharpe={sh:+.3f}   buy-hold Sharpe={bh_sh:+.3f}")

    n_positive = sum(1 for r in results.values() if r["alpha_vs_own_pct"] > 0)
    overall_strat_sharpe = sharpe(holdout_df["strategy_ret"].values)
    overall_bh_sharpe = sharpe(holdout_df["fwd_ret"].values)
    overall_alpha = float((holdout_df["strategy_ret"].mean() - holdout_df["fwd_ret"].mean()) * 252 * 100)
    print(f"\nPositive alpha on {n_positive}/{len(results)} instruments")
    print(f"Pooled across all instruments: alpha={overall_alpha:+.2f}%/yr, "
          f"strategy Sharpe={overall_strat_sharpe:+.3f}, pooled buy-hold Sharpe={overall_bh_sharpe:+.3f}")
    print(f"\nComparison to 82's pooled-fitness result: alpha -10.86%/yr, strategy Sharpe +0.178, "
          f"1/12 instruments positive")

    with open(os.path.join(OUT_DIR, "83_ga_results.json"), "w") as f:
        json.dump({
            "per_instrument": results, "n_positive": n_positive, "n_total": len(results),
            "overall_alpha_pct": overall_alpha, "overall_strategy_sharpe": overall_strat_sharpe,
            "overall_buy_hold_sharpe": overall_bh_sharpe,
            "pop_size": POP_SIZE, "n_generations": N_GENERATIONS, "n_subperiods": N_SUBPERIODS,
            "total_policies_evaluated": POP_SIZE * N_GENERATIONS,
            "best_train_worst_subperiod_fitness": float(final_fitness[best_i]),
            "best_train_per_subperiod_fitness": [float(x) for x in final_sub_fitnesses[:, best_i]],
            "comparison_82_pooled_fitness": {"overall_alpha_pct": -10.86, "overall_strategy_sharpe": 0.178, "n_positive": 1},
        }, f, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(best_fitness_history, color="#2f8a4e", lw=1.3, label="Best individual (worst-subperiod P&L, %/yr)")
    ax.plot(mean_fitness_history, color="#9AA1AD", lw=1.0, label="Population mean")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (worst-subperiod P&L, %/yr)")
    ax.set_title(f"Regime-diverse GA fitness over {N_GENERATIONS} generations, population {POP_SIZE}, {N_SUBPERIODS} sub-periods\n"
                 f"Selection = minimum fitness across sub-periods, not pooled average")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "83_ga_fitness_history.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    tickers_sorted = sorted(results, key=lambda t: -results[t]["alpha_vs_own_pct"])
    colors = ["#2f8a4e" if results[t]["alpha_vs_own_pct"] > 0 else "#B0492F" for t in tickers_sorted]
    ax.bar(range(len(tickers_sorted)), [results[t]["alpha_vs_own_pct"] for t in tickers_sorted], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(tickers_sorted))); ax.set_xticklabels(tickers_sorted)
    ax.set_ylabel("Regime-diverse GA policy alpha vs. own buy-hold, 2022+ holdout (%/yr)")
    ax.set_title(f"Regime-diverse fitness (min across {N_SUBPERIODS} sub-periods), OOS 2022+\n"
                 f"Positive on {n_positive}/{len(results)} instruments, pooled alpha {overall_alpha:+.2f}%/yr "
                 f"(82's pooled-fitness version: 1/12, -10.86%/yr)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "83_ga_decision_layer_equity.png"), dpi=140)
    plt.close(fig)
    print("\nSaved: 83_ga_results.json, 83_ga_fitness_history.png, 83_ga_decision_layer_equity.png")
