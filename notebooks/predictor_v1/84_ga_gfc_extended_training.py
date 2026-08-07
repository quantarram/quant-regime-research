"""
84_ga_gfc_extended_training.py -- drops the forecast signal (capped at
2014), keeps tau*/pocket + CPE (both derivable from raw prices, which go
back to the 1990s/2000s for this universe), extends training to 2005 to
include the real 2008 GFC bear market as a training regime, same GA
budget (800 x 1500), same 2022+ holdout, same architecture minus the two
forecast features.
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
COST_BPS = _ga82.COST_BPS
MAX_LEV = _ga82.MAX_LEV
REPO_DIR = _ga82.REPO_DIR
OUT_DIR = _ga82.OUT_DIR
sharpe = _ga82.sharpe
simulate_population_batched = _ga82.simulate_population_batched

TRAIN_START = pd.Timestamp("2005-01-01")  # includes the real 2008 GFC
POP_SIZE = 800
N_GENERATIONS = 1500
HIDDEN_DIM = 6
ELITE_FRAC = 0.05
MUTATION_SIGMA_START = 0.35
MUTATION_SIGMA_END = 0.03
TOURNAMENT_K = 4
SEED = 7
N_FEATURES = 4  # pocket_flag, pocket_strength, cpe_firing, cpe_probability -- forecast dropped
STATE_COLS = ["pocket_flag", "pocket_strength", "cpe_firing", "cpe_probability"]


def build_state_no_forecast(prices, cpe_series, pocket_features):
    rows = []
    for t in INSTRUMENTS:
        idx = prices[t].dropna().index
        if cpe_series[t] is not None:
            idx = idx.intersection(cpe_series[t].index)
        idx = idx[idx >= TRAIN_START]
        if len(idx) < 100:
            continue
        daily_ret = np.log(prices[t]).diff().reindex(idx)
        df = pd.DataFrame(index=idx)
        df["pocket_flag"] = pocket_features[t]["pocket_flag"]
        df["pocket_strength"] = pocket_features[t]["pocket_strength"]
        if cpe_series[t] is not None:
            df["cpe_firing"] = cpe_series[t]["cpe_firing"].reindex(idx).fillna(0.0)
            df["cpe_probability"] = cpe_series[t]["cpe_probability"].reindex(idx).fillna(0.0)
        else:
            df["cpe_firing"] = 0.0
            df["cpe_probability"] = 0.0
        df["fwd_ret"] = daily_ret.shift(-1)
        df["instrument"] = t
        df["is_holdout"] = idx >= HOLDOUT_START
        rows.append(df.dropna(subset=["fwd_ret"]))
    full = pd.concat(rows)
    max_strength = full["pocket_strength"].max()
    full["pocket_strength"] = full["pocket_strength"] / (max_strength + 1e-9) if max_strength > 0 else full["pocket_strength"]
    return full


if __name__ == "__main__":
    t0 = time.time()
    prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
    n_bad = int((prices[INSTRUMENTS] <= 0).sum().sum())
    if n_bad:
        prices = prices.copy()
        prices[INSTRUMENTS] = prices[INSTRUMENTS].mask(prices[INSTRUMENTS] <= 0)

    pocket_features = _ga82.load_tau_pocket_features()
    cpe_series = _ga82.load_cpe_series(prices)

    full = build_state_no_forecast(prices, cpe_series, pocket_features)
    inst_map = {t: i for i, t in enumerate(INSTRUMENTS)}
    full["inst_id"] = full["instrument"].map(inst_map)
    train_mask = ~full["is_holdout"].values
    holdout_mask = full["is_holdout"].values
    print(f"Rows: {len(full)} ({train_mask.sum()} train {TRAIN_START.date()}+ incl. 2008 GFC, {holdout_mask.sum()} holdout)")

    rng = np.random.default_rng(SEED)
    W1 = rng.normal(0, 0.5, size=(POP_SIZE, N_FEATURES, HIDDEN_DIM)).astype(np.float32)
    b1 = rng.normal(0, 0.1, size=(POP_SIZE, HIDDEN_DIM)).astype(np.float32)
    W2 = rng.normal(0, 0.5, size=(POP_SIZE, HIDDEN_DIM)).astype(np.float32)
    b2 = rng.normal(0, 0.1, size=(POP_SIZE,)).astype(np.float32)

    train_state = full.loc[train_mask, STATE_COLS].values.astype(np.float32)
    train_fwd = full.loc[train_mask, "fwd_ret"].values.astype(np.float32)
    train_inst = full.loc[train_mask, "inst_id"].values.astype(np.int32)
    train_boundaries = np.zeros(len(train_inst), dtype=bool)
    train_boundaries[0] = True
    train_boundaries[1:] = train_inst[1:] != train_inst[:-1]

    best_hist, mean_hist = [], []
    n_elite = max(2, int(POP_SIZE * ELITE_FRAC))
    for gen in range(N_GENERATIONS):
        net_ret, positions = simulate_population_batched(train_state, train_fwd, train_inst, len(INSTRUMENTS), W1, b1, W2, b2, train_boundaries)
        avg_exposure = np.abs(positions).mean(axis=1) + 1e-6
        fitness = (net_ret.mean(axis=1) / avg_exposure) * 252 * 100
        order = np.argsort(-fitness)
        best_hist.append(float(fitness[order[0]])); mean_hist.append(float(fitness.mean()))
        if gen % 200 == 0 or gen == N_GENERATIONS - 1:
            print(f"  gen {gen:>5}  best={fitness[order[0]]:+.2f}%  mean={fitness.mean():+.2f}%  elapsed={time.time()-t0:.1f}s")
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

    net_ret_f, pos_f = simulate_population_batched(train_state, train_fwd, train_inst, len(INSTRUMENTS), W1, b1, W2, b2, train_boundaries)
    final_fitness = (net_ret_f.mean(axis=1) / (np.abs(pos_f).mean(axis=1) + 1e-6)) * 252 * 100
    best_i = int(np.argmax(final_fitness))
    print(f"Best train fitness: {final_fitness[best_i]:+.2f}%/yr, elapsed {time.time()-t0:.1f}s")
    bW1, bb1, bW2, bb2 = W1[best_i:best_i+1], b1[best_i:best_i+1], W2[best_i:best_i+1], b2[best_i:best_i+1]

    h_state = full.loc[holdout_mask, STATE_COLS].values.astype(np.float32)
    h_fwd = full.loc[holdout_mask, "fwd_ret"].values.astype(np.float32)
    h_inst = full.loc[holdout_mask, "inst_id"].values.astype(np.int32)
    h_bound = np.zeros(len(h_inst), dtype=bool); h_bound[0] = True; h_bound[1:] = h_inst[1:] != h_inst[:-1]
    h_net = simulate_population_batched(h_state, h_fwd, h_inst, len(INSTRUMENTS), bW1, bb1, bW2, bb2, h_bound)[0][0]

    holdout_df = full.loc[holdout_mask].copy()
    holdout_df["strategy_ret"] = h_net
    results = {}
    for t in INSTRUMENTS:
        sub = holdout_df[holdout_df["instrument"] == t]
        if len(sub) < 30: continue
        strat, bh = sub["strategy_ret"].values, sub["fwd_ret"].values
        alpha = float((strat.mean() - bh.mean()) * 252 * 100)
        results[t] = {"alpha_vs_own_pct": alpha, "strategy_sharpe": sharpe(strat), "buy_hold_sharpe": sharpe(bh)}
        print(f"  {t:<10} alpha={alpha:+7.2f}%/yr  strat_sharpe={sharpe(strat):+.3f}  bh_sharpe={sharpe(bh):+.3f}")
    n_pos = sum(1 for r in results.values() if r["alpha_vs_own_pct"] > 0)
    overall_alpha = float((holdout_df["strategy_ret"].mean() - holdout_df["fwd_ret"].mean()) * 252 * 100)
    overall_sharpe = sharpe(holdout_df["strategy_ret"].values)
    print(f"\nPositive on {n_pos}/{len(results)}. Pooled alpha={overall_alpha:+.2f}%/yr, Sharpe={overall_sharpe:+.3f}")
    print(f"vs 82 (pooled fitness, 2015+, w/ forecasts): 1/12, -10.86%/yr, Sharpe +0.178")
    print(f"vs 83 (regime-diverse, 2015+, w/ forecasts):  1/12, -12.90%/yr, Sharpe +0.038")

    with open(os.path.join(OUT_DIR, "84_ga_results.json"), "w") as f:
        json.dump({"per_instrument": results, "n_positive": n_pos, "n_total": len(results),
                    "overall_alpha_pct": overall_alpha, "overall_strategy_sharpe": overall_sharpe}, f, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(best_hist, color="#2f8a4e", label="Best individual"); ax.plot(mean_hist, color="#9AA1AD", label="Population mean")
    ax.set_xlabel("Generation"); ax.set_ylabel("Fitness (train P&L, %/yr)")
    ax.set_title(f"GFC-inclusive training (2005+, no forecast signal), {POP_SIZE}x{N_GENERATIONS}")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "84_ga_fitness_history.png"), dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    ts = sorted(results, key=lambda t: -results[t]["alpha_vs_own_pct"])
    colors = ["#2f8a4e" if results[t]["alpha_vs_own_pct"] > 0 else "#B0492F" for t in ts]
    ax.bar(range(len(ts)), [results[t]["alpha_vs_own_pct"] for t in ts], color=colors)
    ax.axhline(0, color="black", lw=0.8); ax.set_xticks(range(len(ts))); ax.set_xticklabels(ts)
    ax.set_ylabel("Alpha vs own buy-hold, 2022+ (%/yr)")
    ax.set_title(f"GFC-inclusive training (tau*+CPE only, no forecast), OOS 2022+\nPositive {n_pos}/{len(results)}, pooled {overall_alpha:+.2f}%/yr")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "84_ga_decision_layer_equity.png"), dpi=140); plt.close(fig)
    print("Saved 84_ga_results.json, 84_ga_fitness_history.png, 84_ga_decision_layer_equity.png")
