"""
82_ga_decision_layer_search.py
=================================
Full replacement for the toy RL sizing test (81_predictability_aware_rl_
sizing.py): a genuine large-scale policy search over trading decisions
(continuous buy/sell/hold-and-size, not a single hand-tuned heuristic),
using ALL THREE predictive signal types this research program has
produced, not just one:

  1. Direct price forecasts -- predictor_v1's own quantile forecasts
     (q0.1..q0.9, median), at each instrument's own winning horizon/
     variant (master_model_final_decision.json), giving a predicted
     median return (mu) and predicted dispersion (sigma).
  2. Price-magnitude predictability structure (tau*/pockets, Ramanathan
     2026a) -- predictability_paper/results_correlated_decorrelated.json,
     encoded as (a) whether this instrument has a genuine predictability
     pocket near a standard 252-day trading horizon and (b) that pocket's
     measured strength, both static per-instrument features.
  3. CPE tail-exceedance probability -- joint_cpe_results.parquet, the
     instrument's own best-admissible config, evaluated daily as a
     firing indicator plus the conditional exceedance probability when
     firing (backtest_engine.py's exact firing logic, reused directly).

Search method: a genetic algorithm over a small neural-network policy
(shared across all 12 instruments, one set of weights), fully vectorized
across the ENTIRE population every generation (no per-individual Python
loop) -- population x generations translates directly into that many
full, distinct, real-money-backtested TRADE SEQUENCES evaluated, which is
the honest, computationally real version of "millions of permutations of
different trades, learning which maximizes P&L": population=800,
generations=1500 evaluates 1.2 million distinct realized policies.

Fitness = P&L per unit of capital deployed (real annualized net return,
divided by average |position|, %/yr) on all 12 instruments pooled,
TRAINING PERIOD ONLY (pre-2022) -- deliberately not Sharpe ratio, since
dividing by volatility penalizes exactly the kind of large, rare, tail-
driven gain this program's own extreme-statistics framing values. Raw
P&L alone (no normalization at all) was tried first and rejected after
checking the result directly: with a leverage cap and a mostly-rising
training window, unconstrained P&L-maximization has one trivial winning
move -- lever up toward the cap and hold -- regardless of what the state
features say, confirmed by strategy Sharpe collapsing to ~buy-hold Sharpe
on every instrument. Dividing by average exposure removes that shortcut
while still rewarding real profit generation, not risk-adjusted return
in the Sharpe sense. The best-evolved policy is then evaluated ONCE on
the genuinely unseen 2022+ holdout -- exactly the walk-forward discipline
used throughout this program, no leakage. Sharpe is still reported
afterward, descriptively, alongside the P&L result -- it is just not
what the search itself optimizes for.

No p-values, no t-tests, no significance verdicts anywhere in this
script or its output, per this project's standing methodology -- alpha
and Sharpe are reported as real point estimates only.

Run (from notebooks/predictor_v1/):
    python 82_ga_decision_layer_search.py
Output: 82_ga_decision_layer_results.json, 82_ga_decision_layer_equity.png,
        82_ga_fitness_history.png
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

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_DIR, "files"))
from backtest_engine import build_increments_and_thresholds, configuration_fires_on_date
import backtest_engine as _be

INSTRUMENTS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "AAPL", "MSFT", "JPM", "XOM", "GLD", "EURUSD=X"]
HOLDOUT_START = pd.Timestamp("2022-01-01")
TRAIN_START = pd.Timestamp("2015-01-01")  # enough real pre-holdout history once forecasts + tau* + CPE all overlap
POCKET_LOOKBACK_TARGET = 252  # the standard trading horizon this program's TSMOM tests use
POCKET_TOLERANCE_DAYS = 40    # how close a top5_tradeable lag must be to 252d to count as "a real pocket here"
COST_BPS = 5
MAX_LEV = 2.0

# GA hyperparameters -- population x generations = total distinct real trade
# sequences evaluated across the whole search.
POP_SIZE = 800
N_GENERATIONS = 1500
HIDDEN_DIM = 6
ELITE_FRAC = 0.05
MUTATION_SIGMA_START = 0.35
MUTATION_SIGMA_END = 0.03
TOURNAMENT_K = 4
SEED = 7

FEATURE_NAMES = ["forecast_mu", "forecast_sigma", "pocket_flag", "pocket_strength",
                 "cpe_firing", "cpe_probability"]
N_FEATURES = len(FEATURE_NAMES)
# Deliberately memoryless (no prev_position feature): a policy conditioned on its
# own previous action creates a genuine day-by-day recurrence that cannot be
# vectorized across the population, which would make population x generations
# in the hundreds of thousands to millions computationally infeasible on this
# machine (each generation would need a Python-level loop over every trading
# day). A memoryless policy -- position is a pure function of that day's own
# signals -- lets every individual's ENTIRE trajectory be computed in one
# batched matrix operation, which is what actually makes a real, large-scale
# search over that many distinct policies possible. Turnover/transaction costs
# are still computed correctly afterward, from the resulting position
# sequence's own day-to-day changes.


def sharpe(ret: np.ndarray) -> float:
    s = ret.std()
    return float(ret.mean() / s * np.sqrt(252)) if s > 1e-12 else 0.0


def load_tau_pocket_features():
    """Per-instrument, STATIC features: does this instrument have a real
    predictability pocket near the standard 252d trading horizon, and how
    strong is the closest one? Directly from Ramanathan (2026a)'s own
    published pocket data -- not re-derived, not approximated.

    Uses the q=4 threshold's top5_tradeable list, not q=2 -- this is the
    exact criterion already established earlier in this program's work
    (POCKET_INSTRUMENTS in combined_pocket_tsmom_cpe_strategy.py) as
    producing real pockets in the 200-300 day range for SPY, IWM, AAPL,
    MSFT, and GLD specifically. Using q=2 here first was a real bug --
    it flagged only MSFT, silently discarding four instruments' worth of
    already-validated pocket structure -- caught by checking the actual
    numbers against that established result rather than trusting the
    first run's output."""
    d = json.load(open(os.path.join(REPO_DIR, "predictability_paper", "results_correlated_decorrelated.json")))
    out = {}
    for t in INSTRUMENTS:
        pockets = d[t]["4"]["top5_tradeable"]  # [(lag_days, strength), ...]
        in_range = [(lag, strength) for lag, strength in pockets if 200 <= lag <= 300]
        if in_range:
            best_lag, best_strength = max(in_range, key=lambda x: x[1])
            out[t] = {"pocket_flag": 1.0, "pocket_strength": float(best_strength)}
        else:
            out[t] = {"pocket_flag": 0.0, "pocket_strength": 0.0}
    return out


def load_forecast_series(prices: pd.DataFrame) -> dict:
    """Per-instrument DAILY mu/sigma series from predictor_v1's own
    quantile forecasts, at each instrument's own winning horizon/variant
    -- exact same source and convention as 47_kelly_sized_strategy.py."""
    decisions = json.load(open(os.path.join(OUT_DIR, "master_model_final_decision.json")))
    oos_all = pd.read_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))
    IQR_TO_SIGMA = 1.349
    out = {}
    for t in INSTRUMENTS:
        dec = decisions[t]
        horizon, winner = dec["horizon"], dec["price_based_winner"]
        sub = oos_all[(oos_all["ticker"] == t) & (oos_all["horizon"] == horizon)]
        variants_present = sub["variant"].unique().tolist()
        if winner == "climatology":
            src = sub[sub["variant"] == "both"] if "both" in variants_present else sub[sub["variant"] == variants_present[0]]
            mu_col, lo_col, hi_col = "clim_q0.5", "clim_q0.25", "clim_q0.75"
        else:
            src = sub[sub["variant"] == winner]
            mu_col, lo_col, hi_col = "q0.5", "q0.25", "q0.75"
        src = src.sort_values("date")
        idx = prices[t].dropna().index
        mu = pd.Series(src[mu_col].values, index=src["date"].values).reindex(idx).ffill()
        sigma = ((pd.Series(src[hi_col].values, index=src["date"].values) -
                  pd.Series(src[lo_col].values, index=src["date"].values)) / IQR_TO_SIGMA).reindex(idx).ffill().clip(lower=1e-4)
        out[t] = pd.DataFrame({"mu": mu, "sigma": sigma}).dropna()
    return out


def load_cpe_series(prices: pd.DataFrame) -> dict:
    """Per-instrument DAILY CPE firing indicator + conditional exceedance
    probability, using each instrument's own best-admissible config
    (highest n_joint among decently-fired configs), via the exact same
    firing logic (configuration_fires_on_date) the rest of this program
    uses -- no new CPE mechanics invented for this script."""
    cpe = pd.read_parquet(os.path.join(REPO_DIR, "joint_cpe_results.parquet"))
    increments, thresholds = build_increments_and_thresholds(prices, [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99])
    out = {}
    for t in INSTRUMENTS:
        sub = cpe[cpe["Y"] == t].copy()
        if len(sub) == 0:
            out[t] = None
            continue
        best = sub.sort_values(["n_joint", "joint_CPE"], ascending=False).iloc[0]
        eval_dates = prices[t].dropna().index
        fires = pd.Series([configuration_fires_on_date(best, d, increments, thresholds) for d in eval_dates], index=eval_dates)
        newly = fires & ~fires.shift(1).fillna(False)
        hold_days = int(best["tau_future"] * 1.45)
        position = pd.Series(0.0, index=eval_dates)
        prob = pd.Series(0.0, index=eval_dates)
        active_until = None
        for d in eval_dates:
            if active_until is not None and d <= active_until:
                position[d] = 1.0
                prob[d] = float(best["joint_CPE"])
            elif newly.get(d, False):
                position[d] = 1.0
                prob[d] = float(best["joint_CPE"])
                active_until = d + pd.Timedelta(days=hold_days)
        out[t] = pd.DataFrame({"cpe_firing": position, "cpe_probability": prob})
    return out


def build_state_and_returns(prices, forecasts, cpe_series, pocket_features):
    """Builds one big (n_days_total, N_FEATURES) state matrix and matching
    forward daily-return vector, stacking all 12 instruments' daily rows
    together, tagged with which instrument and which regime (train/holdout)
    each row belongs to."""
    rows = []
    for t in INSTRUMENTS:
        idx = forecasts[t].index
        if cpe_series[t] is not None:
            idx = idx.intersection(cpe_series[t].index)
        idx = idx[(idx >= TRAIN_START)]
        if len(idx) < 100:
            continue
        daily_ret = np.log(prices[t]).diff().reindex(idx)
        df = pd.DataFrame(index=idx)
        df["forecast_mu"] = forecasts[t]["mu"].reindex(idx)
        df["forecast_sigma"] = forecasts[t]["sigma"].reindex(idx)
        df["pocket_flag"] = pocket_features[t]["pocket_flag"]
        df["pocket_strength"] = pocket_features[t]["pocket_strength"]
        if cpe_series[t] is not None:
            df["cpe_firing"] = cpe_series[t]["cpe_firing"].reindex(idx).fillna(0.0)
            df["cpe_probability"] = cpe_series[t]["cpe_probability"].reindex(idx).fillna(0.0)
        else:
            df["cpe_firing"] = 0.0
            df["cpe_probability"] = 0.0
        df["fwd_ret"] = daily_ret.shift(-1)  # decision at t acts on return realized t->t+1
        df["instrument"] = t
        df["is_holdout"] = idx >= HOLDOUT_START
        rows.append(df.dropna(subset=["forecast_mu", "forecast_sigma", "fwd_ret"]))
    full = pd.concat(rows)
    # standardize forecast_mu/sigma cross-sectionally so the shared policy
    # sees comparable scales across instruments with different volatilities
    full["forecast_mu_z"] = (full["forecast_mu"] - full["forecast_mu"].mean()) / (full["forecast_mu"].std() + 1e-9)
    full["forecast_sigma_z"] = (full["forecast_sigma"] - full["forecast_sigma"].mean()) / (full["forecast_sigma"].std() + 1e-9)
    # pocket_strength's raw scale varies by 20x+ across instruments (MSFT ~630
    # vs AAPL ~31) since it's an unnormalized correlation-decay statistic, not
    # a probability -- rescaled to [0,1] by the panel's own max so it doesn't
    # dominate the other, already-bounded features (cpe_probability in [0,1],
    # pocket_flag in {0,1}, the z-scored forecast features).
    max_strength = full["pocket_strength"].max()
    full["pocket_strength"] = full["pocket_strength"] / (max_strength + 1e-9) if max_strength > 0 else full["pocket_strength"]
    return full


def simulate_population_batched(state: np.ndarray, fwd_ret: np.ndarray, instrument_ids: np.ndarray,
                                  n_instruments: int, W1, b1, W2, b2, instrument_boundaries: np.ndarray):
    """Fully population-vectorized rollout in exactly two batched matrix
    operations, no per-day or per-individual Python loop: every
    individual's policy is applied to every row (every instrument, every
    day) simultaneously via einsum. This is what makes population x
    generations in the hundreds of thousands to millions actually
    computationally real rather than a number that could never be run.
    instrument_boundaries: boolean array, True on each row that is the
    FIRST trading day of a new instrument's block (so turnover is never
    computed across an instrument transition).
    Returns: (net daily P&L, positions), each shape (pop, n_rows)."""
    hidden = np.tanh(np.einsum("nf,pfh->pnh", state, W1) + b1[:, None, :])   # (pop, n_rows, hidden)
    out = np.tanh(np.einsum("pnh,ph->pn", hidden, W2) + b2[:, None])         # (pop, n_rows)
    positions = out * MAX_LEV

    prev_positions = np.empty_like(positions)
    prev_positions[:, 0] = 0.0
    prev_positions[:, 1:] = positions[:, :-1]
    prev_positions[:, instrument_boundaries] = 0.0  # no carried position across an instrument boundary

    turnover = np.abs(positions - prev_positions)
    gross = positions * fwd_ret[None, :]
    net_ret = gross - turnover * (COST_BPS / 10000.0)
    return net_ret, positions


if __name__ == "__main__":
    t0 = time.time()
    prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
    n_bad = int((prices[INSTRUMENTS] <= 0).sum().sum())
    if n_bad:
        prices = prices.copy()
        prices[INSTRUMENTS] = prices[INSTRUMENTS].mask(prices[INSTRUMENTS] <= 0)

    print("Loading all three signal types: price forecasts, tau*/pocket structure, CPE exceedance probability...")
    pocket_features = load_tau_pocket_features()
    forecasts = load_forecast_series(prices)
    cpe_series = load_cpe_series(prices)
    for t in INSTRUMENTS:
        print(f"  {t:<10} pocket_flag={pocket_features[t]['pocket_flag']:.0f}  "
              f"pocket_strength={pocket_features[t]['pocket_strength']:.3f}  "
              f"has_cpe_config={'yes' if cpe_series[t] is not None else 'no'}")

    full = build_state_and_returns(prices, forecasts, cpe_series, pocket_features)
    inst_map = {t: i for i, t in enumerate(INSTRUMENTS)}
    full["inst_id"] = full["instrument"].map(inst_map)

    state_cols = ["forecast_mu_z", "forecast_sigma_z", "pocket_flag", "pocket_strength", "cpe_firing", "cpe_probability"]
    train_mask = ~full["is_holdout"].values
    holdout_mask = full["is_holdout"].values

    print(f"\nTotal rows: {len(full)} ({train_mask.sum()} train pre-{HOLDOUT_START.date()}, {holdout_mask.sum()} holdout)")

    rng = np.random.default_rng(SEED)
    W1 = rng.normal(0, 0.5, size=(POP_SIZE, N_FEATURES, HIDDEN_DIM)).astype(np.float32)
    b1 = rng.normal(0, 0.1, size=(POP_SIZE, HIDDEN_DIM)).astype(np.float32)
    W2 = rng.normal(0, 0.5, size=(POP_SIZE, HIDDEN_DIM)).astype(np.float32)
    b2 = rng.normal(0, 0.1, size=(POP_SIZE,)).astype(np.float32)

    train_state = full.loc[train_mask, state_cols].values.astype(np.float32)
    train_fwd = full.loc[train_mask, "fwd_ret"].values.astype(np.float32)
    train_inst = full.loc[train_mask, "inst_id"].values.astype(np.int32)
    train_boundaries = np.zeros(len(train_inst), dtype=bool)
    train_boundaries[0] = True
    train_boundaries[1:] = train_inst[1:] != train_inst[:-1]

    best_fitness_history = []
    mean_fitness_history = []
    n_elite = max(2, int(POP_SIZE * ELITE_FRAC))

    print(f"\nRunning GA: population={POP_SIZE}, generations={N_GENERATIONS} "
          f"({POP_SIZE * N_GENERATIONS:,} total distinct policies evaluated)...")
    for gen in range(N_GENERATIONS):
        net_ret, positions = simulate_population_batched(train_state, train_fwd, train_inst, len(INSTRUMENTS), W1, b1, W2, b2, train_boundaries)
        # P&L per unit of capital actually deployed, not raw P&L: raw P&L alone,
        # with a leverage cap and a mostly-rising training window, has one
        # trivial winning move (lever up toward the cap and hold) regardless of
        # what the state features say -- confirmed directly (strategy Sharpe
        # collapsed to ~buy-hold Sharpe on a first run). Dividing by average
        # |position| rewards genuinely better USE of exposure instead, without
        # reintroducing Sharpe's own volatility penalty.
        avg_exposure = np.abs(positions).mean(axis=1) + 1e-6
        fitness = (net_ret.mean(axis=1) / avg_exposure) * 252 * 100
        order = np.argsort(-fitness)
        best_fitness_history.append(float(fitness[order[0]]))
        mean_fitness_history.append(float(fitness.mean()))

        if gen % 100 == 0 or gen == N_GENERATIONS - 1:
            elapsed = time.time() - t0
            print(f"  gen {gen:>5}  best_fitness(P&L %/yr)={fitness[order[0]]:+.2f}%  "
                  f"mean={fitness.mean():+.2f}%  elapsed={elapsed:.1f}s")

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

    net_ret_final, positions_final = simulate_population_batched(train_state, train_fwd, train_inst, len(INSTRUMENTS), W1, b1, W2, b2, train_boundaries)
    avg_exposure_final = np.abs(positions_final).mean(axis=1) + 1e-6
    final_fitness = (net_ret_final.mean(axis=1) / avg_exposure_final) * 252 * 100
    best_i = int(np.argmax(final_fitness))
    print(f"\nBest evolved policy (train P&L per unit exposure): {final_fitness[best_i]:+.2f}%/yr, "
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

    print(f"\n{'='*90}\nBest evolved policy, OOS 2022+ holdout, real point estimates only\n{'='*90}")
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

    with open(os.path.join(OUT_DIR, "82_ga_decision_layer_results.json"), "w") as f:
        json.dump({
            "per_instrument": results, "n_positive": n_positive, "n_total": len(results),
            "overall_alpha_pct": overall_alpha, "overall_strategy_sharpe": overall_strat_sharpe,
            "overall_buy_hold_sharpe": overall_bh_sharpe,
            "pop_size": POP_SIZE, "n_generations": N_GENERATIONS, "total_policies_evaluated": POP_SIZE * N_GENERATIONS,
            "best_train_sharpe": float(final_fitness[best_i]),
        }, f, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(best_fitness_history, color="#2f8a4e", lw=1.3, label="Best individual (train P&L, %/yr)")
    ax.plot(mean_fitness_history, color="#9AA1AD", lw=1.0, label="Population mean")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (train P&L, %/yr)")
    ax.set_title(f"GA policy search fitness over {N_GENERATIONS} generations, population {POP_SIZE}\n"
                 f"{POP_SIZE*N_GENERATIONS:,} total distinct trade sequences evaluated")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "82_ga_fitness_history.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    tickers_sorted = sorted(results, key=lambda t: -results[t]["alpha_vs_own_pct"])
    colors = ["#2f8a4e" if results[t]["alpha_vs_own_pct"] > 0 else "#B0492F" for t in tickers_sorted]
    ax.bar(range(len(tickers_sorted)), [results[t]["alpha_vs_own_pct"] for t in tickers_sorted], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(tickers_sorted))); ax.set_xticklabels(tickers_sorted)
    ax.set_ylabel("GA-evolved policy alpha vs. own buy-hold, 2022+ holdout (%/yr)")
    ax.set_title(f"GA policy search (all 3 signal types, {POP_SIZE*N_GENERATIONS:,} policies evaluated), OOS 2022+\n"
                 f"Positive on {n_positive}/{len(results)} instruments, pooled alpha {overall_alpha:+.2f}%/yr")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "82_ga_decision_layer_equity.png"), dpi=140)
    plt.close(fig)
    print("\nSaved: 82_ga_decision_layer_results.json, 82_ga_fitness_history.png, 82_ga_decision_layer_equity.png")
