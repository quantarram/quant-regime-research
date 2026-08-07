"""
85_ppo_decision_layer.py
===========================
Real RL, not the GA-based workaround (82/83/84): the memoryless policy
used throughout those scripts (no previous-position or unrealized-P&L
feature) was a computational shortcut specific to GA -- vectorizing a
population of 800 across a day-by-day recurrence isn't tractable, so the
policy was made a pure function of same-day signals only. That constraint
does not apply to actual RL: a single agent trained via proper temporal
credit assignment (PPO, generalized advantage estimation, a learned value
function) processes each instrument as one sequential episode and can use
genuinely path-dependent state -- current position, unrealized P&L, time
in trade -- the "game" framing this test is actually suited to, unlike
memoryless GA.

Same three predictive signal types as 82/83 (price forecasts, tau*/pocket
structure, CPE probability), same 12-instrument universe, same pre-2022
training / 2022+ holdout split, same real point-estimate reporting (no
p-values, t-stats, or significance verdicts).

Algorithm: PPO (Schulman et al. 2017) with a Gaussian continuous-action
policy (position size, tanh-squashed to [-MAX_LEV, MAX_LEV]) and a
learned value function baseline, GAE(lambda) advantages. Each of the 12
instruments is one sequential episode per epoch (full walk through its
own training-period date range, in order, with real running position and
running unrealized P&L as state) -- true sequential rollouts, not
independent day-samples.

Run (from notebooks/predictor_v1/):
    python 85_ppo_decision_layer.py
Output: 85_ppo_results.json, 85_ppo_training_curve.png, 85_ppo_equity.png
"""
import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
_ga82 = import_module("82_ga_decision_layer_search")

INSTRUMENTS = _ga82.INSTRUMENTS
HOLDOUT_START = _ga82.HOLDOUT_START
TRAIN_START = _ga82.TRAIN_START
COST_BPS = _ga82.COST_BPS
MAX_LEV = _ga82.MAX_LEV
REPO_DIR = _ga82.REPO_DIR
OUT_DIR = _ga82.OUT_DIR
sharpe = _ga82.sharpe

DEVICE = torch.device("cpu")  # batch size is only K=12 instruments per step -- MPS dispatch overhead dominates at this scale
SEED = 7
torch.manual_seed(SEED)
np.random.seed(SEED)

# Base signal features (same as GA) + genuinely sequential ones GA couldn't use
BASE_COLS = ["forecast_mu_z", "forecast_sigma_z", "pocket_flag", "pocket_strength", "cpe_firing", "cpe_probability"]
N_SEQ_FEATURES = 2  # prev_position, unrealized_pnl -- filled in during rollout, not precomputed
N_FEATURES = len(BASE_COLS) + N_SEQ_FEATURES
HIDDEN = 32

N_EPOCHS = 3000          # each epoch = one full sequential rollout over all 12 instruments + several PPO update passes
PPO_UPDATE_EPOCHS = 4   # minibatch passes per collected rollout batch
CLIP_EPS = 0.2
GAMMA = 0.999            # per-day discount, ~63-day effective horizon
GAE_LAMBDA = 0.95
LR = 3e-4
ENTROPY_COEF = 0.08
VALUE_COEF = 0.5
LOG_STD_INIT = -0.2


class ActorCritic(nn.Module):
    def __init__(self, n_features, hidden):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(n_features, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.mean_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.ones(1) * LOG_STD_INIT)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        mean = torch.tanh(self.mean_head(h))
        value = self.value_head(h).squeeze(-1)
        return mean, self.log_std.expand_as(mean), value


def collect_rollout(model, per_instrument_data, device):
    """All 12 instruments stepped IN LOCKSTEP, one batched forward pass
    per day rather than one tiny forward pass per (instrument, day) pair
    -- the recurrence that actually matters is within an instrument's own
    day-to-day sequence, not across instruments, so the 12-instrument
    dimension is embarrassingly parallel and belongs in the batch
    dimension of a single tensor op per step. Shorter episodes are masked
    out (via an active flag) once they run out of days, rather than
    requiring every instrument to have identical length. This is the same
    class of fix as the GA scripts' population-batching, applied to the
    instrument dimension instead of the population dimension."""
    names = list(per_instrument_data.keys())
    K = len(names)
    lengths = [len(per_instrument_data[t][1]) for t in names]
    T = max(lengths)
    n_base = per_instrument_data[names[0]][0].shape[1]

    state_base = np.zeros((K, T, n_base), dtype=np.float32)
    fwd_ret = np.zeros((K, T), dtype=np.float32)
    active = np.zeros((K, T), dtype=np.float32)
    for k, t in enumerate(names):
        sb, fr = per_instrument_data[t]
        n = len(fr)
        state_base[k, :n] = sb
        fwd_ret[k, :n] = fr
        active[k, :n] = 1.0

    prev_pos = np.zeros(K, dtype=np.float32)
    unreal_pnl = np.zeros(K, dtype=np.float32)

    all_states, all_actions, all_logprobs, all_rewards, all_values, all_dones, all_active = [], [], [], [], [], [], []
    positions_out = np.zeros((K, T), dtype=np.float32)

    for i in range(T):
        feat = np.concatenate([state_base[:, i, :], prev_pos[:, None], np.clip(unreal_pnl, -1, 1)[:, None]], axis=1).astype(np.float32)
        x = torch.from_numpy(feat).to(device)
        with torch.no_grad():
            mean, log_std, value = model(x)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            raw_action = dist.sample()
            logprob = dist.log_prob(raw_action).sum(-1)
        action = (torch.tanh(raw_action).squeeze(-1) * MAX_LEV).cpu().numpy()
        r = fwd_ret[:, i]
        turnover_cost = np.abs(action - prev_pos) * (COST_BPS / 10000.0)
        # Reward is EXCESS return over passive 1x-long holding, not raw P&L:
        # raw P&L with an uncapped-below leverage action (MAX_LEV=2) trivially
        # rewards levering toward the cap in a mostly-rising training window,
        # confirmed directly (strategy Sharpe matched buy-hold Sharpe almost
        # exactly, ratio 0.94-1.0, on a first run -- the exact leverage-gaming
        # signature already caught and fixed in the GA scripts). (action-1)*r
        # is zero reward for simply holding 1x long, so the agent is only
        # rewarded for genuinely deviating from passive exposure in a way
        # that adds real excess return -- removes the beta-capture shortcut.
        reward = (action - 1.0) * r - turnover_cost

        all_states.append(feat)
        all_actions.append(raw_action.squeeze(-1).cpu().numpy())
        all_logprobs.append(logprob.cpu().numpy())
        all_rewards.append(reward)
        all_values.append(value.cpu().numpy())
        is_last = np.array([1.0 if i == lengths[k] - 1 else 0.0 for k in range(K)], dtype=np.float32)
        all_dones.append(is_last)
        all_active.append(active[:, i].copy())
        positions_out[:, i] = action

        unreal_pnl = unreal_pnl * 0.98 + reward
        prev_pos = action

    per_instrument_positions = {t: positions_out[k, :lengths[k]] for k, t in enumerate(names)}

    # Keep as (T, K, ...) -- time-major, K instruments in the batch dimension --
    # so GAE can be computed per-instrument-row (vectorized across K, looped
    # over T in reverse), THEN flattened+masked for the PPO update. Flattening
    # before GAE would interleave independent episodes in time order and
    # silently corrupt the backward recursion across episode boundaries.
    states_arr = np.stack(all_states, axis=0)      # (T, K, F)
    actions_arr = np.stack(all_actions, axis=0)     # (T, K)
    logprobs_arr = np.stack(all_logprobs, axis=0)   # (T, K)
    rewards_arr = np.stack(all_rewards, axis=0)     # (T, K)
    values_arr = np.stack(all_values, axis=0)       # (T, K)
    dones_arr = np.stack(all_dones, axis=0)         # (T, K)
    active_arr = np.stack(all_active, axis=0)       # (T, K)

    return states_arr, actions_arr, logprobs_arr, rewards_arr, values_arr, dones_arr, active_arr, per_instrument_positions


def compute_gae(rewards, values, dones, gamma, lam):
    """rewards/values/dones: (T, K) -- vectorized across the K instruments
    (independent episodes), looped over T in reverse. Each column k is one
    instrument's own episode; done[t,k]=1 correctly zeroes the bootstrap
    and resets the running GAE accumulator for that column only."""
    T, K = rewards.shape
    advantages = np.zeros((T, K), dtype=np.float32)
    last_gae = np.zeros(K, dtype=np.float32)
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else np.zeros(K, dtype=np.float32)
        next_value = next_value * (1.0 - dones[t])
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def ppo_update(model, optimizer, states, actions, old_logprobs, advantages, returns, device):
    states_t = torch.from_numpy(states).to(device)
    actions_t = torch.from_numpy(actions).to(device).unsqueeze(-1)
    old_logprobs_t = torch.from_numpy(old_logprobs).to(device)
    advantages_t = torch.from_numpy((advantages - advantages.mean()) / (advantages.std() + 1e-8)).to(device)
    returns_t = torch.from_numpy(returns).to(device)

    for _ in range(PPO_UPDATE_EPOCHS):
        mean, log_std, values = model(states_t)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        new_logprobs = dist.log_prob(actions_t).sum(-1)
        entropy = dist.entropy().sum(-1).mean()

        ratio = torch.exp(new_logprobs - old_logprobs_t)
        surr1 = ratio * advantages_t
        surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages_t
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = ((values - returns_t) ** 2).mean()
        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()


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

    train_full = full.loc[~full["is_holdout"].values]
    holdout_full = full.loc[full["is_holdout"].values]

    train_by_instrument = {}
    for t in INSTRUMENTS:
        sub = train_full[train_full["instrument"] == t]
        if len(sub) < 100:
            continue
        train_by_instrument[t] = (sub[BASE_COLS].values.astype(np.float32), sub["fwd_ret"].values.astype(np.float32))
    print(f"Training instruments: {list(train_by_instrument.keys())}, "
          f"episode lengths: {[len(v[1]) for v in train_by_instrument.values()]}")

    model = ActorCritic(N_FEATURES, HIDDEN).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    reward_history = []
    print(f"\nTraining PPO: {N_EPOCHS} epochs, {len(train_by_instrument)} sequential episodes/epoch on device={DEVICE}...")
    for epoch in range(N_EPOCHS):
        states, actions, logprobs, rewards, values, dones, active, _ = collect_rollout(model, train_by_instrument, DEVICE)
        advantages, returns = compute_gae(rewards, values, dones, GAMMA, GAE_LAMBDA)
        mask = active > 0.5
        T, K, F = states.shape
        states_flat = states.reshape(T * K, F)[mask.reshape(T * K)]
        actions_flat = actions.reshape(T * K)[mask.reshape(T * K)]
        logprobs_flat = logprobs.reshape(T * K)[mask.reshape(T * K)]
        advantages_flat = advantages.reshape(T * K)[mask.reshape(T * K)]
        returns_flat = returns.reshape(T * K)[mask.reshape(T * K)]
        ppo_update(model, optimizer, states_flat, actions_flat, logprobs_flat, advantages_flat, returns_flat, DEVICE)
        mean_daily_reward = float(rewards[mask].mean()) * 252 * 100  # annualized %, real point estimate
        reward_history.append(mean_daily_reward)
        if epoch % 25 == 0 or epoch == N_EPOCHS - 1:
            print(f"  epoch {epoch:>4}  mean reward (annualized P&L %/yr)={mean_daily_reward:+7.2f}%  "
                  f"elapsed={time.time()-t0:.1f}s")

    print(f"\nTraining done, {time.time()-t0:.1f}s elapsed. Evaluating once on 2022+ holdout...")

    holdout_by_instrument = {}
    for t in INSTRUMENTS:
        sub = holdout_full[holdout_full["instrument"] == t]
        if len(sub) < 30:
            continue
        holdout_by_instrument[t] = (sub[BASE_COLS].values.astype(np.float32), sub["fwd_ret"].values.astype(np.float32))

    model.eval()
    results = {}
    all_strat, all_bh = [], []
    with torch.no_grad():
        for t, (state_base, fwd_ret) in holdout_by_instrument.items():
            n = len(fwd_ret)
            prev_pos, unreal_pnl = 0.0, 0.0
            strat_rets, positions_log = [], []
            for i in range(n):
                feat = np.concatenate([state_base[i], [prev_pos, np.clip(unreal_pnl, -1, 1)]]).astype(np.float32)
                x = torch.from_numpy(feat).unsqueeze(0).to(DEVICE)
                mean, log_std, _ = model(x)
                action = torch.tanh(mean).item() * MAX_LEV  # deterministic (mean) action at eval time
                r = fwd_ret[i]
                turnover_cost = abs(action - prev_pos) * (COST_BPS / 10000.0)
                strat_ret = action * r - turnover_cost  # REAL P&L, for reporting -- not reward-shaped
                training_reward = (action - 1.0) * r - turnover_cost  # matches the training reward exactly, for the state feature only
                strat_rets.append(strat_ret)
                positions_log.append(action)
                unreal_pnl = unreal_pnl * 0.98 + training_reward
                prev_pos = action
            strat_rets = np.array(strat_rets)
            positions_log = np.array(positions_log)
            alpha = float((strat_rets.mean() - fwd_ret.mean()) * 252 * 100)
            results[t] = {"alpha_vs_own_pct": alpha, "strategy_sharpe": sharpe(strat_rets), "buy_hold_sharpe": sharpe(fwd_ret),
                          "mean_position": float(positions_log.mean()), "std_position": float(positions_log.std())}
            all_strat.append(strat_rets); all_bh.append(fwd_ret)
            print(f"  {t:<10} alpha={alpha:+7.2f}%/yr  strat_sharpe={sharpe(strat_rets):+.3f}  bh_sharpe={sharpe(fwd_ret):+.3f}  "
                  f"mean_pos={positions_log.mean():+.3f}  std_pos={positions_log.std():.3f}")

    all_strat, all_bh = np.concatenate(all_strat), np.concatenate(all_bh)
    n_pos = sum(1 for r in results.values() if r["alpha_vs_own_pct"] > 0)
    overall_alpha = float((all_strat.mean() - all_bh.mean()) * 252 * 100)
    overall_sharpe = sharpe(all_strat)
    print(f"\nPositive on {n_pos}/{len(results)}. Pooled alpha={overall_alpha:+.2f}%/yr, strategy Sharpe={overall_sharpe:+.3f}, "
          f"pooled buy-hold Sharpe={sharpe(all_bh):+.3f}")
    print(f"vs GA pooled-fitness (82): 1/12, -10.86%/yr, Sharpe +0.178")
    print(f"vs GA regime-diverse (83): 1/12, -12.90%/yr, Sharpe +0.038")
    print(f"vs GA GFC-inclusive (84):  2/12, -9.73%/yr, Sharpe +0.213")

    with open(os.path.join(OUT_DIR, "85_ppo_results.json"), "w") as f:
        json.dump({"per_instrument": results, "n_positive": n_pos, "n_total": len(results),
                    "overall_alpha_pct": overall_alpha, "overall_strategy_sharpe": overall_sharpe,
                    "overall_buy_hold_sharpe": sharpe(all_bh), "n_epochs": N_EPOCHS}, f, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(reward_history, color="#2f8a4e", lw=1.2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean training reward (annualized P&L, %/yr)")
    ax.set_title(f"PPO actor-critic training curve, {N_EPOCHS} epochs, {len(train_by_instrument)} sequential episodes/epoch\n"
                 f"Real temporal credit assignment (GAE), path-dependent state (position, unrealized P&L)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "85_ppo_training_curve.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    ts = sorted(results, key=lambda t: -results[t]["alpha_vs_own_pct"])
    colors = ["#2f8a4e" if results[t]["alpha_vs_own_pct"] > 0 else "#B0492F" for t in ts]
    ax.bar(range(len(ts)), [results[t]["alpha_vs_own_pct"] for t in ts], color=colors)
    ax.axhline(0, color="black", lw=0.8); ax.set_xticks(range(len(ts))); ax.set_xticklabels(ts)
    ax.set_ylabel("PPO policy alpha vs. own buy-hold, 2022+ holdout (%/yr)")
    ax.set_title(f"PPO actor-critic (real RL, path-dependent state), OOS 2022+\nPositive {n_pos}/{len(results)}, pooled {overall_alpha:+.2f}%/yr")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "85_ppo_equity.png"), dpi=140)
    plt.close(fig)
    print("\nSaved: 85_ppo_results.json, 85_ppo_training_curve.png, 85_ppo_equity.png")
