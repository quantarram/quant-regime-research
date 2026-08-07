"""
Paper 16 candidate: five levers have now failed to buy PREDICTION skill past
the measured ceiling (Ramanathan 2026d, 2026e). This script asks a different
question at the DECISION layer instead: given we already know each
instrument's own predictability limit tau* (2026a) and that climatology's
own raw return distribution is a strong, honest uncertainty estimate
(2026e), can a policy that is explicitly given that information make
better sizing/timing decisions than Paper 12's original strategies did --
which is the real gap in Paper 12, not "RL might find alpha where five
other strategies didn't." Paper 12's Kelly-sized strategy (47_) already
sizes positions by conviction (mu/sigma^2, the Kelly fraction) -- but mu
and sigma there come from the ORIGINAL forecasting model's fixed training
window, never re-estimated on the honest, non-stale half_window =
floor(tau*/2) budget Papers 13-15 established, and never conditioned on
whether the information behind it is fresh or stale relative to tau*.
This script gives a policy-gradient RL agent exactly that: the same
mu/sigma conviction signal Kelly uses, but estimated only from an honest,
tau*-respecting window, plus a same fresh-vs-stale contrast as Papers
13/14 used for prediction -- to test whether DECISION quality collapses
at the same wall PREDICTION quality does, and whether tau*-awareness
itself (not RL sophistication) is what any resulting edge comes from.

Design
------
- 12-instrument panel, tau* from Ramanathan (2026a), horizon and winning
  regime feature reused from Paper 12's master_model_final_decision.json.
- At each half_window = floor(tau*/2) block boundary, the policy observes
  climatology's own mu (mean) and sigma (std) of the horizon-day-ahead
  forward log return, estimated from only the most recent half_window
  RESOLVED observations (target date strictly before the decision date --
  no lookahead), plus the previous block's position.
- FRESH: that training window is the immediately preceding half_window of
  resolved observations. STALE (the decision-ceiling control): identical
  in every other respect, but the training window is instead the most
  recent half_window of observations resolving >= 2*tau* before the
  decision date -- the same staleness definition used in
  64_good_vs_stale_test.py for the prediction-layer version of this test.
- Policy: linear-Gaussian, action = tanh(w.state + b) in [-1, 1] (long or
  short, unlike Kelly's long-only design), trained via REINFORCE with a
  running baseline (same style as RLPolicyForecaster in
  65_architecture_bakeoff.py). Reward = the block's realized, cost-
  adjusted net return (COST_BPS charged once per block on the position
  change).
- Trained ONLINE, block by block, walking forward through pre-holdout
  history only -- block t's reward is only known, and only used to
  update the policy, after block t has actually played out, before block
  t+1's decision is made. Frozen at HOLDOUT_START (2022-01-01, the same
  cutoff as 44/45/46/47), then evaluated purely out-of-sample with the
  policy's mean action (no exploration noise), exactly like a deployed
  model.
- Benchmarks: the same instrument's own buy-and-hold, scored via the same
  Jensen-alpha regression used in 44_alpha_test_own_benchmark.py, and
  Paper 12's existing Kelly-sized strategy result (kelly_strategy_results
  .json) as the tau*-BLIND baseline -- same underlying conviction-sizing
  idea, but never re-estimated on an honest window or conditioned on
  freshness.

No significance/randomization-test games, per this program's standing
rule -- one real, walk-forward, cost-adjusted OOS evaluation, reported
honestly whichever way it comes out.

Run: python 81_predictability_aware_rl_sizing.py
Output: 81_rl_sizing_results.json, 81_rl_sizing_alpha_vs_own.png
"""
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTABILITY_JSON = os.path.join(NB_DIR, "predictability_paper", "results_correlated_decorrelated.json")
DECISIONS_JSON = os.path.join(OUT_DIR, "master_model_final_decision.json")
KELLY_JSON = os.path.join(OUT_DIR, "kelly_strategy_results.json")

INSTRUMENTS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "AAPL", "MSFT", "JPM", "XOM", "GLD", "EURUSD=X"]
HOLDOUT_START = pd.Timestamp("2022-01-01")
COST_BPS = 5
STALE_MULT = 2  # staleness threshold, same convention as 64_good_vs_stale_test.py: >= 2*tau*


def load_meta():
    tau = json.load(open(PREDICTABILITY_JSON))
    dec = json.load(open(DECISIONS_JSON))
    meta = {}
    for t in INSTRUMENTS:
        tau_star = tau[t]["2"]["top5_tradeable"][0][0]
        horizon = dec[t]["horizon"]
        meta[t] = {"tau_star": tau_star, "half_window": max(tau_star // 2, 5), "horizon": horizon}
    return meta


def fwd_ret_series(price, horizon):
    """log(price[t+horizon]/price[t]), indexed by the RESOLUTION date t+horizon
    (the date this observation actually becomes known), not the base date."""
    p = price.dropna()
    vals = p.values
    n = len(vals)
    if n <= horizon:
        return pd.Series(dtype=float)
    ret = np.log(vals[horizon:] / vals[:-horizon])
    resolve_dates = p.index[horizon:]
    return pd.Series(ret, index=resolve_dates)


class SizingPolicy:
    """Linear-Gaussian policy over [mu/sigma (conviction), sigma (scaled),
    prev_position]; action = tanh(w.state+b) in [-1,1]. REINFORCE with a
    running baseline, updated one block at a time as blocks are observed."""

    def __init__(self, lr=0.08, sigma0=0.35, sigma_min=0.05, anneal_blocks=120, seed=0):
        self.lr, self.sigma0, self.sigma_min, self.anneal_blocks = lr, sigma0, sigma_min, anneal_blocks
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros(3)
        self.b = 0.0
        self.baseline = 0.0
        self.n_updates = 0

    def _explore_sigma(self):
        frac = min(self.n_updates / self.anneal_blocks, 1.0)
        return max(self.sigma0 * (1 - frac), self.sigma_min)

    def act(self, state, explore=True):
        raw_mean = float(self.w @ state + self.b)
        if not explore:
            return float(np.tanh(raw_mean)), raw_mean, 0.0
        eps_sigma = self._explore_sigma()
        noise = self.rng.normal(0.0, eps_sigma)
        raw = raw_mean + noise
        return float(np.tanh(raw)), raw_mean, noise

    def update(self, state, noise, reward):
        self.baseline = 0.9 * self.baseline + 0.1 * reward
        adv = np.clip(reward - self.baseline, -5.0, 5.0)
        self.w = self.w + self.lr * adv * noise * state
        self.b = self.b + self.lr * adv * noise
        self.n_updates += 1


def run_instrument(tkr, meta, prices, kelly_results):
    tau_star, half_window, horizon = meta["tau_star"], meta["half_window"], meta["horizon"]
    price = prices[tkr].dropna()
    daily_log_ret = np.log(price / price.shift(1)).dropna()
    fwd = fwd_ret_series(price, horizon)
    if len(fwd) < 4 * half_window:
        return None

    all_dates = daily_log_ret.index
    stale_gap = pd.Timedelta(days=int(STALE_MULT * tau_star * 1.5))  # trading->calendar day buffer
    daily_vol = daily_log_ret.std()

    results = {}
    for condition in ["fresh", "stale"]:
        # Precompute per-block (mu, sigma, realized block return) once -- these
        # depend only on price/predictability data, never on the policy, so
        # they are identical across every training epoch below.
        blocks = []
        t0_positions = list(range(half_window * 2, len(all_dates), half_window))
        for pos_idx in t0_positions:
            t0 = all_dates[pos_idx]
            block_end_idx = min(pos_idx + half_window, len(all_dates))
            block_dates = all_dates[pos_idx:block_end_idx]
            if len(block_dates) == 0:
                continue
            resolved = fwd[fwd.index < t0]
            if condition == "fresh":
                train = resolved.tail(half_window)
            else:
                cutoff = t0 - stale_gap
                train = resolved[resolved.index <= cutoff].tail(half_window)
            if len(train) < half_window:
                continue
            mu, sigma = float(train.mean()), float(train.std())
            sigma = max(sigma, 1e-4)
            conviction = float(np.clip(mu / sigma, -5, 5))
            sigma_scaled = float(np.clip(sigma / (daily_vol * np.sqrt(horizon)), 0, 5))
            block_ret_sum = float(daily_log_ret.reindex(block_dates).fillna(0.0).sum())
            blocks.append({
                "t0": t0, "mu": mu, "sigma": sigma, "conviction": conviction,
                "sigma_scaled": sigma_scaled, "block_ret_sum": block_ret_sum,
                "is_holdout": bool(t0 >= HOLDOUT_START),
            })

        pre_holdout = [b for b in blocks if not b["is_holdout"]]
        holdout = [b for b in blocks if b["is_holdout"]]

        # Many passes over the fixed pre-holdout block sequence -- each pass
        # replays the same historical environment with the policy's current
        # (evolving) parameters, giving REINFORCE enough gradient steps to
        # actually learn (one pass alone gives only as many updates as there
        # are pre-holdout blocks, 35-190 here -- far too few). No holdout data
        # is touched at any point in this loop.
        policy = SizingPolicy(anneal_blocks=max(len(pre_holdout) * 15, 200))
        n_epochs = 120
        for _epoch in range(n_epochs):
            prev_pos = 0.0
            for b in pre_holdout:
                state = np.array([b["conviction"], b["sigma_scaled"], prev_pos])
                action, raw_mean, noise = policy.act(state, explore=True)
                turnover_cost = abs(action - prev_pos) * (COST_BPS / 10000.0)
                net = action * b["block_ret_sum"] - turnover_cost
                policy.update(state, noise, net)
                prev_pos = action

        # One final deterministic (frozen, no-exploration) pass over the
        # pre-holdout blocks to establish the position the policy would
        # actually be holding the moment the holdout period begins, then
        # continue deterministically through the holdout blocks themselves.
        rows = []
        prev_pos = 0.0
        for b in pre_holdout + holdout:
            state = np.array([b["conviction"], b["sigma_scaled"], prev_pos])
            action, raw_mean, noise = policy.act(state, explore=False)
            turnover_cost = abs(action - prev_pos) * (COST_BPS / 10000.0)
            gross = action * b["block_ret_sum"]
            net = gross - turnover_cost
            rows.append({
                "t0": b["t0"], "action": action, "mu": b["mu"], "sigma": b["sigma"],
                "block_gross_ret": gross, "block_net_ret": net,
                "block_bh_ret": b["block_ret_sum"], "is_holdout": b["is_holdout"],
            })
            prev_pos = action

        df = pd.DataFrame(rows)
        results[condition] = df

    out = {}
    for condition, df in results.items():
        oos = df[df["is_holdout"]]
        if len(oos) < 3:
            out[condition] = None
            continue
        strat_ret = oos.set_index("t0")["block_net_ret"]
        own_ret = oos.set_index("t0")["block_bh_ret"]
        X = np.column_stack([np.ones(len(own_ret)), own_ret.values])
        beta_hat, *_ = np.linalg.lstsq(X, strat_ret.values, rcond=None)
        alpha_block, beta = beta_hat
        resid = strat_ret.values - X @ beta_hat
        n, k = len(strat_ret), 2
        sigma2 = (resid @ resid) / max(n - k, 1)
        se_alpha = float(np.sqrt(max(sigma2 * np.linalg.inv(X.T @ X)[0, 0], 0)))
        blocks_per_year = 252.0 / half_window
        out[condition] = {
            "n_blocks_oos": int(n),
            "strategy_cum_ret_pct": float((np.exp(strat_ret.sum()) - 1) * 100),
            "buyhold_cum_ret_pct": float((np.exp(own_ret.sum()) - 1) * 100),
            "alpha_per_block_pct": float(alpha_block * 100),
            "alpha_annualized_pct": float(alpha_block * blocks_per_year * 100),
            "t_alpha": float(alpha_block / se_alpha) if se_alpha > 0 else float("nan"),
            "beta_vs_own": float(beta),
            "significant_95": bool(abs(alpha_block / se_alpha) >= 1.96) if se_alpha > 0 else False,
            "mean_abs_position": float(oos["action"].abs().mean()),
        }
    out["tau_star"] = tau_star
    out["half_window"] = half_window
    out["horizon"] = horizon
    out["kelly_alpha_vs_own_pct"] = kelly_results.get(tkr, {}).get("alpha_vs_own_pct")
    out["kelly_beats_buy_hold_net"] = kelly_results.get(tkr, {}).get("beats_buy_hold_net")
    return out


if __name__ == "__main__":
    meta = load_meta()
    prices = pd.read_parquet(os.path.join(NB_DIR, "multiasset_prices.parquet"))
    kelly_results = json.load(open(KELLY_JSON)) if os.path.exists(KELLY_JSON) else {}

    all_results = {}
    for tkr in INSTRUMENTS:
        r = run_instrument(tkr, meta[tkr], prices, kelly_results)
        if r is None:
            print(f"{tkr}: skipped, insufficient history")
            continue
        all_results[tkr] = r
        fresh, stale = r.get("fresh"), r.get("stale")
        f_a = fresh["alpha_annualized_pct"] if fresh else float("nan")
        f_t = fresh["t_alpha"] if fresh else float("nan")
        s_a = stale["alpha_annualized_pct"] if stale else float("nan")
        kelly_a = r["kelly_alpha_vs_own_pct"]
        print(f"{tkr}: tau*={r['tau_star']} half_window={r['half_window']} horizon={r['horizon']} | "
              f"RL-fresh alpha={f_a:+.2f}%/yr t={f_t:+.2f} | RL-stale alpha={s_a:+.2f}%/yr | "
              f"Kelly(tau*-blind) alpha={kelly_a}")

    with open(os.path.join(OUT_DIR, "81_rl_sizing_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    n_sig = sum(1 for r in all_results.values() if r.get("fresh") and r["fresh"]["significant_95"])
    print(f"\n{n_sig}/{len(all_results)} instruments: RL-fresh significant alpha vs own buy-and-hold at 95%")

    tickers = [t for t in INSTRUMENTS if t in all_results and all_results[t].get("fresh")]
    fresh_alpha = [all_results[t]["fresh"]["alpha_annualized_pct"] for t in tickers]
    stale_alpha = [all_results[t]["stale"]["alpha_annualized_pct"] if all_results[t].get("stale") else np.nan for t in tickers]
    kelly_alpha = [all_results[t]["kelly_alpha_vs_own_pct"] if all_results[t]["kelly_alpha_vs_own_pct"] is not None else np.nan for t in tickers]
    order = np.argsort(fresh_alpha)
    tickers = [tickers[i] for i in order]
    fresh_alpha = [fresh_alpha[i] for i in order]
    stale_alpha = [stale_alpha[i] for i in order]
    kelly_alpha = [kelly_alpha[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(tickers))
    h = 0.25
    ax.barh(y + h, fresh_alpha, height=h, color="#2E6DA4", label="RL, fresh (tau*-aware)")
    ax.barh(y, stale_alpha, height=h, color="#B0492F", label="RL, stale (>=2 tau* old, control)")
    ax.barh(y - h, kelly_alpha, height=h, color="#9AA1AD", label="Kelly, tau*-blind (Paper 12)")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(tickers, fontsize=9)
    ax.set_xlabel("Annualized alpha vs own instrument's buy-and-hold, net of costs (%/yr)")
    ax.set_title("Does knowing tau* help an RL sizing policy beat Paper 12's tau*-blind Kelly strategy?\n"
                  f"OOS from {HOLDOUT_START.date()}, all {len(tickers)} instruments")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "81_rl_sizing_alpha_vs_own.png"), dpi=140)
    plt.close(fig)
    print("Saved: 81_rl_sizing_results.json, 81_rl_sizing_alpha_vs_own.png")
