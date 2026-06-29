"""
Paper 7 — Fast Permutation Test
================================
Optimised version — runs in ~10-15 minutes.

Key speedups:
1. Pre-compute all CPE numerators/denominators as numpy arrays
2. Only test the pre-specified key signals (not all 5920 configs)
3. Sign-flip test for top 5 signals (most statistically powerful)
4. Within-year shuffle (correct permutation method)
5. N_PERMS = 500 (sufficient for p-value precision to 2 decimal places)

Run on your LOCAL machine.
"""

import pandas as pd
import numpy as np
import warnings, os, time
warnings.filterwarnings('ignore')

TRAIN_END  = '2024-12-31'
MIN_N_COND = 30
N_PERMS    = 500   # sufficient for p-value ± 0.02

# ── PRE-SPECIFIED KEY SIGNALS to test ────────────────────────────────
# (pred_col, ticker, horizon_days, q_target, expected_lift)
# These are the economically motivated signals from the paper
KEY_SIGNALS = [
    ('EU_Urban_Energy_heatstress_30C',        'UNG',  21,  0.65, '1.95×'),
    ('EU_Urban_Energy_heatstress_30C',        'UNG',  21,  0.60, '1.87×'),
    ('Ukraine_Russia_Wheat_heatstress_30C',   'GC=F', 126, 0.65, '1.72×'),
    ('EU_Urban_Energy_heatstress_30C',        'NG=F', 126, 0.65, '1.66×'),
    ('EU_Urban_Energy_heatstress_30C',        'DBB',  126, 0.55, '1.62×'),
    ('US_Great_Plains_Wheat_heatstress_32C',  'WEAT',  63, 0.60, '1.60×'),
    ('US_Corn_Belt_seasonal_q90',             'WEAT',  63, 0.60, '1.50×'),
    ('EU_Urban_Energy_GDD30_q90',             'DBB',  126, 0.50, '1.54×'),
    ('US_Corn_Belt_GDD30_q90',               'WEAT',  63, 0.55, '1.49×'),
    ('Ukraine_Russia_Wheat_heatstress_30C',  'GC=F',  63, 0.55, '1.43×'),
]

# ── LOAD ──────────────────────────────────────────────────────────────
def load_data():
    print("Loading...")
    returns = pd.read_parquet('multiasset_returns.parquet')
    temp    = pd.read_parquet('data/paper7_agri_exceedances_aligned.parquet')

    all_h       = sorted(set(s[2] for s in KEY_SIGNALS))
    all_tickers = list(set(s[1] for s in KEY_SIGNALS))

    cum_fwd = {}
    for ticker in all_tickers:
        if ticker not in returns.columns: continue
        dc = returns[ticker].dropna()
        cum_fwd[ticker] = {}
        for h in all_h:
            td   = max(1, int(round(h * 252 / 365)))
            roll = dc.rolling(window=td, min_periods=td).sum().shift(-td)
            cum_fwd[ticker][h] = roll.reindex(returns[ticker].index)
    # Align to common trading-day index (returns=18238, temp=18247)
    common_idx = returns.index.intersection(temp.index)
    temp = temp.loc[common_idx]
    for ticker in cum_fwd:
        for h in cum_fwd[ticker]:
            cum_fwd[ticker][h] = cum_fwd[ticker][h].loc[common_idx]
    print(f"  Returns: {returns.shape}, Temp: {temp.shape}")
    print(f"  Common index: {len(common_idx)} dates")
    return temp, cum_fwd

# ── FAST CPE for one signal ────────────────────────────────────────────
def compute_cpe_fast(temp_vals, fwd_vals, q):
    """Pure numpy — no DataFrame overhead."""
    valid = ~(np.isnan(temp_vals) | np.isnan(fwd_vals))
    tv    = temp_vals[valid]
    fv    = fwd_vals[valid]
    if len(fv) < MIN_N_COND:
        return None, None, None, None
    thr  = np.nanquantile(fv, q)
    up   = np.mean(fv > thr)
    if up <= 0:
        return None, None, None, None
    cond_mask = (tv == 1)
    n_cond    = cond_mask.sum()
    if n_cond < MIN_N_COND:
        return None, None, None, None
    cpe  = np.mean(fv[cond_mask] > thr)
    lift = cpe / up
    return cpe, lift, n_cond, thr

# ── WITHIN-YEAR SHUFFLE ───────────────────────────────────────────────
def within_year_shuffle(temp_vals, dates, rng):
    """Shuffle temp flags within each calendar year."""
    shuffled = temp_vals.copy()
    years    = np.unique(dates.year)
    for y in years:
        mask = dates.year == y
        idx  = np.where(mask)[0]
        perm = rng.permutation(len(idx))
        shuffled[idx] = temp_vals[idx[perm]]
    return shuffled

# ── SIGN-FLIP TEST for one signal ────────────────────────────────────
def sign_flip_test(temp_vals, fwd_vals, q, n_perms=10000, rng=None):
    """
    Under null: the conditioning events have no special return distribution.
    Randomly sample n_cond returns from the full training set
    and measure how often CPE >= observed.
    """
    if rng is None: rng = np.random.default_rng(42)
    valid = ~(np.isnan(temp_vals) | np.isnan(fwd_vals))
    tv, fv = temp_vals[valid], fwd_vals[valid]
    if len(fv) < MIN_N_COND: return np.nan, np.nan, np.nan, 0

    thr    = np.nanquantile(fv, q)
    up     = np.mean(fv > thr)
    if up <= 0: return np.nan, np.nan, np.nan, 0

    cond   = fv[tv == 1]
    n_cond = len(cond)
    if n_cond < MIN_N_COND: return np.nan, np.nan, np.nan, 0

    obs_cpe = np.mean(cond > thr)
    # Null: draw n_cond samples from all training returns
    perm_cpes = np.array([
        np.mean(rng.choice(fv, size=n_cond, replace=False) > thr)
        for _ in range(n_perms)
    ])
    p_val = np.mean(perm_cpes >= obs_cpe)
    return p_val, obs_cpe, up, n_cond

# ── MAIN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("PAPER 7 — FAST PERMUTATION TEST")
    print("Sign-flip test on 10 pre-specified key signals")
    print("Within-year permutation on aggregate signal count (N=500)")
    print("=" * 65)
    t0  = time.time()
    rng = np.random.default_rng(42)

    temp, cum_fwd = load_data()
    dates = temp.index

    # ── PART 1: Sign-flip test for each key signal ────────────────────
    print("\n── PART 1: Individual sign-flip tests (N=10,000 each) ──")
    print(f"{'Signal':<45} {'Lift':>6} {'CPE':>6} {'Uncond':>7} {'N':>5} {'p-val':>7} {'Sig':>5}")
    print("-" * 80)

    results = []
    for pred_col, ticker, h, q, expected_lift in KEY_SIGNALS:
        if pred_col not in temp.columns or ticker not in cum_fwd:
            print(f"  MISSING: {pred_col}")
            continue
        if h not in cum_fwd[ticker]:
            continue

        # Training period only for sign-flip test
        train_mask = dates <= TRAIN_END
        tv = temp[pred_col].values[train_mask]
        fv = cum_fwd[ticker][h].values[train_mask]

        p_val, cpe, uncond, n_cond = sign_flip_test(tv, fv, q, n_perms=10000, rng=rng)

        if np.isnan(p_val):
            sig = '—'
            lift = np.nan
        else:
            lift = cpe / uncond if uncond > 0 else 0
            sig  = ('***' if p_val < 0.001 else
                    '**'  if p_val < 0.01  else
                    '*'   if p_val < 0.05  else
                    '.'   if p_val < 0.10  else 'n.s.')

        label = f"{pred_col[-28:]:28} →{ticker} h={h:3}d q={q:.0%}"
        if np.isnan(p_val):
            print(f"  {label:<45} {'n/a':>6} {'n/a':>6} {'n/a':>7} {'n/a':>5} {'n/a':>7} {'—':>5}")
        else:
            print(f"  {label:<45} {lift:>6.2f}× {cpe:>6.3f} {uncond:>7.3f} {n_cond:>5} {p_val:>7.4f} {sig:>5}")

        results.append({
            'signal': label, 'expected_lift': expected_lift,
            'cpe': cpe, 'lift': lift, 'uncond': uncond,
            'n_cond': n_cond, 'p_val': p_val, 'sig': sig,
            'ticker': ticker, 'horizon': h, 'q_target': q,
        })

    print(f"\nPart 1 done in {time.time()-t0:.0f}s")

    # ── PART 2: Within-year permutation on aggregate count ─────────────
    print(f"\n── PART 2: Within-year permutation, N={N_PERMS} ──")
    print("(counting signals with CPE≥0.60, lift≥1.30× across all 10 key signals)")
    print("Within-year shuffle preserves seasonality, destroys day-level alignment")

    MIN_CPE_STRICT  = 0.60
    MIN_LIFT_STRICT = 1.30

    def count_key_signals(temp, cum_fwd, dates, tv_override=None):
        n = 0
        for pred_col, ticker, h, q, _ in KEY_SIGNALS:
            if pred_col not in temp.columns or ticker not in cum_fwd: continue
            if h not in cum_fwd[ticker]: continue
            train_mask = dates <= TRAIN_END
            tv = (tv_override[pred_col] if tv_override is not None
                  else temp[pred_col].values)
            fv = cum_fwd[ticker][h].values
            # restrict to training
            tv_t = tv[train_mask]; fv_t = fv[train_mask]
            cpe, lift, nc, _ = compute_cpe_fast(tv_t, fv_t, q)
            if cpe is not None and cpe >= MIN_CPE_STRICT and lift >= MIN_LIFT_STRICT:
                n += 1
        return n

    n_empirical = count_key_signals(temp, cum_fwd, dates)
    print(f"  Empirical (key signals, strict filters): {n_empirical}")

    perm_counts = []
    for i in range(N_PERMS):
        # Build shuffled temp dict
        tv_shuf = {}
        for pred_col in temp.columns:
            tv_shuf[pred_col] = within_year_shuffle(
                temp[pred_col].values, dates, rng)
        n_perm = count_key_signals(temp, cum_fwd, dates, tv_override=tv_shuf)
        perm_counts.append(n_perm)
        if (i+1) % 100 == 0:
            elapsed = time.time()-t0
            eta     = elapsed/(i+1)*(N_PERMS-i-1)
            print(f"  [{i+1:3d}/{N_PERMS}] median={np.median(perm_counts):.0f} "
                  f"max={max(perm_counts)} ETA={eta/60:.1f}min")

    perm_counts = np.array(perm_counts)
    p_agg   = np.mean(perm_counts >= n_empirical)
    med_p   = np.median(perm_counts)
    p95     = np.percentile(perm_counts, 95)
    ratio   = n_empirical / med_p if med_p > 0 else float('inf')

    print(f"\n  Empirical signals  : {n_empirical}")
    print(f"  Perm median        : {med_p:.1f}")
    print(f"  Perm 95th pct      : {p95:.1f}")
    print(f"  Ratio              : {ratio:.2f}×")
    print(f"  Aggregate p-value  : {p_agg:.4f}")

    # ── SUMMARY ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)

    sig_results = [r for r in results if not np.isnan(r.get('p_val', np.nan))]
    n_sig   = sum(1 for r in sig_results if r['p_val'] < 0.05)
    n_total = len(sig_results)

    print(f"\nSign-flip tests: {n_sig}/{n_total} key signals significant at p<0.05")
    for r in sig_results:
        if r['p_val'] < 0.05:
            print(f"  ✓ {r['signal']}: lift={r['lift']:.2f}×, p={r['p_val']:.4f} {r['sig']}")
        else:
            print(f"  ✗ {r['signal']}: lift={r['lift']:.2f}×, p={r['p_val']:.4f}")

    print(f"\nWithin-year permutation: p={p_agg:.4f} "
          f"({'SIGNIFICANT' if p_agg<0.05 else 'NOT SIGNIFICANT'})")
    print(f"  Empirical {n_empirical} signals vs perm median {med_p:.0f} ({ratio:.2f}× ratio)")

    # Save
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(results).to_csv('results/paper7_perm_fast_signals.csv', index=False)
    pd.DataFrame({'perm_count': perm_counts}).to_csv(
        'results/paper7_perm_fast_counts.csv', index=False)
    pd.Series({
        'n_empirical': n_empirical, 'perm_median': med_p,
        'perm_p95': p95, 'ratio': ratio, 'p_agg': p_agg,
        'n_sig_signals': n_sig, 'n_total_signals': n_total,
    }).to_csv('results/paper7_perm_fast_summary.csv')

    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} minutes")
    print("Saved: results/paper7_perm_fast_*.csv")
