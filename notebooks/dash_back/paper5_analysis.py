"""
Paper 5: Corrected Inference for the CPE Portfolio Tilt Strategy
Newey-West HAC Standard Errors and Robustness Checks

This script produces all results for Paper 5.
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv('/mnt/user-data/uploads/dashboard_portfolio_replay.csv')
df['signal_date'] = pd.to_datetime(df['signal_date'])
df = df.sort_values('signal_date').reset_index(drop=True)

HORIZONS = ['21d', '63d', '126d']
HORIZON_DAYS = {'21d': 21, '63d': 63, '126d': 126}

print("=" * 70)
print("PAPER 5: CORRECTED INFERENCE FOR CPE PORTFOLIO TILT")
print("=" * 70)

# ─────────────────────────────────────────────
# SECTION 1: NEWEY-WEST HAC CORRECTION
# ─────────────────────────────────────────────
print("\n" + "─" * 70)
print("SECTION 1: NEWEY-WEST HAC CORRECTION")
print("─" * 70)

nw_results = {}

for h in HORIZONS:
    resolved = df[df[f'status_{h}'] == 'RESOLVED'].copy()
    edge = resolved[f'tilt_pnl_{h}'].values - resolved[f'neutral_pnl_{h}'].values
    n = len(edge)
    T_days = HORIZON_DAYS[h]
    T_weeks = T_days / 5  # approximate weeks

    # OLS t-stat (Paper 4 result)
    ols_t, ols_p = stats.ttest_1samp(edge, 0)

    # Newey-West HAC - two lag choices
    # Rule of thumb: floor(4*(T/100)^(2/9))
    rot_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    # Horizon-matched lag: horizon in weeks
    hor_lag = int(np.ceil(T_weeks))

    # Both lag specs
    nw_results[h] = {}
    for lag_name, n_lags in [('Rule-of-thumb', rot_lag), ('Horizon-matched', hor_lag)]:
        # OLS regression with HAC standard errors
        X = sm.add_constant(np.ones(n))
        model = sm.OLS(edge, X)
        res = model.fit()
        hac_cov = cov_hac(res, nlags=n_lags)
        hac_se = np.sqrt(np.diag(hac_cov))[0]  # SE of intercept (mean)
        mean_edge = np.mean(edge)
        hac_t = mean_edge / hac_se
        hac_p = 2 * (1 - stats.t.cdf(abs(hac_t), df=n - 1))

        nw_results[h][lag_name] = {
            'n': n, 'mean': mean_edge, 'std': np.std(edge, ddof=1),
            'n_lags': n_lags, 'hac_se': hac_se,
            'ols_t': ols_t, 'ols_p': ols_p,
            'hac_t': hac_t, 'hac_p': hac_p
        }

    print(f"\nHorizon {h} (n={n}, OLS t={ols_t:.3f}, p={ols_p:.6f}):")
    print(f"  Mean edge: {np.mean(edge):.4f}%  Std: {np.std(edge, ddof=1):.4f}%")
    print(f"  Overlap: ~{int(T_weeks - 1)}/{int(T_weeks)} weeks per observation")
    print(f"  {'Lag spec':<20} {'Lags':>6} {'HAC SE':>10} {'HAC t':>8} {'HAC p':>12} {'Significant':>12}")
    for lag_name, r in nw_results[h].items():
        sig = "YES (p<0.05)" if r['hac_p'] < 0.05 else "NO"
        sig = "YES (p<0.01)" if r['hac_p'] < 0.01 else sig
        sig = "YES (p<0.001)" if r['hac_p'] < 0.001 else sig
        print(f"  {lag_name:<20} {r['n_lags']:>6} {r['hac_se']:>10.4f} {r['hac_t']:>8.3f} {r['hac_p']:>12.6f} {sig:>12}")

# ─────────────────────────────────────────────
# SECTION 2: ROBUSTNESS CHECK 1 - NON-OVERLAPPING
# ─────────────────────────────────────────────
print("\n" + "─" * 70)
print("SECTION 2: ROBUSTNESS CHECK 1 — NON-OVERLAPPING OBSERVATIONS")
print("─" * 70)

for h in HORIZONS:
    resolved = df[df[f'status_{h}'] == 'RESOLVED'].copy().reset_index(drop=True)
    T_weeks = HORIZON_DAYS[h] / 5

    # Select every T_weeks-th observation (non-overlapping)
    step = max(1, int(np.floor(T_weeks)))
    non_overlap = resolved.iloc[::step].copy()
    edge = non_overlap[f'tilt_pnl_{h}'].values - non_overlap[f'neutral_pnl_{h}'].values
    n = len(edge)

    if n >= 3:
        t_stat, p_val = stats.ttest_1samp(edge, 0)
        beat_rate = np.mean(non_overlap[f'tilt_beat_{h}'])
    else:
        t_stat, p_val, beat_rate = np.nan, np.nan, np.nan

    print(f"\nHorizon {h}: every {step} weeks → n={n} non-overlapping obs")
    print(f"  Mean edge: {np.mean(edge):.4f}%  Beat rate: {beat_rate:.1%}")
    print(f"  t={t_stat:.3f}  p={p_val:.4f}  Significant: {'YES' if p_val < 0.05 else 'NO'}")

# ─────────────────────────────────────────────
# SECTION 3: ROBUSTNESS CHECK 2 - BLOCK BOOTSTRAP
# ─────────────────────────────────────────────
print("\n" + "─" * 70)
print("SECTION 3: ROBUSTNESS CHECK 2 — BLOCK BOOTSTRAP")
print("─" * 70)

np.random.seed(42)
N_BOOT = 5000

for h in HORIZONS:
    resolved = df[df[f'status_{h}'] == 'RESOLVED'].copy().reset_index(drop=True)
    edge = (resolved[f'tilt_pnl_{h}'] - resolved[f'neutral_pnl_{h}']).values
    n = len(edge)
    T_weeks = HORIZON_DAYS[h] / 5
    block_size = max(2, int(np.ceil(T_weeks)))  # block = horizon length

    # Circular block bootstrap
    boot_means = []
    for _ in range(N_BOOT):
        n_blocks = int(np.ceil(n / block_size))
        starts = np.random.randint(0, n, size=n_blocks)
        boot_sample = []
        for s in starts:
            block = [edge[(s + i) % n] for i in range(block_size)]
            boot_sample.extend(block)
        boot_means.append(np.mean(boot_sample[:n]))

    boot_means = np.array(boot_means)
    # p-value: fraction of bootstrap means <= 0
    p_boot = np.mean(boot_means <= 0)

    ols_t, ols_p = stats.ttest_1samp(edge, 0)

    print(f"\nHorizon {h} (n={n}, block_size={block_size} weeks):")
    print(f"  OLS p={ols_p:.6f}  Block-bootstrap p={p_boot:.4f}")
    print(f"  Bootstrap 95th CI: [{np.percentile(boot_means, 2.5):.4f}, {np.percentile(boot_means, 97.5):.4f}]")
    print(f"  Significant at p<0.05: {'YES' if p_boot < 0.05 else 'NO'}")

# ─────────────────────────────────────────────
# SECTION 4: ROBUSTNESS CHECK 3 - SUBPERIOD ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "─" * 70)
print("SECTION 4: ROBUSTNESS CHECK 3 — SUBPERIOD ANALYSIS")
print("─" * 70)

for h in HORIZONS:
    resolved = df[df[f'status_{h}'] == 'RESOLVED'].copy().reset_index(drop=True)
    n = len(resolved)
    mid = n // 2

    print(f"\nHorizon {h} (n={n}):")
    for label, subset in [('First half', resolved.iloc[:mid]),
                           ('Second half', resolved.iloc[mid:])]:
        edge = (subset[f'tilt_pnl_{h}'] - subset[f'neutral_pnl_{h}']).values
        n_sub = len(edge)
        if n_sub >= 3:
            t, p = stats.ttest_1samp(edge, 0)
            dates = f"{subset['signal_date'].iloc[0].strftime('%Y-%m-%d')} to {subset['signal_date'].iloc[-1].strftime('%Y-%m-%d')}"
            print(f"  {label} (n={n_sub}, {dates}):")
            print(f"    Mean edge={np.mean(edge):.4f}%  t={t:.3f}  p={p:.4f}  "
                  f"Beat rate={np.mean(subset[f'tilt_beat_{h}']):.1%}  "
                  f"Sig: {'YES' if p < 0.05 else 'NO'}")

# ─────────────────────────────────────────────
# SECTION 5: ROBUSTNESS CHECK 4 - WILSON SCORE CALIBRATION CIs
# ─────────────────────────────────────────────
print("\n" + "─" * 70)
print("SECTION 5: ROBUSTNESS CHECK 4 — CALIBRATION WILSON SCORE CIs")
print("─" * 70)

# From Paper 4 reported numbers
calibration_data = {
    'Bullish': {'n': 100397, 'hit_rate': 0.970, 'stated_cpe': 0.930},
    'Bearish': {'n': 3586,   'hit_rate': 0.290, 'stated_cpe': 0.837},
    'Joint':   {'n': 3402,   'hit_rate': 0.995, 'stated_cpe': 0.979},
}

# Bitcoin ETF effective sample (IBIT, FBTC, BITB account for ~96% of bullish)
# Effective n adjusted for 3 near-identical instruments
btc_concentration = 0.96
n_effective_btc = int(100397 * (1 - btc_concentration) + 100397 * btc_concentration / 3)

print(f"\nBitcoin ETF concentration adjustment:")
print(f"  Nominal n (bullish): 100,397")
print(f"  BTC ETF share: ~96% from IBIT/FBTC/BITB")
print(f"  Effective independent n: ~{n_effective_btc:,}")

z95 = 1.96
for label, d in calibration_data.items():
    n, p = d['n'], d['hit_rate']
    # Wilson score interval
    denom = 1 + z95**2 / n
    center = (p + z95**2 / (2*n)) / denom
    half = (z95 * np.sqrt(p*(1-p)/n + z95**2/(4*n**2))) / denom
    lo, hi = center - half, center + half

    # Also for effective n (bullish only)
    if label == 'Bullish':
        n_eff = n_effective_btc
        denom_e = 1 + z95**2 / n_eff
        center_e = (p + z95**2 / (2*n_eff)) / denom_e
        half_e = (z95 * np.sqrt(p*(1-p)/n_eff + z95**2/(4*n_eff**2))) / denom_e
        lo_e, hi_e = center_e - half_e, center_e + half_e
        print(f"\n{label} signals (n={n:,}, effective n≈{n_eff:,}):")
        print(f"  Hit rate: {p:.1%}  Stated CPE: {d['stated_cpe']:.1%}")
        print(f"  Wilson 95% CI (nominal n):    [{lo:.4f}, {hi:.4f}]")
        print(f"  Wilson 95% CI (effective n):  [{lo_e:.4f}, {hi_e:.4f}]")
        print(f"  Stated CPE within CI (nominal):   {'YES' if lo <= d['stated_cpe'] <= hi else 'NO'}")
        print(f"  Stated CPE within CI (effective): {'YES' if lo_e <= d['stated_cpe'] <= hi_e else 'NO'}")
    else:
        print(f"\n{label} signals (n={n:,}):")
        print(f"  Hit rate: {p:.1%}  Stated CPE: {d['stated_cpe']:.1%}")
        print(f"  Wilson 95% CI: [{lo:.4f}, {hi:.4f}]")
        print(f"  Stated CPE within CI: {'YES' if lo <= d['stated_cpe'] <= hi else 'NO'}")

# ─────────────────────────────────────────────
# SECTION 6: SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY: WHAT SURVIVES CORRECTED INFERENCE")
print("=" * 70)

print("""
┌─────────────┬──────────┬──────────────┬──────────────┬──────────────┐
│ Test        │ 21-Day   │ 63-Day       │ 126-Day      │ Verdict      │
├─────────────┼──────────┼──────────────┼──────────────┼──────────────┤
│ OLS t-stat  │ 0.84     │ 4.07***      │ 5.51***      │ Paper 4      │
│ NW (RoT)    │ see above│ see above    │ see above    │ This paper   │
│ NW (Hor)    │ see above│ see above    │ see above    │ This paper   │
│ Non-overlap │ see above│ see above    │ see above    │ This paper   │
│ Block boot  │ see above│ see above    │ see above    │ This paper   │
│ Subperiod   │ see above│ see above    │ see above    │ This paper   │
└─────────────┴──────────┴──────────────┴──────────────┴──────────────┘
""")

print("\nDone. All results above feed directly into Paper 5.")
