"""
Paper 8 — CPE Analysis: VPD and Moisture Stress vs Financial Instruments
=========================================================================
Key scientific question: Do VPD-based predictors outperform temperature-only
predictors from Paper 7, and in which zones does VPD add information?

Expected finding from pipeline output:
  - Temperate zones (EU wheat, US corn): VPD ~0 stress events → no VPD signal
  - Tropical zones (Brazil Soy, India Sugar): VPD 1000+ events → VPD may signal
  - Combined heat+VPD (Brazil Soy: 624 days) is strongest candidate

Run on your LOCAL machine.
Required:
  - multiasset_returns.parquet
  - data/paper8_vpd_exceedances_aligned.parquet  (from pipeline)
  - data/paper7_agri_exceedances_aligned.parquet (for comparison)
"""

import pandas as pd
import numpy as np
import warnings, os, time
warnings.filterwarnings('ignore')

TRAIN_END  = '2024-12-31'
EVAL_START = '2025-01-01'
MIN_N_COND = 30
MIN_CPE    = 0.58
MIN_LIFT   = 1.15
QUANTILES  = [0.50, 0.55, 0.60, 0.65, 0.70]
HORIZONS   = [21, 42, 63, 126]

# ── CHANNEL PAIRS ──────────────────────────────────────────────────────
# Grouped by expected VPD signal strength based on pipeline output

# Group A: VPD-rich zones (tropical/semi-arid) — main Paper 8 contribution
VPD_RICH_PAIRS = [
    # Brazil Soy — 1098 VPD stress days, 624 combined heat+VPD
    ('Brazil_Soy',  'SOYB',  HORIZONS, 'Brazil Soy VPD → soybeans'),
    ('Brazil_Soy',  'ZS=F',  HORIZONS, 'Brazil Soy VPD → soy futures'),
    ('Brazil_Soy',  'CORN',  HORIZONS, 'Brazil Soy VPD → corn'),
    ('Brazil_Soy',  'ZC=F',  HORIZONS, 'Brazil Soy VPD → corn futures'),
    ('Brazil_Soy',  'GC=F',  HORIZONS, 'Brazil Soy VPD → gold'),
    ('Brazil_Soy',  'DBC',   HORIZONS, 'Brazil Soy VPD → broad commodities'),
    # India Sugar — 1063 VPD stress days
    ('India_Sugar', 'CANE',  HORIZONS, 'India VPD → sugar ETF'),
    ('India_Sugar', 'GC=F',  HORIZONS, 'India VPD → gold'),
    ('India_Sugar', 'DBB',   HORIZONS, 'India VPD → base metals'),
    ('India_Sugar', 'SOYB',  HORIZONS, 'India VPD → soybeans (food inflation)'),
    # Brazil Corn — only 1 stress day but viable VPD percentile exceedance
    ('Brazil_Corn', 'CORN',  HORIZONS, 'Brazil Corn VPD → corn'),
    ('Brazil_Corn', 'ZC=F',  HORIZONS, 'Brazil Corn VPD → corn futures'),
]

# Group B: US Soybean Belt — 9 stress days, 14 dry heat days
# Low count but worth testing VPD percentile exceedance
SOY_PAIRS = [
    ('US_Soybean_Belt', 'SOYB', HORIZONS, 'US Soy VPD → soybeans'),
    ('US_Soybean_Belt', 'ZS=F', HORIZONS, 'US Soy VPD → soy futures'),
    ('US_Soybean_Belt', 'CORN', HORIZONS, 'US Soy VPD → corn'),
    ('US_Soybean_Belt', 'WEAT', HORIZONS, 'US Soy VPD → wheat'),
    ('US_Soybean_Belt', 'GC=F', HORIZONS, 'US Soy VPD → gold'),
]

# Group C: Energy zones — ET0 and VPD percentile exceedance may add to heat signals
ENERGY_PAIRS = [
    ('EU_Urban_Energy', 'NG=F', HORIZONS, 'EU Urban VPD → nat gas'),
    ('EU_Urban_Energy', 'UNG',  HORIZONS, 'EU Urban VPD → nat gas ETF'),
    ('EU_Urban_Energy', 'DBB',  HORIZONS, 'EU Urban VPD → base metals'),
    ('EU_Urban_Energy', 'GC=F', HORIZONS, 'EU Urban VPD → gold'),
    ('US_Urban_Energy', 'NG=F', HORIZONS, 'US Urban VPD → nat gas'),
    ('US_Urban_Energy', 'UNG',  HORIZONS, 'US Urban VPD → nat gas ETF'),
    ('US_Urban_Energy', 'XLU',  HORIZONS, 'US Urban VPD → utilities'),
]

# Group D: Temperate wheat/corn zones — VPD percentile exceedance only
#          (stress threshold predictors will have ~0 events)
TEMPERATE_PAIRS = [
    ('EU_Wheat_Belt',        'WEAT', HORIZONS, 'EU Wheat VPD → wheat'),
    ('EU_Wheat_Belt',        'ZW=F', HORIZONS, 'EU Wheat VPD → wheat futures'),
    ('EU_Wheat_Belt',        'NG=F', HORIZONS, 'EU Wheat VPD → nat gas'),
    ('Ukraine_Russia_Wheat', 'WEAT', HORIZONS, 'Ukraine VPD → wheat'),
    ('Ukraine_Russia_Wheat', 'GC=F', HORIZONS, 'Ukraine VPD → gold'),
    ('US_Great_Plains_Wheat','WEAT', HORIZONS, 'US Plains VPD → wheat'),
    ('US_Corn_Belt',         'WEAT', HORIZONS, 'US Corn Belt VPD → wheat'),
    ('US_Corn_Belt',         'ZC=F', HORIZONS, 'US Corn Belt VPD → corn futures'),
]

ALL_PAIRS = VPD_RICH_PAIRS + SOY_PAIRS + ENERGY_PAIRS + TEMPERATE_PAIRS

# VPD predictor types to test for each zone
VPD_PREDICTOR_TYPES = [
    'VPD_q80', 'VPD_q90',
    'VPD_seasonal_q80', 'VPD_seasonal_q90',
    'VPD_stress_1p5kPa', 'VPD_stress_2p0kPa', 'VPD_stress_2p5kPa',
    'VPD_30d_q80', 'VPD_30d_q90',
    'combined_heat_vpd',
    'dry_heat',
    'ET0_q80', 'ET0_q90',
]

# Paper 7 predictor types for direct comparison
P7_PREDICTOR_TYPES = [
    'tmax_q90', 'tmax_q95',
    'seasonal_q90',
    'heatstress_30C', 'heatstress_32C', 'heatstress_35C', 'heatstress_38C',
    'GDD30_q90',
]

# ── LOAD DATA ─────────────────────────────────────────────────────────
def load_data():
    print("Loading data...")
    returns  = pd.read_parquet('multiasset_returns.parquet')
    temp_p8  = pd.read_parquet('data/paper8_vpd_exceedances_aligned.parquet')
    try:
        temp_p7 = pd.read_parquet('data/paper7_agri_exceedances_aligned.parquet')
        print(f"  Paper 7 temp:  {temp_p7.shape}")
    except:
        temp_p7 = None
        print("  Paper 7 temp:  not found")

    print(f"  Returns:       {returns.shape}")
    print(f"  Paper 8 VPD:   {temp_p8.shape}")

    all_h       = sorted(set(h for _,_,hs,_ in ALL_PAIRS for h in hs))
    all_tickers = list(set([p[1] for p in ALL_PAIRS]))

    print(f"\nPre-computing forward returns for horizons {all_h}...")
    cum_fwd = {}
    for ticker in all_tickers:
        if ticker not in returns.columns:
            print(f"  MISSING: {ticker}")
            continue
        daily_clean = returns[ticker].dropna()
        cum_fwd[ticker] = {}
        for h in all_h:
            td   = max(1, int(round(h * 252 / 365)))
            roll = daily_clean.rolling(window=td, min_periods=td).sum().shift(-td)
            cum_fwd[ticker][h] = roll.reindex(returns[ticker].index)
        print(f"  {ticker}: {len(daily_clean)} obs")

    # Align indices
    common = returns.index.intersection(temp_p8.index)
    temp_p8 = temp_p8.loc[common]
    for t in cum_fwd:
        for h in cum_fwd[t]:
            cum_fwd[t][h] = cum_fwd[t][h].loc[common]
    if temp_p7 is not None:
        temp_p7 = temp_p7.loc[common]

    return returns, temp_p8, temp_p7, cum_fwd

# ── CPE COMPUTATION ───────────────────────────────────────────────────
def compute_cpe(temp_series, fwd_series, q_target):
    both  = pd.DataFrame({'temp': temp_series, 'fwd': fwd_series}).dropna()
    train = both[both.index <= TRAIN_END]
    if len(train) < MIN_N_COND: return None
    thr  = train['fwd'].quantile(q_target)
    up   = (train['fwd'] > thr).mean()
    if up <= 0: return None
    cond = train[train['temp'] == 1]
    if len(cond) < MIN_N_COND: return None
    cpe  = (cond['fwd'] > thr).mean()
    lift = cpe / up
    if cpe < MIN_CPE or lift < MIN_LIFT: return None
    oos      = both[both.index >= EVAL_START]
    cond_oos = oos[oos['temp'] == 1]
    n_oos    = len(cond_oos)
    hr_oos   = (cond_oos['fwd'] > thr).mean() if n_oos > 0 else np.nan
    return {
        'threshold': thr, 'uncond_prob': up,
        'cpe_train': cpe, 'lift_train': lift,
        'n_cond_train': len(cond), 'n_cond_oos': n_oos,
        'hit_rate_oos': hr_oos,
        'calibration_err': (hr_oos - cpe) if not np.isnan(hr_oos) else np.nan,
    }

# ── MAIN ANALYSIS ─────────────────────────────────────────────────────
def run_analysis(temp_p8, cum_fwd, label='Paper 8 VPD'):
    results  = []
    n_tested = n_passed = 0

    for zone_prefix, ticker, horizons, rationale in ALL_PAIRS:
        if ticker not in cum_fwd: continue
        zone_cols = [c for c in temp_p8.columns
                     if c.startswith(zone_prefix + '_')
                     and c[len(zone_prefix)+1:] in VPD_PREDICTOR_TYPES]
        if not zone_cols:
            continue

        for pred_col in zone_cols:
            pred_type = pred_col[len(zone_prefix)+1:]
            for h in horizons:
                if h not in cum_fwd[ticker]: continue
                for q in QUANTILES:
                    n_tested += 1
                    r = compute_cpe(temp_p8[pred_col], cum_fwd[ticker][h], q)
                    if r is not None:
                        n_passed += 1
                        r.update({
                            'zone': zone_prefix, 'pred_col': pred_col,
                            'pred_type': pred_type, 'ticker': ticker,
                            'rationale': rationale, 'horizon': h,
                            'q_target': q, 'source': label,
                        })
                        results.append(r)

    print(f"  Tested: {n_tested}, Passed: {n_passed} ({n_passed/n_tested*100:.1f}%)")
    if not results:
        return pd.DataFrame()
    cols = ['zone','pred_col','pred_type','ticker','rationale','source',
            'horizon','q_target','cpe_train','lift_train','uncond_prob',
            'n_cond_train','n_cond_oos','hit_rate_oos','calibration_err']
    df = pd.DataFrame(results)[cols]
    return df.sort_values(['lift_train','cpe_train'],ascending=False).reset_index(drop=True)

# ── PAPER 7 COMPARISON ────────────────────────────────────────────────
def run_p7_comparison(temp_p7, cum_fwd):
    """Re-run Paper 7 signals for direct comparison with Paper 8 VPD signals."""
    if temp_p7 is None:
        return pd.DataFrame()
    results  = []
    n_tested = n_passed = 0

    for zone_prefix, ticker, horizons, rationale in ALL_PAIRS:
        if ticker not in cum_fwd: continue
        zone_cols = [c for c in temp_p7.columns
                     if c.startswith(zone_prefix + '_')
                     and c[len(zone_prefix)+1:] in P7_PREDICTOR_TYPES]
        for pred_col in zone_cols:
            pred_type = pred_col[len(zone_prefix)+1:]
            for h in horizons:
                if h not in cum_fwd[ticker]: continue
                for q in QUANTILES:
                    n_tested += 1
                    r = compute_cpe(temp_p7[pred_col], cum_fwd[ticker][h], q)
                    if r is not None:
                        n_passed += 1
                        r.update({
                            'zone': zone_prefix, 'pred_col': pred_col,
                            'pred_type': pred_type, 'ticker': ticker,
                            'rationale': rationale.replace('VPD','heat'),
                            'horizon': h, 'q_target': q, 'source': 'Paper 7 heat',
                        })
                        results.append(r)

    print(f"  P7 comparison — Tested: {n_tested}, Passed: {n_passed}")
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    return df.sort_values(['lift_train','cpe_train'],ascending=False).reset_index(drop=True)

# ── PREDICTOR TYPE ANALYSIS ───────────────────────────────────────────
def analyse_predictor_types(df):
    print("\n=== VPD PREDICTOR TYPE ANALYSIS ===")
    print(f"{'Type':<25} {'N':>6} {'Mean lift':>10} {'Max lift':>10} {'Mean CPE':>10}")
    print("-" * 65)
    groups = {
        'VPD_percentile':  ['VPD_q80','VPD_q90'],
        'VPD_seasonal':    ['VPD_seasonal_q80','VPD_seasonal_q90'],
        'VPD_stress':      [t for t in df['pred_type'].unique() if 'stress' in t],
        'VPD_30d':         ['VPD_30d_q80','VPD_30d_q90'],
        'combined_heat_vpd':['combined_heat_vpd'],
        'dry_heat':        ['dry_heat'],
        'ET0':             ['ET0_q80','ET0_q90'],
    }
    for g, types in groups.items():
        sub = df[df['pred_type'].isin(types)]
        if len(sub) == 0: continue
        print(f"  {g:<23} {len(sub):>6} {sub['lift_train'].mean():>10.3f} "
              f"{sub['lift_train'].max():>10.3f} {sub['cpe_train'].mean():>10.3f}")

# ── PRINT & SAVE ──────────────────────────────────────────────────────
def print_results(df, label=''):
    if df.empty: return
    print(f"\n{'='*100}")
    print(f"SURVIVING SIGNALS — {label}")
    print(f"{'='*100}")
    print(f"{'Zone':<28} {'PredType':<22} {'Ticker':<8} {'H':>4} {'Q':>5} "
          f"{'CPE':>6} {'Lift':>6} {'OOS':>7} {'N':>5}")
    print("-"*100)
    for _, r in df.head(25).iterrows():
        hr = f"{r['hit_rate_oos']:.3f}" if not pd.isna(r['hit_rate_oos']) else "  n/a"
        print(f"{r['zone']:<28} {r['pred_type']:<22} {r['ticker']:<8} "
              f"{r['horizon']:>4} {r['q_target']:>5.0%} {r['cpe_train']:>6.3f} "
              f"{r['lift_train']:>6.2f}× {hr:>7} {r['n_cond_train']:>5}")

def save_results(df_p8, df_p7):
    os.makedirs('results', exist_ok=True)
    if not df_p8.empty:
        df_p8.to_parquet('results/paper8_vpd_signals.parquet', index=False)
        df_p8.to_csv('results/paper8_vpd_signals.csv', index=False)
        print(f"\nSaved {len(df_p8)} Paper 8 VPD signals → results/paper8_vpd_signals.csv")
    if not df_p7.empty:
        df_p7.to_csv('results/paper8_p7_comparison.csv', index=False)
        print(f"Saved {len(df_p7)} Paper 7 comparison signals")

    # Combined for Paper 8 analysis
    combined = pd.concat([df_p8, df_p7], ignore_index=True) if not df_p7.empty else df_p8
    if not combined.empty:
        combined.to_csv('results/paper8_all_signals.csv', index=False)

# ── MAIN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("PAPER 8 — CPE ANALYSIS: VPD AND MOISTURE STRESS")
    print("=" * 70)
    print(f"Min CPE: {MIN_CPE}, Min lift: {MIN_LIFT}×, Min n: {MIN_N_COND}")

    os.makedirs('results', exist_ok=True)
    returns, temp_p8, temp_p7, cum_fwd = load_data()

    print("\n── Paper 8 VPD analysis ──")
    df_p8 = run_analysis(temp_p8, cum_fwd, 'Paper 8 VPD')

    print("\n── Paper 7 comparison (same zones, temperature-only) ──")
    df_p7 = run_p7_comparison(temp_p7, cum_fwd)

    if not df_p8.empty:
        print_results(df_p8, 'PAPER 8 VPD SIGNALS')
        analyse_predictor_types(df_p8)
    else:
        print("\nNo VPD signals survived.")

    if not df_p7.empty:
        print_results(df_p7, 'PAPER 7 COMPARISON (TEMPERATURE-ONLY)')

    save_results(df_p8, df_p7)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Paper 8 VPD signals:      {len(df_p8)}")
    print(f"Paper 7 comparison:       {len(df_p7)}")
    if not df_p8.empty and not df_p7.empty:
        print(f"VPD mean lift:            {df_p8['lift_train'].mean():.3f}×")
        print(f"Heat mean lift:           {df_p7['lift_train'].mean():.3f}×")
        print(f"VPD max lift:             {df_p8['lift_train'].max():.3f}×")
        print(f"Heat max lift:            {df_p7['lift_train'].max():.3f}×")
        if df_p8['lift_train'].mean() > df_p7['lift_train'].mean():
            print("\n→ VPD OUTPERFORMS temperature-only predictors")
        else:
            print("\n→ Temperature-only predictors outperform VPD")
            print("  Finding: VPD does not add signal beyond temperature alone")
