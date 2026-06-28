"""
Paper 7 — CPE Analysis: Crop-Zone Temperature vs Financial Instruments
======================================================================
Tests whether Paper 6 agricultural signals survive on properly specified
crop-zone temperature data (ERA5 gridded bounding boxes vs city indices).

Key research questions:
  Q1: Do wheat signals survive on EU Plain / Ukraine / US Great Plains data?
  Q2: Do corn/soy signals survive on US Corn Belt data?
  Q3: Do sugar signals survive on Thailand / India crop-zone data?
  Q4: Do energy signals improve or change with dedicated urban zone data?
  Q5: Which predictor type is strongest: tmax, seasonal, heat stress, or GDD?

Also tests Paper 6's city-temperature proxies side-by-side for comparison.

Run on your LOCAL machine.
Required files in same directory:
  - multiasset_returns.parquet
  - data/paper7_agri_exceedances_aligned.parquet   (from pipeline)
  - data/temperature_exceedances_aligned.parquet   (Paper 6 city data)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings('ignore')

# ── CONFIGURATION ──────────────────────────────────────────────────────
TRAIN_END  = '2024-12-31'
EVAL_START = '2025-01-01'
MIN_N_COND = 30
MIN_CPE    = 0.58   # slightly relaxed — exploratory
MIN_LIFT   = 1.15
QUANTILES  = [0.50, 0.55, 0.60, 0.65, 0.70]
HORIZONS   = [21, 42, 63, 126]

# ── CHANNEL PAIRS: crop-zone predictor → financial instrument ──────────
# Format: (zone_prefix, ticker, rationale, horizon_list)
CHANNEL_PAIRS = [
    # ── WHEAT ─────────────────────────────────────────────────────────
    ('EU_Wheat_Belt',         'WEAT',  'EU wheat zone → wheat ETF',       HORIZONS),
    ('EU_Wheat_Belt',         'ZW=F',  'EU wheat zone → wheat futures',   HORIZONS),
    ('EU_Wheat_Belt',         'GC=F',  'EU wheat zone → gold futures',    HORIZONS),
    ('EU_Wheat_Belt',         'NG=F',  'EU wheat zone → nat gas',         HORIZONS),
    ('Ukraine_Russia_Wheat',  'WEAT',  'Black Sea wheat → wheat ETF',     HORIZONS),
    ('Ukraine_Russia_Wheat',  'ZW=F',  'Black Sea wheat → wheat futures', HORIZONS),
    ('Ukraine_Russia_Wheat',  'GC=F',  'Black Sea wheat → gold',          HORIZONS),
    ('US_Great_Plains_Wheat', 'WEAT',  'US Plains wheat → wheat ETF',     HORIZONS),
    ('US_Great_Plains_Wheat', 'ZW=F',  'US Plains wheat → wheat futures', HORIZONS),
    ('US_Great_Plains_Wheat', 'CORN',  'US Plains wheat → corn',          HORIZONS),
    ('US_Great_Plains_Wheat', 'ZC=F',  'US Plains wheat → corn futures',  HORIZONS),

    # ── CORN ──────────────────────────────────────────────────────────
    ('US_Corn_Belt',   'CORN',  'US Corn Belt → corn ETF',       HORIZONS),
    ('US_Corn_Belt',   'ZC=F',  'US Corn Belt → corn futures',   HORIZONS),
    ('US_Corn_Belt',   'SOYB',  'US Corn Belt → soybeans',       HORIZONS),
    ('US_Corn_Belt',   'ZS=F',  'US Corn Belt → soy futures',    HORIZONS),
    ('US_Corn_Belt',   'WEAT',  'US Corn Belt → wheat',          HORIZONS),
    ('US_Corn_Belt',   'GC=F',  'US Corn Belt → gold',           HORIZONS),
    ('Brazil_Corn',    'CORN',  'Brazil corn → corn ETF',        HORIZONS),
    ('Brazil_Corn',    'ZC=F',  'Brazil corn → corn futures',    HORIZONS),
    ('Brazil_Corn',    'SOYB',  'Brazil corn → soybeans',        HORIZONS),

    # ── SUGAR ─────────────────────────────────────────────────────────
    ('Thailand_Sugar', 'CANE',  'Thailand sugar → sugar ETF',    HORIZONS),
    ('Thailand_Sugar', 'GC=F',  'Thailand sugar → gold',         HORIZONS),
    ('Thailand_Sugar', 'DBB',   'Thailand sugar → base metals',  HORIZONS),
    ('India_Sugar',    'CANE',  'India sugar → sugar ETF',       HORIZONS),
    ('India_Sugar',    'GC=F',  'India sugar → gold',            HORIZONS),

    # ── ENERGY (urban — correct variable) ─────────────────────────────
    ('EU_Urban_Energy', 'NG=F', 'EU urban heat → nat gas',       HORIZONS),
    ('EU_Urban_Energy', 'UNG',  'EU urban heat → nat gas ETF',   HORIZONS),
    ('EU_Urban_Energy', 'XLU',  'EU urban heat → utilities',     HORIZONS),
    ('EU_Urban_Energy', 'XLE',  'EU urban heat → energy',        HORIZONS),
    ('EU_Urban_Energy', 'ICLN', 'EU urban heat → clean energy',  HORIZONS),
    ('EU_Urban_Energy', 'DBB',  'EU urban heat → base metals',   HORIZONS),
    ('EU_Urban_Energy', 'GC=F', 'EU urban heat → gold',          HORIZONS),
    ('US_Urban_Energy', 'NG=F', 'US urban heat → nat gas',       HORIZONS),
    ('US_Urban_Energy', 'UNG',  'US urban heat → nat gas ETF',   HORIZONS),
    ('US_Urban_Energy', 'XLU',  'US urban heat → utilities',     HORIZONS),
    ('US_Urban_Energy', 'CORN', 'US urban heat → corn',          HORIZONS),
    ('US_Urban_Energy', 'GC=F', 'US urban heat → gold',          HORIZONS),
]

# Predictor suffix types to test for each zone
PREDICTOR_TYPES = [
    'tmax_q80', 'tmax_q90', 'tmax_q95',
    'seasonal_q80', 'seasonal_q90',
    'heatstress_30C', 'heatstress_32C', 'heatstress_35C', 'heatstress_38C',
    'GDD30_q80', 'GDD30_q90',
]

# ── LOAD DATA ─────────────────────────────────────────────────────────
def load_data():
    print("\nLoading data...")
    returns   = pd.read_parquet('multiasset_returns.parquet')
    temp_p7   = pd.read_parquet('data/paper7_agri_exceedances_aligned.parquet')
    try:
        temp_p6 = pd.read_parquet('data/temperature_exceedances_aligned.parquet')
        print(f"  Paper 6 city temp: {temp_p6.shape}")
    except FileNotFoundError:
        temp_p6 = None
        print("  Paper 6 city temp: not found (skipping comparison)")

    print(f"  Returns:           {returns.shape}")
    print(f"  Paper 7 crop temp: {temp_p7.shape}")
    print(f"  Available predictors: {len(temp_p7.columns)}")

    # Pre-compute cumulative forward returns
    all_h = sorted(set(h for _, _, _, hs in CHANNEL_PAIRS for h in hs))
    print(f"\nPre-computing forward returns for horizons {all_h}...")
    cum_fwd = {}
    tickers = list(set([p[1] for p in CHANNEL_PAIRS]))
    for ticker in tickers:
        if ticker not in returns.columns:
            print(f"  MISSING: {ticker}")
            continue
        daily_clean = returns[ticker].dropna()
        cum_fwd[ticker] = {}
        for h in all_h:
            td   = max(1, int(round(h * 252 / 365)))
            roll = daily_clean.rolling(window=td, min_periods=td).sum().shift(-td)
            cum_fwd[ticker][h] = roll.reindex(returns[ticker].index)
        print(f"  {ticker}: {len(daily_clean)} clean obs")

    return returns, temp_p7, temp_p6, cum_fwd

# ── CPE COMPUTATION ───────────────────────────────────────────────────
def compute_cpe(temp_series, fwd_series, q_target):
    both  = pd.DataFrame({'temp': temp_series, 'fwd': fwd_series}).dropna()
    train = both[both.index <= TRAIN_END]
    if len(train) < MIN_N_COND:
        return None
    thr  = train['fwd'].quantile(q_target)
    up   = (train['fwd'] > thr).mean()
    if up <= 0:
        return None
    cond = train[train['temp'] == 1]
    if len(cond) < MIN_N_COND:
        return None
    cpe  = (cond['fwd'] > thr).mean()
    lift = cpe / up
    if cpe < MIN_CPE or lift < MIN_LIFT:
        return None
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
def run_analysis(temp_p7, cum_fwd):
    results  = []
    n_tested = 0
    n_passed = 0

    print(f"\nTesting {len(CHANNEL_PAIRS)} pairs × up to {len(PREDICTOR_TYPES)} "
          f"predictor types × {len(HORIZONS)} horizons × {len(QUANTILES)} quantiles")

    for zone_prefix, ticker, rationale, horizons in CHANNEL_PAIRS:
        if ticker not in cum_fwd:
            continue

        # Find all available predictor columns for this zone
        zone_cols = [c for c in temp_p7.columns
                     if c.startswith(zone_prefix + '_')]
        if not zone_cols:
            print(f"  No columns for zone: {zone_prefix}")
            continue

        for pred_col in zone_cols:
            pred_type = pred_col[len(zone_prefix)+1:]  # strip zone prefix
            if pred_type not in PREDICTOR_TYPES:
                continue

            for h in horizons:
                if h not in cum_fwd[ticker]:
                    continue
                for q in QUANTILES:
                    n_tested += 1
                    r = compute_cpe(temp_p7[pred_col], cum_fwd[ticker][h], q)
                    if r is not None:
                        n_passed += 1
                        r.update({
                            'zone':          zone_prefix,
                            'pred_col':      pred_col,
                            'pred_type':     pred_type,
                            'ticker':        ticker,
                            'rationale':     rationale,
                            'horizon':       h,
                            'q_target':      q,
                        })
                        results.append(r)

    print(f"\nTested:  {n_tested}")
    print(f"Passed:  {n_passed}  ({n_passed/n_tested*100:.1f}%)")

    if not results:
        return pd.DataFrame()

    cols = ['zone','pred_col','pred_type','ticker','rationale',
            'horizon','q_target','cpe_train','lift_train','uncond_prob',
            'threshold','n_cond_train','n_cond_oos','hit_rate_oos','calibration_err']
    df = pd.DataFrame(results)[cols]
    return df.sort_values(['lift_train','cpe_train'], ascending=False).reset_index(drop=True)

# ── PAPER 6 COMPARISON ────────────────────────────────────────────────
def compare_with_paper6(temp_p6, cum_fwd):
    """Re-run Paper 6 signals for direct comparison."""
    if temp_p6 is None:
        return pd.DataFrame()

    p6_pairs = [
        ('Europe_tmax_q90_exceed', 'WEAT', 'P6: EU city → wheat'),
        ('Europe_tmax_q90_exceed', 'ZW=F', 'P6: EU city → wheat futures'),
        ('Europe_tmax_q90_exceed', 'CORN', 'P6: EU city → corn'),
        ('Europe_tmax_q90_exceed', 'ZC=F', 'P6: EU city → corn futures'),
        ('Europe_tmax_q90_exceed', 'CANE', 'P6: EU city → sugar'),
        ('North_America_tmax_q90_exceed', 'CORN', 'P6: NA city → corn'),
        ('North_America_tmax_q90_exceed', 'ZC=F', 'P6: NA city → corn futures'),
        ('Asia_tmax_q90_exceed',   'CANE', 'P6: Asia city → sugar'),
    ]

    results = []
    for pred_col, ticker, rationale in p6_pairs:
        if pred_col not in temp_p6.columns or ticker not in cum_fwd:
            continue
        for h in [63, 126]:
            if h not in cum_fwd[ticker]:
                continue
            for q in [0.55, 0.60]:
                r = compute_cpe(temp_p6[pred_col], cum_fwd[ticker][h], q)
                if r is not None:
                    r.update({'pred_col': pred_col, 'ticker': ticker,
                               'rationale': rationale, 'horizon': h, 'q_target': q,
                               'source': 'Paper6_city'})
                    results.append(r)

    return pd.DataFrame(results) if results else pd.DataFrame()

# ── PREDICTOR TYPE ANALYSIS ───────────────────────────────────────────
def analyse_predictor_types(df):
    """Which predictor type (tmax, seasonal, heatstress, GDD) is strongest?"""
    if df.empty:
        return
    print("\n=== PREDICTOR TYPE ANALYSIS ===")
    print(f"{'Type':<25} {'N signals':>10} {'Mean lift':>10} {'Max lift':>10} {'Mean CPE':>10}")
    print("-" * 60)

    type_groups = {
        'tmax_exceedance':  [t for t in df['pred_type'].unique() if t.startswith('tmax')],
        'seasonal':         [t for t in df['pred_type'].unique() if t.startswith('seasonal')],
        'heat_stress':      [t for t in df['pred_type'].unique() if t.startswith('heatstress')],
        'GDD_exceedance':   [t for t in df['pred_type'].unique() if t.startswith('GDD')],
    }

    for group_name, types in type_groups.items():
        sub = df[df['pred_type'].isin(types)]
        if len(sub) == 0:
            continue
        print(f"  {group_name:<23} {len(sub):>10} {sub['lift_train'].mean():>10.3f} "
              f"{sub['lift_train'].max():>10.3f} {sub['cpe_train'].mean():>10.3f}")

# ── FIGURE: Crop-zone vs City comparison ─────────────────────────────
def fig_comparison(df_p7, df_p6):
    """Compare Paper 7 crop-zone signals vs Paper 6 city-proxy signals."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        'Figure 6.  Crop-Zone vs City-Temperature Signals: CPE Lift Comparison\n'
        'Paper 7 (proper agricultural zones) vs Paper 6 (city proxies) for wheat, corn, sugar',
        fontsize=10, fontweight='bold'
    )

    ax = axes[0]
    # Lift distribution comparison
    if not df_p7.empty and not df_p6.empty:
        ax.hist(df_p7['lift_train'], bins=20, alpha=0.7, color='#2E8B57',
                label=f'Paper 7 crop-zone (n={len(df_p7)})', density=True)
        ax.hist(df_p6['lift_train'], bins=20, alpha=0.7, color='#1B6CA8',
                label=f'Paper 6 city proxy (n={len(df_p6)})', density=True)
        ax.axvline(1.0, color='black', lw=1.5, ls='--', label='No lift')
        ax.set_xlabel('CPE Lift over Unconditional')
        ax.set_ylabel('Density')
        ax.set_title('Lift Distribution: Crop-Zone vs City Proxy')
        ax.legend(fontsize=8)
    elif not df_p7.empty:
        ax.hist(df_p7['lift_train'], bins=20, alpha=0.8, color='#2E8B57',
                label=f'Paper 7 crop-zone (n={len(df_p7)})', density=True)
        ax.axvline(1.0, color='black', lw=1.5, ls='--')
        ax.set_xlabel('CPE Lift'); ax.set_ylabel('Density')
        ax.set_title('Lift Distribution: Crop-Zone Signals')
        ax.legend(fontsize=8)

    # Right panel: predictor type breakdown
    ax2 = axes[1]
    if not df_p7.empty:
        type_map = lambda t: ('tmax' if t.startswith('tmax') else
                              'seasonal' if t.startswith('seasonal') else
                              'heat_stress' if t.startswith('heatstress') else
                              'GDD')
        df_p7['type_group'] = df_p7['pred_type'].apply(type_map)
        type_lift = df_p7.groupby('type_group')['lift_train'].agg(['mean','max','count'])
        colors = ['#1B6CA8','#2E8B57','#C0392B','#D4A017']
        bars = ax2.bar(type_lift.index, type_lift['mean'],
                       color=colors[:len(type_lift)], alpha=0.85, width=0.6)
        ax2.axhline(1.0, color='black', lw=1.2, ls='--')
        for bar, (_, row) in zip(bars, type_lift.iterrows()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'n={int(row["count"])}\n{row["mean"]:.3f}×',
                     ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax2.set_xlabel('Predictor Type')
        ax2.set_ylabel('Mean CPE Lift')
        ax2.set_title('Mean Lift by Predictor Type (Paper 7)')

    plt.tight_layout()
    plt.savefig('figures/fig6_cropzone_vs_city.png', dpi=180, bbox_inches='tight')
    plt.close()
    print("  Fig 6 saved: figures/fig6_cropzone_vs_city.png")

# ── FIGURE: Zone × Instrument heatmap ────────────────────────────────
def fig_zone_heatmap(df):
    if df.empty:
        print("  Fig 7 skipped — no signals")
        return

    pivot = df.groupby(['zone','ticker'])['lift_train'].max().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(
        'Figure 7.  Maximum CPE Lift: Crop Zone × Financial Instrument\n'
        '(Paper 7: proper agricultural zone temperatures, best predictor type and horizon)',
        fontsize=10, fontweight='bold'
    )

    import matplotlib.colors as mcolors
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=1.0, vmax=max(pivot.values.max(), 1.5))
    im = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if val > 0:
                ax.text(j, i, f'{val:.2f}×', ha='center', va='center',
                        fontsize=7.5, fontweight='bold',
                        color='white' if val > 1.35 else 'black')

    plt.colorbar(im, ax=ax, label='CPE Lift', shrink=0.8)
    plt.tight_layout()
    plt.savefig('figures/fig7_zone_heatmap.png', dpi=180, bbox_inches='tight')
    plt.close()
    print("  Fig 7 saved: figures/fig7_zone_heatmap.png")

# ── PRINT RESULTS ─────────────────────────────────────────────────────
def print_results(df):
    if df.empty:
        print("\nNo signals survived.")
        return
    print(f"\n{'='*100}")
    print("SURVIVING SIGNALS — PAPER 7 (CROP-ZONE TEMPERATURES)")
    print(f"{'='*100}")
    print(f"{'Zone':<30} {'PredType':<20} {'Ticker':<8} {'H':>4} {'Q':>5} "
          f"{'CPE':>6} {'Lift':>6} {'OOS':>7} {'N':>5}")
    print("-"*100)
    for _, r in df.head(30).iterrows():
        hr = f"{r['hit_rate_oos']:.3f}" if not pd.isna(r['hit_rate_oos']) else "  n/a"
        print(f"{r['zone']:<30} {r['pred_type']:<20} {r['ticker']:<8} "
              f"{r['horizon']:>4} {r['q_target']:>5.0%} {r['cpe_train']:>6.3f} "
              f"{r['lift_train']:>6.2f}× {hr:>7} {r['n_cond_train']:>5}")

# ── SAVE ──────────────────────────────────────────────────────────────
def save_results(df, df_p6):
    os.makedirs('results', exist_ok=True)
    if not df.empty:
        df.to_parquet('results/paper7_signals.parquet', index=False)
        df.to_csv('results/paper7_signals.csv', index=False)
        print(f"\nSaved {len(df)} Paper 7 signals → results/paper7_signals.csv")
    if not df_p6.empty:
        df_p6.to_csv('results/paper7_paper6_comparison.csv', index=False)
        print(f"Saved {len(df_p6)} Paper 6 comparison signals → results/paper7_paper6_comparison.csv")

# ── MAIN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("PAPER 7 — CPE ANALYSIS: CROP-ZONE TEMPERATURES")
    print("=" * 70)
    print(f"Training cutoff : {TRAIN_END}")
    print(f"Min CPE         : {MIN_CPE}")
    print(f"Min lift        : {MIN_LIFT}×")
    print(f"Quantiles       : {QUANTILES}")
    print(f"Horizons        : {HORIZONS}")

    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    returns, temp_p7, temp_p6, cum_fwd = load_data()

    print("\n── Paper 7 crop-zone analysis ──")
    df_p7 = run_analysis(temp_p7, cum_fwd)

    print("\n── Paper 6 city-proxy comparison ──")
    df_p6_comp = compare_with_paper6(temp_p6, cum_fwd)
    print(f"  Paper 6 comparison signals: {len(df_p6_comp)}")

    print("\n── Predictor type breakdown ──")
    analyse_predictor_types(df_p7)

    print("\n── Generating figures ──")
    fig_comparison(df_p7, df_p6_comp)
    fig_zone_heatmap(df_p7)

    print("\n── Results ──")
    print_results(df_p7)

    print("\n── Paper 6 comparison signals ──")
    if not df_p6_comp.empty:
        print(df_p6_comp[['pred_col','ticker','horizon','q_target',
                           'cpe_train','lift_train']].to_string())

    save_results(df_p7, df_p6_comp)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Paper 7 signals: {len(df_p7)}")
    print(f"Paper 6 comparison signals: {len(df_p6_comp)}")
    if not df_p7.empty:
        print(f"Paper 7 lift range: {df_p7['lift_train'].min():.2f}× – {df_p7['lift_train'].max():.2f}×")
    if not df_p6_comp.empty:
        print(f"Paper 6 lift range: {df_p6_comp['lift_train'].min():.2f}× – {df_p6_comp['lift_train'].max():.2f}×")
