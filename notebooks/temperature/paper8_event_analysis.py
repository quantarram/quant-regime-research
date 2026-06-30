"""
Paper 8 — Event Clustering Analysis
Shows when dry_heat and combined_heat_vpd events occur.
Run on your LOCAL machine.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':9,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':0.25,'grid.linestyle':'--',
    'figure.facecolor':'white','axes.facecolor':'#FAFAFA',
})
C_BLUE='#1B6CA8'; C_RED='#C0392B'; C_GOLD='#D4A017'; C_GREEN='#2E8B57'

temp = pd.read_parquet('data/paper8_vpd_exceedances_aligned.parquet')
temp = temp[temp.index <= '2024-12-31']  # training period only

key_flags = [
    'EU_Urban_Energy_dry_heat',
    'EU_Urban_Energy_combined_heat_vpd',
    'EU_Urban_Energy_VPD_stress_2p0kPa',
    'EU_Urban_Energy_heatstress_30C',       # Paper 7 comparison
    'US_Great_Plains_Wheat_combined_heat_vpd',
    'India_Sugar_VPD_seasonal_q90',
]

colors = [C_RED, '#8E44AD', C_GOLD, C_BLUE, C_GREEN, '#16A085']

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle(
    'Figure 13.  Event Clustering Analysis: When Do Conditioning Events Occur?\n'
    'Confirms signals are not driven by 1-2 isolated heatwave years',
    fontsize=11, fontweight='bold')

for idx, (flag, color) in enumerate(zip(key_flags, colors)):
    ax = axes[idx//2][idx%2]
    if flag not in temp.columns:
        ax.text(0.5, 0.5, f'{flag}\nNot found', ha='center', transform=ax.transAxes)
        continue

    events = temp[temp[flag] == 1].index
    by_year = pd.Series(1, index=events).resample('Y').sum()

    ax.bar(by_year.index.year, by_year.values, color=color, alpha=0.8, width=0.8)
    ax.set_title(f'{flag}\nn={len(events)} total events, {len(by_year[by_year>0])} years with events',
                 fontsize=8.5, fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Events per year')

    # Stats
    n_years_with_events = (by_year > 0).sum()
    max_year = by_year.idxmax().year if len(by_year) > 0 else 'N/A'
    max_val  = by_year.max() if len(by_year) > 0 else 0
    pct_concentrated = by_year.nlargest(3).sum() / len(events) * 100 if len(events) > 0 else 0

    ax.text(0.98, 0.95,
            f'Years with events: {n_years_with_events}/25\n'
            f'Peak year: {max_year} ({max_val:.0f} events)\n'
            f'Top-3 years: {pct_concentrated:.0f}% of total',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, alpha=0.9))

plt.tight_layout()
plt.savefig('figures/fig13_event_clustering.png', dpi=180, bbox_inches='tight')
print("Saved: figures/fig13_event_clustering.png")

# Print summary table
print("\n=== EVENT CLUSTERING SUMMARY ===")
print(f"{'Flag':<45} {'N events':>9} {'N years':>8} {'Top-3 yr %':>11}")
print("-" * 75)
for flag in key_flags:
    if flag not in temp.columns: continue
    events  = temp[temp[flag] == 1].index
    by_year = pd.Series(1, index=events).resample('Y').sum()
    n_yrs   = (by_year > 0).sum()
    pct_top3= by_year.nlargest(3).sum()/len(events)*100 if len(events) > 0 else 0
    print(f"  {flag:<43} {len(events):>9} {n_yrs:>8} {pct_top3:>10.1f}%")
