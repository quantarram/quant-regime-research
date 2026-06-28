"""
Paper 7 — Permutation Test Figure Generator
Run this AFTER paper7_permutation_test.py completes.
Reads results/paper7_permutation_test.csv and builds Figure 9.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 10, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--',
    'figure.facecolor': 'white', 'axes.facecolor': '#FAFAFA',
})

C_BLUE = '#1B6CA8'; C_GREEN = '#2E8B57'; C_RED = '#C0392B'
C_GOLD = '#D4A017'

# Load results
results = pd.read_csv('results/paper7_permutation_test.csv')
summary = pd.read_csv('results/paper7_permutation_summary.csv',
                      index_col=0, header=None).squeeze()

n_empirical = int(summary['n_empirical'])
perm_median = float(summary['perm_median'])
p_value     = float(summary['p_value'])
p95         = float(summary['perm_p95'])
p99         = float(summary['perm_p99'])
ratio       = float(summary['ratio'])
n_perms     = int(summary['n_perms'])
perm_counts = results['n_signals'].values

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    f'Figure 9.  Permutation Test: Are the 172 Surviving Signals a Multiple Testing Artefact?\n'
    f'Circular block permutation (N={n_perms:,}): shuffle temperature dates, re-run full analysis\n'
    f'Empirical count = {n_empirical} signals  |  Permutation median = {perm_median:.0f}  |  '
    f'Ratio = {ratio:.1f}×  |  p = {p_value:.4f}',
    fontsize=10, fontweight='bold')

# Left panel: histogram of permutation distribution
ax = axes[0]
ax.hist(perm_counts, bins=40, color=C_BLUE, alpha=0.75, density=False,
        label=f'Permutation distribution\n(N={n_perms:,} shuffles)')
ax.axvline(n_empirical, color=C_GREEN, lw=2.5, label=f'Empirical: {n_empirical} signals')
ax.axvline(perm_median, color=C_BLUE,  lw=1.5, ls='--', label=f'Perm median: {perm_median:.0f}')
ax.axvline(p95,         color=C_GOLD,  lw=1.2, ls=':',  label=f'Perm 95th pct: {p95:.0f}')
ax.axvline(p99,         color=C_RED,   lw=1.2, ls=':',  label=f'Perm 99th pct: {p99:.0f}')

# Shade right tail (signals ≥ empirical)
tail = perm_counts[perm_counts >= n_empirical]
if len(tail) > 0:
    ax.hist(tail, bins=20, color=C_RED, alpha=0.4,
            label=f'p-value region: {len(tail)}/{n_perms}')

ax.set_xlabel('Number of surviving signals (CPE ≥ 0.58, lift ≥ 1.15×)')
ax.set_ylabel('Count of permutations')
ax.set_title('Permutation Distribution vs Empirical Signal Count')
ax.legend(fontsize=7.5, loc='upper right')

# Significance annotation
sig_text = (f'p = {p_value:.4f}\n'
            f'{"SIGNIFICANT ✓" if p_value < 0.05 else "NOT SIGNIFICANT ✗"}')
color = C_GREEN if p_value < 0.05 else C_RED
ax.text(0.02, 0.97, sig_text, transform=ax.transAxes,
        ha='left', va='top', fontsize=11, fontweight='bold', color=color,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor=color, linewidth=1.5, alpha=0.9))

# Right panel: cumulative distribution + empirical
ax2 = axes[1]
sorted_counts = np.sort(perm_counts)
cdf = np.arange(1, len(sorted_counts)+1) / len(sorted_counts)
ax2.plot(sorted_counts, cdf, color=C_BLUE, lw=2.5, label='Permutation CDF')
ax2.axvline(n_empirical, color=C_GREEN, lw=2.5,
            label=f'Empirical: {n_empirical} signals')
ax2.axhline(1 - p_value, color=C_RED, lw=1.2, ls='--',
            label=f'1 - p = {1-p_value:.4f}')

ax2.set_xlabel('Number of surviving signals')
ax2.set_ylabel('Cumulative probability')
ax2.set_title('Permutation CDF: Fraction of Shuffles Below Empirical Count')
ax2.legend(fontsize=8)
ax2.set_ylim(0, 1.05)

# Annotate where empirical lands on CDF
ax2.annotate(
    f'{(1-p_value)*100:.1f}% of permutations\nhave fewer signals\nthan empirical',
    xy=(n_empirical, 1-p_value),
    xytext=(n_empirical * 0.6, 0.6),
    fontsize=8.5, color=C_GREEN, fontweight='bold',
    arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=1.5),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
              edgecolor=C_GREEN, alpha=0.9))

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/fig9_permutation_test.png', dpi=180, bbox_inches='tight')
plt.close()
print("Figure 9 saved: figures/fig9_permutation_test.png")
print(f"\nKey result: {(1-p_value)*100:.1f}% of permutations have fewer signals than empirical.")
print(f"Empirical is {ratio:.1f}× the permutation median.")
