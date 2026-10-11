"""
paper17_graphical_abstract.py
Graphical abstract for Paper 17 (weather_impacts_paper_draft.md). Left: forecast skill horizon, in days, for demand, wind and solar
(read from paper17_figs/numbers.json, produced by paper17_figures.py). Right: the 614 weather-tail configurations admitted in formation, their
tail probability in formation and in the test, against the test base rate (weather_finance/scan_results.parquet).

Run (from notebooks/):
    ../.venv/bin/python paper17_graphical_abstract.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
num = json.load(open(os.path.join(HERE, "paper17_figs", "numbers.json")))
GREEN, BLUE, AMBER, GREY, RED = "#2e7d4f", "#2f5f8f", "#c98a1b", "#8a94a3", "#c0392b"

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.15]})

ax = axes[0]
names = ["Electricity\ndemand", "Wind\ngeneration", "Solar\ngeneration"]
days = [num["demand"]["crossover_lead"], num["renewables"]["crossover"]["wind"], num["renewables"]["crossover"]["solar"]]
bars = ax.bar(range(3), days, color=[GREEN, BLUE, AMBER], width=0.55)
for i, d in enumerate(days):
    ax.text(i, d + 0.15, f"{d} days", ha="center", fontsize=14, fontweight="bold")
ax.set_xticks(range(3))
ax.set_xticklabels(names, fontsize=11)
ax.set_ylim(0, 11)
ax.set_ylabel("Days ahead the forecast still beats climatology", fontsize=10.5)
ax.set_title("Free weather forecasts beat climatology\nfor about a week, then do worse", fontsize=12)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

ax2 = axes[1]
import pandas as pd
sc = pd.read_parquet(os.path.join(HERE, "weather_finance", "scan_results.parquet"))
adm = sc[(sc.eps_form >= 5) & (sc.p_form >= 0.30)]
vals = [adm.p_form.mean(), adm.p_test.mean(), adm.base_test.mean()]
ax2.bar(range(3), vals, color=[GREEN, RED, GREY], width=0.55)
for i, v in enumerate(vals):
    ax2.text(i, v + 0.008, f"{v:.1%}", ha="center", fontsize=13, fontweight="bold")
ax2.set_xticks(range(3))
ax2.set_xticklabels(["Formation\n1996-2012", "Test\n2013-2025", "Test base rate\n(chance)"], fontsize=11)
ax2.set_ylim(0, 0.42)
ax2.set_ylabel("P(big price move | weather tail)", fontsize=10.5)
ax2.set_title(f"{len(adm)} weather-and-price pairs that looked predictive\nin 1996-2012 (of {len(sc):,} tested) fall to chance", fontsize=12)
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)

fig.suptitle("A Free, Self-Scoring Weather Impacts Pipeline", fontsize=14.5, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "paper17_graphical_abstract.png"), dpi=170, bbox_inches="tight")
