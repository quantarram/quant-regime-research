"""
paper16_graphical_abstract.py
==============================
Graphical abstract for Paper 16 (decision_layer_ceiling_paper_draft.md).
Centered on the paper's main claim: predictability-informed instrument
selection improves standard TSMOM. Left (primary, large): the win-rate
and Sharpe improvement from restricting TSMOM to predictable instruments.
Right (secondary, brief): the two instruments (JPM, XLB) where a
regime-conditioned single-instrument signal beats standard TSMOM outright,
reported as real mean-difference alpha, not the originally-published
Jensen's-alpha figures.

Run (from notebooks/):
    python paper16_graphical_abstract.py
"""
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})

# ---- Left panel (primary): predictability-filtered TSMOM ----
ax = axes[0]
labels = ["Unfiltered\nTSMOM baseline\n(12 instruments)", "Predictability-\nfiltered TSMOM\n(5 instruments)"]
wins = [2, 3]
totals = [5, 5]
bar_colors = ["#8a94a3", "#2e7d4f"]
xpos = [0, 1]
ax.bar(xpos, [w / t for w, t in zip(wins, totals)], color=bar_colors, width=0.55)
for xi, w, t in zip(xpos, wins, totals):
    ax.text(xi, w / t + 0.03, f"{w}/{t}", ha="center", fontsize=15, fontweight="bold")
ax.set_xticks(xpos)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Real historical blocks won (of 5)", fontsize=10.5)
ax.set_title("Picking predictable stocks improves TSMOM\nSharpe 0.32 vs. 0.29, full sample (1994-2026)", fontsize=12)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# ---- Right panel (secondary): JPM + XLB, real mean-difference ----
ax2 = axes[1]
inst_labels = ["JPM", "XLB"]
own_signal = [0.78, 0.97]   # master-model, real mean-difference
tsmom = [-10.99, -9.64]     # standard TSMOM, same instrument/period
x = np.arange(2)
w_ = 0.32
ax2.bar(x - w_ / 2, own_signal, width=w_, color="#2e7d4f", label="Master-model\n(own signal)")
ax2.bar(x + w_ / 2, tsmom, width=w_, color="#c0392b", label="Standard TSMOM\n(published)")
ax2.axhline(0, color="black", lw=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(inst_labels, fontsize=11)
ax2.set_ylabel("Alpha vs. own buy-and-hold\n2022+ holdout (%/yr)", fontsize=10.5)
ax2.set_title("Two instruments, real point estimates:\nsame direction, every way we checked", fontsize=12)
ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)

fig.suptitle(
    "Predictability-Informed TSMOM Beats the Standard Momentum Trading Strategy",
    fontsize=14.5, y=1.03,
)
fig.tight_layout()
fig.savefig("paper16_graphical_abstract.png", dpi=150, bbox_inches="tight")
print("Saved paper16_graphical_abstract.png")
