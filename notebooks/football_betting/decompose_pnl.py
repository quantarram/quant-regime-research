import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import binomtest

OUT_DIR = Path(__file__).parent / "output"
q = pd.read_parquet(OUT_DIR / "qualifying_bets.parquet")

fig, ax = plt.subplots(figsize=(12, 6))
colors = {"1x2": "#2166ac", "double_chance": "#999999", "over_under_2.5": "#d6604d", "ah0_dnb": "#f4a582"}
for mkt, grp in q.groupby("market"):
    grp = grp.sort_values("date")
    cum = grp["pnl"].cumsum()
    label = mkt + ("  (synthetic no-vig price -- not realistic)" if mkt == "double_chance" else "")
    ax.plot(grp["date"], cum, label=label, color=colors.get(mkt))
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Cumulative P&L (SGD)")
ax.set_title("Cumulative P&L by market, model P>=80% strategy, S$50 flat stake")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "pnl_by_market.png", dpi=150)

print("Statistical significance check (is hit rate actually > 80%, or noise?):")
for mkt, grp in q.groupby("market"):
    n = len(grp)
    k = int(grp["won"].sum())
    p_hat = k / n
    test = binomtest(k, n, 0.80, alternative="greater")
    print(f"  {mkt:16s} n={n:5d}  hit_rate={p_hat:.3f}  p-value (H0: true rate<=80%)={test.pvalue:.4f}")

print("\n1x2-only (the one market with real, sharp-book market odds):")
mkt_1x2 = q[q["market"] == "1x2"]
print(f"  n={len(mkt_1x2)}, total pnl=S${mkt_1x2['pnl'].sum():.2f}, "
      f"mean pnl/bet=S${mkt_1x2['pnl'].mean():.3f}, "
      f"se of hit rate={np.sqrt(0.8*0.2/len(mkt_1x2)):.4f}")
