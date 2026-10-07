"""Plots the real 2026 IMD cumulative deficit trajectory from imd_2026_monsoon_timeline.csv
(final point = IMD's end-of-season figure, as reported by news outlets)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd
df = pd.read_csv("imd_2026_monsoon_timeline.csv"); df["as_of_date"] = pd.to_datetime(df["as_of_date"])
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["as_of_date"], df["cumulative_deficit_pct_lpa"], "o-", color="#c96a3a", lw=2, markersize=7)
for _, r in df.iterrows():
    ax.annotate(f"{r['cumulative_deficit_pct_lpa']:g}%", (r["as_of_date"], r["cumulative_deficit_pct_lpa"]),
                textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
ax.axhline(-10, color="#2E7D4F", ls="--", lw=1.2, label="IMD's own full-season forecast (90% of LPA = -10%)")
ax.axhline(0, color="#444", lw=0.8)
ax.set_ylabel("Cumulative all-India rainfall deficit (% of LPA)")
ax.set_title("India's 2026 monsoon: real trajectory, June to final season total\nStarted severe, finished at -12.6% -- close to IMD's own forecast, the same shape as 2023")
ax.legend(loc="lower right", fontsize=9); fig.autofmt_xdate(); fig.tight_layout()
fig.savefig("imd_2026_monsoon_trajectory_plot.png", dpi=140); print("saved")
