"""
Where does the >=80% rule actually fail? The aggregate hit rate (82.1% for
1x2, the one market with real validated edge) sounds reassuring, but every
one of those ~18% losses pays out a full stake at once while wins trickle in
a few cents at a time (thin odds on heavy favorites) -- so the loss tail is
doing almost all the work in whether the strategy nets positive or negative
over any given stretch. This looks directly at that ~18%: is it uniform
noise, or does it cluster somewhere identifiable (a league, a probability
band, a selection type, a time period)?

Primary sample: the 1x2 market's qualifying bets (n=1149, the one market
in this whole analysis validated against real market odds with a real, if
thin, edge). double_chance (n=12184) is used as a secondary, higher-power
check on the same question, since the win/loss PATTERN is real even though
its P&L numbers used a synthetic price (per the existing caveat in
backtest.py/results_by_market.csv).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2_contingency

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"


def load():
    q = pd.read_parquet(OUT_DIR / "qualifying_bets.parquet")
    matches = pd.read_parquet(DATA_DIR / "matches.parquet")[
        ["league", "date", "home_team", "away_team", "fthg", "ftag"]
    ]
    q = q.merge(matches, on=["league", "date", "home_team", "away_team"], how="left")
    return q


def breakdown(df, market_name, group_col, label):
    sub = df[df["market"] == market_name].copy()
    n_total = len(sub)
    n_loss = (~sub["won"]).sum()
    print(f"\n--- {market_name}: loss rate by {label} ---")
    g = sub.groupby(group_col, observed=True).agg(
        n=("won", "size"), losses=("won", lambda x: (~x).sum()), loss_rate=("won", lambda x: 1 - x.mean())
    ).sort_values("n", ascending=False)
    g["share_of_all_losses"] = g["losses"] / n_loss
    print(g.to_string())
    return g


def main():
    q = load()
    x2 = q[q["market"] == "1x2"].copy()
    dc = q[q["market"] == "double_chance"].copy()

    n = len(x2)
    n_loss = (~x2["won"]).sum()
    print(f"1x2 qualifying bets: {n}, losses: {n_loss} ({n_loss/n:.1%})")

    # 1. Loss rate by predicted-probability band -- the actionable question:
    #    does tightening the threshold above 80% actually cut losses disproportionately?
    x2["p_band"] = pd.cut(x2["model_p"], bins=[0.80, 0.85, 0.90, 0.95, 1.01],
                            labels=["80-85%", "85-90%", "90-95%", "95-100%"], right=False)
    breakdown(x2, "1x2", "p_band", "predicted-probability band")

    # 2. Loss rate by league
    breakdown(x2, "1x2", "league", "league")

    # 3. Loss rate by selection (H/D/A)
    breakdown(x2, "1x2", "selection", "selection type")

    # 4. Loss rate by year
    x2["year"] = x2["date"].dt.year
    breakdown(x2, "1x2", "year", "year")

    # 5. Margin analysis: for the H-selection losses specifically, was it close or a blowout?
    h_losses = x2[(x2["selection"] == "H") & (~x2["won"])].copy()
    h_losses["margin"] = h_losses["ftag"] - h_losses["fthg"]  # away goals - home goals, positive = away won by that much
    print(f"\n--- 1x2 selection=H losses (n={len(h_losses)}): margin of defeat/draw ---")
    print(h_losses["margin"].value_counts().sort_index().to_string())
    print(f"Draws: {(h_losses['margin']==0).sum()} ({(h_losses['margin']==0).mean():.1%})")
    print(f"Away win by 1: {(h_losses['margin']==1).sum()} ({(h_losses['margin']==1).mean():.1%})")
    print(f"Away win by 2+: {(h_losses['margin']>=2).sum()} ({(h_losses['margin']>=2).mean():.1%})")

    # 6. Odds level: are losses concentrated at the SHORTER-priced end (biggest "sure thing"
    #    favorites, where the market itself was most confident) or the LONGER-priced end
    #    (bets that only just cleared 80% and had more built-in doubt)?
    x2["odds_band"] = pd.cut(x2["odds"], bins=[1.0, 1.15, 1.30, 1.50, 10],
                               labels=["<=1.15", "1.15-1.30", "1.30-1.50", ">1.50"])
    breakdown(x2, "1x2", "odds_band", "odds band")

    # 7. Cross-check the probability-band pattern on the larger double_chance sample (real
    #    outcomes, just synthetic pricing -- the win/loss pattern itself is still real data)
    dc["p_band"] = pd.cut(dc["model_p"], bins=[0.80, 0.85, 0.90, 0.95, 1.01],
                            labels=["80-85%", "85-90%", "90-95%", "95-100%"], right=False)
    breakdown(dc, "double_chance", "p_band", "predicted-probability band (double_chance, n=12184, cross-check)")

    # significance check: does p_band actually relate to loss rate, or is the pattern noise?
    ct = pd.crosstab(x2["p_band"], x2["won"])
    chi2, p, dof, _ = chi2_contingency(ct)
    print(f"\nChi-square test, 1x2 loss rate vs probability band: chi2={chi2:.2f}, p={p:.4f}")
    ct_dc = pd.crosstab(dc["p_band"], dc["won"])
    chi2_dc, p_dc, dof_dc, _ = chi2_contingency(ct_dc)
    print(f"Chi-square test, double_chance loss rate vs probability band: chi2={chi2_dc:.2f}, p={p_dc:.4f}")

    # -- plots --
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    g = x2.groupby("p_band", observed=True).agg(hit_rate=("won", "mean"), n=("won", "size"))
    bars = ax.bar(g.index.astype(str), g["hit_rate"], color="#2166ac")
    ax.axhline(0.80, color="red", linestyle=":")
    for rect, nn in zip(bars, g["n"]):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02, f"n={nn}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("1x2: hit rate by predicted-probability band")

    ax = axes[0, 1]
    g = x2.groupby("league", observed=True).agg(hit_rate=("won", "mean"), n=("won", "size")).sort_values("n", ascending=False)
    bars = ax.bar(g.index.astype(str), g["hit_rate"], color="#2166ac")
    ax.axhline(0.80, color="red", linestyle=":")
    for rect, nn in zip(bars, g["n"]):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02, f"n={nn}", ha="center", fontsize=7, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("1x2: hit rate by league")

    ax = axes[0, 2]
    g = x2.groupby("year", observed=True).agg(hit_rate=("won", "mean"), n=("won", "size"))
    bars = ax.bar(g.index.astype(str), g["hit_rate"], color="#2166ac")
    ax.axhline(0.80, color="red", linestyle=":")
    for rect, nn in zip(bars, g["n"]):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02, f"{nn}", ha="center", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("1x2: hit rate by year")

    ax = axes[1, 0]
    g = x2.groupby("odds_band", observed=True).agg(hit_rate=("won", "mean"), n=("won", "size"))
    bars = ax.bar(g.index.astype(str), g["hit_rate"], color="#2166ac")
    ax.axhline(0.80, color="red", linestyle=":")
    for rect, nn in zip(bars, g["n"]):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02, f"n={nn}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("1x2: hit rate by odds band (shorter = bigger favorite)")

    ax = axes[1, 1]
    margin_counts = h_losses["margin"].clip(upper=4).value_counts().sort_index()
    ax.bar(margin_counts.index.astype(str), margin_counts.values, color="#d6604d")
    ax.set_title(f"Margin when an 'H' pick lost (n={len(h_losses)})\n0=draw, N=away won by N (4=4+)")
    ax.set_xlabel("Away margin of victory (0 = draw)")

    ax = axes[1, 2]
    g_dc = dc.groupby("p_band", observed=True).agg(hit_rate=("won", "mean"), n=("won", "size"))
    bars = ax.bar(g_dc.index.astype(str), g_dc["hit_rate"], color="#4393c3")
    ax.axhline(0.80, color="red", linestyle=":")
    for rect, nn in zip(bars, g_dc["n"]):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02, f"n={nn}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("double_chance (n=12184): hit rate by\nprobability band -- cross-check")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "loss_pattern_analysis.png", dpi=150)
    print(f"\nSaved plot -> {OUT_DIR / 'loss_pattern_analysis.png'}")


if __name__ == "__main__":
    main()
