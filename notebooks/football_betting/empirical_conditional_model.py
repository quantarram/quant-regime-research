"""
Pure empirical / non-parametric approach, as requested: no Poisson goal model,
no assumed score distribution. Just count what actually happened in the past
under comparable conditions, and use that raw historical frequency as the
probability estimate going forward.

Conditioning variable: each team's actual trailing form, measured as points
per game (PPG = (3*wins + 1*draws) / games) over their last N real results
-- the same number you'd get by literally looking at the league table over a
recent stretch. Home team's trailing PPG and away team's trailing PPG are each
bucketed; for every match, we look up "in the past, when a team in this form
bucket played at home against a team in that form bucket, what fraction of
the time did the home team actually win / draw / the away team actually win"
-- computed ONLY from matches that happened strictly before the match being
evaluated (walk-forward, no lookahead), pooled across all 13 leagues so each
bucket combination has enough real matches behind it to be meaningful.

Rule: only bet when that actual historical conditional probability is >= 80%
AND at least MIN_N comparable historical matches back it up (otherwise "80%"
is just noise from a handful of games). Real Singapore-Pools-relevant market
odds are still used to compute what betting on it would actually have paid --
because a historically-80%-likely outcome only makes for a good bet if the
payout beats the bookmaker's margin too, not just because it clears 80%.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import binomtest

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

STAKE_SGD = 50.0
PROB_THRESHOLD = 0.80
FORM_WINDOW = 10          # trailing matches used to compute a team's current PPG "form"
MIN_N = 30                # minimum comparable historical matches before trusting a bucket's rate
PPG_BINS = [0, 0.75, 1.25, 1.75, 2.25, 3.01]
PPG_LABELS = ["0-0.75 (poor)", "0.75-1.25", "1.25-1.75", "1.75-2.25", "2.25-3.0 (excellent)"]


def compute_trailing_ppg(matches):
    """For every match, attach each team's actual PPG over their last FORM_WINDOW matches
    (any competition/venue, using real results only, strictly before this match's date)."""
    matches = matches.sort_values("date").reset_index(drop=True)
    team_history = {}  # team -> list of points earned, chronological
    home_ppg, away_ppg = [], []

    for row in matches.itertuples():
        h, a = row.home_team, row.away_team
        h_hist = team_history.get(h, [])
        a_hist = team_history.get(a, [])
        home_ppg.append(np.mean(h_hist[-FORM_WINDOW:]) if len(h_hist) >= FORM_WINDOW else np.nan)
        away_ppg.append(np.mean(a_hist[-FORM_WINDOW:]) if len(a_hist) >= FORM_WINDOW else np.nan)

        if row.ftr == "H":
            h_pts, a_pts = 3, 0
        elif row.ftr == "A":
            h_pts, a_pts = 0, 3
        else:
            h_pts, a_pts = 1, 1
        team_history.setdefault(h, []).append(h_pts)
        team_history.setdefault(a, []).append(a_pts)

    matches["home_ppg"] = home_ppg
    matches["away_ppg"] = away_ppg
    return matches


def run():
    matches = pd.read_parquet(DATA_DIR / "matches.parquet")
    matches = compute_trailing_ppg(matches)
    matches = matches.dropna(subset=["home_ppg", "away_ppg"]).sort_values("date").reset_index(drop=True)
    matches["home_bucket"] = pd.cut(matches["home_ppg"], bins=PPG_BINS, labels=PPG_LABELS, include_lowest=True)
    matches["away_bucket"] = pd.cut(matches["away_ppg"], bins=PPG_BINS, labels=PPG_LABELS, include_lowest=True)

    # walk-forward empirical outcome table: (home_bucket, away_bucket) -> counts of H/D/A seen so far
    counts = {}  # key: (home_bucket, away_bucket) -> [n_H, n_D, n_A]
    rows = []

    for row in matches.itertuples():
        key = (row.home_bucket, row.away_bucket)
        c = counts.get(key)
        if c is not None and sum(c) >= MIN_N:
            n = sum(c)
            p_h, p_d, p_a = c[0] / n, c[1] / n, c[2] / n
            for sel, p, odds_col in [("H", p_h, row.odds_h), ("D", p_d, row.odds_d), ("A", p_a, row.odds_a)]:
                if p >= PROB_THRESHOLD and pd.notna(odds_col) and odds_col > 1:
                    won = (row.ftr == sel)
                    rows.append(dict(
                        date=row.date, league=row.league, home_team=row.home_team, away_team=row.away_team,
                        home_bucket=row.home_bucket, away_bucket=row.away_bucket, n_comparable=n,
                        selection=sel, empirical_p=p, odds=odds_col, won=won,
                        pnl=STAKE_SGD * (odds_col - 1) if won else -STAKE_SGD,
                    ))

        # update the table with this match's actual result (after prediction, so no lookahead)
        c = counts.setdefault(key, [0, 0, 0])
        c["HDA".index(row.ftr)] += 1

    bets = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    bets["cum_pnl"] = bets["pnl"].cumsum()
    return matches, bets


def report(bets):
    n = len(bets)
    hit_rate = bets["won"].mean() if n else np.nan
    total_pnl = bets["pnl"].sum() if n else 0.0
    roi = total_pnl / (STAKE_SGD * n) if n else np.nan

    print(f"Qualifying bets (empirical conditional P >= 80%, n_comparable >= {MIN_N}): {n}")
    print(f"Realized hit rate: {hit_rate:.1%}")
    print(f"Total P&L on S${STAKE_SGD:.0f} flat stakes: S${total_pnl:,.2f}")
    print(f"ROI per bet: {roi:.2%}")

    if n:
        test = binomtest(int(bets["won"].sum()), n, 0.80, alternative="greater")
        print(f"p-value (H0: true rate <= 80%): {test.pvalue:.4f}")
        avg_market_p = (1 / bets["odds"]).mean()  # rough, not devigged, just for context
        print(f"Average market-implied probability (raw, not devigged) on these same bets: {avg_market_p:.1%}")
        vig_note = bets["empirical_p"].mean() - avg_market_p
        print(f"Average (empirical_p - raw market-implied p): {vig_note:+.1%}")

        by_sel = bets.groupby("selection").agg(n=("won", "size"), hit_rate=("won", "mean"), pnl=("pnl", "sum"))
        by_sel["roi"] = by_sel["pnl"] / (STAKE_SGD * by_sel["n"])
        print("\nBy selection:")
        print(by_sel)

    bets.to_parquet(OUT_DIR / "empirical_qualifying_bets.parquet", index=False)
    return hit_rate, total_pnl, roi


def make_plots(matches, bets):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Heatmap: empirical P(home win) by (home_bucket, away_bucket), using ALL data
    #    (not walk-forward -- just to visualize the actual historical pattern)
    pivot = matches.groupby(["home_bucket", "away_bucket"], observed=True).apply(
        lambda g: (g["ftr"] == "H").mean(), include_groups=False
    ).unstack()
    pivot = pivot.reindex(index=PPG_LABELS, columns=PPG_LABELS)
    ax = axes[0, 0]
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(PPG_LABELS)))
    ax.set_xticklabels(PPG_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(PPG_LABELS)))
    ax.set_yticklabels(PPG_LABELS, fontsize=8)
    ax.set_xlabel("Away team trailing form")
    ax.set_ylabel("Home team trailing form")
    ax.set_title("Actual P(home win), full history\nby trailing-form bucket combo")
    for i in range(len(PPG_LABELS)):
        for j in range(len(PPG_LABELS)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v - 0.5) > 0.25 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 2. Calibration: predicted (empirical_p at bet time) vs realized, for ALL candidate
    #    bucket/outcome combos evaluated walk-forward (not just qualifying >=80% ones)
    ax = axes[0, 1]
    if len(bets):
        b = bets.copy()
        b["bucket"] = pd.cut(b["empirical_p"], bins=np.arange(0.75, 1.01, 0.02))
        calib = b.groupby("bucket", observed=True).agg(
            predicted=("empirical_p", "mean"), realized=("won", "mean"), n=("won", "size")
        ).dropna()
        ax.plot([0.75, 1], [0.75, 1], "k--", alpha=0.5, label="perfect calibration")
        sizes = np.clip(calib["n"] / max(calib["n"].max(), 1) * 300, 10, 300)
        ax.scatter(calib["predicted"], calib["realized"], s=sizes, alpha=0.7, color="#2166ac")
    ax.axvline(0.80, color="red", linestyle=":", label="80% threshold")
    ax.set_xlim(0.75, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Empirical (historical) conditional probability")
    ax.set_ylabel("Realized frequency, forward")
    ax.set_title("Calibration of the >=80% empirical rule\n(qualifying bets only, zoomed to 0.75-1.0)")
    ax.legend()

    # 3. Cumulative P&L
    ax = axes[1, 0]
    if len(bets):
        ax.plot(bets["date"], bets["cum_pnl"], color="#2166ac")
        ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(f"Cumulative P&L: empirical P>=80% rule (S${STAKE_SGD:.0f} flat stake)")
    ax.set_ylabel("Cumulative P&L (SGD)")
    ax.tick_params(axis="x", rotation=30)

    # 4. Hit rate by selection
    ax = axes[1, 1]
    if len(bets):
        by_sel = bets.groupby("selection").agg(hit_rate=("won", "mean"), n=("won", "size"))
        bars = ax.bar(by_sel.index, by_sel["hit_rate"], color="#4393c3")
        ax.axhline(0.80, color="red", linestyle=":", label="80% threshold")
        for rect, nn in zip(bars, by_sel["n"]):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02, f"n={nn}",
                    ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Realized hit rate by selection (H/D/A)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "empirical_model_summary.png", dpi=150)
    print(f"\nSaved plots -> {OUT_DIR / 'empirical_model_summary.png'}")


if __name__ == "__main__":
    matches, bets = run()
    report(bets)
    make_plots(matches, bets)
