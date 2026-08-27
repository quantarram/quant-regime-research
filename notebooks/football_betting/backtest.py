"""
Walk-forward backtest of the rule: bet only when model probability > 80% AND
the model beats the market's devigged (overround-stripped) probability.

For every match, the model is fit ONLY on that league's matches strictly
before it (rolling, refit every N days to keep runtime sane) -- no lookahead.
Stakes are flat at STAKE_SGD (<=50 SGD per the user's own rule), and results
are reported per market type since hit-rate/edge differs a lot between e.g.
double-chance and 1X2.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dixon_coles import DixonColesModel, implied_prob_devigged, ah_settle
from data_download import CORE_LEAGUES

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

STAKE_SGD = 50.0
PROB_THRESHOLD = 0.90
EDGE_THRESHOLD = 0.0   # require model prob to exceed devigged market prob by at least this much
REFIT_EVERY_DAYS = 14  # refit each league's model every 2 weeks rather than every single match (speed)
MIN_TRAIN_MATCHES = 380  # >= 1 season of history before a league enters the backtest


def build_market_rows(match, probs, model):
    """Turn one match's model probs + market odds into candidate (market, selection, prob, odds, won) rows."""
    rows = []
    h, d, a = match["odds_h"], match["odds_d"], match["odds_a"]
    if pd.notna(h) and pd.notna(d) and pd.notna(a) and min(h, d, a) > 1:
        mkt = implied_prob_devigged([h, d, a])
        outcomes = ["H", "D", "A"]
        odds = [h, d, a]
        won_outcome = match["ftr"]
        for o, mp, mo in zip(outcomes, mkt, odds):
            rows.append(dict(market="1x2", selection=o, model_p=probs["1x2"][o],
                              market_p=mp, odds=mo, won=(o == won_outcome)))
        # double chance derived from same 1x2 odds (devigged) by summing pairs
        dc_pairs = {"1X": ("H", "D"), "X2": ("D", "A"), "12": ("H", "A")}
        for sel, (o1, o2) in dc_pairs.items():
            mp_dc = mkt[outcomes.index(o1)] + mkt[outcomes.index(o2)]
            won_dc = won_outcome in (o1, o2)
            # approximate double-chance payout from devigged prob (no direct market odds needed
            # for a probability-threshold rule check, but we still need *some* price to size P&L;
            # use 1/mp_dc as a fair-ish price proxy net of vig, since DC vig is typically small)
            rows.append(dict(market="double_chance", selection=sel, model_p=probs["double_chance"][sel],
                              market_p=mp_dc, odds=1.0 / mp_dc if mp_dc > 0 else np.nan, won=won_dc))

    ou, uu = match["odds_over25"], match["odds_under25"]
    if pd.notna(ou) and pd.notna(uu) and min(ou, uu) > 1:
        mkt = implied_prob_devigged([ou, uu])
        total_goals = match["fthg"] + match["ftag"]
        won_over = total_goals > 2.5
        rows.append(dict(market="over_under_2.5", selection="O", model_p=probs["over_under_2.5"]["O"],
                          market_p=mkt[0], odds=ou, won=won_over))
        rows.append(dict(market="over_under_2.5", selection="U", model_p=probs["over_under_2.5"]["U"],
                          market_p=mkt[1], odds=uu, won=(not won_over)))

    # General Asian handicap: use whatever line the market actually quoted for this match
    # (usually not 0 -- see ah_line distribution, most matches get -0.25 to -1 favoring the
    # favorite). Settlement (win/push/half-win/half-loss) handled by ah_settle for arbitrary
    # quarter/half/whole lines; the line itself is kept on each row so results can be bucketed
    # by |line| afterwards to see whether lines away from 0 behave differently.
    ah_h, ah_a, ah_line = match["odds_ah_home"], match["odds_ah_away"], match["ah_line"]
    if pd.notna(ah_h) and pd.notna(ah_a) and pd.notna(ah_line) and min(ah_h, ah_a) > 1:
        ah_probs = model.ah_probs(match["home_team"], match["away_team"], ah_line)
        if ah_probs is not None and pd.notna(ah_probs["H"]) and pd.notna(ah_probs["A"]):
            mkt = implied_prob_devigged([ah_h, ah_a])
            diff = match["fthg"] - match["ftag"]
            s_home = ah_settle(diff, ah_line)
            if s_home != 0:  # drop full pushes from the candidate sample, as elsewhere
                won_home = s_home > 0
                rows.append(dict(market="asian_handicap", selection="H", model_p=ah_probs["H"],
                                  market_p=mkt[0], odds=ah_h, won=won_home, ah_line=ah_line,
                                  settle_frac=s_home))
                rows.append(dict(market="asian_handicap", selection="A", model_p=ah_probs["A"],
                                  market_p=mkt[1], odds=ah_a, won=(not won_home), ah_line=ah_line,
                                  settle_frac=-s_home))
    return rows


def run_backtest():
    matches = pd.read_parquet(DATA_DIR / "matches.parquet")
    # Restricted to the validated deep/liquid league set -- see CORE_LEAGUES in
    # data_download.py for why: the wider 37-league set was tested and actively
    # destroys the edge (thin/non-European markets are far less well-calibrated).
    matches = matches[matches["league"].isin(CORE_LEAGUES)]
    matches = matches.sort_values("date").reset_index(drop=True)

    all_rows = []
    for league, lg_df in matches.groupby("league"):
        lg_df = lg_df.sort_values("date").reset_index(drop=True)
        if len(lg_df) < MIN_TRAIN_MATCHES + 50:
            continue
        print(f"Backtesting {league} ({len(lg_df)} matches)...")

        model = None
        last_fit_date = None
        for i in range(MIN_TRAIN_MATCHES, len(lg_df)):
            match = lg_df.iloc[i]
            train = lg_df.iloc[:i]

            if model is None or last_fit_date is None or (match["date"] - last_fit_date).days >= REFIT_EVERY_DAYS:
                model = DixonColesModel().fit(train)
                last_fit_date = match["date"]

            probs = model.match_probs(match["home_team"], match["away_team"])
            if probs is None:
                continue
            for row in build_market_rows(match, probs, model):
                row.update(league=league, date=match["date"],
                           home_team=match["home_team"], away_team=match["away_team"])
                all_rows.append(row)

    bets_df = pd.DataFrame(all_rows)
    bets_df.to_parquet(OUT_DIR / "all_candidate_bets.parquet", index=False)
    print(f"\n{len(bets_df)} candidate (market, selection) rows evaluated across all matches.")
    return bets_df


def apply_strategy_and_report(bets_df):
    bets_df = bets_df.dropna(subset=["model_p", "market_p", "odds"]).copy()
    qualifying = bets_df[
        (bets_df["model_p"] >= PROB_THRESHOLD) &
        (bets_df["model_p"] - bets_df["market_p"] >= EDGE_THRESHOLD)
    ].copy()
    qualifying = qualifying.sort_values("date").reset_index(drop=True)

    if "settle_frac" not in qualifying.columns:
        qualifying["settle_frac"] = np.nan
    sf = qualifying["settle_frac"].where(
        qualifying["settle_frac"].notna(), np.where(qualifying["won"], 1.0, -1.0)
    )
    qualifying["pnl"] = STAKE_SGD * (np.maximum(sf, 0) * (qualifying["odds"] - 1) + np.minimum(sf, 0))
    qualifying["cum_pnl"] = qualifying["pnl"].cumsum()
    qualifying["cum_staked"] = STAKE_SGD * (qualifying.index + 1)

    n_bets = len(qualifying)
    hit_rate = qualifying["won"].mean() if n_bets else np.nan
    total_pnl = qualifying["pnl"].sum() if n_bets else 0.0
    roi = total_pnl / (STAKE_SGD * n_bets) if n_bets else np.nan

    print(f"\n=== Strategy result: model P >= {PROB_THRESHOLD:.0%}, model edge over devigged market >= 0 ===")
    print(f"Qualifying bets: {n_bets}")
    print(f"Realized hit rate: {hit_rate:.1%}  (should be >={PROB_THRESHOLD:.0%} if the model is well-calibrated)")
    print(f"Total P&L on S${STAKE_SGD:.0f} flat stakes: S${total_pnl:,.2f}")
    print(f"ROI per bet: {roi:.2%}")
    print("\nBy market:")
    by_mkt = qualifying.groupby("market").agg(
        n=("won", "size"), hit_rate=("won", "mean"), pnl=("pnl", "sum")
    )
    by_mkt["roi"] = by_mkt["pnl"] / (STAKE_SGD * by_mkt["n"])
    print(by_mkt)
    if "double_chance" in by_mkt.index:
        print("\nNOTE: double_chance has no real historical odds source (football-data.co.uk doesn't\n"
              "publish it), so its 'odds' are a synthetic no-vig price derived from the 1X2 devigged\n"
              "probabilities. Its hit-rate is a valid calibration check, but its P&L/ROI ignores the\n"
              "bookmaker's actual margin on that market and should be read as an upper bound, not a\n"
              "realistic return.")

    by_mkt.to_csv(OUT_DIR / "results_by_market.csv")
    qualifying.to_parquet(OUT_DIR / "qualifying_bets.parquet", index=False)

    ah_bucket_summary = None
    if "asian_handicap" in qualifying["market"].values:
        ah = qualifying[qualifying["market"] == "asian_handicap"].copy()
        ah["abs_line"] = ah["ah_line"].abs()
        bins = [-0.01, 0.01, 0.5, 1.0, 1.5, 10]
        labels = ["0 (DNB-equivalent)", "0.25-0.5", "0.75-1.0", "1.25-1.5", "1.75+"]
        ah["line_bucket"] = pd.cut(ah["abs_line"], bins=bins, labels=labels)
        ah_bucket_summary = ah.groupby("line_bucket", observed=True).agg(
            n=("won", "size"), hit_rate=("won", "mean"), pnl=("pnl", "sum")
        )
        ah_bucket_summary["roi"] = ah_bucket_summary["pnl"] / (STAKE_SGD * ah_bucket_summary["n"])
        print("\n=== Asian handicap: qualifying bets by |line| bucket (0 = DNB-equivalent) ===")
        print(ah_bucket_summary)
        ah_bucket_summary.to_csv(OUT_DIR / "ah_by_line_bucket.csv")
        ah.to_parquet(OUT_DIR / "ah_qualifying_bets.parquet", index=False)

    return qualifying, by_mkt, ah_bucket_summary


def make_plots(bets_df, qualifying):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Calibration plot: predicted prob bucket vs realized frequency, ALL candidate bets
    #    (not just the >=80% ones) so you can see calibration across the whole probability range
    valid = bets_df.dropna(subset=["model_p", "won"]).copy()
    valid["bucket"] = pd.cut(valid["model_p"], bins=np.arange(0, 1.05, 0.05))
    calib = valid.groupby("bucket", observed=True).agg(
        predicted=("model_p", "mean"), realized=("won", "mean"), n=("won", "size")
    ).dropna()
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    sizes = np.clip(calib["n"] / calib["n"].max() * 300, 10, 300)
    ax.scatter(calib["predicted"], calib["realized"], s=sizes, alpha=0.7, color="#2166ac")
    ax.axvline(PROB_THRESHOLD, color="red", linestyle=":", label=f"{PROB_THRESHOLD:.0%} threshold")
    ax.set_xlabel("Model predicted probability")
    ax.set_ylabel("Realized frequency")
    ax.set_title("Calibration: all candidate bets (all markets)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 2. Cumulative P&L of the qualifying (>=80%) strategy over time
    ax = axes[0, 1]
    if len(qualifying):
        ax.plot(qualifying["date"], qualifying["cum_pnl"], color="#2166ac")
        ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(f"Cumulative P&L: model P>={PROB_THRESHOLD:.0%} strategy (S${STAKE_SGD:.0f} flat stake)")
    ax.set_ylabel("Cumulative P&L (SGD)")
    ax.tick_params(axis="x", rotation=30)

    # 3. Hit rate and bet count by market, for qualifying bets only
    ax = axes[1, 0]
    if len(qualifying):
        by_mkt = qualifying.groupby("market").agg(hit_rate=("won", "mean"), n=("won", "size"))
        bars = ax.bar(by_mkt.index, by_mkt["hit_rate"], color="#4393c3")
        ax.axhline(PROB_THRESHOLD, color="red", linestyle=":", label=f"{PROB_THRESHOLD:.0%} threshold")
        for rect, n in zip(bars, by_mkt["n"]):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02, f"n={n}",
                    ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Realized hit rate by market (qualifying bets only)")
    ax.legend()

    # 4. Distribution of odds actually available on qualifying (>=80% prob) bets
    ax = axes[1, 1]
    if len(qualifying):
        ax.hist(qualifying["odds"], bins=30, color="#4393c3", edgecolor="white")
    ax.set_title("Decimal odds distribution on qualifying bets")
    ax.set_xlabel("Decimal odds")
    ax.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "backtest_summary.png", dpi=150)
    print(f"\nSaved plots -> {OUT_DIR / 'backtest_summary.png'}")


def make_ah_line_plot(ah_bucket_summary, bets_df):
    if ah_bucket_summary is None or not len(ah_bucket_summary):
        return
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    bars = ax.bar(ah_bucket_summary.index.astype(str), ah_bucket_summary["hit_rate"], color="#4393c3")
    ax.axhline(PROB_THRESHOLD, color="red", linestyle=":", label=f"{PROB_THRESHOLD:.0%} threshold")
    for rect, n in zip(bars, ah_bucket_summary["n"]):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02, f"n={n}",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Asian handicap: realized hit rate by |line|\n(qualifying bets: model P>={PROB_THRESHOLD:.0%})")
    ax.set_xlabel("|handicap line|")
    ax.legend()

    ax = axes[1]
    bars = ax.bar(ah_bucket_summary.index.astype(str), ah_bucket_summary["roi"] * 100, color="#4393c3")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Asian handicap: ROI per bet by |line|")
    ax.set_ylabel("ROI (%)")
    ax.set_xlabel("|handicap line|")

    # calibration specifically for the AH market across its FULL probability range (not just
    # qualifying bets) so you can see whether AH calibration differs from 1x2/O-U generally
    ax = axes[2]
    ah_all = bets_df[bets_df["market"] == "asian_handicap"].dropna(subset=["model_p", "won"]).copy()
    ah_all["bucket"] = pd.cut(ah_all["model_p"], bins=np.arange(0, 1.05, 0.05))
    calib = ah_all.groupby("bucket", observed=True).agg(
        predicted=("model_p", "mean"), realized=("won", "mean"), n=("won", "size")
    ).dropna()
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    sizes = np.clip(calib["n"] / calib["n"].max() * 300, 10, 300) if len(calib) else []
    ax.scatter(calib["predicted"], calib["realized"], s=sizes, alpha=0.7, color="#2166ac")
    ax.axvline(PROB_THRESHOLD, color="red", linestyle=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model predicted probability")
    ax.set_ylabel("Realized frequency")
    ax.set_title("Asian handicap: calibration (all candidate bets)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "ah_line_analysis.png", dpi=150)
    print(f"Saved plots -> {OUT_DIR / 'ah_line_analysis.png'}")


if __name__ == "__main__":
    bets_df = run_backtest()
    qualifying, by_mkt, ah_bucket_summary = apply_strategy_and_report(bets_df)
    make_plots(bets_df, qualifying)
    make_ah_line_plot(ah_bucket_summary, bets_df)
