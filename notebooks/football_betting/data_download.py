"""
Pull historical results + closing odds from football-data.co.uk (free, public CSVs).

Two source formats on the same site:
  1. "mmz4281" seasonal files -- one CSV per league per season, rich odds
     columns (1x2, over/under 2.5, Asian handicap). Originally just the
     top-5 Europe + EFL Championship + Scottish Prem + Eredivisie +
     Portugal/Turkey/Greece/Belgium (checked live against SG Pools' board);
     widened here to also include the English/Scottish LOWER tiers
     (League One/Two, Scottish Championship/League One/Two), which use the
     identical column format.
  2. "new/<COUNTRY>.csv" extra-country files -- one combined CSV per
     country covering all seasons at once, 1x2 odds only (no AH/O-U
     columns at all). This is where the non-European leagues actually seen
     on Singapore Pools' live board live: Japan (J-League), Mexico (Liga
     MX), USA (MLS), Brazil, Argentina, China, Russia, plus several more
     European countries (Austria, Denmark, Finland, Ireland, Norway,
     Poland, Romania, Sweden, Switzerland) not in the mmz4281 set.
     Checked and confirmed live: Korea (K-League) and Chile are NOT
     available from this source (both redirect/404) -- still excluded,
     for the same reason as before: no free historical-odds source.
"""
import io
import time
import warnings
import requests
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", message="Could not infer format")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Closed seasons never change once football-data.co.uk has posted a season's final
# results, so caching them locally turns refresh_core_leagues()'s daily re-fetch from
# 13 leagues x 11 seasons = 143 HTTP requests into just 13 (only the live season still
# needs a fresh pull). See refresh_core_leagues() for how this is used.
SEASON_CACHE_DIR = DATA_DIR / "season_cache"
SEASON_CACHE_DIR.mkdir(exist_ok=True)

LEAGUES = {
    "E0": "England - Premier League",
    "E1": "England - Championship",
    "E2": "England - League One",
    "E3": "England - League Two",
    "EC": "England - National League",
    "D1": "Germany - Bundesliga",
    "D2": "Germany - 2. Bundesliga",
    "I1": "Italy - Serie A",
    "SP1": "Spain - La Liga",
    "F1": "France - Ligue 1",
    "N1": "Netherlands - Eredivisie",
    "P1": "Portugal - Primeira Liga",
    "SC0": "Scotland - Premiership",
    "SC1": "Scotland - Championship",
    "SC2": "Scotland - League One",
    "SC3": "Scotland - League Two",
    "T1": "Turkey - Super Lig",
    "G1": "Greece - Super League",
    "B1": "Belgium - Pro League",
}

# Validated production set. The full 37-league backtest at the 90% threshold showed the
# edge is real ONLY here (n=183, hit rate 91.8%, ROI +5.22%, holds up in a discovery/holdout
# split) and actively breaks when pooled with the other 24 leagues (that combined group: n=112,
# hit rate 58.9%, ROI -13.9% -- e.g. England League Two hit only 33% on bets the model called
# 90%+ likely). The common thread among these 13 is deep, liquid, heavily-traded markets
# (top-5 Europe, EFL Championship, Scottish Premiership, Eredivisie, Portugal/Turkey/Greece/
# Belgium's top flights) -- lower divisions and thinner non-European markets are excluded not
# because the data is unavailable, but because the strategy demonstrably doesn't transfer to
# them. backtest.py and live_checklist.py both filter to this set by default; EXTRA_COUNTRIES
# and the English/Scottish lower tiers stay in LEAGUES/matches.parquet for reference and future
# comparison, but are not part of the validated, tradeable configuration.
CORE_LEAGUES = ("E0", "E1", "D1", "D2", "I1", "SP1", "F1", "N1", "P1", "SC0", "T1", "G1", "B1")

EXTRA_COUNTRIES = {
    "ARG": "Argentina", "AUT": "Austria", "BRA": "Brazil", "CHN": "China",
    "DNK": "Denmark", "FIN": "Finland", "IRL": "Ireland", "JPN": "Japan",
    "MEX": "Mexico", "NOR": "Norway", "POL": "Poland", "ROU": "Romania",
    "RUS": "Russia", "SWE": "Sweden", "SWZ": "Switzerland", "USA": "USA (MLS)",
}

# season codes: "2324" = 2023-24. football-data.co.uk keeps ~ the last 10-12 seasons at this path.
SEASONS = ["1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
EXTRA_URL = "https://www.football-data.co.uk/new/{country}.csv"

KEEP_COLS = [
    "Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "B365H", "B365D", "B365A",
    "PSH", "PSD", "PSA",
    "AvgH", "AvgD", "AvgA",
    "Avg>2.5", "Avg<2.5", "AvgAHH", "AvgAHA", "AHh",
    "BbAvH", "BbAvD", "BbAvA", "BbAv>2.5", "BbAv<2.5", "BbAvAHH", "BbAvAHA", "BbAHh",
]


def _pick(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(index=df.index, dtype=float)


def fetch_one(league_code, season):
    url = BASE_URL.format(season=season, league=league_code)
    r = requests.get(url, timeout=20)
    if r.status_code != 200 or len(r.content) < 200:
        return None
    try:
        df = pd.read_csv(io.BytesIO(r.content), encoding="latin1", on_bad_lines="skip")
    except Exception:
        return None
    if "HomeTeam" not in df.columns or "FTR" not in df.columns:
        return None
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
    out = pd.DataFrame({
        "league": league_code,
        "season": season,
        "date": pd.to_datetime(df.get("Date"), dayfirst=True, errors="coerce"),
        "home_team": df["HomeTeam"].str.strip(),
        "away_team": df["AwayTeam"].str.strip(),
        "fthg": pd.to_numeric(df.get("FTHG"), errors="coerce"),
        "ftag": pd.to_numeric(df.get("FTAG"), errors="coerce"),
        "ftr": df["FTR"],
        # closing/average odds -- prefer Pinnacle (sharpest), fall back to market average, then Bet365
        "odds_h": pd.to_numeric(_pick(df, "PSH", "AvgH", "BbAvH", "B365H"), errors="coerce"),
        "odds_d": pd.to_numeric(_pick(df, "PSD", "AvgD", "BbAvD", "B365D"), errors="coerce"),
        "odds_a": pd.to_numeric(_pick(df, "PSA", "AvgA", "BbAvA", "B365A"), errors="coerce"),
        "odds_over25": pd.to_numeric(_pick(df, "Avg>2.5", "BbAv>2.5"), errors="coerce"),
        "odds_under25": pd.to_numeric(_pick(df, "Avg<2.5", "BbAv<2.5"), errors="coerce"),
        "ah_line": pd.to_numeric(_pick(df, "AHh", "BbAHh"), errors="coerce"),
        "odds_ah_home": pd.to_numeric(_pick(df, "AvgAHH", "BbAvAHH"), errors="coerce"),
        "odds_ah_away": pd.to_numeric(_pick(df, "AvgAHA", "BbAvAHA"), errors="coerce"),
    })
    out = out.dropna(subset=["date", "fthg", "ftag"])
    return out


def fetch_extra_country(country_code):
    """The 'new/<COUNTRY>.csv' format: one file, all seasons, 1x2 odds only
    (columns: Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PSCH/D/A,
    MaxCH/D/A,AvgCH/D/A,BFECH/D/A,B365CH/D/A -- the 'C' suffix means closing).
    No Asian handicap or over/under columns exist in this format at all."""
    url = EXTRA_URL.format(country=country_code)
    r = requests.get(url, timeout=30)
    if r.status_code != 200 or len(r.content) < 200:
        return None
    try:
        df = pd.read_csv(io.BytesIO(r.content), encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        return None
    if "Home" not in df.columns or "Res" not in df.columns:
        return None
    df = df.dropna(subset=["Home", "Away", "Res"])
    # "League" holds the actual competition name (e.g. "J1 League" vs "J2 League" within JPN.csv,
    # multiple divisions can share one country file) -- keep it as the league code so a country's
    # divisions don't get pooled together into one Dixon-Coles team-strength model incorrectly.
    league_col = country_code + "_" + df["League"].astype(str).str.strip().str.replace(r"\s+", "", regex=True)
    out = pd.DataFrame({
        "league": league_col,
        "season": df["Season"].astype(str),
        "date": pd.to_datetime(df.get("Date"), dayfirst=True, errors="coerce"),
        "home_team": df["Home"].astype(str).str.strip(),
        "away_team": df["Away"].astype(str).str.strip(),
        "fthg": pd.to_numeric(df.get("HG"), errors="coerce"),
        "ftag": pd.to_numeric(df.get("AG"), errors="coerce"),
        "ftr": df["Res"],
        "odds_h": pd.to_numeric(_pick(df, "PSCH", "AvgCH", "B365CH"), errors="coerce"),
        "odds_d": pd.to_numeric(_pick(df, "PSCD", "AvgCD", "B365CD"), errors="coerce"),
        "odds_a": pd.to_numeric(_pick(df, "PSCA", "AvgCA", "B365CA"), errors="coerce"),
        "odds_over25": pd.Series(index=df.index, dtype=float),
        "odds_under25": pd.Series(index=df.index, dtype=float),
        "ah_line": pd.Series(index=df.index, dtype=float),
        "odds_ah_home": pd.Series(index=df.index, dtype=float),
        "odds_ah_away": pd.Series(index=df.index, dtype=float),
    })
    out = out.dropna(subset=["date", "fthg", "ftag"])
    return out


def _fetch_one_cached(league_code, season):
    """Like fetch_one(), but closed seasons (every SEASONS entry except the current,
    live one) are read from a local parquet cache after their first fetch instead of
    hitting the network again -- their results are final and never change. Only the
    live season (SEASONS[-1]) always fetches fresh, since it's still being played."""
    is_live_season = season == SEASONS[-1]
    cache_path = SEASON_CACHE_DIR / f"{league_code}_{season}.parquet"
    if not is_live_season and cache_path.exists():
        return pd.read_parquet(cache_path)

    df = fetch_one(league_code, season)
    time.sleep(0.1)  # be polite to a free public data source -- only reached on an actual HTTP fetch

    if not is_live_season and df is not None and len(df):
        df.to_parquet(cache_path, index=False)
    return df


def refresh_core_leagues():
    """Fast daily refresh: re-fetch ONLY CORE_LEAGUES (13 leagues, not the full 37)
    and overwrite matches.parquet with just that -- the other 24 leagues were tested
    and don't survive holdout validation (see CORE_LEAGUES' own comment), so there's
    no reason to keep re-fetching them daily. Called from daily_dashboard.py before
    every run, so trailing form and thresholds are always computed from results as
    of today, not a stale one-time download.

    Closed seasons are served from SEASON_CACHE_DIR after their first fetch (see
    _fetch_one_cached) -- only the current season is re-downloaded every run, cutting
    this from 143 HTTP requests/day down to 13 once the cache is warm."""
    frames = []
    for league_code, league_name in LEAGUES.items():
        if league_code not in CORE_LEAGUES:
            continue
        for season in SEASONS:
            df = _fetch_one_cached(league_code, season)
            if df is not None and len(df):
                frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(["league", "date"]).reset_index(drop=True)
    all_df.to_parquet(DATA_DIR / "matches.parquet", index=False)
    return all_df


def main():
    frames = []
    for league_code, league_name in LEAGUES.items():
        for season in SEASONS:
            df = fetch_one(league_code, season)
            if df is not None and len(df):
                frames.append(df)
                print(f"{league_name:30s} {season}  {len(df):4d} matches")
            time.sleep(0.15)  # be polite to a free public data source

    for country_code, country_name in EXTRA_COUNTRIES.items():
        df = fetch_extra_country(country_code)
        if df is not None and len(df):
            frames.append(df)
            for lg, n in df.groupby("league").size().items():
                print(f"{country_name:30s} {lg:30s} {n:4d} matches")
        time.sleep(0.15)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(["league", "date"]).reset_index(drop=True)
    out_path = DATA_DIR / "matches.parquet"
    all_df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(all_df)} matches across {all_df['league'].nunique()} leagues -> {out_path}")
    print(all_df.groupby("league").size())


if __name__ == "__main__":
    main()
