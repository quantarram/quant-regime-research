"""
Dixon-Coles Poisson model for football match outcomes.

Standard approach (Dixon & Coles 1997): each team has an attack strength and a
defense strength; home team gets a fixed home-advantage multiplier. Expected
goals for home/away are Poisson rates built from these. A low-score
correlation adjustment (rho) corrects the well-known Poisson independence
failure for 0-0/1-0/0-1/1-1 scorelines. Older matches are down-weighted with
an exponential time decay so the model tracks current form rather than
treating a team's whole history as equally informative.

Fit is walk-forward per league: to predict any given match, only matches
strictly before it (within that league) are used, so the backtest has no
lookahead.
"""
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


def _settle_half(diff, line):
    """line must be a multiple of 0.5. Returns +1 (win), -1 (loss), or 0 (push, integer lines only)."""
    adj = diff + line
    if adj > 0:
        return 1.0
    elif adj < 0:
        return -1.0
    return 0.0


def ah_settle(diff, line):
    """General Asian handicap settlement for the home side, from the home team's goal
    difference (diff = home_goals - away_goals) against `line` (home's handicap, e.g. -1.5
    means home must win by 2+). Quarter lines (-0.25, -0.75, ...) are split into two equal
    half-stakes on the neighboring half-lines, matching how Asian handicap actually settles.
    Returns one of {-1, -0.5, 0, 0.5, 1}. The away side's settlement is always -ah_settle(...)."""
    l1 = math.floor(line * 2) / 2
    l2 = math.ceil(line * 2) / 2
    if l1 == l2:
        return _settle_half(diff, l1)
    return 0.5 * _settle_half(diff, l1) + 0.5 * _settle_half(diff, l2)


def _rho_correction(x, y, lam, mu, rho):
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


class DixonColesModel:
    def __init__(self, xi=0.0018):
        # xi: daily exponential time-decay rate (~half-life of a bit over a year;
        # standard Dixon-Coles range is 0.001-0.003)
        self.xi = xi
        self.teams = []
        self.params = None
        self.fit_date = None

    def fit(self, df):
        """df: columns home_team, away_team, fthg, ftag, date. Fits on everything given."""
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.teams = teams
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}

        max_date = df["date"].max()
        days_ago = (max_date - df["date"]).dt.days.values
        weights = np.exp(-self.xi * days_ago)

        home_idx = df["home_team"].map(idx).values
        away_idx = df["away_team"].map(idx).values
        hg = df["fthg"].values.astype(int)
        ag = df["ftag"].values.astype(int)

        def unpack(params):
            attack = params[:n]
            defense = params[n:2 * n]
            home_adv = params[2 * n]
            rho = params[2 * n + 1]
            return attack, defense, home_adv, rho

        def neg_log_lik(params):
            attack, defense, home_adv, rho = unpack(params)
            lam = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
            mu = np.exp(attack[away_idx] + defense[home_idx])
            ll = poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
            # low-score correlation adjustment, only meaningfully applies to 0/1 scorelines
            corr = np.ones(len(hg))
            low_mask = (hg <= 1) & (ag <= 1)
            for i in np.where(low_mask)[0]:
                c = _rho_correction(hg[i], ag[i], lam[i], mu[i], rho)
                corr[i] = max(c, 1e-6)
            ll = ll + np.log(corr)
            return -np.sum(weights * ll)

        x0 = np.concatenate([
            np.zeros(n),  # attack
            np.zeros(n),  # defense
            [0.25],       # home_adv (log-scale, ~ +28% goal rate at home)
            [0.0],        # rho
        ])

        # constrain average attack to 0 to make params identifiable (added as penalty)
        def objective(params):
            attack = params[:n]
            penalty = 1000 * (np.mean(attack)) ** 2
            return neg_log_lik(params) + penalty

        res = minimize(objective, x0, method="L-BFGS-B",
                        options={"maxiter": 300, "ftol": 1e-8})
        self.params = res.x
        self.fit_date = max_date
        return self

    def _team_strengths(self):
        n = len(self.teams)
        attack = self.params[:n]
        defense = self.params[n:2 * n]
        home_adv = self.params[2 * n]
        rho = self.params[2 * n + 1]
        return attack, defense, home_adv, rho

    def score_matrix(self, home_team, away_team):
        if home_team not in self.teams or away_team not in self.teams:
            return None
        attack, defense, home_adv, rho = self._team_strengths()
        i, j = self.teams.index(home_team), self.teams.index(away_team)
        lam = np.exp(attack[i] + defense[j] + home_adv)
        mu = np.exp(attack[j] + defense[i])
        lam = np.clip(lam, 1e-3, 8)
        mu = np.clip(mu, 1e-3, 8)
        hg = np.arange(0, MAX_GOALS + 1)
        ag = np.arange(0, MAX_GOALS + 1)
        ph = poisson.pmf(hg, lam)
        pa = poisson.pmf(ag, mu)
        mat = np.outer(ph, pa)
        for x in range(2):
            for y in range(2):
                mat[x, y] *= _rho_correction(x, y, lam, mu, rho)
        mat = mat / mat.sum()
        return mat

    def match_probs(self, home_team, away_team):
        """Returns a dict of market -> {selection: probability} for a single match."""
        mat = self.score_matrix(home_team, away_team)
        if mat is None:
            return None
        n = mat.shape[0]
        gi, gj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

        p_home = mat[gi > gj].sum()
        p_draw = mat[gi == gj].sum()
        p_away = mat[gi < gj].sum()

        p_over25 = mat[(gi + gj) > 2.5].sum()
        p_under25 = 1 - p_over25

        # Asian handicap at 0 (= draw-no-bet): home win outright wins, draw pushes (stake back),
        # away win outright wins. Probability expressed conditional on not pushing, matching how
        # a DNB/AH0 bet actually settles.
        denom = p_home + p_away
        p_ah0_home = p_home / denom if denom > 0 else np.nan
        p_ah0_away = p_away / denom if denom > 0 else np.nan

        return {
            "1x2": {"H": p_home, "D": p_draw, "A": p_away},
            "double_chance": {"1X": p_home + p_draw, "X2": p_draw + p_away, "12": p_home + p_away},
            "over_under_2.5": {"O": p_over25, "U": p_under25},
            "ah0_dnb": {"H": p_ah0_home, "A": p_ah0_away},
        }

    def ah_probs(self, home_team, away_team, line):
        """Model probability of covering an arbitrary Asian handicap `line` (home team's
        perspective, e.g. -1.5). Returns {'H': p_home_covers, 'A': p_away_covers, 'push': p_push},
        where p_home_covers/p_away_covers are conditional on not pushing (comparable to how
        1x2/DNB probabilities are reported elsewhere in this module)."""
        mat = self.score_matrix(home_team, away_team)
        if mat is None:
            return None
        n = mat.shape[0]
        gi, gj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        diff = gi - gj
        s = np.vectorize(lambda d: ah_settle(d, line))(diff)
        p_full_win = mat[s == 1].sum()
        p_half_win = mat[s == 0.5].sum()
        p_push = mat[s == 0].sum()
        p_half_loss = mat[s == -0.5].sum()
        p_full_loss = mat[s == -1].sum()
        denom = 1 - p_push
        p_home_covers = (p_full_win + 0.5 * p_half_win) / denom if denom > 0 else np.nan
        p_away_covers = (p_full_loss + 0.5 * p_half_loss) / denom if denom > 0 else np.nan
        return {"H": p_home_covers, "A": p_away_covers, "push": p_push}


def implied_prob_devigged(odds_list):
    """Given decimal odds for all outcomes of a market, strip the overround (vig)
    proportionally and return true-probability estimates that sum to 1."""
    odds = np.array(odds_list, dtype=float)
    raw = 1.0 / odds
    return raw / raw.sum()
