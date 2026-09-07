# -*- coding: utf-8 -*-
"""
==============================================================
 UEFA CHAMPIONS LEAGUE XG PREDICTOR PRO V2
 Season focus: 2026/27
 Android / Pydroid
 No pandas / no requests

 V2 improvements:
 - Competition-specific Champions League data
 - Recency weighting
 - Home/Away context
 - Opponent-strength adjustment
 - Elo-style team strength
 - Shrinkage for small samples
 - Separate HT model
 - Poisson score matrix
 - 1X2 / O-U / BTTS / Double Chance
 - Exact scores
 - Corners / cards when available
 - Confidence score
 - Backtest with chronological training
 - Cache + refresh
 - Robust CSV fallback
==============================================================
"""

import csv
import io
import math
import os
import sys
from datetime import datetime

try:
    import urllib.request
except Exception:
    urllib = None


# ==============================================================
# UI
# ==============================================================

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
WHITE = "\033[37m"
BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_WHITE = "\033[97m"


def c(text, color):
    return color + str(text) + RESET


def title(text):
    print()
    print(c("╔" + "═" * 72 + "╗", BRIGHT_CYAN))
    print(c("║" + text.center(72) + "║", BOLD + BRIGHT_CYAN))
    print(c("╚" + "═" * 72 + "╝", BRIGHT_CYAN))


def section(text):
    print()
    print(c("┌" + "─" * 72 + "┐", BLUE))
    print(c("│ " + text.ljust(70) + "│", BOLD + BRIGHT_WHITE))
    print(c("└" + "─" * 72 + "┘", BLUE))


# ==============================================================
# CONFIG
# ==============================================================

CACHE_DIR = "ucl_cache_v2"
os.makedirs(CACHE_DIR, exist_ok=True)

# Football-Data season notation:
# 2223 = 2022/23 ... 2627 = 2026/27
SEASONS = ["2223", "2324", "2425", "2526", "2627"]

# Champions League code used by Football-Data.
# If a future source changes its naming, SOURCE_ALIASES are tried.
SOURCE_ALIASES = [
    "https://www.football-data.co.uk/mmz4281/{}/CL.csv",
    "https://www.football-data.co.uk/mmz4281/{}/CL.csv?x=1",
]

MAX_GOALS = 8

# More recent matches matter more.
HALF_LIFE_DAYS = 120.0

# Competition baseline. These are only starting priors;
# fit() replaces them with observed data whenever possible.
PRIOR_HOME_GOALS = 1.55
PRIOR_AWAY_GOALS = 1.20

PRIOR_HT_HOME = 0.68
PRIOR_HT_AWAY = 0.48

PRIOR_CORNERS_HOME = 5.20
PRIOR_CORNERS_AWAY = 4.40
PRIOR_YELLOW_HOME = 1.55
PRIOR_YELLOW_AWAY = 1.85
PRIOR_RED_HOME = 0.07
PRIOR_RED_AWAY = 0.08

# Home advantage is deliberately modest for elite competition.
HOME_ADVANTAGE = 0.07

# Elo settings.
ELO_BASE = 1500.0
ELO_K = 22.0
ELO_HOME = 35.0

# Shrink team estimates toward league priors.
MIN_TEAM_WEIGHT = 2.5
SHRINK_RATE = 6.0

# Current-season matches get an extra emphasis if available.
CURRENT_SEASON_BOOST = 1.20


# ==============================================================
# INTERNET
# ==============================================================

def http_get(url):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Android; Pydroid) UCL-XG-Predictor/2.0",
            "Accept": "text/csv,text/plain,application/csv,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read()


def looks_like_csv(data):
    if not data:
        return False
    text = data[:7000].decode("utf-8-sig", errors="ignore").lower()
    return all(x in text for x in ["hometeam", "awayteam", "fthg", "ftag"])


def download_data(season, refresh=False):
    cache = os.path.join(CACHE_DIR, "CL_{}.csv".format(season))

    if os.path.exists(cache) and not refresh:
        try:
            with open(cache, "rb") as f:
                data = f.read()
            if looks_like_csv(data):
                print(c("✓ Cache: Champions League {}".format(season), GREEN))
                return data
        except Exception:
            pass

    for url_template in SOURCE_ALIASES:
        url = url_template.format(season)
        try:
            print(c("Downloading UCL {} ...".format(season), CYAN))
            data = http_get(url)
            if looks_like_csv(data):
                try:
                    with open(cache, "wb") as f:
                        f.write(data)
                except Exception:
                    pass
                print(c("✓ CSV loaded.", GREEN))
                return data
        except Exception as e:
            print(c("⚠ Source attempt failed: {}".format(e), YELLOW))

    if os.path.exists(cache):
        try:
            with open(cache, "rb") as f:
                data = f.read()
            if looks_like_csv(data):
                print(c("✓ Old cache used: {}".format(season), GREEN))
                return data
        except Exception:
            pass

    return None


# ==============================================================
# CSV
# ==============================================================

def parse_date(value):
    value = (value or "").strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def parse_csv(data):
    if not data:
        return []

    text = data.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = []

    def num(row, key):
        try:
            v = row.get(key)
            if v in (None, ""):
                return None
            return float(v)
        except Exception:
            return None

    for r in reader:
        try:
            date = parse_date(r.get("Date"))
            home = (r.get("HomeTeam") or "").strip()
            away = (r.get("AwayTeam") or "").strip()
            if not date or not home or not away:
                continue

            hg = int(float(r["FTHG"]))
            ag = int(float(r["FTAG"]))

            rows.append({
                "date": date,
                "home": home,
                "away": away,
                "hg": hg,
                "ag": ag,
                "hthg": num(r, "HTHG"),
                "htag": num(r, "HTAG"),
                "hc": num(r, "HC"),
                "ac": num(r, "AC"),
                "hy": num(r, "HY"),
                "ay": num(r, "AY"),
                "hr": num(r, "HR"),
                "ar": num(r, "AR"),
            })
        except (ValueError, TypeError, KeyError):
            continue

    return rows


def load_history(refresh=False):
    all_rows = []

    for season in SEASONS:
        data = download_data(season, refresh)
        if not data:
            print(c("⚠ No UCL data for {}".format(season), YELLOW))
            continue

        rows = parse_csv(data)
        for r in rows:
            r["season"] = season

        print(c("✓ UCL {}: {} matches".format(season, len(rows)), GREEN))
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError(
            "No Champions League data loaded. Check Internet permission in Pydroid."
        )

    all_rows.sort(key=lambda x: x["date"])
    return all_rows


# ==============================================================
# MATH
# ==============================================================

def poisson(lam, goals):
    try:
        return math.exp(-lam) * (lam ** goals) / math.factorial(goals)
    except Exception:
        return 0.0


def score_matrix(xh, xa):
    m = {}
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            m[(h, a)] = poisson(xh, h) * poisson(xa, a)

    total = sum(m.values())
    if total > 0:
        for k in m:
            m[k] /= total
    return m


def probabilities(matrix):
    hp = dp = ap = 0.0
    for (h, a), p in matrix.items():
        if h > a:
            hp += p
        elif h == a:
            dp += p
        else:
            ap += p
    return hp, dp, ap


def over_under(matrix, line):
    under = sum(
        p for (h, a), p in matrix.items()
        if h + a <= line
    )
    return 1.0 - under, under


def btts(xh, xa):
    yes = (
        1.0
        - math.exp(-xh)
        - math.exp(-xa)
        + math.exp(-(xh + xa))
    )
    yes = max(0.0, min(1.0, yes))
    return yes, 1.0 - yes


def top_scores(matrix, count=5):
    return sorted(matrix.items(), key=lambda x: x[1], reverse=True)[:count]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def weighted_mean(values, weights, fallback):
    usable = [
        (v, w) for v, w in zip(values, weights)
        if v is not None and w > 0
    ]
    if not usable:
        return fallback

    den = sum(w for _, w in usable)
    if den <= 0:
        return fallback

    return sum(v * w for v, w in usable) / den


# ==============================================================
# MODEL V2
# ==============================================================

class UCLModel:

    def __init__(self, history):
        self.history = history

        self.league_home = PRIOR_HOME_GOALS
        self.league_away = PRIOR_AWAY_GOALS
        self.ht_home = PRIOR_HT_HOME
        self.ht_away = PRIOR_HT_AWAY

        self.corner_home = PRIOR_CORNERS_HOME
        self.corner_away = PRIOR_CORNERS_AWAY
        self.yellow_home = PRIOR_YELLOW_HOME
        self.yellow_away = PRIOR_YELLOW_AWAY
        self.red_home = PRIOR_RED_HOME
        self.red_away = PRIOR_RED_AWAY

        self.team = {}
        self.ht_team = {}
        self.elo = {}

        self.latest_date = None

    # ----------------------------------------------------------
    # FIT
    # ----------------------------------------------------------

    def fit(self, rows):
        self.team = {}
        self.ht_team = {}
        self.elo = {}

        if not rows:
            return

        self.latest_date = max(r["date"] for r in rows)

        # League baselines from available observations.
        self.league_home = sum(r["hg"] for r in rows) / len(rows)
        self.league_away = sum(r["ag"] for r in rows) / len(rows)

        ht_rows = [
            r for r in rows
            if r["hthg"] is not None and r["htag"] is not None
        ]
        if ht_rows:
            self.ht_home = sum(r["hthg"] for r in ht_rows) / len(ht_rows)
            self.ht_away = sum(r["htag"] for r in ht_rows) / len(ht_rows)

        corner_rows = [
            r for r in rows
            if r["hc"] is not None and r["ac"] is not None
        ]
        if corner_rows:
            self.corner_home = sum(r["hc"] for r in corner_rows) / len(corner_rows)
            self.corner_away = sum(r["ac"] for r in corner_rows) / len(corner_rows)

        yellow_rows = [
            r for r in rows
            if r["hy"] is not None and r["ay"] is not None
        ]
        if yellow_rows:
            self.yellow_home = sum(r["hy"] for r in yellow_rows) / len(yellow_rows)
            self.yellow_away = sum(r["ay"] for r in yellow_rows) / len(yellow_rows)

        red_rows = [
            r for r in rows
            if r["hr"] is not None and r["ar"] is not None
        ]
        if red_rows:
            self.red_home = sum(r["hr"] for r in red_rows) / len(red_rows)
            self.red_away = sum(r["ar"] for r in red_rows) / len(red_rows)

        # Chronological Elo.
        for r in rows:
            self._ensure_team(r["home"])
            self._ensure_team(r["away"])

            eh = self._elo_expected(
                self.elo[r["home"]] + ELO_HOME,
                self.elo[r["away"]]
            )

            actual = (
                1.0 if r["hg"] > r["ag"]
                else 0.5 if r["hg"] == r["ag"]
                else 0.0
            )

            margin = min(2.0, abs(r["hg"] - r["ag"]) + 1.0)
            k = ELO_K * margin

            self.elo[r["home"]] += k * (actual - eh)
            self.elo[r["away"]] += k * ((1.0 - actual) - (1.0 - eh))

            self._add_team_match(r)
            self._add_ht_match(r)

    def _ensure_team(self, team):
        if team not in self.elo:
            self.elo[team] = ELO_BASE

        if team not in self.team:
            self.team[team] = []

        if team not in self.ht_team:
            self.ht_team[team] = []

    @staticmethod
    def _elo_expected(a, b):
        try:
            return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))
        except Exception:
            return 0.5

    def _add_team_match(self, r):
        self._ensure_team(r["home"])
        self._ensure_team(r["away"])

        boost = CURRENT_SEASON_BOOST if r.get("season") == "2627" else 1.0

        self.team[r["home"]].append({
            "date": r["date"],
            "gf": r["hg"],
            "ga": r["ag"],
            "cf": r["hc"],
            "ca": r["ac"],
            "yf": r["hy"],
            "ya": r["ay"],
            "rf": r["hr"],
            "ra": r["ar"],
            "home": True,
            "boost": boost,
            "opp": r["away"],
        })

        self.team[r["away"]].append({
            "date": r["date"],
            "gf": r["ag"],
            "ga": r["hg"],
            "cf": r["ac"],
            "ca": r["hc"],
            "yf": r["ay"],
            "ya": r["hy"],
            "rf": r["ar"],
            "ra": r["hr"],
            "home": False,
            "boost": boost,
            "opp": r["home"],
        })

    def _add_ht_match(self, r):
        if r["hthg"] is None or r["htag"] is None:
            return

        self._ensure_team(r["home"])
        self._ensure_team(r["away"])

        self.ht_team[r["home"]].append({
            "date": r["date"],
            "gf": r["hthg"],
            "ga": r["htag"],
            "home": True,
            "opp": r["away"],
        })

        self.ht_team[r["away"]].append({
            "date": r["date"],
            "gf": r["htag"],
            "ga": r["hthg"],
            "home": False,
            "opp": r["home"],
        })

    # ----------------------------------------------------------
    # WEIGHTING
    # ----------------------------------------------------------

    def _weight(self, date, boost=1.0):
        age = max(0, (self.latest_date - date).days)
        w = math.exp(-math.log(2.0) * age / HALF_LIFE_DAYS)
        return w * boost

    def _records(self, team, home_context=None):
        records = self.team.get(team, [])
        if home_context is not None:
            records = [x for x in records if x["home"] == home_context]
        return records

    def _team_stats(self, team, home_context=None):
        records = self._records(team, home_context)
        if not records:
            return None

        weights = [self._weight(x["date"], x["boost"]) for x in records]
        nweight = sum(weights)

        def avg(key, fallback):
            vals = [x[key] for x in records]
            return weighted_mean(vals, weights, fallback)

        return {
            "gf": avg("gf", self.league_home if home_context else self.league_away),
            "ga": avg("ga", self.league_away if home_context else self.league_home),
            "cf": avg("cf", self.corner_home if home_context else self.corner_away),
            "ca": avg("ca", self.corner_away if home_context else self.corner_home),
            "yf": avg("yf", self.yellow_home if home_context else self.yellow_away),
            "ya": avg("ya", self.yellow_away if home_context else self.yellow_home),
            "rf": avg("rf", self.red_home if home_context else self.red_away),
            "ra": avg("ra", self.red_away if home_context else self.red_home),
            "weight": nweight,
            "n": len(records),
        }

    def _ht_stats(self, team, home_context=None):
        records = self.ht_team.get(team, [])
        if home_context is not None:
            records = [x for x in records if x["home"] == home_context]

        if not records:
            return None

        weights = [self._weight(x["date"]) for x in records]
        return {
            "gf": weighted_mean(
                [x["gf"] for x in records], weights,
                self.ht_home if home_context else self.ht_away
            ),
            "ga": weighted_mean(
                [x["ga"] for x in records], weights,
                self.ht_away if home_context else self.ht_home
            ),
            "weight": sum(weights),
            "n": len(records),
        }

    # ----------------------------------------------------------
    # OPPONENT STRENGTH
    # ----------------------------------------------------------

    def opponent_factor(self, opponent, attacking=False):
        """
        Uses Elo as a soft adjustment, not as a replacement for goals.
        Strong opponent => lower expected attack and higher defensive demand.
        """
        rating = self.elo.get(opponent, ELO_BASE)
        delta = (rating - ELO_BASE) / 400.0

        if attacking:
            # Opponent quality changes attacking expectation modestly.
            return clamp(1.0 - 0.22 * delta, 0.78, 1.22)

        # Defensive expectation: stronger opponent tends to raise danger.
        return clamp(1.0 + 0.18 * delta, 0.82, 1.18)

    def _shrink(self, estimate, prior, weight):
        alpha = weight / (weight + SHRINK_RATE)
        alpha = clamp(alpha, 0.0, 0.92)
        return alpha * estimate + (1.0 - alpha) * prior

    # ----------------------------------------------------------
    # EXPECTED GOALS
    # ----------------------------------------------------------

    def expected_goals(self, home, away):
        h_all = self._team_stats(home)
        a_all = self._team_stats(away)
        h_home = self._team_stats(home, True)
        a_away = self._team_stats(away, False)

        # Global estimates.
        h_attack = h_all["gf"] if h_all else self.league_home
        h_def = h_all["ga"] if h_all else self.league_away
        a_attack = a_all["gf"] if a_all else self.league_away
        a_def = a_all["ga"] if a_all else self.league_home

        # Context-specific blend.
        if h_home:
            h_attack = 0.72 * h_home["gf"] + 0.28 * h_attack
            h_def = 0.72 * h_home["ga"] + 0.28 * h_def

        if a_away:
            a_attack = 0.72 * a_away["gf"] + 0.28 * a_attack
            a_def = 0.72 * a_away["ga"] + 0.28 * a_def

        # Shrink toward league priors.
        h_weight = (h_home["weight"] if h_home else 0.0)
        a_weight = (a_away["weight"] if a_away else 0.0)

        h_attack = self._shrink(h_attack, self.league_home, h_weight)
        h_def = self._shrink(h_def, self.league_away, h_weight)
        a_attack = self._shrink(a_attack, self.league_away, a_weight)
        a_def = self._shrink(a_def, self.league_home, a_weight)

        # Multiplicative attack x opponent defense.
        xh = self.league_home
        xh *= h_attack / max(self.league_home, 0.10)
        xh *= a_def / max(self.league_home, 0.10)

        xa = self.league_away
        xa *= a_attack / max(self.league_away, 0.10)
        xa *= h_def / max(self.league_away, 0.10)

        # Soft opponent/Elo correction.
        xh *= self.opponent_factor(away, attacking=True)
        xa *= self.opponent_factor(home, attacking=True)

        xh += HOME_ADVANTAGE

        # Keep predictions realistic.
        xh = clamp(xh, 0.20, 4.20)
        xa = clamp(xa, 0.15, 3.80)

        # Elo blend: only a small contribution, avoiding overfitting.
        he = self.elo.get(home, ELO_BASE) + ELO_HOME
        ae = self.elo.get(away, ELO_BASE)
        elo_home = self._elo_expected(he, ae)
        elo_away = 1.0 - elo_home

        xh *= (0.90 + 0.20 * elo_home)
        xa *= (0.90 + 0.20 * elo_away)

        return clamp(xh, 0.20, 4.50), clamp(xa, 0.15, 4.50)

    # ----------------------------------------------------------
    # HALF TIME
    # ----------------------------------------------------------

    def expected_ht_goals(self, home, away):
        h = self._ht_stats(home)
        a = self._ht_stats(away)
        hh = self._ht_stats(home, True)
        aa = self._ht_stats(away, False)

        h_attack = h["gf"] if h else self.ht_home
        h_def = h["ga"] if h else self.ht_away
        a_attack = a["gf"] if a else self.ht_away
        a_def = a["ga"] if a else self.ht_home

        if hh:
            h_attack = 0.70 * hh["gf"] + 0.30 * h_attack
            h_def = 0.70 * hh["ga"] + 0.30 * h_def

        if aa:
            a_attack = 0.70 * aa["gf"] + 0.30 * a_attack
            a_def = 0.70 * aa["ga"] + 0.30 * a_def

        hw = hh["weight"] if hh else 0.0
        aw = aa["weight"] if aa else 0.0

        h_attack = self._shrink(h_attack, self.ht_home, hw)
        h_def = self._shrink(h_def, self.ht_away, hw)
        a_attack = self._shrink(a_attack, self.ht_away, aw)
        a_def = self._shrink(a_def, self.ht_home, aw)

        xh = self.ht_home
        xh *= h_attack / max(self.ht_home, 0.05)
        xh *= a_def / max(self.ht_home, 0.05)

        xa = self.ht_away
        xa *= a_attack / max(self.ht_away, 0.05)
        xa *= h_def / max(self.ht_away, 0.05)

        return clamp(xh, 0.05, 2.30), clamp(xa, 0.05, 2.20)

    # ----------------------------------------------------------
    # CORNERS / CARDS
    # ----------------------------------------------------------

    def expected_match_stats(self, home, away):
        h = self._team_stats(home)
        a = self._team_stats(away)
        hh = self._team_stats(home, True)
        aa = self._team_stats(away, False)

        h = h or {}
        a = a or {}
        hh = hh or {}
        aa = aa or {}

        hcf = hh.get("cf", h.get("cf", self.corner_home))
        hca = hh.get("ca", h.get("ca", self.corner_away))
        acf = aa.get("cf", a.get("cf", self.corner_away))
        aca = aa.get("ca", a.get("ca", self.corner_home))

        hyf = hh.get("yf", h.get("yf", self.yellow_home))
        hya = hh.get("ya", h.get("ya", self.yellow_away))
        ayf = aa.get("yf", a.get("yf", self.yellow_away))
        aya = aa.get("ya", a.get("ya", self.yellow_home))

        hrf = hh.get("rf", h.get("rf", self.red_home))
        hra = hh.get("ra", h.get("ra", self.red_away))
        arf = aa.get("rf", a.get("rf", self.red_away))
        ara = aa.get("ra", a.get("ra", self.red_home))

        return {
            "home_corners": clamp(0.55 * hcf + 0.45 * aca, 0, 15),
            "away_corners": clamp(0.55 * acf + 0.45 * hca, 0, 15),
            "home_yellow": clamp(0.55 * hyf + 0.45 * aya, 0, 6),
            "away_yellow": clamp(0.55 * ayf + 0.45 * hya, 0, 6),
            "home_red": clamp(0.50 * hrf + 0.50 * ara, 0, 1),
            "away_red": clamp(0.50 * arf + 0.50 * hra, 0, 1),
        }

    # ----------------------------------------------------------
    # CONFIDENCE
    # ----------------------------------------------------------

    def confidence(self, home, away, hp, dp, ap, xh, xa):
        h_n = len(self.team.get(home, []))
        a_n = len(self.team.get(away, []))

        best = max(hp, dp, ap)
        gap = sorted([hp, dp, ap], reverse=True)
        separation = gap[0] - gap[1]

        sample = min(1.0, (h_n + a_n) / 20.0)
        certainty = clamp((best - 0.34) / 0.40, 0.0, 1.0)
        sep = clamp(separation / 0.30, 0.0, 1.0)

        score = 100.0 * (
            0.45 * certainty
            + 0.30 * sep
            + 0.25 * sample
        )

        if xh + xa > 4.0 or xh + xa < 1.3:
            score *= 0.92

        return clamp(score, 0, 100)


# ==============================================================
# DISPLAY
# ==============================================================

def prob_color(p):
    if p >= 0.70:
        return BRIGHT_GREEN
    if p >= 0.55:
        return GREEN
    if p >= 0.45:
        return YELLOW
    return RED


def print_probability(label, p):
    print(
        "{} {:>7.1f}% {}".format(
            label, p * 100,
            c("●", prob_color(p))
        )
    )


def best_market(ou):
    best = None
    for line, over, under in ou:
        for name, p in (("OVER " + str(line), over), ("UNDER " + str(line), under)):
            if best is None or p > best[1]:
                best = (name, p)
    return best


# ==============================================================
# VERDICT
# ==============================================================

def final_verdict(
    home, away,
    hp, dp, ap,
    btts_yes, btts_no,
    ou, top,
    ht_hp, ht_dp, ht_ap,
    xh, xa,
    confidence
):
    section("⭐ FINAL VERDICT V2 / الخلاصة")

    result_options = [("HOME", hp), ("DRAW", dp), ("AWAY", ap)]
    result_name, result_prob = max(result_options, key=lambda x: x[1])

    if result_name == "HOME":
        result_text = "🏠 فوز " + home
    elif result_name == "AWAY":
        result_text = "✈️ فوز " + away
    else:
        result_text = "🤝 تعادل"

    print(
        c("🏆 1X2 الأقرب : ", BOLD + BRIGHT_YELLOW)
        + c(result_text, prob_color(result_prob))
        + "  {:.1f}%".format(result_prob * 100)
    )

    market, market_p = best_market(ou)
    print(
        c("⚽ أفضل خط أهداف : ", BOLD + BRIGHT_YELLOW)
        + c(market, prob_color(market_p))
        + "  {:.1f}%".format(market_p * 100)
    )

    if btts_yes >= btts_no:
        bt = "BTTS YES"
        bp = btts_yes
    else:
        bt = "BTTS NO"
        bp = btts_no

    print(
        c("🤝 BTTS : ", BOLD + BRIGHT_YELLOW)
        + c(bt, prob_color(bp))
        + "  {:.1f}%".format(bp * 100)
    )

    if top:
        (sh, sa), sp = top[0]
        print(
            c("🎯 Exact Score : ", BOLD + BRIGHT_YELLOW)
            + c("{} - {}".format(sh, sa), BRIGHT_GREEN)
            + "  {:.1f}%".format(sp * 100)
        )

    ht_options = [("HOME", ht_hp), ("DRAW", ht_dp), ("AWAY", ht_ap)]
    ht_name, ht_p = max(ht_options, key=lambda x: x[1])
    ht_text = {
        "HOME": "تقدم صاحب الأرض",
        "DRAW": "تعادل",
        "AWAY": "تقدم الضيف",
    }[ht_name]

    print(
        c("🏁 HT : ", BOLD + BRIGHT_YELLOW)
        + c(ht_text, prob_color(ht_p))
        + "  {:.1f}%".format(ht_p * 100)
    )

    print(
        c("🧠 Model Confidence : ", BOLD + BRIGHT_MAGENTA)
        + c("{:.0f}/100".format(confidence), BRIGHT_WHITE)
    )

    print()
    if result_name == "HOME" and bp >= 0.55:
        scenario = "أفضلية صاحب الأرض مع سيناريو أهداف متماسك."
    elif result_name == "AWAY" and bp >= 0.55:
        scenario = "أفضلية الضيف مع سيناريو أهداف متماسك."
    elif result_name == "DRAW":
        scenario = "مباراة متقاربة؛ التعادل حاضر بقوة."
    else:
        scenario = "المباراة غير واضحة؛ الأفضل تجنب الثقة الزائدة."

    print(c("🔥 السيناريو : ", BOLD + BRIGHT_MAGENTA))
    print(c("   " + scenario, BRIGHT_WHITE))
    print()
    print(
        c("📈 XG : ", BOLD + CYAN)
        + "{} {:.2f} | {} {:.2f}".format(home, xh, away, xa)
    )


# ==============================================================
# PREDICT
# ==============================================================

def predict(history, home, away):
    model = UCLModel(history)
    model.fit(history)

    xh, xa = model.expected_goals(home, away)
    matrix = score_matrix(xh, xa)
    hp, dp, ap = probabilities(matrix)

    btts_yes, btts_no = btts(xh, xa)

    ht_xh, ht_xa = model.expected_ht_goals(home, away)
    ht_matrix = score_matrix(ht_xh, ht_xa)
    ht_hp, ht_dp, ht_ap = probabilities(ht_matrix)

    stats = model.expected_match_stats(home, away)
    confidence = model.confidence(
        home, away, hp, dp, ap, xh, xa
    )

    title("🇪🇺 UEFA CHAMPIONS LEAGUE XG PREDICTOR PRO V2")

    print(c("🏆 Competition : ", BOLD + CYAN) + "UEFA Champions League")
    print(
        c("⚽ Match : ", BOLD + CYAN)
        + c(home, BRIGHT_GREEN)
        + "  VS  "
        + c(away, BRIGHT_RED)
    )

    section("📈 EXPECTED GOALS")
    print("🏠 {} : {}".format(home, c("{:.2f}".format(xh), BRIGHT_GREEN)))
    print("✈️ {} : {}".format(away, c("{:.2f}".format(xa), BRIGHT_RED)))
    print("⚽ TOTAL : {}".format(c("{:.2f}".format(xh + xa), BRIGHT_YELLOW)))

    section("🏆 1X2")
    print_probability("🏠 HOME", hp)
    print_probability("🤝 DRAW", dp)
    print_probability("✈️ AWAY", ap)

    section("⚽ OVER / UNDER")
    ou = []
    for line in (0.5, 1.5, 2.5, 3.5):
        over, under = over_under(matrix, line)
        ou.append((line, over, under))
        print(
            "O {:.1f}: {} | U {:.1f}: {}".format(
                line,
                c("{:.1f}%".format(over * 100), prob_color(over)),
                line,
                c("{:.1f}%".format(under * 100), prob_color(under)),
            )
        )

    section("🤝 BTTS")
    print("🟢 YES : {}".format(
        c("{:.1f}%".format(btts_yes * 100), prob_color(btts_yes))
    ))
    print("🔴 NO  : {}".format(
        c("{:.1f}%".format(btts_no * 100), prob_color(btts_no))
    ))

    section("🛡️ DOUBLE CHANCE")
    print("1X : {}".format(c("{:.1f}%".format((hp + dp) * 100), prob_color(hp + dp))))
    print("12 : {}".format(c("{:.1f}%".format((hp + ap) * 100), prob_color(hp + ap))))
    print("X2 : {}".format(c("{:.1f}%".format((dp + ap) * 100), prob_color(dp + ap))))

    section("🎯 TOP 5 EXACT SCORES")
    top = top_scores(matrix, 5)
    for i, ((h, a), p) in enumerate(top, 1):
        print(
            "{}. {} - {} → {}".format(
                i, h, a,
                c("{:.1f}%".format(p * 100), prob_color(p))
            )
        )

    section("🏁 FIRST HALF")
    print("📈 HT XG : {} {:.2f} | {} {:.2f}".format(
        home, ht_xh, away, ht_xa
    ))
    print_probability("🏠 HT HOME", ht_hp)
    print_probability("🤝 HT DRAW", ht_dp)
    print_probability("✈️ HT AWAY", ht_ap)

    print()
    print(c("🎯 TOP 5 HT SCORES", BOLD + BRIGHT_YELLOW))
    for i, ((h, a), p) in enumerate(top_scores(ht_matrix, 5), 1):
        print("{}. {} - {} → {}".format(
            i, h, a,
            c("{:.1f}%".format(p * 100), prob_color(p))
        ))

    print()
    print(c("⚽ HT OVER / UNDER", BOLD + BRIGHT_YELLOW))
    for line in (0.5, 1.5):
        over, under = over_under(ht_matrix, line)
        print(
            "HT {:.1f} → O {} | U {}".format(
                line,
                c("{:.1f}%".format(over * 100), prob_color(over)),
                c("{:.1f}%".format(under * 100), prob_color(under)),
            )
        )

    section("🚩 EXPECTED CORNERS")
    ct = stats["home_corners"] + stats["away_corners"]
    print("🏠 {} : {:.2f}".format(home, stats["home_corners"]))
    print("✈️ {} : {:.2f}".format(away, stats["away_corners"]))
    print("🚩 TOTAL : {}".format(c("{:.2f}".format(ct), BRIGHT_YELLOW)))

    section("🟨 YELLOW / 🟥 RED")
    yt = stats["home_yellow"] + stats["away_yellow"]
    rt = stats["home_red"] + stats["away_red"]

    print("🟨 {} : {:.2f}".format(home, stats["home_yellow"]))
    print("🟨 {} : {:.2f}".format(away, stats["away_yellow"]))
    print("🟨 TOTAL : {}".format(c("{:.2f}".format(yt), BRIGHT_YELLOW)))

    print("🟥 {} : {:.2f}".format(home, stats["home_red"]))
    print("🟥 {} : {:.2f}".format(away, stats["away_red"]))
    print("🟥 TOTAL : {}".format(c("{:.2f}".format(rt), BRIGHT_RED)))

    final_verdict(
        home, away,
        hp, dp, ap,
        btts_yes, btts_no,
        ou, top,
        ht_hp, ht_dp, ht_ap,
        xh, xa,
        confidence
    )

    print()
    print(c("⚠️ ملاحظة: النموذج إحصائي وليس ضمانًا للنتيجة.", YELLOW))
    print(c("=" * 72, BRIGHT_CYAN))


# ==============================================================
# BACKTEST
# ==============================================================

def backtest(history, warmup=80):
    if len(history) <= warmup:
        print(c("❌ بيانات غير كافية للـ backtest.", RED))
        return

    tested = 0
    correct = 0
    logloss = 0.0
    btts_correct = 0
    btts_tested = 0

    for i in range(warmup, len(history)):
        row = history[i]
        train = history[:i]

        model = UCLModel(train)
        model.fit(train)

        xh, xa = model.expected_goals(row["home"], row["away"])
        matrix = score_matrix(xh, xa)
        hp, dp, ap = probabilities(matrix)

        probs = {"H": hp, "D": dp, "A": ap}
        actual = (
            "H" if row["hg"] > row["ag"]
            else "D" if row["hg"] == row["ag"]
            else "A"
        )

        pred = max(probs, key=probs.get)
        if pred == actual:
            correct += 1

        logloss += -math.log(max(probs[actual], 1e-12))

        actual_btts = row["hg"] > 0 and row["ag"] > 0
        by, bn = btts(xh, xa)
        pred_btts = by >= bn

        if pred_btts == actual_btts:
            btts_correct += 1
        btts_tested += 1
        tested += 1

    title("📊 UCL V2 BACKTEST")

    accuracy = 100.0 * correct / tested
    ll = logloss / tested
    bacc = 100.0 * btts_correct / max(1, btts_tested)

    print("Matches tested : {}".format(tested))
    print("1X2 accuracy   : {}".format(c("{:.2f}%".format(accuracy), BRIGHT_GREEN)))
    print("1X2 log loss   : {:.4f}".format(ll))
    print("BTTS accuracy  : {}".format(c("{:.2f}%".format(bacc), BRIGHT_GREEN)))
    print(c("=" * 72, BRIGHT_CYAN))


# ==============================================================
# TEAMS
# ==============================================================

def teams_list(history):
    teams = set()
    for r in history:
        teams.add(r["home"])
        teams.add(r["away"])
    return sorted(teams)


def current_teams(history):
    teams = set()
    for r in history:
        if r.get("season") == "2627":
            teams.add(r["home"])
            teams.add(r["away"])
    return sorted(teams)


# ==============================================================
# MAIN
# ==============================================================

def main():
    title("🇪🇺 UEFA CHAMPIONS LEAGUE XG PREDICTOR PRO V2")
    print(c("Season target: 2026/2027", BOLD + BRIGHT_CYAN))
    print(c("Source: Football-Data Champions League CSV", CYAN))
    print()

    refresh = "--refresh" in sys.argv
    do_backtest = "--backtest" in sys.argv

    try:
        history = load_history(refresh)
    except Exception as e:
        print(c("❌ ERROR: {}".format(e), RED))
        print(c("تأكد من Internet permission في Pydroid.", YELLOW))
        return

    print()
    print(c(
        "✓ Total UCL matches loaded: {}".format(len(history)),
        BRIGHT_GREEN
    ))

    if do_backtest:
        backtest(history)
        return

    teams = current_teams(history)
    if not teams:
        teams = teams_list(history)

    section("📋 CHAMPIONS LEAGUE TEAMS")
    print(", ".join(teams))

    home = input("\n🏠 الفريق المضيف: ").strip()
    away = input("✈️ الفريق الضيف: ").strip()

    if home not in teams:
        print(c("❌ الفريق المضيف غير موجود في البيانات الحالية.", RED))
        return

    if away not in teams:
        print(c("❌ الفريق الضيف غير موجود في البيانات الحالية.", RED))
        return

    if home == away:
        print(c("❌ اختار فريقين مختلفين.", RED))
        return

    predict(history, home, away)


if __name__ == "__main__":
    main()
