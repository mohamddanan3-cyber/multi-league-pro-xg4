import csv
import io
import json
import math
import os
import time
import urllib.request
import urllib.parse
from datetime import datetime

LEAGUES = {
    "E0": ("Premier League", "football-data"),
    "E1": ("Championship", "football-data"),
    "E2": ("League One", "football-data"),
    "SP1": ("La Liga", "football-data"),
    "I1": ("Serie A", "football-data"),
    "F1": ("Ligue 1", "football-data"),
    "D1": ("Bundesliga", "football-data"),
    "AUT": ("Austrian Bundesliga", "football-data"),
    "B1": ("Belgian First Division", "football-data"),
    "ARG": ("Argentina", "football-data"),
    "BRA": ("Brazil Serie A", "football-data"),
    "SPL": ("Saudi Pro League", "saudi"),
    "DEN": ("Danish Superliga", "football-data"),
}

SEASONS = ("2223", "2324", "2425", "2526", "2627")
FD_BASE = "https://www.football-data.co.uk/mmz4281/"
SAUDI_URL = "https://raw.githubusercontent.com/ichraf10/football-datasets/main/saudi-pro-league.json"
CACHE_TTL = 6 * 60 * 60

ALIASES = {
    "man utd": "Manchester United",
    "manchester united": "Manchester United",
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "spurs": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "newcastle": "Newcastle",
    "west ham united": "West Ham",
    "wolves": "Wolves",
    "wolverhampton wanderers": "Wolves",
    "brighton and hove albion": "Brighton",
    "nottingham forest": "Nott'm Forest",
    "nottm forest": "Nott'm Forest",
    "psg": "Paris Saint-Germain",
    "paris sg": "Paris Saint-Germain",
    "inter": "Inter",
    "internazionale": "Inter",
    "ac milan": "Milan",
    "atletico madrid": "Ath Madrid",
    "athletic bilbao": "Ath Bilbao",
    "barcelona": "Barcelona",
    "real madrid": "Real Madrid",
    "bayern": "Bayern Munich",
    "bayern munich": "Bayern Munich",
    "dortmund": "Dortmund",
    "borussia dortmund": "Dortmund",
}

def norm(s):
    return " ".join(str(s or "").lower().replace("'", "").replace('"', "").replace(".", "").split())

def canon(s):
    n = norm(s)
    return ALIASES.get(n, str(s or "").strip())

def fnum(v):
    try:
        x = float(str(v).replace(",", "."))
        return x
    except Exception:
        return None

def parse_date(v):
    s = str(v or "").strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            d = datetime.strptime(s, fmt)
            if d.year < 100:
                d = d.replace(year=d.year + 2000)
            return d
        except ValueError:
            pass
    return None

def poisson(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    p = math.exp(-lam)
    for i in range(1, k + 1):
        p *= lam / i
    return p

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

class Model:
    def __init__(self, records, decay=0.006):
        self.records = records
        self.decay = decay
        self.latest = max((r["date"] for r in records), default=datetime.now())

    def recency(self, date):
        age = max(0, (self.latest - date).total_seconds() / 86400)
        return math.exp(-self.decay * age)

    def weighted(self, team, role, window=10):
        arr = []
        for r in self.records:
            if role == "home" and r["home"] == team:
                arr.append((r, r["hg"], r["ag"], r.get("hc"), r.get("ac"), r.get("hy"), r.get("ay"), r.get("hr"), r.get("ar")))
            elif role == "away" and r["away"] == team:
                arr.append((r, r["ag"], r["hg"], r.get("ac"), r.get("hc"), r.get("ay"), r.get("hy"), r.get("ar"), r.get("hr")))
        arr = arr[-window:]
        if not arr:
            return {}
        names = ("gf","ga","cf","ca","yf","ya","rf","ra")
        out = {}
        for idx, name in enumerate(names, start=1):
            pairs = [(x[idx], self.recency(x[0])) for x in arr if x[idx] is not None]
            if pairs:
                sw = sum(w for _, w in pairs)
                out[name] = sum(v*w for v,w in pairs) / sw
        return out

    def ht_weighted(self, team, role, window=10):
        arr = []
        for r in self.records:
            if r.get("hthg") is None or r.get("htag") is None:
                continue
            if role == "home" and r["home"] == team:
                arr.append((r, r["hthg"], r["htag"]))
            elif role == "away" and r["away"] == team:
                arr.append((r, r["htag"], r["hthg"]))
        arr = arr[-window:]
        if not arr:
            return {}
        ws = [self.recency(x[0]) for x in arr]
        sw = sum(ws)
        return {
            "gf": sum(x[1]*w for x,w in zip(arr,ws))/sw,
            "ga": sum(x[2]*w for x,w in zip(arr,ws))/sw,
        }

    def expected_goals(self, home, away):
        hh = self.weighted(home, "home")
        aa = self.weighted(away, "away")
        ah = self.weighted(away, "home")
        ha = self.weighted(home, "away")

        hgf, hga = hh.get("gf", 1.3), hh.get("ga", 1.0)
        agf, aga = aa.get("gf", 1.0), aa.get("ga", 1.2)

        eh = .5*(hgf + aga)
        ea = .5*(agf + hga)
        ctx_h = .5*hh.get("gf", hgf) + .5*ha.get("ga", hga)
        ctx_a = .5*aa.get("gf", agf) + .5*ah.get("ga", aga)
        eh = .65*eh + .35*ctx_h
        ea = .65*ea + .35*ctx_a
        return clamp(eh,.2,4.5), clamp(ea,.2,4.5)

    def matrix(self, lh, la, max_goals=7):
        items = []
        for h in range(max_goals+1):
            for a in range(max_goals+1):
                items.append({"h":h,"a":a,"p":poisson(h,lh)*poisson(a,la)})
        z = sum(x["p"] for x in items) or 1
        for x in items:
            x["p"] /= z
        return items

    @staticmethod
    def prob(m, fn):
        return sum(x["p"] for x in m if fn(x))

    def expected_stats(self, home, away):
        hh, aa = self.weighted(home,"home"), self.weighted(away,"away")
        return {
            "home_corners": hh.get("cf",5.0),
            "away_corners": aa.get("cf",4.0),
            "home_yellows": hh.get("yf",1.7),
            "away_yellows": aa.get("yf",1.8),
            "home_reds": hh.get("rf",0.05),
            "away_reds": aa.get("rf",0.05),
        }

    def predict(self, home, away):
        lh, la = self.expected_goals(home, away)
        m = self.matrix(lh, la)
        p1 = {
            "home": self.prob(m, lambda x:x["h"]>x["a"]),
            "draw": self.prob(m, lambda x:x["h"]==x["a"]),
            "away": self.prob(m, lambda x:x["h"]<x["a"]),
        }
        ou = {}
        for line in (.5,1.5,2.5,3.5):
            ou[str(line)] = {
                "over": self.prob(m, lambda x, l=line:x["h"]+x["a"]>l),
                "under": self.prob(m, lambda x, l=line:x["h"]+x["a"]<=l),
            }
        btts_yes = self.prob(m, lambda x:x["h"]>0 and x["a"]>0)
        scores = sorted(m,key=lambda x:x["p"],reverse=True)[:5]

        ht_h, ht_a = .6, .5
        h1, a1 = self.ht_weighted(home,"home"), self.ht_weighted(away,"away")
        if h1 or a1:
            ht_h = .5*(h1.get("gf",.6)+a1.get("ga",.5))
            ht_a = .5*(a1.get("gf",.5)+h1.get("ga",.6))
        ht_m = self.matrix(clamp(ht_h,.05,2.5),clamp(ht_a,.05,2.5),5)

        confidence = 55
        hh, aa = self.weighted(home,"home"), self.weighted(away,"away")
        confidence += 5 if "gf" in hh else 0
        confidence += 5 if "gf" in aa else 0
        confidence += 4 if "ga" in hh else 0
        confidence += 4 if "ga" in aa else 0
        if .8 < lh+la < 3.8: confidence += 8

        return {
            "home":home, "away":away, "xg_home":round(lh,3), "xg_away":round(la,3),
            "xg_total":round(lh+la,3), "confidence":clamp(confidence,0,95),
            "p1x2":p1, "ou":ou,
            "btts":{"yes":btts_yes,"no":1-btts_yes},
            "double_chance":{
                "1x":p1["home"]+p1["draw"],
                "12":p1["home"]+p1["away"],
                "x2":p1["draw"]+p1["away"],
            },
            "scores":scores,
            "ht":{
                "home":self.prob(ht_m,lambda x:x["h"]>x["a"]),
                "draw":self.prob(ht_m,lambda x:x["h"]==x["a"]),
                "away":self.prob(ht_m,lambda x:x["h"]<x["a"]),
                "scores":sorted(ht_m,key=lambda x:x["p"],reverse=True)[:5],
            },
            "stats":self.expected_stats(home,away)
        }

    def value_check(self, result, oh, od, oa):
        out=[]
        for name, odd, prob in (("HOME",oh,result["p1x2"]["home"]),("DRAW",od,result["p1x2"]["draw"]),("AWAY",oa,result["p1x2"]["away"])):
            if odd is None or odd <= 1: continue
            out.append({"market":name,"odds":odd,"model_probability":prob,"fair_odds":1/prob,"edge":prob*odd-1})
        return out

    def backtest(self, home, away, warmup=120):
        if len(self.records) < warmup+1:
            return {"matches":0}
        correct=0; total=0; brier=0.0
        for i in range(min(warmup,len(self.records)-1),len(self.records)):
            r=self.records[i]
            if r["home"]!=home or r["away"]!=away:
                continue
            hist=self.records[:i]
            m=Model(hist,self.decay)
            p=m.predict(home,away)["p1x2"]
            actual="home" if r["hg"]>r["ag"] else "away" if r["hg"]<r["ag"] else "draw"
            pred=max(("home","draw","away"),key=lambda k:p[k])
            correct += pred==actual
            brier += sum((p[k]-(1 if k==actual else 0))**2 for k in ("home","draw","away"))
            total += 1
        if not total:
            return {"matches":0}
        return {"matches":total,"accuracy":correct/total,"brier":brier/total}

class Predictor:
    def __init__(self):
        self.cache = {}

    def fetch(self, url):
        req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0 xG-Predictor/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode("utf-8", errors="replace")

    def load_league(self, code, refresh=False):
        code = code.upper()
        if code not in LEAGUES:
            raise ValueError(f"Unknown league: {code}")
        key = code
        if not refresh and key in self.cache and time.time()-self.cache[key][0] < CACHE_TTL:
            return self.cache[key][1]

        rows=[]
        if LEAGUES[code][1]=="football-data":
            for season in SEASONS:
                url=f"{FD_BASE}{season}/{code}.csv"
                try:
                    text=self.fetch(url)
                except Exception:
                    continue
                rows.extend(self.parse_csv(text))
        else:
            text=self.fetch(SAUDI_URL)
            rows=self.parse_saudi(text)

        rows.sort(key=lambda x:x["date"])
        seen=set(); clean=[]
        for r in rows:
            k=(r["date"].date().isoformat(),norm(r["home"]),norm(r["away"]),r["hg"],r["ag"])
            if k not in seen:
                seen.add(k); clean.append(r)
        self.cache[key]=(time.time(),clean)
        return clean

    def parse_csv(self,text):
        out=[]
        for r in csv.DictReader(io.StringIO(text)):
            home=r.get("Home"); away=r.get("Away")
            hg=fnum(r.get("FTHG")); ag=fnum(r.get("FTAG")); d=parse_date(r.get("Date"))
            if not home or not away or hg is None or ag is None or not d:
                continue
            def opt(k): return fnum(r.get(k))
            out.append({
                "date":d,"home":canon(home),"away":canon(away),"hg":hg,"ag":ag,
                "hthg":opt("HTHG"),"htag":opt("HTAG"),"hc":opt("HC"),"ac":opt("AC"),
                "hy":opt("HY"),"ay":opt("AY"),"hr":opt("HR"),"ar":opt("AR")
            })
        return out

    def parse_saudi(self,text):
        data=json.loads(text)
        if isinstance(data,list): arr=data
        else:
            arr=[]
            for k in ("matches","fixtures","data","results"):
                if isinstance(data.get(k),list):
                    arr=data[k]; break
        out=[]
        for r in arr:
            home=canon(r.get("home_team") or r.get("home") or r.get("Home") or "")
            away=canon(r.get("away_team") or r.get("away") or r.get("Away") or "")
            hg=fnum(r.get("home_score",r.get("home_goals",r.get("FTHG"))))
            ag=fnum(r.get("away_score",r.get("away_goals",r.get("FTAG"))))
            d=parse_date(r.get("date") or r.get("Date") or r.get("match_date"))
            if home and away and hg is not None and ag is not None and d:
                out.append({"date":d,"home":home,"away":away,"hg":hg,"ag":ag,
                            "hthg":None,"htag":None,"hc":None,"ac":None,"hy":None,"ay":None,"hr":None,"ar":None})
        return out

    def model(self, rows):
        return Model(rows)
