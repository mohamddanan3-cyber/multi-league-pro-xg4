from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
from predictor import LEAGUES, Predictor

app = FastAPI(title="MULTI-LEAGUE PRO xG PREDICTOR V7", version="1.0.0")
predictor = Predictor()

class PredictRequest(BaseModel):
    league: str = Field(..., min_length=2)
    home: str
    away: str
    warmup: int = Field(120, ge=30, le=1000)
    odd_home: Optional[float] = None
    odd_draw: Optional[float] = None
    odd_away: Optional[float] = None

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/api/leagues")
def leagues():
    return {k: v[0] for k, v in LEAGUES.items()}

@app.get("/api/league/{code}")
def league(code: str, refresh: bool = False):
    try:
        rows = predictor.load_league(code.upper(), refresh=refresh)
        teams = sorted(set(x["home"] for x in rows) | set(x["away"] for x in rows))
        return {"league": code.upper(), "matches": len(rows), "teams": teams}
    except Exception as exc:
        raise HTTPException(502, str(exc))

@app.post("/api/predict")
def predict(req: PredictRequest):
    try:
        rows = predictor.load_league(req.league.upper(), refresh=False)
        if req.home == req.away:
            raise HTTPException(400, "Home and away teams must be different.")
        if not any(r["home"] == req.home and r["away"] == req.away for r in rows):
            # Still allow a prediction for a team pairing not previously played.
            known = {r["home"] for r in rows} | {r["away"] for r in rows}
            if req.home not in known or req.away not in known:
                raise HTTPException(400, "One or both teams are not present in the selected league data.")
        model = predictor.model(rows)
        result = model.predict(req.home, req.away)
        result["backtest"] = model.backtest(req.home, req.away, req.warmup)
        result["value"] = model.value_check(
            result,
            req.odd_home, req.odd_draw, req.odd_away
        )
        result["data_matches"] = len(rows)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))

@app.get("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
