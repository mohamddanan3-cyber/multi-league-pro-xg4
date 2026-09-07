# MULTI-LEAGUE PRO xG PREDICTOR V7 — Full Web App

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000

## API

- GET /api/leagues
- GET /api/league/E0
- GET /api/league/E0?refresh=true
- POST /api/predict
- GET /health

## Deploy

Push this folder to GitHub, create a Render Web Service, then use:

Build:
`pip install -r requirements.txt`

Start:
`uvicorn main:app --host 0.0.0.0 --port $PORT`

A render.yaml is included for Blueprint deployment.
