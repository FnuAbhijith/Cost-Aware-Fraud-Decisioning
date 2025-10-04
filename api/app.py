import os, joblib, numpy as np, pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

ART_DIR = "model/artifacts"
API_KEY = os.getenv("API_KEY", "change-me")
pipe    = joblib.load(f"{ART_DIR}/pipeline.pkl")
metrics = joblib.load(f"{ART_DIR}/metrics.pkl")
best_thr = float(metrics["threshold"])

class Txn(BaseModel):
    amount: float
    device_risk: float
    geo_distance_km: float
    bin_risk: float
    country: str | None = None

app = FastAPI(title="Fraud Scoring API (XGBoost)")

@app.get("/health")
def health(): return {"status":"ok","pr_auc":metrics["ap"],"threshold":best_thr}

@app.post("/score")
def score(txn:Txn, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    x_df = pd.DataFrame([txn.model_dump()])
    proba = float(pipe.predict_proba(x_df)[:,1][0])
    return {"fraud_proba": round(proba,6), "decision": int(proba>=best_thr), "threshold": best_thr}
