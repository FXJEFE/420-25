from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import json

app = FastAPI()
scaler = joblib.load("05_Models/scaler.pkl")
model = joblib.load("05_Models/11_feature_rf.pkl")

class Features(BaseModel):
    features: list[float]
    symbol: str

@app.post("/predict")
def predict(data: Features):
    X = np.array([data.features])
    X = scaler.transform(X)
    prob = model.predict_proba(X)[0][1]
    signal = "BUY" if prob >= 0.98 else "SELL" if prob <= 0.02 else "HOLD"
    return {"signal": signal, "prob": round(prob, 4), "model": "11_feature_rf"}







