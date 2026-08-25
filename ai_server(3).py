import zmq
import json
import joblib
import numpy as np
import logging
import os
from datetime import datetime

# Setup logging
os.makedirs('03_Data/Logs', exist_ok=True)
logging.basicConfig(filename='03_Data/Logs/ai_server.log', level=logging.INFO,
                    format='%(asctime)s | %(message)s')

# Load models
models = {}
try:
    scaler = joblib.load("05_Models/scaler.pkl")
    print("Scaler loaded successfully")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

# Load all available models
model_files = [
    ("rf", "05_Models/11_feature_rf.pkl"),
    ("ens", "05_Models/ensemble_model.pkl"),
    ("my", "05_Models/my_model.pkl"),
    ("xgb", "05_Models/xgboost_model.json")
]

for name, file_path in model_files:
    try:
        if file_path.endswith(".json"):
            import xgboost as xgb
            m = xgb.XGBClassifier()
            m.load_model(file_path)
        else:
            m = joblib.load(file_path)
        models[name] = m
        print(f"Model {name} loaded successfully")
    except Exception as e:
        print(f"Error loading model {name}: {e}")
        continue

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5560")

print("AI_SERVER LIVE â€"" Listening on 5560")
print(f"Available models: {list(models.keys())}")

while True:
    try:
        msg = socket.recv_json()
        feats = np.array([msg["features"]])
       
        if scaler:
            feats = scaler.transform(feats)
       
        best_p, best_m = 0.5, "none"
        for n, m in models.items():
            try:
                p = m.predict_proba(feats)[0][1]
                if p > best_p:
                    best_p, best_m = p, n
            except Exception as e:
                print(f"Error predicting with model {n}: {e}")
                continue
       
        signal = "BUY" if best_p >= 0.98 else "SELL" if best_p <= 0.02 else "HOLD"
       
        reply = {"signal": signal, "prob": round(best_p, 4), "model": best_m}
        socket.send_json(reply)
       
        logging.info(f"SIGNAL | {msg['symbol']} | {best_m} | {best_p:.4f} | {signal}")
        print(f"SIGNAL | {msg['symbol']} | {best_m} | {best_p:.4f} | {signal}")
       
    except Exception as e:
        print(f"Error processing request: {e}")
        reply = {"signal": "ERROR", "prob": 0.0, "model": "ERROR"}
        socket.send_json(reply)

