import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
from datetime import datetime
import logging

# === AUTO USB DETECTION ===
USB_ROOTS = ["X:/", "Y:/", "Z:/", "W:/"]
def get_usb_root():
    for root in USB_ROOTS:
        if os.path.exists(root):
            return root.rstrip("/\\") + "/"
    return "X:/"

usb_root = get_usb_root()
data_path = os.path.join(usb_root, "03_Data", "Live")
models_path = os.path.join(usb_root, "05_Models")
os.makedirs(models_path, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
print("TRAINING 11-FEATURE RF â€"" 2025 PRODUCTION PIPELINE")
logging.info("Starting 11-feature RF training...")

# === EXACT 11 FEATURES IN ORDER (SACRED ORDER â€"" DO NOT CHANGE) ===
FEATURES_11 = [
    "price", "atr", "ema_diff", "rsi", "macd_diff",
    "vwap", "price_vwap_diff", "momentum", "volume_delta",
    "spread", "sentiment"
]

# === LOAD ALL LIVE DATA ===
dfs = []
for symbol in ["EURUSD", "XAUUSD", "BTCUSD"]:
    for tf in ["M1", "M5", "M15", "M30", "H1", "H4"]:
        file = os.path.join(data_path, symbol, f"{tf}.csv")
        if os.path.exists(file):
            df = pd.read_csv(file)
            df['symbol'] = symbol
            df['tf'] = tf
            dfs.append(df)

if not dfs:
    print("NO DATA FOUND â€"" Run LIVE_DATA_COLLECTOR.ex5 first!")
    exit()

data = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(data):,} rows from live streams")

# === TARGET: NEXT BAR DIRECTION ===
data['return'] = data['close'].shift(-1) - data['close']
data['target'] = (data['return'] > 0).astype(int)
data = data.dropna().reset_index(drop=True)

# === EXTRACT 11 FEATURES ===
X = data[FEATURES_11].copy()
y = data['target']

# === TRAIN-TEST SPLIT (last 20% as test) ===
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# === SCALE ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === TRAIN RANDOM FOREST (OPTIMIZED HYPERPARAMETERS) ===
print("Training 11-feature Random Forest...")
rf = RandomForestClassifier(
    n_estimators=2000,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

# === EVALUATE ===
pred = rf.predict(X_test_scaled)
probs = rf.predict_proba(X_test_scaled)[:, 1]
accuracy = (pred == y_test).mean()
high_conf = probs[probs >= 0.98]
win_rate = (pred[probs >= 0.98] == y_test[probs >= 0.98]).mean() if len(high_conf) > 0 else 0

print("\n11-FEATURE RF TRAINING COMPLETE")
print(f"Accuracy: {accuracy:.1%}")
print(f"Signals â‰¥0.98 prob: {len(high_conf)}")
print(f"Win Rate on 0.98+ signals: {win_rate:.1%}")

# === SAVE MODEL + SCALER ===
joblib.dump(rf, os.path.join(models_path, "11_feature_rf.pkl"))
joblib.dump(scaler, os.path.join(models_path, "scaler.pkl"))
logging.info("11_feature_rf.pkl + scaler.pkl SAVED")

print(f"\nMODEL READY â†’ X:/05_Models/11_feature_rf.pkl")
print("WIN RATE ON 0.98+ SIGNALS:", f"{win_rate:.1%}")
print("\nNext: Run prediction_server.py â†’ get live signals via FastAPI + ZeroMQ")

# === AUTO LAUNCHER ===
with open("X:/TRAIN_11_COMPLETE.bat", "w") as f:
    f.write('@echo off\npython 02_Scripts\\Python\\prediction_server.py\npause')
print("\nLAUNCHER CREATED: X:/TRAIN_11_COMPLETE.bat")



