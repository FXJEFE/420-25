#!/usr/bin/env python3
"""
ai_server_golden.py — FXJEFE Golden Multi-Model Ensemble Server v5.03
Port 8080 | 0.77 confidence gate | Consensus voting (needle-threading EDGE)
Date: 2026-05-05 | Aligned to 29 features, no retraining required
"""

import os
from pathlib import Path
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import onnxruntime as ort

# ====================== CONFIG ======================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

MODELS_DIR = config.get("models_path", PROJECT_ROOT)  # FIXED: flat key, project root
LOG_DIR = config.get("log_path", os.path.join(PROJECT_ROOT, "Logs"))
LOG_DIR = LOG_DIR or str(Path(__file__).resolve().parent / 'Logs')
os.makedirs(LOG_DIR, exist_ok=True)

MIN_CONF = config.get("min_confidence_threshold", 0.77)
GOLDEN_WEIGHTS = config.get("golden_weights", {"xgb_6": 0.35, "avg_9feat": 0.40, "rf_28": 0.25})

# 29 features — EXACT ORDER for all models
FEATURES = config.get("features", [
    "price","atr","ema_diff","rsi","garch_vol","macd_diff",
    "vwap","price_vwap_diff","bb_position","roc","stochastic",
    "cci","williams","momentum","realized_vol","chaikin_vol",
    "adx","rvi","obv","volume_delta","ad_line","vol_osc",
    "supertrend","hma","ichimoku_tenkan","sar","dpo","spread","sentiment"
])

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "golden_server.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GoldenServer")

# ====================== MODEL LOADING ======================
models = {}
model_meta = {}

def load_model_safe(name, path, mtype, feat_count):
    full_path = os.path.join(MODELS_DIR, path) if not os.path.isabs(path) else path
    try:
        if mtype == "pkl":
            m = joblib.load(full_path)
            models[name] = m
            model_meta[name] = {"type": "pkl", "features": feat_count}
            logger.info(f"✅ Loaded {name} ({mtype}) — {feat_count} feats")
        elif mtype == "json":  # XGBoost Booster
            m = xgb.Booster()
            m.load_model(full_path)
            models[name] = m
            model_meta[name] = {"type": "xgb", "features": feat_count}
            logger.info(f"✅ Loaded {name} (XGBoost JSON) — {feat_count} feats")
        elif mtype == "onnx":
            m = ort.InferenceSession(full_path)
            models[name] = m
            model_meta[name] = {"type": "onnx", "features": feat_count}
            logger.info(f"✅ Loaded {name} (ONNX) — {feat_count} feats")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Skipped {name}: {str(e)}")
        return False

# Core golden models (root directory)
load_model_safe("xgb_6", "xgboost_model.json", "json", 6)
load_model_safe("ensemble_9a", "ensamble_model.pkl", "pkl", 9)
load_model_safe("rf_9b", "my_model (2).pkl", "pkl", 9)
load_model_safe("voting_9c", "my_model (3).pkl", "pkl", 9)
load_model_safe("rf_9d", "my_model - Copy.pkl", "pkl", 9)
load_model_safe("rf_28", "my_model.pkl", "pkl", 28)

# Optional extra voters (models/ subdir, half weight)
OPTIONAL_FILES = {
    "rf_big":    ("models/my_modelbig.pkl",        "pkl", 28),
    "stacking":  ("models/stacking_model.pkl",     "pkl", 28),
    "lgbm_onnx": ("models/lightgbm_model.onnx",   "onnx", 28),
}
for name, (path, mtype, fc) in OPTIONAL_FILES.items():
    load_model_safe(name, path, mtype, fc)

logger.info(f"Golden Server ready — {len(models)} models loaded | Gate={MIN_CONF}")

# ====================== HELPER FUNCTIONS ======================
def prepare_features(payload):
    """Extract 29 features in exact order. Fills missing with 0.0."""
    vec = []
    for f in FEATURES:
        val = payload.get(f, 0.0)
        if isinstance(val, (int, float)):
            vec.append(float(val))
        else:
            vec.append(0.0)
    return np.array(vec).reshape(1, -1)

def predict_model(name, X):
    meta = model_meta.get(name, {})
    m = models.get(name)
    if m is None:
        return "hold", 0.0

    try:
        if meta["type"] == "xgb":
            dmat = xgb.DMatrix(X)
            proba = m.predict(dmat)[0]
            signal = "buy" if proba > 0.5 else "sell" if proba < 0.5 else "hold"
            conf = float(abs(proba - 0.5) * 2)
        elif meta["type"] == "onnx":
            input_name = m.get_inputs()[0].name
            proba = m.run(None, {input_name: X.astype(np.float32)})[0][0][1]
            signal = "buy" if proba > 0.5 else "sell"
            conf = float(abs(proba - 0.5) * 2)
        else:  # pkl (sklearn)
            if hasattr(m, "predict_proba"):
                proba = m.predict_proba(X)[0][1]
            else:
                proba = m.predict(X)[0]
            signal = "buy" if proba > 0.5 else "sell" if proba < 0.5 else "hold"
            conf = float(abs(proba - 0.5) * 2)
        return signal, conf
    except Exception as e:
        logger.error(f"Prediction error in {name}: {e}")
        return "hold", 0.0

def compute_stop_loss(signal, price, atr):
    """Simple ATR-based stop (student can replace with model output)."""
    if signal == "buy":
        return round(price - 1.5 * atr, 5)
    elif signal == "sell":
        return round(price + 1.5 * atr, 5)
    return price

# ====================== ENDPOINTS ======================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "loaded_models": list(models.keys()),
        "gate": MIN_CONF,
        "features_count": len(FEATURES),
        "version": "5.03-golden"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        symbol = payload.get("symbol", "UNKNOWN")
        price = float(payload.get("price", 0.0))
        atr = float(payload.get("atr", 0.0001))

        X = prepare_features(payload)

        # === GROUP PREDICTIONS ===
        # Group 1: XGB 6-feat (fast filter)
        xgb_sig, xgb_conf = predict_model("xgb_6", X[:, :6]) if "xgb_6" in models else ("hold", 0.0)

        # Group 2: 9-feature models (average vote)
        nine_sigs = []
        for n in ["ensemble_9a", "rf_9b", "voting_9c", "rf_9d"]:
            if n in models:
                s, c = predict_model(n, X[:, :9])
                nine_sigs.append(s)
        nine_signal = max(set(nine_sigs), key=nine_sigs.count) if nine_sigs else "hold"
        nine_conf = np.mean([c for s, c in [predict_model(n, X[:, :9]) for n in ["ensemble_9a", "rf_9b", "voting_9c", "rf_9d"] if n in models]] or [0.0])

        # Group 3: 28-feature models
        full_sigs = []
        for n in ["rf_28"] + [k for k in ["rf_big", "stacking", "lgbm_onnx"] if k in models]:
            if n in models:
                s, c = predict_model(n, X)
                full_sigs.append(s)
        full_signal = max(set(full_sigs), key=full_sigs.count) if full_sigs else "hold"
        full_conf = np.mean([c for s, c in [predict_model(n, X) for n in ["rf_28"] + [k for k in ["rf_big", "stacking", "lgbm_onnx"] if k in models]] or [0.0]])

        # === NEEDLE-THREADING CONSENSUS ===
        group_signals = [xgb_sig, nine_signal, full_signal]
        active = [s for s in group_signals if s != "hold"]
        if len(active) == len(group_signals) and len(set(active)) == 1:
            final_signal = active[0]
            # Weighted confidence (respect golden_weights)
            final_conf = (xgb_conf * GOLDEN_WEIGHTS.get("xgb_6", 0.35) +
                          nine_conf * GOLDEN_WEIGHTS.get("avg_9feat", 0.40) +
                          full_conf * GOLDEN_WEIGHTS.get("rf_28", 0.25))
        else:
            final_signal = "hold"
            final_conf = 0.0

        # === 0.77 GATE ===
        probability = max(final_conf, 0.5)  # conservative
        if final_signal != "hold" and final_conf < MIN_CONF:
            logger.info(f"[{symbol}] GATE BLOCK: conf={final_conf:.4f} < {MIN_CONF} → HOLD")
            final_signal = "hold"
            final_conf = 0.0

        n_models = len([m for m in models if m in ["xgb_6", "ensemble_9a", "rf_9b", "voting_9c", "rf_9d", "rf_28"]])

        stop_loss = compute_stop_loss(final_signal, price, atr)

        response = {
            "symbol": symbol,
            "signal": final_signal,
            "confidence": round(final_conf, 4),
            "probability": round(probability, 6),
            "n_models": n_models,
            "stop_loss": stop_loss,
            "timestamp": int(datetime.utcnow().timestamp()),
            "gate": MIN_CONF
        }

        logger.info(f"[{symbol}] {final_signal.upper()} | conf={final_conf:.4f} | models={n_models} | groups={group_signals}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Predict error: {str(e)}")
        return jsonify({"signal": "hold", "confidence": 0.0, "error": str(e)}), 500

if __name__ == "__main__":
    logger.info("🚀 Starting FXJEFE Golden Server on port 8080...")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
