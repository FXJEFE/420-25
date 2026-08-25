# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import joblib
import requests
import numpy as np
import logging
from pathlib import Path
import json

from path_resolver import get_paths, get_config

# Initialize paths dynamically
paths = get_paths()
config = get_config()

app = Flask(__name__)
try:
    from tracing import instrument_flask_app
    instrument_flask_app(app)
except Exception:
    pass

# Dynamic paths
MODEL_PATH = paths.get_model_path("my_model.pkl")
XGB_API_URL = config.get('api_configuration', {}).get('api_endpoints', {}).get('ai_predict', 'http://127.0.0.1:5562/predict')
LOG_DIR = paths.logs_path
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "ai_server_ensemble.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("OG ELITE MAIN ENSEMBLE SERVER REBIRTH - MIRACLE + XGB INTEGRATION")
logger.info(f"Project root: {paths.project_root}")

# Load legacy miracle model
legacy_model = None
try:
    if MODEL_PATH.exists():
        legacy_model = joblib.load(MODEL_PATH)
        logger.info("SACRED LEGACY ENSEMBLE MODEL LOADED: my_model.pkl")
    else:
        logger.warning(f"Legacy model not found at {MODEL_PATH}")
except Exception as e:
    logger.error(f"Legacy load failed: {e}")

# Per-model feature sets from config
ml_cfg = config.get('ml_configuration', {})
models_cfg = ml_cfg.get('models', {})
FEATURES_LEGACY = ml_cfg.get(models_cfg.get('legacy_ensemble', {}).get('features', ''), ml_cfg.get('legacy_9_features', [
    "price", "atr", "rsi", "macd_diff", "vwap", "momentum", "volume_delta", "realized_vol", "spread"
]))
FEATURES_XGB = ml_cfg.get(models_cfg.get('xgboost', {}).get('features', ''), ml_cfg.get('full_43_features', [
    "price", "atr", "ema_diff", "rsi", "garch_vol", "macd_diff", "vwap", "price_vwap_diff",
    "bb_position", "roc", "stochastic", "cci", "williams", "momentum", "realized_vol",
    "chaikin_vol", "adx", "rvi", "obv", "volume_delta", "ad_line", "vol_osc",
    "supertrend", "hma", "ichimoku_tenkan", "sar", "dpo", "spread", "sentiment",
    "rsi_m5", "rsi_h1", "macd_diff_m5", "macd_diff_h1", "atr_m5", "atr_h1",
    "vwap_m5", "vwap_h1", "roc_m5", "roc_h1", "stochastic_m5", "stochastic_h1",
    "cci_m5", "cci_h1"
]))


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "running",
        "legacy_model": "loaded" if legacy_model else "missing",
        "xgb_api": "available" if legacy_model else "fallback",
        "confidence_gate": "98%",
        "project_root": str(paths.project_root)
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON"}), 400

        symbol = data.get("symbol", "UNKNOWN")

        # Build feature vectors per model
        legacy_vector = [float(data.get(f, 0.0)) for f in FEATURES_LEGACY]

        predictions = []
        confidences = []
        models_used = []

        # 1. Legacy Ensemble Prediction (Primary)
        if legacy_model:
            try:
                pred = legacy_model.predict([legacy_vector])[0]
                if hasattr(legacy_model, "predict_proba"):
                    probs = legacy_model.predict_proba([legacy_vector])[0]
                    conf = float(np.max(probs))
                else:
                    conf = 0.85
                signal = {1: "buy", 0: "hold", -1: "sell"}.get(int(pred), "hold")
                predictions.append(signal)
                confidences.append(conf)
                models_used.append("legacy_ensemble")
                logger.info(f"Legacy: {signal.upper()} | CONF: {conf:.4f}")
            except Exception as e:
                logger.warning(f"Legacy predict failed: {e}")

        # 2. XGBoost API Integration (Secondary/Fallback)
        try:
            # Forward original data; XGB service handles its 43-feature set
            xgb_resp = requests.post(XGB_API_URL, json=data, timeout=5)
            if xgb_resp.status_code == 200:
                xgb_pred = xgb_resp.json()
                predictions.append(xgb_pred["signal"])
                confidences.append(xgb_pred["confidence"])
                models_used.append("xgboost")
                logger.info(f"XGB: {xgb_pred['signal'].upper()} | CONF: {xgb_pred['confidence']:.4f}")
        except Exception as e:
            logger.warning(f"XGB API unreachable: {e}")

        # Ensemble Voting
        if not predictions:
            return jsonify({"error": "No predictions available"}), 500

        # Weighted average confidence (legacy preferred)
        final_conf = np.average(confidences, weights=[0.7 if "legacy" in m else 0.3 for m in models_used])

        # Majority vote signal
        from collections import Counter
        vote = Counter(predictions).most_common(1)[0][0]

        # SACRED 98% GATE
        if final_conf < 0.98:
            final_signal = "hold"
            final_conf = 0.0
            logger.info(f"98% GATE BLOCKED | {symbol} | Vote: {vote.upper()} {final_conf:.3f} -> HOLD")
        else:
            final_signal = vote
            logger.info(f"ELITE ENSEMBLE SIGNAL | {symbol} | {final_signal.upper()} | CONF: {final_conf:.4f} | Models: {models_used}")

        price = float(data.get("price", 1.0))
        atr = float(data.get("atr", 0.001))
        sl = price - (2 * atr) if final_signal == "buy" else price + (2 * atr) if final_signal == "sell" else 0.0

        return jsonify({
            "signal": final_signal,
            "confidence": round(final_conf, 4),
            "stop_loss": round(sl, 5),
            "model_used": "+".join(models_used),
            "gate_passed": final_conf >= 0.98
        })

    except Exception as e:
        logger.error(f"Ensemble error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = config.get('api_configuration', {}).get('ports', {}).get('ai_server', 5561)
    logger.info(f"OG ELITE ENSEMBLE SERVER STARTING ON PORT {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
