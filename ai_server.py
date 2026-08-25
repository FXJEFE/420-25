# -*- coding: utf-8 -*-
"""
FXJEFE AI Server — serves predictions from per-symbol models (full_pipeline)
with fallback to legacy my_model.pkl for symbols without dedicated models.
"""
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
import os
import logging
from textblob import TextBlob

# Path to the config file
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

# Load the config file safely
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find config file at {CONFIG_PATH}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Config file has invalid format - {e}")
    exit(1)

# Set up logging using config
log_file = os.path.join(config['log_path'], 'ai_server.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info("AI server starting...")

app = Flask(__name__)

# ── Model loading ────────────────────────────────────────────────────────────

# Per-symbol models from full_pipeline.py (XGBoost .json or LightGBM .pkl)
symbol_models = {}   # {symbol: {'model': ..., 'features': [...], 'model_type': 'xgb'|'lgb'}}

# Legacy fallback model
legacy_model = None


def _load_symbol_models():
    """Scan models/ for per-symbol models produced by full_pipeline.py."""
    import glob
    models_dir = config['models_path']
    feature_files = glob.glob(os.path.join(models_dir, '*_features.json'))

    for feat_path in feature_files:
        try:
            with open(feat_path, 'r') as f:
                meta = json.load(f)
            symbol = meta['symbol']
            model_type = meta['model_type']
            features = meta['features']
            base = feat_path.replace('_features.json', '')

            model = None
            if model_type == 'xgb':
                model_path = base + '_xgb.json'
                if os.path.exists(model_path):
                    import xgboost as xgb
                    model = xgb.Booster()
                    model.load_model(model_path)
            elif model_type == 'lgb':
                model_path = base + '_lgb.pkl'
                if os.path.exists(model_path):
                    model = joblib.load(model_path)

            if model is not None:
                symbol_models[symbol] = {
                    'model': model,
                    'features': features,
                    'model_type': model_type,
                    'accuracy': meta.get('accuracy', 0),
                    'sharpe': meta.get('sharpe', 0),
                }
                logging.info(f"  Loaded {model_type.upper()} model for {symbol} "
                             f"({len(features)} features, acc={meta.get('accuracy',0):.1%})")
        except Exception as e:
            logging.warning(f"  Failed to load model from {feat_path}: {e}")


def _load_legacy_model():
    """Load the legacy my_model.pkl (RandomForest with 28 config features)."""
    global legacy_model
    model_path = os.path.join(config['models_path'], 'my_model.pkl')
    if os.path.exists(model_path):
        try:
            legacy_model = joblib.load(model_path)
            logging.info(f"  Legacy model loaded: {model_path}")
        except Exception as e:
            logging.error(f"  Failed to load legacy model: {e}")


# Load all models at startup
_load_symbol_models()
_load_legacy_model()

if symbol_models:
    logging.info(f"Per-symbol models ready: {list(symbol_models.keys())}")
if legacy_model is not None:
    logging.info("Legacy my_model.pkl ready as fallback")
if not symbol_models and legacy_model is None:
    logging.error("NO MODELS LOADED -- run full_pipeline.py or train_models.py first")

logging.info("AI server started and configuration loaded successfully")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "running",
        "models": list(symbol_models.keys()),
        "legacy_model": legacy_model is not None,
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        symbol = str(data.get('symbol', '')).replace('.r', '').strip()
        price = float(data.get('price', 0))
        atr = float(data.get('atr', 0.001))

        # ── Try per-symbol model first ──
        if symbol in symbol_models:
            entry = symbol_models[symbol]
            model = entry['model']
            model_features = entry['features']
            model_type = entry['model_type']

            # Build feature vector: use EA data where available, NaN for missing
            feat_values = []
            for feat_name in model_features:
                val = data.get(feat_name)
                if val is not None:
                    feat_values.append(float(val))
                else:
                    feat_values.append(np.nan)

            feat_array = np.array(feat_values, dtype=np.float32).reshape(1, -1)

            if model_type == 'xgb':
                import xgboost as xgb
                dmatrix = xgb.DMatrix(feat_array, feature_names=model_features)
                prob = float(model.predict(dmatrix)[0])
            else:  # lgb
                # LightGBM handles NaN natively
                prob = float(model.predict_proba(feat_array)[0][1])

            # Binary model: prob > 0.5 = buy, < 0.5 = sell
            # Use confidence band for hold (near 0.5 = uncertain)
            min_conf = config.get('min_confidence_threshold', 0.65)
            if prob >= min_conf:
                signal = 'buy'
                confidence = prob
            elif prob <= (1.0 - min_conf):
                signal = 'sell'
                confidence = 1.0 - prob
            else:
                signal = 'hold'
                confidence = 0.5

            stop_loss = price - (2 * atr) if signal == 'buy' else price + (2 * atr)

            logging.info(f"[{symbol}] {entry['model_type'].upper()} prob={prob:.3f} -> {signal} "
                         f"(conf={confidence:.2f})")
            return jsonify({
                "signal": signal,
                "confidence": float(confidence),
                "stop_loss": float(stop_loss),
                "model": f"{symbol}_{model_type}",
            })

        # ── Fallback to legacy model ──
        if legacy_model is not None:
            features = [data.get(feat, 0) for feat in config['features']]

            prediction = legacy_model.predict([features])[0]
            confidence = legacy_model.predict_proba([features])[0].max() if hasattr(legacy_model, 'predict_proba') else 0.5
            signal = {1: 'buy', 0: 'sell', -1: 'hold'}.get(prediction, 'hold')
            stop_loss = price - (2 * atr) if signal == 'buy' else price + (2 * atr)

            logging.info(f"[{symbol}] LEGACY pred={prediction} -> {signal} (conf={confidence:.2f})")
            return jsonify({
                "signal": signal,
                "confidence": float(confidence),
                "stop_loss": float(stop_loss),
                "model": "legacy",
            })

        logging.error("No model available for prediction")
        return jsonify({"error": "No model loaded"}), 500

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/sentiment', methods=['GET'])
def sentiment():
    symbol = request.args.get('symbol', '').replace('.r', '').strip()
    posts = {
        "EURUSD": "Bullish trend expected",
        "USDJPY": "Neutral market",
        "XAUUSD": "Bearish sentiment",
        "AUDUSD": "Positive outlook",
        "GBPUSD": "Strong buy signals",
        "USDCAD": "Sell pressure",
        "BTCUSD": "Volatile bullish momentum",
        "XRPUSD": "Speculative neutral",
    }
    text = posts.get(symbol, "Neutral")
    try:
        sentiment_score = TextBlob(text).sentiment.polarity
    except Exception as e:
        logging.warning(f"Sentiment analysis failed for {symbol}: {str(e)}")
        sentiment_score = 0.0
    logging.info(f"Sentiment for {symbol}: {sentiment_score}")
    return jsonify({"sentiment": float(sentiment_score)})


@app.route('/reload', methods=['POST'])
def reload_models():
    """Hot-reload models without restarting the server."""
    global symbol_models, legacy_model
    symbol_models = {}
    legacy_model = None
    _load_symbol_models()
    _load_legacy_model()
    logging.info("Models reloaded")
    return jsonify({
        "status": "reloaded",
        "models": list(symbol_models.keys()),
        "legacy_model": legacy_model is not None,
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)
