# -*- coding: utf-8 -*-
"""
AI Server for FXJEFE -- Multi-model prediction server for Forex + Crypto.

Auto-discovers all trained models (XGBoost, LightGBM, CatBoost) in models/ dir.
Routes predictions by symbol + timeframe. Supports ensemble predictions when
multiple model types exist for the same symbol/tf.

Endpoints:
    GET  /health              -- {"status": "running", "models": {...}}
    POST /predict             -- {"signal": "buy"/"sell"/"hold", "confidence": 0.XX, ...}
    GET  /models              -- list of all loaded models
    GET  /sentiment           -- {"sentiment": 0.0}  (placeholder)
"""
from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import numpy as np
import json
import os
import glob
import logging
import joblib
from datetime import datetime

# -- Config -------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.json')
CRYPTO_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'crypto_config.json')

try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error loading config: {e}")
    exit(1)

# Load crypto config if available (not required for forex-only)
crypto_config = {}
if os.path.exists(CRYPTO_CONFIG_PATH):
    try:
        with open(CRYPTO_CONFIG_PATH, 'r') as f:
            crypto_config = json.load(f)
    except Exception:
        pass

MODELS_DIR = config['models_path']
HIST_DIR = os.path.join(config['data_output_path'], 'Historical', 'enhanced')
LOG_DIR = config['log_path']
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, 'ai_server.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
)

# -- Import feature computation from full_pipeline ---------------------------
import sys
sys.path.insert(0, PROJECT_ROOT)
from full_pipeline import compute_all_features


# -----------------------------------------------------------------------------
# MODEL REGISTRY -- auto-discovers all models
# -----------------------------------------------------------------------------

_models = {}  # {(symbol, tf): {'xgb': model, 'lgb': model, 'cat': model, 'features': [...], 'weights': [...]}}


def _load_all_models():
    """Auto-discover and load all trained models from models/ directory."""
    global _models
    feature_files = glob.glob(os.path.join(MODELS_DIR, '*_binary_features.json'))

    for fpath in feature_files:
        try:
            with open(fpath, 'r') as f:
                meta = json.load(f)

            symbol = meta['symbol']
            tf = meta['timeframe']
            features = meta['features']
            model_type = meta.get('model_type', 'xgb')
            name = f"{symbol}_{tf}_binary"

            entry = _models.get((symbol, tf), {
                'features': features,
                'n_features': len(features),
                'weights': meta.get('ensemble_weights', [1.0, 0.0, 0.0]),
                'xgb': None, 'lgb': None, 'cat': None,
            })

            # Update features if this is a newer/better metadata file
            if len(features) >= entry['n_features']:
                entry['features'] = features
                entry['n_features'] = len(features)
                if 'ensemble_weights' in meta:
                    entry['weights'] = meta['ensemble_weights']

            # Load XGBoost
            xgb_path = os.path.join(MODELS_DIR, f'{name}_xgb.json')
            if os.path.exists(xgb_path) and entry['xgb'] is None:
                m = xgb.Booster()
                m.load_model(xgb_path)
                entry['xgb'] = m

            # Load LightGBM
            lgb_path = os.path.join(MODELS_DIR, f'{name}_lgb.pkl')
            if os.path.exists(lgb_path) and entry['lgb'] is None:
                entry['lgb'] = joblib.load(lgb_path)

            # Load CatBoost
            cat_path = os.path.join(MODELS_DIR, f'{name}_catboost.cbm')
            if os.path.exists(cat_path) and entry['cat'] is None:
                try:
                    from catboost import CatBoostClassifier
                    m = CatBoostClassifier()
                    m.load_model(cat_path)
                    entry['cat'] = m
                except ImportError:
                    pass

            loaded = [k for k in ('xgb', 'lgb', 'cat') if entry[k] is not None]
            if loaded:
                _models[(symbol, tf)] = entry
                logging.info(f"Loaded {symbol} {tf}: {'+'.join(loaded)} ({len(features)} features)")

        except Exception as e:
            logging.warning(f"Failed to load {os.path.basename(fpath)}: {e}")

    logging.info(f"Model registry: {len(_models)} symbol/tf combos loaded")


# -- Feature cache per symbol/tf ---------------------------------------------
_feature_cache = {}  # {(symbol, tf): {'df': DataFrame, 'time': datetime}}
CACHE_TTL_SECONDS = 3600


def _load_and_compute_features(symbol, timeframe):
    """Load OHLCV from enhanced CSV, compute all features."""
    cache_key = (symbol, timeframe)
    now = datetime.now()

    if cache_key in _feature_cache:
        cached = _feature_cache[cache_key]
        elapsed = (now - cached['time']).total_seconds()
        if elapsed < CACHE_TTL_SECONDS:
            return cached['df']

    tf_map = {'D1': 'Daily', 'W1': 'Weekly'}
    tf_name = tf_map.get(timeframe, timeframe)
    pattern = os.path.join(HIST_DIR, f'enhanced_{symbol}_{tf_name}_*.csv')
    matches = glob.glob(pattern)

    if not matches:
        logging.error(f"No OHLCV data found for {symbol} {timeframe} in {HIST_DIR}")
        return None

    df = pd.read_csv(matches[0])
    logging.info(f"Loaded {len(df)} rows from {os.path.basename(matches[0])}")

    if 'open' in df.columns and 'close' in df.columns:
        computed = compute_all_features(df)
        for c in computed.columns:
            if c not in df.columns:
                df[c] = computed[c].values
            else:
                mask = df[c].isna()
                if mask.any():
                    df.loc[mask, c] = computed[c].values[mask.values]
        logging.info(f"Features computed: {len(computed.columns)} total, {len(df.columns)} in dataframe")

    _feature_cache[cache_key] = {'df': df, 'time': now}
    return df


def _predict_with_model(entry, latest):
    """Run prediction through available models, return ensemble probability."""
    features = entry['features']

    X = pd.DataFrame()
    for feat in features:
        if feat in latest.columns:
            X[feat] = latest[feat].values
        else:
            X[feat] = [0.0]

    X_arr = X.values.astype(np.float32)
    probs = {}
    weights = entry['weights']

    # XGBoost
    if entry['xgb'] is not None:
        try:
            dmat = xgb.DMatrix(X_arr, feature_names=features)
            probs['xgb'] = float(entry['xgb'].predict(dmat)[0])
        except Exception as e:
            logging.warning(f"XGB prediction failed: {e}")

    # LightGBM
    if entry['lgb'] is not None:
        try:
            probs['lgb'] = float(entry['lgb'].predict_proba(X_arr)[:, 1][0])
        except Exception as e:
            logging.warning(f"LGB prediction failed (feature mismatch?): {e}")

    # CatBoost
    if entry['cat'] is not None:
        try:
            probs['cat'] = float(entry['cat'].predict_proba(X_arr)[:, 1][0])
        except Exception as e:
            logging.warning(f"CatBoost prediction failed: {e}")

    if not probs:
        return None, {}

    # Ensemble
    if len(probs) == 3 and all(k in probs for k in ('xgb', 'lgb', 'cat')):
        prob = weights[0] * probs['xgb'] + weights[1] * probs['lgb'] + weights[2] * probs['cat']
    elif len(probs) >= 2:
        prob = np.mean(list(probs.values()))
    else:
        prob = list(probs.values())[0]

    return float(prob), probs


# -- Flask App ----------------------------------------------------------------
app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    model_info = {}
    for (sym, tf), entry in _models.items():
        loaded = [k for k in ('xgb', 'lgb', 'cat') if entry[k] is not None]
        model_info[f"{sym}_{tf}"] = {
            'models': loaded,
            'n_features': entry['n_features'],
        }
    return jsonify({
        "status": "running",
        "n_model_combos": len(_models),
        "models": model_info,
    })


@app.route('/models', methods=['GET'])
def list_models():
    """List all loaded models with details."""
    result = []
    for (sym, tf), entry in _models.items():
        loaded = [k for k in ('xgb', 'lgb', 'cat') if entry[k] is not None]
        result.append({
            'symbol': sym,
            'timeframe': tf,
            'model_types': loaded,
            'n_features': entry['n_features'],
            'ensemble_weights': entry['weights'] if len(loaded) > 1 else None,
        })
    return jsonify(result)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'EURUSD')
        timeframe = data.get('timeframe', 'D1')

        # Normalize symbol: strip .r suffix (FTMO compat)
        clean_symbol = symbol.replace('.r', '').replace('.R', '')

        # Find matching model
        entry = _models.get((clean_symbol, timeframe))
        if entry is None:
            # Try just D1 for this symbol
            entry = _models.get((clean_symbol, 'D1'))
            if entry is not None:
                timeframe = 'D1'

        if entry is None:
            logging.info(f"No model for {clean_symbol} {timeframe} -- returning hold")
            return jsonify({
                "signal": "hold",
                "confidence": 0.0,
                "stop_loss": 0.0,
                "price": float(data.get('price', 0)),
                "atr": float(data.get('atr', 0)),
                "reason": f"No model for {clean_symbol} {timeframe}",
                "available_models": [f"{s}_{t}" for s, t in _models.keys()]
            })

        # Load OHLCV and compute features
        df = _load_and_compute_features(clean_symbol, timeframe)
        if df is None or df.empty:
            return jsonify({"error": f"No data available for {clean_symbol} {timeframe}"}), 500

        # Use the last COMPLETED bar for prediction (no lookahead)
        if 'time' in df.columns:
            last_time = pd.to_datetime(df['time'].iloc[-1])
            today = pd.Timestamp.now().normalize()
            if last_time >= today and len(df) > 1:
                latest = df.iloc[-2:-1]
                logging.info(f"Using previous completed bar ({df['time'].iloc[-2]}) -- today's bar incomplete")
            else:
                latest = df.iloc[-1:]
        else:
            latest = df.iloc[-1:]

        # Run prediction
        prob, model_probs = _predict_with_model(entry, latest)
        if prob is None:
            return jsonify({"error": "No predictions available"}), 500

        signal = 'buy' if prob > 0.5 else 'sell'
        confidence = float(prob if prob > 0.5 else 1.0 - prob)

        # ATR and price
        atr = float(latest['atr'].values[0]) if 'atr' in latest.columns else 0.0
        price = float(data.get('price', latest['price'].values[0] if 'price' in latest.columns else 0.0))

        # Stop loss
        if atr > 0 and price > 0:
            stop_loss = price - (2 * atr) if signal == 'buy' else price + (2 * atr)
        else:
            stop_loss = 0.0

        loaded_models = [k for k in ('xgb', 'lgb', 'cat') if entry[k] is not None]
        logging.info(f"Prediction for {clean_symbol} {timeframe}: {signal} "
                     f"(prob={prob:.4f}, conf={confidence:.4f}, models={'+'.join(loaded_models)})")

        return jsonify({
            "signal": signal,
            "confidence": confidence,
            "probability": prob,
            "stop_loss": float(stop_loss),
            "price": price,
            "atr": atr,
            "symbol": clean_symbol,
            "timeframe": timeframe,
            "models_used": loaded_models,
            "model_probs": model_probs,
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/sentiment', methods=['GET'])
def sentiment():
    """Placeholder -- sentiment not used by D1 model."""
    return jsonify({"sentiment": 0.0})


if __name__ == "__main__":
    logging.info("Loading all models...")
    _load_all_models()

    # Pre-load features for all loaded symbol/tf combos
    for (sym, tf) in list(_models.keys()):
        logging.info(f"Pre-loading {sym} {tf} features...")
        _load_and_compute_features(sym, tf)

    logging.info(f"AI Server ready -- {len(_models)} model combos loaded")
    app.run(host='0.0.0.0', port=8080, debug=False)
