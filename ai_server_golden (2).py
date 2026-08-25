# -*- coding: utf-8 -*-
"""
ai_server_golden.py
FXJEFE Golden Model Server — serves ONLY the April-June 2025 golden models.

Models loaded (feature count / type):
  xgboost_model.json      6-feat  XGBoost Booster      (primary fast filter)
  ensamble_model.pkl      9-feat  VotingClassifier      (April 2025 golden)
  my_model (2).pkl        9-feat  VotingClassifier      (April 2025 golden)
  my_model (3).pkl        9-feat  VotingClassifier      (April 2025 golden)
  my_model - Copy.pkl     9-feat  RandomForest          (June 2025 golden)
  my_model.pkl           28-feat  RandomForest          (full-feature fallback)

my_modelOG.pkl is SKIPPED — corrupted binary (UTF-8 re-encoding on USB copy).

Ensemble weights (configurable via config.json 'golden_weights'):
  xgb_6     : 0.35
  avg_9feat : 0.40  (mean of 4 nine-feature models)
  rf_28     : 0.25

Feature subsets:
  6-feat  : price, atr, ema_diff, rsi, garch_vol, macd_diff
  9-feat  : price, atr, ema_diff, rsi, garch_vol, macd_diff,
            vwap, price_vwap_diff, bb_position
  28-feat : full config.json['features'] list

Endpoints:
  GET  /health   -- server status + loaded model names
  POST /predict  -- send all 28 features; server extracts subsets automatically
  POST /reload   -- hot-reload models without restart

Usage:
  python ai_server_golden.py
"""

from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np
import json
import os
import csv
import logging
import joblib
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH  = os.path.join(PROJECT_ROOT, 'config.json')

with open(CONFIG_PATH, 'r', encoding='utf-8') as _f:
    config = json.load(_f)

MODELS_DIR = config['models_path']
LOG_DIR    = config['log_path']
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'ai_server_golden.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Exact golden feature sets (locked from inspect_models.py output) ──────────
FEATURES_6 = ['price', 'atr', 'ema_diff', 'rsi', 'garch_vol', 'macd_diff']

FEATURES_9 = ['price', 'atr', 'ema_diff', 'rsi', 'garch_vol', 'macd_diff',
               'vwap', 'price_vwap_diff', 'bb_position']

FEATURES_28 = config['features']   # exact list from config.json

# Ensemble weights: must sum to 1.0
_w = config.get('golden_weights', {})
W_XGB_6   = _w.get('xgb_6',    0.35)
W_AVG_9   = _w.get('avg_9feat', 0.40)
W_RF_28   = _w.get('rf_28',     0.25)

# ── Golden model file registry ────────────────────────────────────────────────
GOLDEN_FILES = {
    'xgb_6':      ('xgboost_model.json',  'xgb', FEATURES_6),
    'ensemble_9a': ('ensamble_model.pkl',  'pkl', FEATURES_9),
    'rf_9b':       ('my_model (2).pkl',    'pkl', FEATURES_9),
    'voting_9c':   ('my_model (3).pkl',    'pkl', FEATURES_9),
    'rf_9d':       ('my_model - Copy.pkl', 'pkl', FEATURES_9),
    'rf_28':       ('my_model.pkl',        'pkl', FEATURES_28),
}

# Populated at startup
_loaded = {}   # {key: {'model': obj, 'type': 'xgb'|'pkl', 'features': list}}


def _load_all():
    global _loaded
    _loaded = {}
    for key, (fname, mtype, feats) in GOLDEN_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            log.warning(f"  MISSING: {fname} -- skipping")
            continue
        try:
            if mtype == 'xgb':
                m = xgb.Booster()
                m.load_model(path)
            else:
                m = joblib.load(path)
            _loaded[key] = {'model': m, 'type': mtype, 'features': feats}
            log.info(f"  Loaded [{key}] {fname}  ({len(feats)} features)")
        except Exception as e:
            log.error(f"  FAILED to load {fname}: {e}")

    log.info(f"Golden registry: {len(_loaded)}/{len(GOLDEN_FILES)} models loaded")
    if not _loaded:
        log.error("NO MODELS LOADED — check models/ directory")


# ── Prediction helpers ────────────────────────────────────────────────────────

def _extract(data: dict, features: list) -> np.ndarray:
    """Pull feature values from request dict, fill missing with 0."""
    vals = [float(data.get(f, 0.0)) for f in features]
    return np.array(vals, dtype=np.float32).reshape(1, -1)


def _safe_float(x) -> float:
    """Robust scalar conversion — handles 0-d/1-d numpy arrays, lists, and Python scalars."""
    if isinstance(x, np.ndarray):
        flat = x.ravel()
        if flat.size == 0:
            return 0.0
        return float(flat[0])
    if isinstance(x, (list, tuple)):
        return float(x[0]) if len(x) else 0.0
    return float(x)


def _prob_from_model(key: str, entry: dict, data: dict) -> float | None:
    """Return buy probability [0, 1] or None on failure."""
    arr   = _extract(data, entry['features'])
    model = entry['model']
    try:
        if entry['type'] == 'xgb':
            dmat = xgb.DMatrix(arr, feature_names=entry['features'])
            raw = model.predict(dmat)
            return _safe_float(raw[0] if hasattr(raw, '__len__') and len(raw) else raw)
        else:
            proba = model.predict_proba(arr)
            # proba shape is (n_samples, n_classes); take positive-class prob for sample 0
            proba_arr = np.asarray(proba)
            if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
                return _safe_float(proba_arr[0, 1])
            # Binary edge case: some models return (n_samples,)
            return _safe_float(proba_arr.ravel()[0])
    except Exception as e:
        log.warning(f"Prediction error [{key}] ({type(e).__name__}): {e}")
        return None


def _ensemble(data: dict) -> tuple[float, dict]:
    """
    Run all loaded models, return (ensemble_prob, per_model_probs).
    Ensemble = W_XGB_6 * xgb_prob + W_AVG_9 * mean(9feat_probs) + W_RF_28 * rf28_prob
    """
    probs = {}
    for key, entry in _loaded.items():
        p = _prob_from_model(key, entry, data)
        if p is not None:
            probs[key] = round(p, 6)

    if not probs:
        return 0.5, {}

    # Group by feature set size
    xgb_probs  = [probs[k] for k in ('xgb_6',) if k in probs]
    feat9_probs = [probs[k] for k in ('ensemble_9a', 'rf_9b', 'voting_9c', 'rf_9d') if k in probs]
    feat28_probs = [probs[k] for k in ('rf_28',) if k in probs]

    components = []
    weights_used = []

    if xgb_probs:
        components.append(xgb_probs[0])
        weights_used.append(W_XGB_6)

    if feat9_probs:
        components.append(float(np.mean(feat9_probs)))
        weights_used.append(W_AVG_9)

    if feat28_probs:
        components.append(feat28_probs[0])
        weights_used.append(W_RF_28)

    # Renormalise weights for however many groups we actually got
    total_w = sum(weights_used)
    ensemble_prob = sum(c * w / total_w for c, w in zip(components, weights_used))

    return float(ensemble_prob), probs


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    models_info = {k: {'features': len(v['features']), 'type': v['type']}
                   for k, v in _loaded.items()}
    return jsonify({
        'status': 'running',
        'loaded_models': len(_loaded),
        'models': models_info,
        'feature_sets': {'6-feat': FEATURES_6, '9-feat': FEATURES_9,
                         '28-feat (count)': len(FEATURES_28)},
    })


def _audit_decision(row: dict) -> None:
    """Append one decision row to Logs/audit_YYYYMMDD.csv (header on first write)."""
    try:
        path = os.path.join(LOG_DIR, f"audit_{datetime.utcnow().strftime('%Y%m%d')}.csv")
        new_file = not os.path.exists(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(['timestamp_utc', 'symbol', 'timeframe', 'signal',
                            'confidence', 'probability', 'price', 'atr',
                            'stop_loss', 'n_models', 'model_probs'])
            w.writerow([row['timestamp_utc'], row['symbol'], row['timeframe'], row['signal'],
                        row['confidence'], row['probability'], row['price'], row['atr'],
                        row['stop_loss'], row['n_models'], json.dumps(row['model_probs'])])
    except Exception as e:
        log.warning(f"Audit write failed: {e}")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True) or {}

        symbol    = str(data.get('symbol', '')).replace('.r', '').replace('.R', '').upper()
        price     = float(data.get('price', 0))
        atr       = float(data.get('atr', 0.001))
        timeframe = str(data.get('timeframe', 'M15')).upper()

        prob, model_probs = _ensemble(data)

        # Hard 0.70 floor — even if config is misconfigured, never go below the user's gate.
        min_conf = max(float(config.get('min_confidence_threshold', 0.70)), 0.70)

        if prob >= min_conf:
            signal, confidence = 'buy',  prob
        elif prob <= (1.0 - min_conf):
            signal, confidence = 'sell', 1.0 - prob
        else:
            signal, confidence = 'hold', max(prob, 1.0 - prob)

        stop_loss = (price - 2 * atr) if signal == 'buy' else (price + 2 * atr)

        log.info(f"[{symbol} {timeframe}] prob={prob:.4f} -> {signal.upper()} "
                 f"conf={confidence:.3f}  models={list(model_probs.keys())}  gate={min_conf}")

        _audit_decision({
            'timestamp_utc': datetime.utcnow().isoformat(timespec='seconds'),
            'symbol': symbol, 'timeframe': timeframe, 'signal': signal,
            'confidence': round(confidence, 4), 'probability': round(prob, 6),
            'price': price, 'atr': atr, 'stop_loss': round(stop_loss, 5),
            'n_models': len(model_probs), 'model_probs': model_probs,
        })

        return jsonify({
            'signal':     signal,
            'confidence': round(confidence, 4),
            'probability': round(prob, 6),
            'stop_loss':  round(stop_loss, 5),
            'symbol':     symbol,
            'timeframe':  timeframe,
            'model_probs': model_probs,
            'n_models':   len(model_probs),
            'min_conf_gate': min_conf,
        })

    except Exception as e:
        log.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'signal': 'hold', 'confidence': 0.0, 'error': str(e)}), 500


@app.route('/predict/sentiment', methods=['GET'])
@app.route('/sentiment', methods=['GET'])
def sentiment():
    symbol = request.args.get('symbol', '').replace('.r', '').strip().upper()
    try:
        from textblob import TextBlob
        texts = {
            'EURUSD': 'Bullish trend expected', 'USDJPY': 'Neutral market',
            'XAUUSD': 'Bearish sentiment',      'AUDUSD': 'Positive outlook',
            'GBPUSD': 'Strong buy signals',     'USDCAD': 'Sell pressure',
            'BTCUSD': 'Volatile bullish momentum', 'XRPUSD': 'Speculative neutral',
        }
        score = float(TextBlob(texts.get(symbol, 'Neutral')).sentiment.polarity)
    except Exception:
        score = 0.0
    return jsonify({'sentiment': score})


@app.route('/reload', methods=['POST'])
def reload_models():
    _load_all()
    return jsonify({'status': 'reloaded', 'loaded_models': len(_loaded)})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    log.info("=" * 60)
    log.info("FXJEFE Golden Model Server")
    log.info(f"Models dir : {MODELS_DIR}")
    log.info(f"Weights    : XGB_6={W_XGB_6}  9-feat={W_AVG_9}  28-feat={W_RF_28}")
    log.info("=" * 60)
    _load_all()
    app.run(host='0.0.0.0', port=8080, debug=False)
