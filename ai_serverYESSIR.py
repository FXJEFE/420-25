from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
import os
import logging
import time
import threading
from collections import defaultdict, deque
from textblob import TextBlob
import feedparser

# Path to the config file
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')

# Load the config file safely
try:
    with open(CONFIG_PATH, 'r') as f:
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
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logging.info("AI server started and configuration loaded successfully")

app = Flask(__name__)

# Crypto symbols -- routed to crypto_model.pkl
CRYPTO_SYMBOLS = set(config.get('crypto_symbols', ['BTCUSD', 'ETHUSD', 'XRPUSD']))

# Rolling history per symbol for computing lagged features + garch_vol
# Stores last N feature snapshots so server can compute lags on-the-fly
HISTORY_LEN = 5
symbol_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))

# Base features (28) that the EA sends -- same as original config minus derived ones
BASE_FEATURES = [
    'price', 'atr', 'ema_diff', 'rsi', 'macd_diff', 'vwap',
    'price_vwap_diff', 'bb_position', 'roc', 'stochastic', 'cci',
    'williams', 'momentum', 'realized_vol', 'chaikin_vol', 'adx',
    'rvi', 'obv', 'volume_delta', 'ad_line', 'vol_osc', 'supertrend',
    'hma', 'ichimoku_tenkan', 'sar', 'dpo', 'spread', 'sentiment'
]

# Load forex model
forex_model_path = os.path.join(config['models_path'], 'my_model.pkl')
forex_model = None
try:
    if os.path.exists(forex_model_path):
        forex_model = joblib.load(forex_model_path)
        logging.info(f"Forex model loaded from {forex_model_path} (expects {getattr(forex_model, 'n_features_in_', '?')} features)")
    else:
        logging.error(f"Forex model not found: {forex_model_path}")
except Exception as e:
    logging.error(f"Failed to load forex model: {e}")

# Load crypto model (may be ensemble wrapper)
crypto_model_path = os.path.join(config['models_path'], 'crypto_model.pkl')
crypto_model = None
try:
    if os.path.exists(crypto_model_path):
        crypto_model = joblib.load(crypto_model_path)
        logging.info(f"Crypto model loaded from {crypto_model_path} (expects {getattr(crypto_model, 'n_features_in_', '?')} features)")
    else:
        logging.warning(f"Crypto model not found: {crypto_model_path} (will use forex model as fallback)")
except Exception as e:
    logging.error(f"Failed to load crypto model: {e}")

# Backward compat
model = forex_model

# Determine if crypto model needs extended features
crypto_n_features = getattr(crypto_model, 'n_features_in_', 28) if crypto_model else 28
forex_n_features = getattr(forex_model, 'n_features_in_', 28) if forex_model else 28
logging.info(f"Forex model expects {forex_n_features} features, crypto model expects {crypto_n_features} features")


def compute_derived_features(data, symbol):
    """Compute garch_vol, future_return, and lagged features from rolling history."""
    history = symbol_history[symbol]

    # Current snapshot of base values
    snapshot = {feat: data.get(feat, 0) for feat in BASE_FEATURES}

    # Add to history
    history.append(snapshot.copy())

    derived = {}

    # garch_vol: rolling std of log returns from price history
    if len(history) >= 3:
        prices = [h['price'] for h in history if h['price'] > 0]
        if len(prices) >= 3:
            log_returns = [np.log(prices[i] / prices[i-1]) for i in range(1, len(prices)) if prices[i-1] > 0]
            derived['garch_vol'] = float(np.std(log_returns)) if log_returns else 0.0
        else:
            derived['garch_vol'] = 0.0
    else:
        derived['garch_vol'] = data.get('realized_vol', 0.0)  # fallback

    # future_return: we don't know the future at prediction time, use 0
    derived['future_return'] = 0.0

    # Lagged features
    lag_sources = ['price', 'rsi', 'macd_diff', 'atr']
    for col in lag_sources:
        for lag in [1, 2, 3]:
            col_name = f'{col}_lag{lag}'
            if len(history) > lag:
                derived[col_name] = history[-(lag+1)].get(col, data.get(col, 0))
            else:
                derived[col_name] = data.get(col, 0)  # fallback to current value

    # Time features
    import datetime
    now = datetime.datetime.now()
    derived['hour_of_day'] = float(now.hour)
    derived['day_of_week'] = float(now.weekday())

    # Volume ratio: current volume_delta vs average from history
    vol_abs = abs(data.get('volume_delta', 0))
    if len(history) >= 3:
        avg_vol = np.mean([abs(h.get('volume_delta', 1)) for h in history]) or 1.0
        derived['volume_ratio'] = min(vol_abs / avg_vol, 10.0)
    else:
        derived['volume_ratio'] = 1.0

    # Regime detection from ADX
    adx = data.get('adx', 25.0)
    if adx < 20:
        derived['regime'] = 0.0   # ranging
    elif adx < 30:
        derived['regime'] = 1.0   # weak trend
    else:
        derived['regime'] = 2.0   # strong trend

    return derived


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running"})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            logging.error("No data provided")
            return jsonify({"error": "No data provided"}), 400

        symbol = data.get('symbol', '').replace('.r', '').replace('.R', '')
        is_crypto = symbol in CRYPTO_SYMBOLS

        # Route to the right model
        active_model = crypto_model if is_crypto and crypto_model else forex_model
        if active_model is None:
            logging.error("No model loaded")
            return jsonify({"error": "Model not loaded"}), 500

        n_expected = getattr(active_model, 'n_features_in_', 28)

        if n_expected > len(BASE_FEATURES):
            # Model expects extended features -- compute derived ones
            derived = compute_derived_features(data, symbol)
            # Build full feature list: base + extended from config
            all_feature_names = list(config.get('features', BASE_FEATURES))
            for f in config.get('features_extended', []):
                if f not in all_feature_names:
                    all_feature_names.append(f)
            features = []
            for feat in all_feature_names:
                if feat in derived:
                    features.append(derived[feat])
                else:
                    features.append(data.get(feat, 0))
            # Pad or trim to match model expectation
            while len(features) < n_expected:
                features.append(0.0)
            features = features[:n_expected]
        else:
            # Model with 28 features -- use base features only
            features = [data.get(feat, 0) for feat in BASE_FEATURES[:n_expected]]

        prediction = active_model.predict([features])[0]
        confidence = active_model.predict_proba([features])[0].max() if hasattr(active_model, 'predict_proba') else 0.5
        signal = {1: 'buy', 0: 'hold', -1: 'sell'}.get(int(prediction), 'hold')
        stop_loss = data['price'] - (2 * data['atr']) if signal == 'buy' else data['price'] + (2 * data['atr'])

        model_name = 'crypto' if is_crypto and crypto_model else 'forex'
        logging.info(f"[{model_name}] {symbol}: {signal} (conf={confidence:.2f}, feats={len(features)})")
        return jsonify({"signal": signal, "confidence": float(confidence), "stop_loss": float(stop_loss), "model": model_name})
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ── Live Sentiment Engine ────────────────────────────────────────────
# Pulls headlines from free RSS feeds, scores with TextBlob, caches results.
# Each symbol maps to keywords that filter relevant headlines.

SENTIMENT_FEEDS = [
    'https://cointelegraph.com/rss',
    'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'https://bitcoinmagazine.com/feed',
]

FOREX_FEEDS = [
    'https://www.forexlive.com/feed',
    'https://www.fxstreet.com/rss',
]

# Keywords to match headlines to symbols
SYMBOL_KEYWORDS = {
    'BTCUSD': ['bitcoin', 'btc', 'crypto market', 'digital asset', 'halving'],
    'ETHUSD': ['ethereum', 'eth', 'ether', 'defi', 'smart contract'],
    'XRPUSD': ['xrp', 'ripple', 'sec lawsuit', 'cross-border'],
    'EURUSD': ['euro', 'eur', 'ecb', 'eurozone', 'eu economy'],
    'USDJPY': ['yen', 'jpy', 'boj', 'japan', 'japanese'],
    'XAUUSD': ['gold', 'xau', 'precious metal', 'safe haven', 'bullion'],
    'GBPUSD': ['pound', 'gbp', 'sterling', 'bank of england', 'boe'],
    'AUDUSD': ['australian dollar', 'aud', 'rba', 'aussie'],
    'USDCAD': ['canadian dollar', 'cad', 'loonie', 'bank of canada'],
}

# Cache: {symbol: (timestamp, score, headline_count)}
sentiment_cache = {}
CACHE_TTL = 300  # 5 minutes

# Background headline store: refreshed periodically
headline_store = {'crypto': [], 'forex': [], 'last_update': 0}
HEADLINE_REFRESH = 120  # 2 minutes


def refresh_headlines():
    """Pull latest headlines from RSS feeds (called in background)."""
    crypto_headlines = []
    for url in SENTIMENT_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                crypto_headlines.append(entry.title)
        except Exception:
            pass

    forex_headlines = []
    for url in FOREX_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                forex_headlines.append(entry.title)
        except Exception:
            pass

    headline_store['crypto'] = crypto_headlines
    headline_store['forex'] = forex_headlines
    headline_store['last_update'] = time.time()
    logging.info(f"Headlines refreshed: {len(crypto_headlines)} crypto, {len(forex_headlines)} forex")


def get_live_sentiment(symbol):
    """Score live sentiment for a symbol from cached headlines."""
    now = time.time()

    # Return cached score if fresh
    if symbol in sentiment_cache:
        ts, score, count = sentiment_cache[symbol]
        if now - ts < CACHE_TTL:
            return score, count

    # Refresh headlines if stale
    if now - headline_store['last_update'] > HEADLINE_REFRESH:
        try:
            refresh_headlines()
        except Exception as e:
            logging.warning(f"Headline refresh failed: {e}")

    # Pick headline pool based on symbol type
    is_crypto = symbol in CRYPTO_SYMBOLS
    headlines = headline_store['crypto'] if is_crypto else headline_store['forex']
    # Also search crypto headlines for forex symbols (general market news)
    if not is_crypto:
        headlines = headlines + headline_store.get('crypto', [])

    # Filter headlines matching this symbol's keywords
    keywords = SYMBOL_KEYWORDS.get(symbol, [symbol.lower()])
    matched = []
    for h in headlines:
        h_lower = h.lower()
        if any(kw in h_lower for kw in keywords):
            matched.append(h)

    # Score matched headlines
    if matched:
        scores = []
        for h in matched:
            try:
                scores.append(TextBlob(h).sentiment.polarity)
            except Exception:
                pass
        if scores:
            avg_score = sum(scores) / len(scores)
            sentiment_cache[symbol] = (now, avg_score, len(scores))
            return avg_score, len(scores)

    # No matching headlines -- return neutral
    sentiment_cache[symbol] = (now, 0.0, 0)
    return 0.0, 0


@app.route('/predict/sentiment', methods=['GET'])
def sentiment():
    symbol = request.args.get('symbol', '')
    symbol_base = symbol.replace('.r', '').replace('.R', '')

    score, headline_count = get_live_sentiment(symbol_base)

    logging.info(f"Sentiment for {symbol_base}: {score:.3f} ({headline_count} headlines matched)")
    return jsonify({
        "sentiment": float(score),
        "headlines_matched": headline_count,
        "source": "live" if headline_count > 0 else "neutral_default"
    })


# Pre-load headlines on startup
try:
    refresh_headlines()
except Exception as e:
    logging.warning(f"Initial headline fetch failed: {e}")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)
