import json
import os
import logging

# Path to the config file
CONFIG_PATH = 'C:\\Users\\Administrator\\Documents\\FXJEFE_Project\\config.json'

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

# Set up logging
log_file = os.path.join(config['log_path'], 'script.log')  # Change 'script.log' to match the script’s name
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logging.info("Script started and configuration loaded successfully")

import json
import os
with open('C:\\Users\\Administrator\\Documents\\FXJEFE_Project\\config.json', 'r') as f:
    config = json.load(f)
from flask import Flask, request, jsonify
import json
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
import joblib
import pandas as pd
import logging
import os
import filelock
from logging_utils import log_trade
from risk_management import calculate_trade_params

app = Flask(__name__)

# Define all 27 technical indicators
FEATURES = [
    'atr', 'ema_diff', 'rsi', 'macd_diff', 'vwap', 'price_vwap_diff', 'bb_position',
    'roc', 'stochastic', 'cci', 'williams', 'momentum', 'realized_vol', 'chaikin_vol',
    'adx', 'rvi', 'obv', 'volume_delta', 'ad_line', 'vol_osc', 'supertrend', 'hma',
    'ichimoku_tenkan', 'sar', 'dpo', 'spread', 'sentiment'
]

# Setup logging
os.makedirs('data', exist_ok=True)
logging.basicConfig(
    filename='data/ai_server.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load configuration
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    model_type = config['model']
    expected_features = config['features']
    sequence_length = config.get('sequence_length', 1)
except Exception as e:
    logging.error(f"Error loading config.json: {e}")
    raise

# Load models
models = {}
if model_type in ['xgboost', 'ensemble']:
    models['xgboost'] = xgb.Booster()
    models['xgboost'].load_model('models/xgboost_model.json')
if model_type in ['randomforest', 'ensemble']:
    models['randomforest'] = joblib.load('models/ensemble_model.pkl')
if model_type in ['lightgbm', 'ensemble']:
    models['lightgbm'] = lgb.Booster(model_file='models/lightgbm_model.txt')
if model_type in ['lstm', 'ensemble']:
    models['lstm'] = tf.keras.models.load_model('models/lstm_model.h5')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            raise ValueError("No JSON data provided")

        # Required fields
        required_fields = ['symbol', 'model']
        model = data.get('model')
        if model == 'lstm':
            if 'sequence' not in data or len(data['sequence']) != sequence_length:
                raise ValueError(f"Expected {sequence_length} sequence frames")
        else:
            required_fields += expected_features

        # Check required fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing fields: {missing_fields}")

        # Extract optional fields
        time = data.get('time', '')
        price = data.get('price', None)
        if price is not None and not isinstance(price, (int, float)):
            raise ValueError("Price must be a number")

        # Validate features
        if model == 'lstm':
            sequence = data['sequence']
            for frame in sequence:
                for feat in expected_features:
                    if feat not in frame or not isinstance(frame[feat], (int, float)):
                        raise ValueError(f"Invalid sequence feature {feat}")
            df = pd.DataFrame(sequence, columns=expected_features)
        else:
            for feat in expected_features:
                if not isinstance(data[feat], (int, float)):
                    raise ValueError(f"Feature {feat} must be a number")
            df = pd.DataFrame([[float(data[feat]) for feat in expected_features]], columns=expected_features)

        symbol = data['symbol']

        # Make predictions
        predictions = {}
        if model == 'ensemble':
            for m in ['xgboost', 'randomforest', 'lightgbm', 'lstm']:
                if m == 'lstm':
                    seq = df.values.reshape(1, sequence_length, len(expected_features))
                    pred = models[m].predict(seq)[0]
                    predictions[m] = 'buy' if pred > 0.5 else 'sell'
                    predictions[m + '_confidence'] = pred if pred > 0.5 else 1 - pred
                elif m == 'xgboost':
                    dmatrix = xgb.DMatrix(df)
                    pred = models[m].predict(dmatrix)[0]
                    predictions[m] = 'buy' if pred > 0.5 else 'sell'
                    predictions[m + '_confidence'] = pred if pred > 0.5 else 1 - pred
                else:
                    pred = models[m].predict(df)[0]
                    predictions[m] = 'buy' if pred == 1 else 'sell'
                    predictions[m + '_confidence'] = models[m].predict_proba(df)[0][1]
            weights = {'xgboost': 0.3, 'randomforest': 0.2, 'lightgbm': 0.3, 'lstm': 0.2}
            vote = sum(weights[m] for m, p in predictions.items() if p == 'buy')
            signal = 'buy' if vote > 0.5 else 'sell'
            confidence = vote if vote > 0.5 else 1 - vote
        else:
            if model not in models:
                raise ValueError(f"Invalid model: {model}")
            if model == 'lstm':
                seq = df.values.reshape(1, sequence_length, len(expected_features))
                pred = models[model].predict(seq)[0]
                signal = 'buy' if pred > 0.5 else 'sell'
                confidence = pred if pred > 0.5 else 1 - pred
            elif model == 'xgboost':
                dmatrix = xgb.DMatrix(df)
                pred = models[model].predict(dmatrix)[0]
                signal = 'buy' if pred > 0.5 else 'sell'
                confidence = pred if pred > 0.5 else 1 - pred
            else:
                pred = models[model].predict(df)[0]
                signal = 'buy' if pred == 1 else 'sell'
                confidence = models[model].predict_proba(df)[0][1]

        # Calculate trade parameters
        atr = df['atr'].iloc[-1] if model != 'lstm' else sequence[-1]['atr']
        account_equity = data.get('account_equity', 10000)  # Use provided equity or default
        trade_params = calculate_trade_params(signal, atr, account_equity)

        # Log to CSV
        log_data = {
            'time': time,
            'symbol': symbol,
            'price': price,
            **{feat: data.get(feat, sequence[-1][feat] if model == 'lstm' else None) for feat in FEATURES},
            'signal': signal,
            'confidence': confidence,
            'lot_size': trade_params['lot_size'],
            'stop_loss': trade_params['stop_loss'],
            'take_profit': trade_params['take_profit']
        }
        df_log = pd.DataFrame([log_data])
        lock = filelock.FileLock('data/FXJEFE_Features.lock')
        with lock:
            df_log.to_csv(
                'data/FXJEFE_Features.csv',
                mode='a',
                header=not os.path.exists('data/FXJEFE_Features.csv'),
                index=False
            )

        # Log to FXJEFElogtxt.txt
        log_trade(log_data, signal, trade_params, 0.01)  # Placeholder drawdown

        logging.info(f"Prediction for {symbol}: signal={signal}, confidence={confidence}, model={model}")
        return jsonify({
            'signal': signal,
            'confidence': confidence,
            'lot_size': trade_params['lot_size'],
            'stop_loss': trade_params['stop_loss'],
            'take_profit': trade_params['take_profit']
        })

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)