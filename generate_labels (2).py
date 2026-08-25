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
import pandas as pd
import numpy as np
import logging
from textblob import TextBlob

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('C:\\Users\\Administrator\\Documents\\FXJEFE_Fresh\\Logs', 'generate_labels.log')),
        logging.StreamHandler()
    ]
)

def load_config():
    config_path = 'C:\\Users\\Administrator\\Documents\\FXJEFE_Fresh\\config.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config.json: {e}")
        raise

config = load_config()

def fix_csv():
    input_path = os.path.join(config['mt5_data_path'], 'FXJEFE_Features.csv')
    output_path = os.path.join(config['data_output_path'], 'FXJEFE_Features_fixed.csv')
    labels_output_path = os.path.join(config['data_output_path'], 'FXJEFE_Features_with_labels.csv')

    try:
        df = pd.read_csv(input_path, encoding='utf-8-sig', low_memory=False)
        logging.info(f"Read CSV with {len(df)} rows from {input_path}")
    except Exception as e:
        logging.error(f"Failed to read CSV from {input_path}: {e}")
        raise

    expected_columns = ['time', 'symbol', 'price', 'atr', 'ema_diff', 'rsi', 'macd_diff', 'vwap',
                        'price_vwap_diff', 'momentum', 'volume_delta', 'spread', 'sentiment', 'signal']
    for col in expected_columns:
        if col not in df.columns:
            df[col] = '' if col in ['time', 'symbol', 'signal'] else 0.0
            logging.info(f"Added missing column: {col}")

    defaults = {
        'price': df['price'].ffill(),
        'atr': df['atr'].mean() if df['atr'].notna().any() else 0.0001,
        'ema_diff': 0.0,
        'rsi': 50.0,
        'macd_diff': 0.0,
        'vwap': df['price'],
        'price_vwap_diff': 0.0,
        'momentum': 0.0,
        'volume_delta': 0.0,
        'spread': 2.0,
        'sentiment': 0.0,
        'signal': 'hold'
    }

    for col, default in defaults.items():
        df[col] = df[col].fillna(default)
        logging.debug(f"Filled NaN in {col} with default")

    def get_sentiment(symbol):
        posts = {
            "EURUSD.r": "Bullish trend expected",
            "USDJPY.r": "Neutral market",
            "XAUUSD.r": "Bearish sentiment",
            "AUDUSD.r": "Positive outlook",
            "GBPUSD.r": "Strong buy signals",
            "USDCAD.r": "Sell pressure"
        }
        text = posts.get(symbol, "Neutral")
        try:
            return TextBlob(text).sentiment.polarity
        except Exception as e:
            logging.warning(f"Sentiment analysis failed for {symbol}: {e}")
            return 0.0

    df['sentiment'] = df['symbol'].apply(get_sentiment)
    logging.info("Computed sentiment for all symbols")

    numeric_cols = [col for col in expected_columns if col not in ['time', 'symbol', 'signal']]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(defaults.get(col, 0.0))

    try:
        corr_matrix = df[numeric_cols].corr()
        high_corr = corr_matrix[corr_matrix.abs() > 0.8]
        logging.info(f"High correlations (>0.8):\n{high_corr[high_corr != 1.0].dropna(how='all')}")
    except Exception as e:
        logging.warning(f"Correlation analysis failed: {e}")

    if df[numeric_cols].isna().any().any():
        logging.warning("NaNs detected after filling; check data source")
    if len(df) < 1000:
        logging.warning(f"Low row count ({len(df)}); run GenerateFeatures.mq5 again")

    try:
        df.to_csv(output_path, encoding='utf-8', index=False)
        logging.info(f"Saved fixed CSV to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save fixed CSV to {output_path}: {e}")
        raise

    df = generate_labels(df)

    try:
        df.to_csv(labels_output_path, encoding='utf-8', index=False)
        logging.info(f"Saved labeled CSV to {labels_output_path}")
    except Exception as e:
        logging.error(f"Failed to save labeled CSV to {labels_output_path}: {e}")
        raise

def generate_labels(df, threshold=0.0005, look_ahead=1):
    try:
        df['future_price'] = df['price'].shift(-look_ahead)
        df['price_change'] = (df['future_price'] - df['price']) / df['price']

        conditions = [
            (df['price_change'] > threshold),
            (df['price_change'] < -threshold),
            (abs(df['price_change']) <= threshold)
        ]
        choices = [1, 0, -1]
        df['label'] = np.select(conditions, choices, default=-1)

        df = df.dropna(subset=['future_price', 'price_change', 'label'])
        logging.info(f"Generated labels; {len(df)} rows remaining after dropping NaNs")
        return df
    except Exception as e:
        logging.error(f"Failed to generate labels: {e}")
        raise

if __name__ == "__main__":
    try:
        fix_csv()
    except Exception as e:
        logging.error(f"Script failed: {e}")
        raise