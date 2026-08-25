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
import pandas as pd
import logging
import os
import chardet
from textblob import TextBlob

# Configure logging
logging.basicConfig(
    filename=f"standardize_columns_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def detect_encoding(file_path):
    """Detect file encoding using chardet."""
    try:
        with open(file_path, 'rb') as file:
            result = chardet.detect(file.read(10000))
        encoding = result['encoding']
        logging.info(f"Detected encoding for {file_path}: {encoding}")
        return encoding
    except Exception as e:
        logging.error(f"Error detecting encoding for {file_path}: {e}")
        return None

def load_csv(file_path):
    """Load CSV with dynamic encoding detection."""
    encoding = detect_encoding(file_path)
    if not encoding:
        encoding = 'utf-8'
    encodings = [encoding, 'utf-8-sig', 'latin1', 'iso-8859-1', 'utf-16']
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc, low_memory=False)
            logging.info(f"Read {file_path} with {enc} encoding, {len(df)} rows")
            return df
        except Exception as e:
            logging.warning(f"Failed to read {file_path} with {enc}: {e}")
    logging.error(f"Could not read {file_path} with any encoding")
    return None

def standardize_columns(input_paths, output_paths, standard_columns):
    """Standardize column names across CSVs."""
    for input_path, output_path in zip(input_paths, output_paths):
        try:
            df = load_csv(input_path)
            if df is None or df.empty:
                logging.error(f"Skipping {input_path}: Empty or unreadable")
                continue

            # Log current columns
            current_columns = df.columns.tolist()
            logging.info(f"Current columns in {input_path}: {current_columns}")

            # Identify missing and extra columns
            missing_cols = [col for col in standard_columns if col not in df.columns]
            extra_cols = [col for col in df.columns if col not in standard_columns]
            logging.info(f"Missing columns: {missing_cols}")
            logging.info(f"Extra columns: {extra_cols}")

            # Add missing columns with defaults
            defaults = {
                'time': '',
                'symbol': '',
                'price': df['price'].ffill() if 'price' in df.columns else 0.0,
                'atr': df['atr'].mean() if 'atr' in df.columns and df['atr'].notna().any() else 0.0001,
                'ema_diff': 0.0,
                'rsi': 50.0,
                'garch_vol': 0.0,
                'macd_diff': 0.0,
                'vwap': df['price'] if 'price' in df.columns else 0.0,
                'price_vwap_diff': 0.0,
                'bb_position': 0.5,
                'signal': 'hold',
                'future_return': 0.0,
                'threshold': 0.0005,
                'label': -1,
                'roc': 0.0,
                'stochastic': 50.0,
                'cci': 0.0,
                'williams': -50.0,
                'momentum': 0.0,
                'realized_vol': 0.0,
                'chaikin_vol': 0.0,
                'adx': 25.0,
                'rvi': 0.0,
                'obv': 0.0,
                'volume_delta': 0.0,
                'ad_line': 0.0,
                'vol_osc': 0.0,
                'supertrend': 0.0,
                'hma': df['price'] if 'price' in df.columns else 0.0,
                'ichimoku_tenkan': df['price'] if 'price' in df.columns else 0.0,
                'sar': df['price'] if 'price' in df.columns else 0.0,
                'dpo': 0.0,
                'spread': 2.0,
                'sentiment': 0.0
            }
            for col in missing_cols:
                df[col] = defaults.get(col, 0.0)
                logging.info(f"Added missing column {col} with default {defaults.get(col)}")

            # Calculate sentiment (from generate_labels.py)
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
                return TextBlob(text).sentiment.polarity

            if 'symbol' in df.columns:
                df['sentiment'] = df['symbol'].apply(get_sentiment)

            # Convert numeric columns
            numeric_cols = [col for col in standard_columns if col not in ['time', 'symbol', 'signal']]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(defaults.get(col, 0.0))

            # Correlation check (from generate_labels.py)
            corr_matrix = df[numeric_cols].corr()
            high_corr = corr_matrix[corr_matrix.abs() > 0.8]
            logging.info(f"High correlations (>0.8):\n{high_corr[high_corr != 1.0].dropna(how='all')}")

            # Save standardized CSV
            df[standard_columns].to_csv(output_path, encoding='utf-8', index=False)
            logging.info(f"Saved standardized CSV to {output_path}, {len(df)} rows")

        except Exception as e:
            logging.error(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    # Define standard columns from training_data_cleaned.csv
    standard_columns = [
        'time', 'symbol', 'price', 'atr', 'ema_diff', 'rsi', 'garch_vol', 'macd_diff',
        'vwap', 'price_vwap_diff', 'bb_position', 'signal', 'future_return', 'threshold',
        'label', 'roc', 'stochastic', 'cci', 'williams', 'momentum', 'realized_vol',
        'chaikin_vol', 'adx', 'rvi', 'obv', 'volume_delta', 'ad_line', 'vol_osc',
        'supertrend', 'hma', 'ichimoku_tenkan', 'sar', 'dpo', 'spread', 'sentiment'
    ]

    # List of input and output CSVs
    input_paths = [
        r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\FXJEFE_Features.csv",
        r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\fxjefe_features_fixed.csv",
        r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\training_data_updated.csv"
    ]
    output_paths = [
        r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\FXJEFE_Features_standardized.csv",
        r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\fxjefe_features_fixed_standardized.csv",
        r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\training_data_updated_standardized.csv"
    ]

    standardize_columns(input_paths, output_paths, standard_columns)