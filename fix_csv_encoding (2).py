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
from textblob import TextBlob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'FXJEFE_Features.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'FXJEFE_Features_fixed.csv')

def fix_csv():
    try:
        # Attempt to read CSV with multiple encodings
        encodings = ['utf-8-sig', 'latin1', 'iso-8859-1']
        df = None
        for enc in encodings:
            try:
                df = pd.read_csv(INPUT_PATH, encoding=enc, low_memory=False)
                logging.info(f"Successfully read CSV with encoding {enc}, {len(df)} rows")
                break
            except (pd.errors.ParserError, UnicodeDecodeError):
                logging.warning(f"Failed to read CSV with encoding {enc}")
                continue
        if df is None:
            raise ValueError("Could not read CSV with any encoding")

        if len(df) == 0:
            logging.error("CSV is empty; check MT5 script (AI_Algo_Quantum.mq5)")
            raise ValueError("Empty CSV file: FXJEFE_Features.csv")

        # Define all expected columns
        expected_columns = [
            'time', 'symbol', 'price', 'atr', 'ema_diff', 'rsi', 'macd_diff', 'vwap',
            'price_vwap_diff', 'bb_position', 'roc', 'stochastic', 'cci', 'williams',
            'momentum', 'realized_vol', 'chaikin_vol', 'adx', 'rvi', 'obv',
            'volume_delta', 'ad_line', 'vol_osc', 'supertrend', 'hma', 'ichimoku_tenkan',
            'sar', 'dpo', 'spread', 'sentiment', 'signal', 'confidence'
        ]
        
        # Add missing columns with defaults
        for col in expected_columns:
            if col not in df.columns:
                df[col] = '' if col in ['time', 'symbol', 'signal'] else 0.0

        # Define default values
        defaults = {
            'price': df['price'].ffill(),
            'atr': df['atr'].mean() if df['atr'].notna().any() else 0.0001,
            'ema_diff': 0.0,
            'rsi': 50.0,
            'macd_diff': 0.0,
            'vwap': df['price'],
            'price_vwap_diff': 0.0,
            'bb_position': 0.5,
            'roc': 0.0,
            'stochastic': 50.0,
            'cci': 0.0,
            'williams': -50.0,
            'momentum': 0.0,
            'realized_vol': 0.0,
            'chaikin_vol': 0.0,
            'adx': 25.0,
            'rvi': 50.0,
            'obv': 0.0,
            'volume_delta': 0.0,
            'ad_line': 0.0,
            'vol_osc': 0.0,
            'supertrend': df['price'],
            'hma': df['price'],
            'ichimoku_tenkan': df['price'],
            'sar': df['price'],
            'dpo': 0.0,
            'spread': 0.0001,
            'sentiment': 0.5,  # Placeholder
            'signal': 'hold',
            'confidence': 0.5
        }
        for col, default in defaults.items():
            df[col] = df[col].fillna(default)

        # Convert numeric columns
        numeric_cols = [col for col in expected_columns if col not in ['time', 'symbol', 'signal']]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(defaults.get(col, 0.0))

        # Save the fixed CSV
        df.to_csv(OUTPUT_PATH, encoding='utf-8', index=False)
        logging.info(f"Saved fixed CSV to {OUTPUT_PATH}")

    except FileNotFoundError:
        logging.error(f"Input file not found: {INPUT_PATH}")
        raise
    except Exception as e:
        logging.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    fix_csv()