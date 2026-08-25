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

import talib
import pandas as pd
import numpy as np

def build_advanced_features(df):
    """
    Add advanced TA-Lib features for high-confidence trading signals.
    Input: DataFrame with OHLCV data (open, high, low, close, volume).
    Output: Enhanced DataFrame with new features.
    """
    # Ensure required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("DataFrame must contain OHLCV columns: open, high, low, close, volume")

    # 1. Volatility-Adjusted Indicators
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['Norm_RSI'] = talib.RSI(df['close'], timeperiod=14) / df['ATR']  # Normalize RSI by volatility

    # 2. Multi-Timeframe Confirmation
    df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
    df['SMA_200'] = talib.SMA(df['close'], timeperiod=200)
    df['Trend_Filter'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)

    # 3. Candlestick Patterns
    df['Engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
    df['Doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])

    # 4. Market Regime Filter (ADX)
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['Tradeable'] = np.where(df['ADX'] > 25, 1, 0)  # Only trade in trending markets

    # 5. Feature Interactions
    df['RSI_ADX'] = df['Norm_RSI'] * df['ADX']  # Combine momentum and trend strength

    # Handle NaN values
    df.fillna(0, inplace=True)

    return df
