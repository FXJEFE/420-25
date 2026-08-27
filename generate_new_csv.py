# -*- coding: utf-8 -*-
import json
import os
import logging

# Path to the config file
CONFIG_PATH = 'C:\\Users\\LarryLocal\\Documents\\FXJEFE_Project\\config.json'

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
log_file = os.path.join(config['log_path'], 'script.log')  # Change 'script.log' to match the script's name
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
with open('C:\\Users\\LarryLocal\\Documents\\FXJEFE_Project\\config.json', 'r') as f:
    config = json.load(f)
import pandas as pd

# Load features from CSV
try:
    features = pd.read_csv("FXJEFE_Features.csv")
except FileNotFoundError:
    print("Error: FXJEFE_Features.csv not found in C:\\Users\\LarryLocal\\Documents\\FXJEFE_Project")
    exit()

# Add signal based on price movement (10 pips threshold)
features['signal'] = 0  # Default: hold
for symbol in features['symbol'].unique():
    symbol_df = features[features['symbol'] == symbol].reset_index(drop=True)
    for i in range(1, len(symbol_df)):
        price_diff = (symbol_df['price'].iloc[i] - symbol_df['price'].iloc[i-1]) / 0.0001
        if price_diff > 10:
            features.loc[symbol_df.index[i], 'signal'] = 1  # Buy
        elif price_diff < -10:
            features.loc[symbol_df.index[i], 'signal'] = -1  # Sell

# Ensure all required columns
required_columns = ["time", "symbol", "price", "direction", "atr", "ema_diff", "rsi", "garch_vol", "macd_diff", "vwap", "price_vwap_diff", "bb_position", "signal"]
features = features[required_columns]

# Save to CSV with signals
features.to_csv("FXJEFE_Features_with_signals.csv", index=False)
print("FXJEFE_Features_with_signals.csv created with signals")