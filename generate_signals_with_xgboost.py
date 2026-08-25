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
import xgboost as xgb
import numpy as np

# Load the trained XGBoost model
model = xgb.Booster()
model.load_model('xgboost_model.json')

# Load the features data and standardize column names
features = pd.read_csv('FXJEFE_Features.csv')
features.columns = [col.lower().strip() for col in features.columns]
print("Standardized columns:", list(features.columns))
feature_columns = ['price', 'atr', 'ema_diff', 'rsi', 'garch_vol', 'macd_diff']
missing = [col for col in feature_columns if col not in features.columns]
if missing:
    print(f"Missing columns: {missing}")
    exit(1)

# Specify the feature columns that the model was trained on
X = features[feature_columns]

# Create DMatrix with feature names
dmat = xgb.DMatrix(X, feature_names=feature_columns)

# Make predictions (this will return probabilities for each class, shape (137, 3))
predictions = model.predict(dmat)

# Print shape and first few predictions for debugging
print("Predictions shape:", predictions.shape)
print("First 5 predictions:\n", predictions[:5])

# Convert probabilities to class labels (0, 1, -1)
# Assuming classes are ordered as [-1, 0, 1] (sell, hold, buy)
features['signal'] = np.argmax(predictions, axis=1) - 1  # Adjust based on class mapping

# Ensure all required columns
required_columns = ["time", "symbol", "price", "direction", "atr", "ema_diff", "rsi", "garch_vol", "macd_diff", "vwap", "price_vwap_diff", "bb_position", "signal"]
features = features[required_columns]

# Save to CSV
features.to_csv('FXJEFE_Features_with_signals.csv', index=False)
print("FXJEFE_Features_with_signals.csv created with model predictions")

# Add logging and backup
import logging
logging.basicConfig(filename='signal_update.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("Started signal generation with XGBoost")
logging.info("Finished signal generation with XGBoost")
import shutil
import datetime
shutil.copy("FXJEFE_Features_with_signals.csv", f"FXJEFE_Features_with_signals_backup_{datetime.datetime.now().strftime('%Y-%m-%d')}.csv")