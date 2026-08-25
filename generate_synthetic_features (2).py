import pandas as pd
import numpy as np
import json
import os
import logging

# Path to the config file
CONFIG_PATH = 'C:\\Users\\Administrator\\Documents\\FXJEFE_Project\\config.json'

# Load the config file safely
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find config file at {CONFIG_PATH}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Config file has invalid format - {e}")
    exit(1)

# Set up logging
log_file = os.path.join(config['log_path'], 'generate_synthetic_features.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info("Script started and configuration loaded successfully")

# Define paths
features_path = os.path.join(config['data_output_path'], 'FXJEFE_Features.csv')

# Generate synthetic feature data
np.random.seed(42)
n_samples = 1000

data = {
    "time": pd.date_range(start='2023-01-01', periods=n_samples, freq='15min'),
    "symbol": ["EURUSD.r"] * n_samples,
    "open": np.random.uniform(1.0, 2.0, n_samples),
    "high": np.random.uniform(1.0, 2.0, n_samples),
    "low": np.random.uniform(1.0, 2.0, n_samples),
    "close": np.random.uniform(1.0, 2.0, n_samples),
    "volume": np.random.randint(100, 1000, n_samples),
    "price": np.random.uniform(1.0, 2.0, n_samples),
    "atr": np.random.uniform(0.001, 0.1, n_samples),
    "ema_diff": np.random.uniform(-0.05, 0.05, n_samples),
    "rsi": np.random.uniform(20, 80, n_samples),
    "macd_diff": np.random.uniform(-0.01, 0.01, n_samples),
    "vwap": np.random.uniform(1.0, 2.0, n_samples),
    "price_vwap_diff": np.random.uniform(-0.05, 0.05, n_samples),
    "momentum": np.random.uniform(-0.1, 0.1, n_samples),
    "volume_delta": np.random.uniform(-1000, 1000, n_samples),
    "spread": np.random.uniform(0.0001, 0.001, n_samples),
    "sentiment": np.random.uniform(-1, 1, n_samples),
}

features = pd.DataFrame(data)

# Generate signals based on price movement
features['signal'] = 0
for i in range(1, len(features)):
    price_diff = (features['price'].iloc[i] - features['price'].iloc[i-1]) / 0.0001
    if price_diff > 10:
        features.loc[i, 'signal'] = 1  # Buy
    elif price_diff < -10:
        features.loc[i, 'signal'] = -1  # Sell

# Save to CSV with UTF-8 encoding
try:
    features.to_csv(features_path, index=False, encoding='utf-8')
    logging.info(f"Features saved to {features_path}")
except Exception as e:
    logging.error(f"Failed to save features to {features_path}: {e}")
    exit(1)