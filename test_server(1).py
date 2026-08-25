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
import requests
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

url = "http://localhost:8080/predict"
data = {
    "time": "2023-10-01 12:00:00",
    "symbol": "EURUSD.r",
    "price": 1.1234,
    "atr": 0.0005,
    "ema_diff": 0.0001,
    "rsi": 60.0,
    "macd_diff": 0.0,
    "vwap": 1.1235,
    "price_vwap_diff": -0.0001,
    "bb_position": 0.5,
    "roc": 0.01,
    "stochastic": 70.0,
    "cci": 100.0,
    "williams": -30.0,
    "momentum": 0.02,
    "realized_vol": 0.001,
    "chaikin_vol": 0.0,
    "adx": 30.0,
    "rvi": 55.0,
    "obv": 1000000.0,
    "volume_delta": 500.0,
    "ad_line": 2000000.0,
    "vol_osc": 0.0,
    "supertrend": 1.1230,
    "hma": 1.1232,
    "ichimoku_tenkan": 1.1233,
    "sar": 1.1229,
    "dpo": 0.0,
    "spread": 0.0002,
    "sentiment": 0.5  # Placeholder
}

try:
    json_string = json.dumps(data).strip()
    json.loads(json_string)  # Validate JSON
    response = requests.post(url, json=data)
    logging.info(f"Response: {response.json()}")
except json.JSONDecodeError as e:
    logging.error(f"Invalid JSON: {e}")
except requests.RequestException as e:
    logging.error(f"Request failed: {e}")