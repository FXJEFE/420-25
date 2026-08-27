# -*- coding: utf-8 -*-
import os as _os_utf8, sys as _sys_utf8
_os_utf8.environ.setdefault('PYTHONUTF8','1')
_os_utf8.environ.setdefault('PYTHONIOENCODING','utf-8')
for _s in (getattr(_sys_utf8,'stdout',None), getattr(_sys_utf8,'stderr',None)):
    try:
        if _s is not None and hasattr(_s,'reconfigure'):
            _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import json
import os
import logging

# Path to the config file
CONFIG_PATH = 'C:\\Users\\locallarry\\Documents\\FXJEFE_Project\\config.json'

# Load the config file safely
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8', errors='replace') as f:
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
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info("Script started and configuration loaded successfully")

import pandas as pd
import numpy as np
import joblib
import logging
import json
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:\\Users\\locallarry\\Documents\\FXJEFE_Project\\Logs\\ensemble_predictions.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def load_config():
    config_path = 'C:\\Users\\locallarry\\Documents\\FXJEFE_Project\\config.json'
    try:
        with open(config_path, 'r', encoding='utf-8', errors='replace') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config.json: {e}")
        exit(1)

def generate_predictions(data, features, model_path):
    try:
        model = joblib.load(model_path)
        X = data[features]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        data['signal'] = np.where(predictions == 1, 'buy', np.where(predictions == -1, 'sell', 'hold'))
        data['confidence'] = probabilities.max(axis=1)
        
        logging.info("Predictions generated")
        return data
    except Exception as e:
        logging.error(f"Error generating predictions: {e}")
        return None

def main():
    config = load_config()
    data_path = os.path.join(config['data_output_path'], 'FXJEFE_Features_with_labels.csv')
    model_path = os.path.join(config['models_path'], 'ensemble_model.pkl')
    output_path = os.path.join(config['data_output_path'], 'FXJEFE_Predictions.csv')
    
    try:
        data = pd.read_csv(data_path, encoding='utf-8')
        result = generate_predictions(data, config['features'], model_path)
        if result is not None:
            result.to_csv(output_path, index=False, encoding='utf-8')
            logging.info(f"Predictions saved to {output_path}")
    except Exception as e:
        logging.error(f"Error processing data: {e}")

if __name__ == "__main__":
    main()