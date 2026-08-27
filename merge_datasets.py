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

import json
import os
with open('C:\\Users\\locallarry\\Documents\\FXJEFE_Project\\config.json', 'r', encoding='utf-8', errors='replace') as f:
    config = json.load(f)
import json
import logging
import pandas as pd
import os

# Load configuration
with open('config.json', 'r', encoding='utf-8', errors='replace') as f:
    config = json.load(f)

# Set up logging
logging.basicConfig(filename=os.path.join(config['log_path'], 'pipeline.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def merge_datasets():
    """Merge features and trades datasets."""
    from fxjefe_paths import write_feature_csv
    features_csv = os.path.join(config['data_output_path'], 'FXJEFE_Features.csv')
    trades_csv = os.path.join(config['data_output_path'], 'FXJEFE_trades.csv')
    if not os.path.exists(features_csv):
        raise FileNotFoundError(features_csv)
    features_df = pd.read_csv(features_csv, encoding='utf-8', low_memory=False)
    if os.path.exists(trades_csv):
        trades_df = pd.read_csv(trades_csv, encoding='utf-8', low_memory=False)
        if "time" in trades_df.columns and "symbol" in trades_df.columns:
            merged_df = pd.merge(features_df, trades_df, on=['time', 'symbol'], how='left')
        else:
            merged_df = features_df.copy()
    else:
        merged_df = features_df.copy()
    write_feature_csv(merged_df, config, "FXJEFE_merged.csv")
    logging.info("Merged datasets rows=%s", len(merged_df))
    print(f"OK merge_datasets: {len(merged_df)} rows", flush=True)

if __name__ == "__main__":
    merge_datasets()