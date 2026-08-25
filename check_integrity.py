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
import json
import logging
import pandas as pd
import os

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Set up logging
logging.basicConfig(filename=os.path.join(config['log_path'], 'pipeline.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def check_integrity():
    """Check data integrity across CSV files."""
    files = ['FXJEFE_Features.csv', 'FXJEFE_trades.csv', 'FXJEFE_trades_outcomes.csv']
    for file in files:
        file_path = os.path.join(config['data_output_path'], file)
        if not os.path.exists(file_path):
            logging.warning(f"File missing: {file_path}")
            continue
        df = pd.read_csv(file_path)
        if df.empty:
            logging.warning(f"File is empty: {file_path}")
        else:
            logging.info(f"Integrity check passed for {file_path}: {len(df)} rows")

if __name__ == "__main__":
    check_integrity()