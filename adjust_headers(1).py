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
import logging
import pandas as pd
import os

config_path = r"C:\Users\LarryLocal\Documents\FXJEFE_Project\config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

logging.basicConfig(
    filename=os.path.join(config['log_path'], 'pipeline.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def adjust_headers(csv_path):
    csv_name = os.path.basename(csv_path)
    
    if 'expected_headers' not in config or csv_name not in config['expected_headers']:
        logging.warning(f"No expected headers defined for {csv_name}")
        return

    expected_headers = config['expected_headers'][csv_name]

    try:
        df = pd.read_csv(csv_path)
        current_headers = list(df.columns)

        if current_headers != expected_headers:
            logging.info(f"Adjusting headers for {csv_name}")
            for header in expected_headers:
                if header not in current_headers:
                    df[header] = '' if header in ['time', 'symbol'] else 0.0
            df = df[expected_headers]
            df.to_csv(csv_path, index=False)
            logging.info(f"Headers adjusted for {csv_name}")
        else:
            logging.info(f"Headers correct for {csv_name}")
    except Exception as e:
        logging.error(f"Error adjusting headers for {csv_name}: {e}")

if __name__ == "__main__":
    data_path = config['data_output_path']
    for csv_file in os.listdir(data_path):
        if csv_file.endswith('.csv'):
            adjust_headers(os.path.join(data_path, csv_file))