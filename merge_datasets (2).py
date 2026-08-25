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

def merge_datasets():
    """Merge features and trades datasets."""
    features_csv = os.path.join(config['data_output_path'], 'FXJEFE_Features.csv')
    trades_csv = os.path.join(config['data_output_path'], 'FXJEFE_trades.csv')
    output_csv = os.path.join(config['data_output_path'], 'FXJEFE_merged.csv')
    
    if not os.path.exists(features_csv) or not os.path.exists(trades_csv):
        logging.error(f"Required files missing: {features_csv}, {trades_csv}")
        return
    
    features_df = pd.read_csv(features_csv)
    trades_df = pd.read_csv(trades_csv)
    merged_df = pd.merge(features_df, trades_df, on=['time', 'symbol'], how='outer')
    merged_df.to_csv(output_csv, index=False)
    logging.info(f"Merged datasets and saved to {output_csv}")

if __name__ == "__main__":
    merge_datasets()
