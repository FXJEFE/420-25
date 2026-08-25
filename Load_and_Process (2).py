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
import pandas as pd

# Define the column names based on the data structure
column_names = ['time', 'symbol', 'price', 'atr', 'ema_diff', 'rsi', 
                'garch_vol', 'macd_diff', 'vwap', 'price_vwap_diff', 
                'bb_position', 'signal']

# Read the CSV, skipping the first invalid row and assigning column names
df = pd.read_csv('fxfeatures_features.csv', 
                 header=None,          # No header in the file
                 skiprows=1,           # Skip the garbled first row
                 names=column_names)   # Assign the column names

# Verify the data
print(df.head())  # Fixed by adding the missing closing parenthesis
