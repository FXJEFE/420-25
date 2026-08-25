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

def analyze_outcomes():
    csv_path = os.path.join(config['data_output_path'], 'FXJEFE_trades_outcomes.csv')
    if not os.path.exists(csv_path):
        logging.error(f"Trade outcomes file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    total_trades = len(df)
    total_profit = df['profit'].sum()
    win_rate = (df['profit'] > 0).mean() * 100 if total_trades > 0 else 0
    
    logging.info(f"Trade Analysis: Total Trades = {total_trades}, Total Profit = {total_profit:.2f}, Win Rate = {win_rate:.2f}%")

if __name__ == "__main__":
    analyze_outcomes()