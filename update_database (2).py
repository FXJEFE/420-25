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
import sqlite3
import pandas as pd
import os

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Set up logging
logging.basicConfig(filename=os.path.join(config['log_path'], 'pipeline.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def update_database():
    """Update SQLite database with trade data."""
    csv_path = os.path.join(config['data_output_path'], 'FXJEFE_trades.csv')
    db_path = os.path.join(config['data_output_path'], 'fxjefe_trades.db')
    
    if not os.path.exists(csv_path):
        logging.error(f"Trades file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(db_path)
    df.to_sql('trades', conn, if_exists='replace', index=False)
    conn.close()
    logging.info(f"Updated database at {db_path} with trade data")

if __name__ == "__main__":
    update_database()
