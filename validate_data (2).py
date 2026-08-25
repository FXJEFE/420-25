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

def validate_data():
    """Validate data integrity in CSV files."""
    csv_path = os.path.join(config['data_output_path'], 'FXJEFE_Features.csv')
    if not os.path.exists(csv_path):
        logging.error(f"Data file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        logging.warning(f"Found {missing_values} missing values in {csv_path}")
    else:
        logging.info(f"Data validation passed for {csv_path}: No missing values")

if __name__ == "__main__":
    validate_data()
