import json
import os
import logging

# Configuration path (updated to absolute path for consistency)
CONFIG_PATH = r'C:\Users\Administrator\Documents\FXJEFE_Project\config.json'

# Load the configuration file
try:
    with open(CONFIG_PATH, 'r') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    print(f"Error: Could not find config file at {CONFIG_PATH}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Config file has invalid format - {e}")
    exit(1)

# Retrieve the terminal ID from the config
terminal_id = config.get('mt5_terminal_id')
if not terminal_id:
    print("Error: 'mt5_terminal_id' not found in config.json")
    exit(1)

# Define the file name
file_name = 'FXJEFE_Features.csv'

# Construct the dynamic path using an f-string
path = f'C:\\Users\\Administrator\\AppData\\Roaming\\MetaQuotes\\Terminal\\{terminal_id}\\MQL5\\Files\\{file_name}'

# Set up basic logging
log_path = config.get('log_path', 'C:\\Users\\Administrator\\Documents\\FXJEFE_Project\\Logs')
log_file = os.path.join(log_path, 'predict_ea.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logging.info("Script started and configuration loaded successfully")

# Open and read the file
try:
    with open(path, 'r') as f:
        data = f.read()
        print(data)
        logging.info(f"File {file_name} read successfully")
except FileNotFoundError:
    logging.error(f"File not found: {path}")
    exit(1)
except Exception as e:
    logging.error(f"Error reading file: {e}")
    exit(1)