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
import re
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def test_regex(log_file_path):
    # Updated regex pattern
    pattern = re.compile(
        r'(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2})\s+Features\s+for\s+(\w+):\s+'
        r'price=(\d*\.\d+),\s+atr=(\d*\.\d+),\s+signal=(buy|sell|hold)'
    )
    
    matched = 0
    unmatched = 0
    try:
        with open(log_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                match = pattern.match(line.strip())
                if match:
                    time, symbol, price, atr, signal = match.groups()
                    logging.info(f"Matched: time={time}, symbol={symbol}, price={price}, atr={atr}, signal={signal}")
                    matched += 1
                else:
                    logging.warning(f"Unmatched line: {line.strip()}")
                    unmatched += 1
        logging.info(f"Total matched: {matched}, unmatched: {unmatched}")
    except FileNotFoundError:
        logging.error(f"Log file {log_file_path} not found")
    except UnicodeDecodeError:
        logging.error(f"Encoding error in {log_file_path}. Ensure UTF-8 encoding.")

if __name__ == "__main__":
    test_regex("FXJEFE_log.txt")
