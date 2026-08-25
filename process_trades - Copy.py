import json
import os
with open('C:\\Users\\Administrator\\Documents\\FXJEFE_Project\\config.json', 'r') as f:
    config = json.load(f)
import logging
import json
import os

# Load configuration
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("Error: config.json not found.")
    exit(1)
except json.JSONDecodeError:
    print("Error: config.json is not a valid JSON file.")
    exit(1)

# Check if 'log_path' exists
if 'log_path' not in config:
    print("Error: 'log_path' not found in config.json.")
    exit(1)

# Set up logging
log_file = os.path.join(config['log_path'], 'pipeline.log')
logging.basicConfig(filename=log_file, level=logging.INFO)

def process_trades():
    logging.info("Processing trades (placeholder)")

if __name__ == "__main__":
    process_trades()