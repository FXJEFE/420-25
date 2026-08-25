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
import os

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Set up logging
logging.basicConfig(filename=os.path.join(config['log_path'], 'pipeline.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def summarize_logs():
    """Summarize pipeline.log into a concise report."""
    log_file = os.path.join(config['log_path'], 'pipeline.log')
    summary_file = os.path.join(config['log_path'], 'log_summary.txt')
    
    if not os.path.exists(log_file):
        logging.error(f"Log file not found: {log_file}")
        return
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    errors = sum(1 for line in lines if 'ERROR' in line)
    warnings = sum(1 for line in lines if 'WARNING' in line)
    successes = sum(1 for line in lines if 'Successfully' in line)
    
    with open(summary_file, 'w') as f:
        f.write(f"Log Summary:\nTotal Lines: {len(lines)}\nErrors: {errors}\nWarnings: {warnings}\nSuccesses: {successes}\n")
    
    logging.info(f"Log summary written to {summary_file}")

if __name__ == "__main__":
    summarize_logs()
