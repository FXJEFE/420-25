# -*- coding: utf-8 -*-
import os as _os_utf8, sys as _sys_utf8
_os_utf8.environ.setdefault('PYTHONUTF8','1')
_os_utf8.environ.setdefault('PYTHONIOENCODING','utf-8')
for _s in (getattr(_sys_utf8,'stdout',None), getattr(_sys_utf8,'stderr',None)):
    try:
        if _s is not None and hasattr(_s,'reconfigure'):
            _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import json
import os
import logging

# Path to the config file
CONFIG_PATH = 'C:\\Users\\locallarry\\Documents\\FXJEFE_Project\\config.json'

# Load the config file safely
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8', errors='replace') as f:
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
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info("Script started and configuration loaded successfully")

import json
import os
with open('C:\\Users\\locallarry\\Documents\\FXJEFE_Project\\config.json', 'r', encoding='utf-8', errors='replace') as f:
    config = json.load(f)
import json
import logging
import os

# Load configuration
with open('config.json', 'r', encoding='utf-8', errors='replace') as f:
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
    
    with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    errors = sum(1 for line in lines if 'ERROR' in line)
    warnings = sum(1 for line in lines if 'WARNING' in line)
    successes = sum(1 for line in lines if 'Successfully' in line)
    
    with open(summary_file, 'w', encoding='utf-8', errors='replace') as f:
        f.write(f"Log Summary:\nTotal Lines: {len(lines)}\nErrors: {errors}\nWarnings: {warnings}\nSuccesses: {successes}\n")
    
    logging.info(f"Log summary written to {summary_file}")

if __name__ == "__main__":
    summarize_logs()