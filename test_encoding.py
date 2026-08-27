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
import pandas as pd
import os

files = [
    'log.txt',
    'FXJEFE_Features.csv',
    'FXJEFE_trades_outcomes.csv',
    'FXJEFE_trades.csv'
]

base_dir = 'C:\\Users\\locallarry\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Files'

for file in files:
    try:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(base_dir, file), encoding='utf-8')
            print(f'Successfully read {file} as UTF-8')
        else:
            with open(os.path.join(base_dir, file), 'r', encoding='utf-8') as f:
                content = f.read()
            print(f'Successfully read {file} as UTF-8')
    except Exception as e:
        print(f'Error reading {file}: {str(e)}')