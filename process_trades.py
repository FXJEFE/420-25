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
import logging
import json
import os

# Load configuration
try:
    with open('config.json', 'r', encoding='utf-8', errors='replace') as f:
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
    from fxjefe_paths import write_feature_csv
    import pandas as pd
    src = os.path.join(config.get("project_root") or "", "FXJEFE_trades.csv")
    if not os.path.isfile(src):
        src = os.path.join(config.get("data_path") or "", "FXJEFE_trades.csv")
    if os.path.isfile(src):
        df = pd.read_csv(src, encoding="utf-8", header=None)
        if df.shape[1] >= 9:
            df.columns = ["ticket", "time", "symbol", "comment", "type", "lots", "price", "sl", "tp"][: df.shape[1]]
        write_feature_csv(df, config, "FXJEFE_trades.csv")
        logging.info("Processed %s trade rows from %s", len(df), src)
        print(f"OK process_trades: {len(df)} rows from {src}", flush=True)
    else:
        logging.info("Processing trades — no trades file yet (schema published)")
        print("OK process_trades: no live trades file (empty schema ok)", flush=True)

if __name__ == "__main__":
    process_trades()