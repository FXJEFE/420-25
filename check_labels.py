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
from fxjefe_paths import features_path, write_feature_csv

cand = [
    os.path.join(config.get("data_path") or "", "training_data_updated.csv"),
    os.path.join(config.get("data_path") or "", "training_data.csv"),
    os.path.join(config.get("data_path") or "", "FXJEFE_Features_with_labels.csv"),
]
src = next((p for p in cand if p and os.path.isfile(p)), None)
if not src:
    raise SystemExit("check_labels: no training_data / labels CSV found")
data = pd.read_csv(src, encoding="utf-8", low_memory=False)
if "label" not in data.columns and "signal" in data.columns:
    data["label"] = pd.to_numeric(data["signal"], errors="coerce").fillna(0)
write_feature_csv(data, config, "training_data_updated.csv")
print("OK check_labels source=", src, flush=True)
print("Current label counts:", flush=True)
print(data["label"].value_counts() if "label" in data.columns else "no label col", flush=True)