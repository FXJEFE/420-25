# -*- coding: utf-8 -*-
import json
import os
import logging

# Path to the config file
CONFIG_PATH = 'C:\\Users\\LarryLocal\\Documents\\FXJEFE_Project\\config.json'

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
log_file = os.path.join(config['log_path'], 'script.log')  # Change 'script.log' to match the script's name
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
with open('C:\\Users\\LarryLocal\\Documents\\FXJEFE_Project\\config.json', 'r') as f:
    config = json.load(f)
import os

root_dir = "FXJEFE_Trading_System"

structure = {
    "config.json": None,
    "models": {
        "xgboost_model.json": None,
        "ensemble_model.pkl": None,
        "lightgbm_model.txt": None,
        "lstm_model.h5": None
    },
    "data": {
        "ai_server.log": None,
        "FXJEFE_Features.csv": None,
        "FXJEFE_Features.lock": None
    },
    "server.py": None,
    "risk_management.py": None,
    "logging_utils.py": None,
    "FXJEFElogtxt.txt": None,
    "FXJEFE_EA_AI_API_5.0.mq5": None,
    "GenerateFeatures.mq5": None,
    "Predict.mq5": None
}

def create_structure(base_path, structure):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if content is None:
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    pass
        else:
            if not os.path.exists(path):
                os.makedirs(path)
            create_structure(path, content)

create_structure(root_dir, structure)
print(f"Folder structure '{root_dir}' created successfully!")