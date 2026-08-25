import os

# Folder where your scripts are
scripts_path = "C:\\Users\\Administrator\\Documents\\FXJEFE_Project\\Scripts"

# The starting code to add
starting_block = """import json
import os
import logging

# Path to the config file
CONFIG_PATH = 'C:\\\\Users\\\\Administrator\\\\Documents\\\\FXJEFE_Project\\\\config.json'

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

"""

# Add the starting block to every Python script
for filename in os.listdir(scripts_path):
    if filename.endswith('.py'):
        filepath = os.path.join(scripts_path, filename)
        with open(filepath, 'r') as f:
            content = f.read()
        if 'CONFIG_PATH' not in content:  # Skip if already added
            with open(filepath, 'w') as f:
                f.write(starting_block + content)