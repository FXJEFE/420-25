import json
import os
import logging
import pandas as pd

# Path to the config file
CONFIG_PATH = 'C:\\Users\\Administrator\\Documents\\FXJEFE_Project\\config.json'

# Load the config file safely
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find config file at {CONFIG_PATH}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Config file has invalid format - {e}")
    exit(1)

# Set up logging
log_file = os.path.join(config['log_path'], 'generate_training_data.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info("Script started and configuration loaded successfully")

# Read the features CSV
features_path = os.path.join(config['data_output_path'], 'FXJEFE_Features.csv')
try:
    # Try UTF-8 with BOM first, then fall back to UTF-8
    try:
        df = pd.read_csv(features_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(features_path, encoding='utf-8')
except Exception as e:
    logging.error(f"Failed to read {features_path}: {e}")
    exit(1)
logging.info(f"Number of rows in FXJEFE_Features.csv: {len(df)}")

# Check if 'signal' column exists
if 'signal' not in df.columns:
    logging.error("Error: 'signal' column is missing in FXJEFE_Features.csv.")
    logging.error(f"Available columns: {df.columns.tolist()}")
    exit(1)

# Ensure signal is numeric
df['signal'] = pd.to_numeric(df['signal'], errors='coerce')

# Map signals to labels
df['label'] = df['signal'].map({1: 1, -1: -1, 0: 0}).fillna(0)
logging.info(f"Unique values in label column: {df['label'].unique()}")

training_data_path = os.path.join(config['data_output_path'], 'training_data.csv')
try:
    df.to_csv(training_data_path, index=False, encoding='utf-8')
    logging.info(f"training_data.csv generated successfully at {training_data_path}")
except Exception as e:
    logging.error(f"Failed to save training_data.csv to {training_data_path}: {e}")
    exit(1)