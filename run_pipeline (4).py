import json
import os
import logging

# Path to the config file
CONFIG_PATH = '/Users/localhugo/FXJEFE_Project/files-18303052/FXJEFE_Project/config.json'

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
log_file = os.path.join(config['log_path'], 'script.log')  # Change 'script.log' to match the script�s name
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
import subprocess
import logging
import time
import requests
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('C:\\Users\\Administrator\\Documents\\FXJEFE_Fresh\\Logs', 'pipeline.log')),
        logging.StreamHandler()
    ]
)

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config.json: {e}")
        exit(1)

pipeline_scripts = [
    'create_structure.py',
    'mt5_data_sync.py',
    'fix_csv.py',
    'fix_csv_encoding.py',
    'convert_encoding.py',
    'generate_new_csv.py',
    'process_trades.py',
    'merge_datasets.py',
    'generate_labels.py',
    'feature_engineering.py',
    'clean_training_data.py',
    'generate_training_data.py',
    'Load_and_Process.py',
    'validate_data.py',
    'train_models.py',
    'ensemble_predictions.py',
    'generate_signals_with_xgboost.py',
    'get_lstm_prediction.py',
    'fxjefe_xgboost_api.py',
    'check_integrity.py',
    'check_labels.py',
    'log_summary.py',
    'parse_log_to_csv.py',
    'risk_management.py',
    'signal_processor.py',
    'mt5_signal_script.py',
    'update_database.py',
    'update_scripts.py',
    'test_encoding.py',
    'test_regex.py',
    'test_server.py',
    'waitress server.py',
    'logging_utils.py'
]

optional_scripts = [
    'adjust_headers.py',
    'analyze_outcomes.py'
]

def check_server_health(config):
    try:
        response = requests.get(f"{config['ai_server_url']}/health", timeout=5)
        if response.status_code == 200 and response.json().get('status') == 'running':
            logging.info("AI server is running")
            return True
        logging.error("AI server not running")
        return False
    except Exception as e:
        logging.error(f"Server health check failed: {e}")
        return False

def start_ai_server(config):
    try:
        script_path = os.path.join(config['scripts_path'], 'ai_server.py')
        subprocess.Popen(['python', script_path], creationflags=subprocess.CREATE_NEW_CONSOLE)
        logging.info("Started AI server")
        time.sleep(5)
        return True
    except Exception as e:
        logging.error(f"Failed to start AI server: {e}")
        return False

def run_script(script, config):
    try:
        script_path = os.path.join(config['scripts_path'], script)
        if not os.path.exists(script_path):
            logging.warning(f"Script {script} not found, skipping")
            return False
        result = subprocess.run(['python', script_path], check=True, capture_output=True, text=True)
        logging.info(f"Successfully executed {script}: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing {script}: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run FXJEFE Trading Pipeline')
    parser.add_argument('--config', default='C:\\Users\\Administrator\\Documents\\FXJEFE_Fresh\\config.json', help='Path to config.json')
    parser.add_argument('--retry', type=int, default=3, help='Number of retries for failed scripts')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--include-optional', action='store_true', help='Run optional scripts')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)
    config['ai_server_url'] = 'http://127.0.0.1:8080'

    if not check_server_health(config):
        logging.info("AI server not running, attempting to start it")
        if not start_ai_server(config) or not check_server_health(config):
            logging.error("AI server failed to start, aborting pipeline")
            exit(1)

    scripts_to_run = pipeline_scripts
    if args.include-optional:
        scripts_to_run += optional_scripts

    for script in scripts_to_run:
        attempts = 0
        while attempts < args.retry:
            if run_script(script, config):
                break
            attempts += 1
            logging.warning(f"Retrying {script} (attempt {attempts}/{args.retry})")
            time.sleep(2)
        if attempts == args.retry:
            logging.error(f"Pipeline aborted due to repeated errors in {script}")
            exit(1)

    logging.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()