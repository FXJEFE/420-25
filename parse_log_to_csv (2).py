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
import pandas as pd
import re
import json
import logging
import os
from multiprocessing import Pool, cpu_count, freeze_support
import json

with open('config.json', 'r') as f:
    config = json.load(f)

log_file = os.path.join(config['data_output_path'], 'log.txt')
output_csv = os.path.join(config['data_output_path'], 'FXJEFE_Features.csv')

# Setup logging
logging.basicConfig(level=logging.INFO, filename='parse_log.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# File paths (adjusted for your folder)
LOG_PATH = "C:/Users/Administrator/Documents/FXJEFE_Project/log.txt"
OUTPUT_PATH = "C:/Users/Administrator/Documents/FXJEFE_Project/FXJEFE_Features.csv"

# Regex patterns
api_pattern = re.compile(r"(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}).*Sending API request for (\w+\.r): ({.*})")
signal_pattern = re.compile(r"(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}).*Server response for (\w+\.r): .*\"signal\":\"(buy|sell|hold)\"")

# Try different encodings
encodings = ['utf-8', 'utf-16le', 'ansi', 'latin1']
log_lines = None
for enc in encodings:
    try:
        with open(LOG_PATH, "r", encoding=enc) as file:
            log_lines = file.readlines()
        logging.info(f"Successfully read log with {enc} encoding")
        break
    except UnicodeDecodeError as e:
        logging.error(f"Failed to decode with {enc}: {e}")
if log_lines is None:
    logging.error("All encodings failed. Check log.txt manually.")
    exit(1)

# Parse individual log lines
def parse_line(line):
    api_match = api_pattern.search(line)
    if api_match:
        time, symbol, json_data = api_match.groups()
        try:
            data = json.loads(json_data)
            return {
                "time": time,
                "symbol": symbol,
                "price": float(data["price"]),
                "atr": float(data["atr"]),
                "ema_diff": float(data["ema_diff"]),
                "rsi": float(data["rsi"]),
                "garch_vol": float(data["garch_vol"]),
                "macd_diff": float(data["macd_diff"]),
                "vwap": float(data["vwap"]),
                "price_vwap_diff": float(data["price_vwap_diff"]),
                "bb_position": float(data["bb_position"]),
                "signal": "hold",
                "future_return": float('nan'),  # Placeholder
                "threshold": float('nan'),      # Placeholder
                "label": 0                      # Placeholder
            }
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON for {symbol} at {time}: {json_data} - {e}")
            return None
    return None

def main():
    with Pool(cpu_count()) as pool:
        results = pool.map(parse_line, log_lines)
    trade_list = [r for r in results if r is not None]

    # Update signals
    for line in log_lines:
        signal_match = signal_pattern.search(line)
        if signal_match:
            time, symbol, signal = signal_match.groups()
            for trade in trade_list:
                if trade["time"] == time and trade["symbol"] == symbol:
                    trade["signal"] = signal
                    break

    # Create DataFrame with full header
    df = pd.DataFrame(trade_list).dropna(subset=["price"])
    if df.empty:
        logging.error("No valid trade data extracted.")
        exit(1)

    df["time"] = pd.to_datetime(df["time"], format="%Y.%m.%d %H:%M:%S.%f")
    df = df.sort_values("time").reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"{OUTPUT_PATH} created with {len(df)} entries")

if __name__ == '__main__':
    freeze_support()
    main()