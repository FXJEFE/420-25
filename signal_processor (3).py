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
import numpy as np 
import time 
import logging 
 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 
 
def generate_signals(df): 
    df['SMA_short'] = df['close'].rolling(window=5).mean() 
    df['SMA_long'] = df['close'].rolling(window=20).mean() 
    
    df['Signal'] = 1  # Default: hold 
    df.loc[df['SMA_short'] > df['SMA_long'], 'Signal'] = 2  # Buy 
    df.loc[df['SMA_short'] < df['SMA_long'], 'Signal'] = 0  # Sell 
    
    return df 
 
def main(): 
    while True: 
        try: 
            data = pd.read_csv("realtime_data.csv", encoding='utf-8') 
            if data.empty: 
                logging.warning("No data available yet") 
            else: 
                processed_data = generate_signals(data) 
                processed_data.to_csv("signals_output.csv", index=False, encoding='utf-8') 
                signal_map = {0: 'sell', 1: 'hold', 2: 'buy'} 
                latest_signal = signal_map[processed_data['Signal'].iloc[-1]] 
                logging.info(f"Latest Signal at {time.ctime()}: {latest_signal}") 
            time.sleep(60) 
        except FileNotFoundError: 
            logging.info("Waiting for MT5 data sync...") 
            time.sleep(10) 
        except Exception as e: 
            logging.error(f"Error: {e}") 
            break 
 
if __name__ == "__main__": 
    main() 
