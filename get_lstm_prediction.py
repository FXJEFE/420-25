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
def get_lstm_prediction(symbol, features_array):
    """Get prediction from LSTM model using sequence data"""
    if not models_loaded or lstm_model is None:
        return None, 0.0
    
    # Initialize history for this symbol if it doesn't exist
    if symbol not in feature_history:
        feature_history[symbol] = deque(maxlen=SEQUENCE_LENGTH)
    
    # Add current features to history
    feature_history[symbol].append(features_array)
    
    # If not enough history, return None
    if len(feature_history[symbol]) < SEQUENCE_LENGTH:
        return None, 0.0
    
    # Prepare sequence for LSTM
    sequence = np.array(list(feature_history[symbol]))
    sequence = sequence.reshape(1, SEQUENCE_LENGTH, len(FEATURES))
    
    # Predict
    lstm_pred = lstm_model.predict(sequence, verbose=0)[0]
    
    # Map prediction to signal
    signal_map = {0: "neutral", 1: "buy", 2: "sell"}
    lstm_class = np.argmax(lstm_pred)
    return signal_map[lstm_class], float(lstm_pred[lstm_class])
