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
import numpy as np
from collections import deque
from fxjefe_paths import load_config, features_path, models_file

config = load_config()
models_loaded = False
lstm_model = None
SEQUENCE_LENGTH = 8
FEATURES = list(config.get("features") or [])
feature_history = {}

def _try_load():
    global models_loaded, lstm_model
    # prefer zoo lstm then OG lstm_model.h5 (read-only)
    zoo = os.path.join(config["models_path"], config.get("model_write_dir") or "og333_runs")
    cands = []
    if os.path.isdir(zoo):
        for fn in sorted(os.listdir(zoo), reverse=True):
            if fn.startswith("zoo_lstm") and fn.endswith(".pkl"):
                cands.append(os.path.join(zoo, fn))
    og = models_file(config, "lstm_model.h5")
    if os.path.isfile(og):
        cands.append(og)
    for p in cands:
        try:
            import joblib
            blob = joblib.load(p)
            lstm_model = blob
            models_loaded = True
            logging.info("LSTM loaded %s", p)
            return
        except Exception as e:
            logging.warning("LSTM load fail %s: %s", p, e)
    logging.info("No LSTM artifact loaded")

_try_load()


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


if __name__ == "__main__":
    import pandas as pd
    src = features_path(config)
    if not os.path.isfile(src):
        print("OK get_lstm_prediction: no CSV yet, loader=", models_loaded, flush=True)
        raise SystemExit(0)
    df = pd.read_csv(src, encoding="utf-8", low_memory=False)
    feats = [c for c in FEATURES if c in df.columns]
    if "garch_vol" in df.columns and "garch_vol" not in feats:
        feats = ["garch_vol"] + feats
    tail = df.tail(SEQUENCE_LENGTH)
    arrs = tail[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    sig, conf = None, 0.0
    if models_loaded and hasattr(lstm_model, "predict"):
        try:
            sig, conf = get_lstm_prediction(str(tail.iloc[-1].get("symbol", "EURUSD")), arrs[-1])
        except Exception as e:
            logging.warning("infer: %s", e)
    print(
        f"OK get_lstm_prediction loaded={models_loaded} last_signal={sig} conf={conf} "
        f"garch={float(tail.iloc[-1]['garch_vol']) if 'garch_vol' in tail.columns else 'missing'}",
        flush=True,
    )
