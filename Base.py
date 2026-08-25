import json
import os
import logging
import sys
import joblib
import xgboost as xgb
import numpy as np
from datetime import datetime

# === DYNAMIC USB AUTO-DETECTION ===
USB_ROOTS = ["X:/", "Y:/", "Z:/", "W:/"]
CONFIG_NAMES = ["config.json", "config_og_random_forest.json"]

def find_config():
    for root in USB_ROOTS:
        if not os.path.exists(root):
            continue
        for name in CONFIG_NAMES:
            path = os.path.join(root, name)
            if os.path.exists(path):
                return path, root
        config_dir = os.path.join(root, "06_Configs")
        if os.path.exists(config_dir):
            for name in CONFIG_NAMES:
                path = os.path.join(config_dir, name)
                if os.path.exists(path):
                    return path, root
    return None, None

# Load config
config_path, usb_root = find_config()
if config_path is None:
    print("ERROR: config.json not found on USB!")
    sys.exit(1)

try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
except Exception as e:
    print(f"ERROR: Cannot read config â†’ {e}")
    sys.exit(1)

# Force correct usb_root
usb_root = config.setdefault('general', {}).setdefault('usb_root', usb_root.rstrip("/\\") + "/")

# Paths
models_path = os.path.join(usb_root, "05_Models")
log_path = config.get('log_path', os.path.join(usb_root, "03_Data/Logs"))
os.makedirs(log_path, exist_ok=True)
log_file = os.path.join(log_path, 'prediction_engine.log')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logging.info("PREDICTION ENGINE STARTED â€"" MULTI-MODEL LOADER v9.9")
logging.info(f"USB Root: {usb_root}")
logging.info(f"Models Folder: {models_path}")

# === MODEL LOADER CLASS ===
class ModelLoader:
    def __init__(self, models_path):
        self.models_path = models_path
        self.models = {}
        self.scaler = None

    def load(self, name, filename):
        path = os.path.join(self.models_path, filename)
        if not os.path.exists(path):
            logging.warning(f"Model not found: {filename}")
            return False
        try:
            if filename.endswith(".pkl"):
                self.models[name] = joblib.load(path)
                logging.info(f"Loaded {name}: {filename}")
            elif filename.endswith(".json"):
                self.models[name] = xgb.XGBClassifier()
                self.models[name].load_model(path)
                logging.info(f"Loaded {name}: {filename}")
            return True
        except Exception as e:
            logging.error(f"Failed to load {filename}: {e}")
            return False

    def load_scaler(self, filename="scaler.pkl"):
        path = os.path.join(self.models_path, filename)
        if os.path.exists(path):
            self.scaler = joblib.load(path)
            logging.info(f"Scaler loaded: {filename}")
        else:
            logging.warning("scaler.pkl not found â€"" predictions may be unstable")

    def predict(self, features_np):
        """Return highest confidence prediction from all loaded models"""
        if not self.models:
            return None, 0.0

        probs = {}
        for name, model in self.models.items():
            try:
                if self.scaler:
                    X = self.scaler.transform(features_np)
                else:
                    X = features_np
                prob = model.predict_proba(X)[0][1]  # Probability of class 1 (BUY)
                probs[name] = prob
            except:
                probs[name] = 0.0

        # Return best model and probability
        best_model = max(probs, key=probs.get)
        best_prob = probs[best_model]
        return best_model, best_prob

# === LOAD ALL MODELS ===
loader = ModelLoader(models_path)
loader.load_scaler()

# Load your 4 models
loader.load("ensemble", "ensemble_model.pkl")
loader.load("my_model", "my_model.pkl")
loader.load("xgboost", "xgboost_model.json")
loader.load("11_feature_rf", "11_feature_rf.pkl")  # â† your new one

# === PREDICTION FUNCTION ===
def predict_signal(features_list):
    """
    Input: list of 41 (or 11) features from MT5
    Output: (direction, probability, model_name)
    """
    if len(features_list) not in [11, 41]:
        logging.error(f"Invalid feature count: {len(features_list)}")
        return "HOLD", 0.0, "ERROR"

    X = np.array([features_list])

    model_name, prob = loader.predict(X)

    if prob >= 0.98:
        direction = "BUY" if prob > 0.5 else "SELL"
    elif prob <= 0.02:
        direction = "SELL" if prob < 0.5 else "BUY"
    else:
        direction = "HOLD"

    logging.info(f"PREDICTION | {model_name} | Prob={prob:.4f} | Signal={direction}")
    return direction, prob, model_name

# === TEST IT RIGHT NOW ===
if __name__ == "__main__":
    print("\n" + "="*60)
    print("MULTI-MODEL PREDICTION ENGINE â€"" LIVE ON USB")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models loaded: {list(loader.models.keys())}")
    if loader.scaler:
        print("Scaler: LOADED")
    print("-"*60)

    # Test with dummy 41-feature vector (all zeros â†’ should be ~0.5)
    dummy_41 = [0.0] * 41
    direction, prob, model = predict_signal(dummy_41)
    print(f"Test Signal â†’ {direction} | {prob:.4f} | {model}")

    print("\nEngine ready. Send features from MT5 â†’ get 0.98+ signals instantly.\n")


