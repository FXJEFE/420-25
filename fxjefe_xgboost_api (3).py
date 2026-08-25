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
import xgboost as xgb
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained XGBoost model
model = xgb.XGBClassifier()
model_path = "xgboost_model.json"
model.load_model(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    """Receive JSON data, make a prediction, and return a trading signal."""
    try:
        data = request.get_json()
        df = pd.DataFrame([data], columns=["price", "atr", "ema_diff", "rsi", "garch_vol", "macd_diff"])
        prediction = model.predict(df)[0]
        signal = "buy" if prediction == 1 else "sell" if prediction == -1 else "hold"
        return jsonify({"signal": signal})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)  # Use port 8080
python your_flask_script.py
curl -X POST http://127.0.0.1:8080/predict -H "Content-Type: application/json" -d "{\"price\":1.2,\"atr\":0.05,\"ema_diff\":0.01,\"rsi\":55,\"garch_vol\":0.02,\"macd_diff\":0.005}"

pip install flask pandas numpy xgboost


app = Flask(__name__)
model = xgb.XGBClassifier()
model_path = "xgboost_model.json"
model.load_model(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data], columns=["price", "atr", "ema_diff", "rsi", "garch_vol", "macd_diff"])
    prediction = model.predict(df)[0]
    signal = "buy" if prediction == 1 else "sell" if prediction == -1 else "hold"
    return jsonify({"signal": signal})

# Main execution
if __name__ == "__main__":
    features_path = "C:/Users/Administrator/Documents/FXJEFE_Project/FXJEFE_Features.csv"
    labels_path = "C:/Users/Administrator/Documents/FXJEFE_Project/FXJEFE_Labeled.csv"
    model_output_path = "xgboost_model.json"
    train_model(features_path, labels_path, model_output_path)
    app.run(http://127.0.0.1:8080")
# fxjefe_xgboost_local_api.py
# Trains XGBoost model and runs a local Flask API for FXJEFE trading system

import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from flask import Flask, request, jsonify
import os

# Part 1: Train and Save the XGBoost Model
def train_model(features_path, labels_path, model_output_path):
    """Train an XGBoost model and save it to disk."""
    try:
        features = pd.read_csv(features_path)
        labels = pd.read_csv(labels_path)
        X = features[["price", "atr", "ema_diff", "rsi", "garch_vol", "macd_diff"]]
        y = labels["signal"]
        X.fillna(0, inplace=True)  # Handle missing values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
        model.fit(X_train, y_train)
        model.save_model(model_output_path)
        print(f"Model trained and saved as {model_output_path}")
        return model
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return None

# Part 2: Flask API Setup
app = Flask(__name__)

# Load or train model at startup
model_path = "C:/Users/Administrator/Documents/FXJEFE_Project/xgboost_model.json"
features_path = "C:/Users/Administrator/Documents/FXJEFE_Project/FXJEFE_Features.csv"
labels_path = "C:/Users/Administrator/Documents/FXJEFE_Project/FXJEFE_Labeled.csv"

if os.path.exists(model_path):
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print(f"Loaded existing model from {model_path}")
else:
    print("Model not found, training a new one...")
    model = train_model(features_path, labels_path, model_path)
    if model is None:
        raise RuntimeError("Failed to train or load model")

@app.route("/predict", methods=["POST"])
def predict():
    """Receive JSON data, make a prediction, and return a trading signal."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate required fields
        required_fields = ["price", "atr", "ema_diff", "rsi", "garch_vol", "macd_diff"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Convert to DataFrame and ensure numeric values
        df = pd.DataFrame([data], columns=required_fields)
        df = df.astype(float).fillna(0)  # Convert to float and handle NaN
        
        # Make prediction
        prediction = model.predict(df)[0]
        signal = "buy" if prediction == 1 else "sell" if prediction == -1 else "hold"
        return jsonify({"signal": signal})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main execution
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=False)  # Syncs with http://127.0.0.1:8080/predict

