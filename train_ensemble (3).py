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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_PATH = r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\FXJEFE_Features_with_labels.csv"
MODEL_PATH = r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\ensemble_model.pkl"

def train_model():
    try:
        df = pd.read_csv(INPUT_PATH, encoding='utf-8', low_memory=False)
        logging.info(f"Read CSV with labels: {len(df)} rows")

        # Define features (27 indicators) and target
        features = [
            'atr', 'ema_diff', 'rsi', 'macd_diff', 'vwap', 'price_vwap_diff',
            'bb_position', 'roc', 'stochastic', 'cci', 'williams', 'momentum',
            'realized_vol', 'chaikin_vol', 'adx', 'rvi', 'obv', 'volume_delta',
            'ad_line', 'vol_osc', 'supertrend', 'hma', 'ichimoku_tenkan', 'sar',
            'dpo', 'spread', 'sentiment'
        ]
        target = 'label'

        # Verify all required columns are present
        missing_cols = [col for col in features + [target] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        X = df[features]
        y = df[target]

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save the model
        joblib.dump(model, MODEL_PATH)
        logging.info(f"Model trained and saved to {MODEL_PATH}")

        # Evaluate model accuracy
        accuracy = model.score(X_test, y_test)
        logging.info(f"Model accuracy on test set: {accuracy:.2f}")

    except Exception as e:
        logging.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    train_model()
