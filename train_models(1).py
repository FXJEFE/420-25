"""
train_models.py
Reads training_data.csv produced by generate_labels.py,
trains a RandomForestClassifier on all 28 features (price + 27 indicators),
and saves my_model.pkl to the models folder.
"""
import os
import json
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

os.makedirs(config['log_path'],   exist_ok=True)
os.makedirs(config['models_path'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config['log_path'], 'train_models.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def train_model(data: pd.DataFrame, features: list, target: str = 'label'):
    X = data[features]
    y = data[target]
    logging.info(f"Training on {len(X)} rows with {len(features)} features.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    logging.info(f"Test accuracy: {score:.4f}")
    return model


def main():
    data_path  = os.path.join(config['data_output_path'], 'training_data.csv')
    model_path = os.path.join(config['models_path'],      'my_model.pkl')

    if not os.path.exists(data_path):
        logging.error(f"training_data.csv not found at {data_path}")
        logging.error("Run generate_labels.py first.")
        raise FileNotFoundError(data_path)

    data = pd.read_csv(data_path, encoding='utf-8')
    logging.info(f"Loaded {len(data)} training rows.")

    features = config['features']   # 28 cols: price + 27 indicators
    missing  = [c for c in features if c not in data.columns]
    if missing:
        logging.error(f"Missing feature columns in training data: {missing}")
        raise ValueError(f"Missing columns: {missing}")
    if 'label' not in data.columns:
        logging.error("'label' column not found in training data.")
        raise ValueError("Missing label column")

    data = data.dropna(subset=features + ['label'])
    if len(data) == 0:
        logging.error("No rows remaining after dropping NaNs.")
        raise ValueError("Empty training set")

    model = train_model(data, features)
    joblib.dump(model, model_path)
    logging.info(f"Model saved → {model_path}  (expects {model.n_features_in_} features)")


if __name__ == '__main__':
    main()
