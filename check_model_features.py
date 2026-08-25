"""
check_model_features.py
Loads my_model.pkl and verifies the number of features matches config.
"""
import os
import json
import logging
import joblib

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

os.makedirs(config['log_path'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config['log_path'], 'check_model_features.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def main():
    model_path = os.path.join(config['models_path'], 'my_model.pkl')

    if not os.path.exists(model_path):
        logging.error(f"Model not found: {model_path}")
        logging.error("Run train_models.py first.")
        raise FileNotFoundError(model_path)

    model = joblib.load(model_path)
    n_model    = model.n_features_in_
    n_config   = len(config['features'])
    status     = "OK" if n_model == n_config else "MISMATCH"

    logging.info(f"Model feature count : {n_model}")
    logging.info(f"Config feature count: {n_config}  ({config['features']})")
    logging.info(f"Feature count check : {status}")

    if status != "OK":
        raise ValueError(
            f"Feature count mismatch — model expects {n_model}, config has {n_config}. "
            "Retrain the model with the current config['features']."
        )


if __name__ == '__main__':
    main()
