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
import pandas as pd
import numpy as np
import logging
import optuna
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('C:\\Users\\Administrator\\Documents\\FXJEFE_Fresh\\Logs', 'feature_engineering.log')),
        logging.StreamHandler()
    ]
)

def load_config():
    config_path = 'C:\\Users\\Administrator\\Documents\\FXJEFE_Fresh\\config.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config.json: {e}")
        raise

config = load_config()

def validate_input_data(df):
    required_columns = config['ohlcv_columns'] + ['time', 'symbol', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    min_rows = 26
    if len(df) < min_rows:
        logging.error(f"Insufficient data: {len(df)} rows, need at least {min_rows}")
        raise ValueError(f"Dataframe has {len(df)} rows, need at least {min_rows}")

def calculate_indicators(df):
    logging.info("Starting indicator calculation")
    
    validate_input_data(df)
    
    input_columns = ['close', 'high', 'low']
    if df[input_columns].isna().any().any():
        logging.warning("Input data contains NaN values, filling with forward fill")
        df[input_columns] = df[input_columns].fillna(method='ffill')
    
    df['price'] = df['close']
    
    if TALIB_AVAILABLE:
        logging.info("Using TA-Lib for indicator calculations")
        df['sma'] = talib.SMA(df['close'], timeperiod=14)
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        macd, signal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd_diff'] = macd - signal
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        ema_fast = talib.EMA(df['close'], timeperiod=12)
        ema_slow = talib.EMA(df['close'], timeperiod=26)
        df['ema_diff'] = ema_fast - ema_slow
        df['vwap'] = talib.WMA(df['close'] * df['volume'], timeperiod=14) / talib.WMA(df['volume'], timeperiod=14)
    else:
        logging.info("Using pandas/numpy for indicator calculations")
        df['sma'] = df['close'].rolling(window=14).mean()
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_diff'] = ema_fast - ema_slow
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        df['ema_diff'] = ema_fast - ema_slow
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    df['price_vwap_diff'] = df['price'] - df['vwap']
    df['momentum'] = df['close'].diff(10)
    df['volume_delta'] = df['volume'].diff()
    
    indicator_columns = ['price', 'atr', 'ema_diff', 'rsi', 'macd_diff', 'vwap', 'price_vwap_diff', 'momentum', 'volume_delta']
    nan_counts = df[indicator_columns].isna().sum()
    logging.info(f"NaN counts after calculation: {nan_counts.to_dict()}")
    df[indicator_columns] = df[indicator_columns].fillna(method='bfill')
    
    logging.info("Indicators calculated successfully")
    return df

def optimize_xgboost(X_train, y_train):
    logging.info("Starting XGBoost hyperparameter tuning")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0)
        }
        model = XGBClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        return accuracy_score(y_train, model.predict(X_train))
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    logging.info(f"Best XGBoost params: {study.best_params}")
    return study.best_params

def optimize_lightgbm(X_train, y_train):
    logging.info("Starting LightGBM hyperparameter tuning")
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0]
    }
    model = LGBMClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    logging.info(f"Best LightGBM params: {grid_search.best_params_}")
    return grid_search.best_params_

def train_stacking_model(X_train, y_train, xgboost_params, lightgbm_params):
    logging.info("Starting stacking model training")
    
    estimators = [
        ('xgboost', XGBClassifier(**xgboost_params, random_state=42)),
        ('lightgbm', LGBMClassifier(**lightgbm_params, random_state=42))
    ]
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=3
    )
    stacking_model.fit(X_train, y_train)
    logging.info("Stacking model trained successfully")
    return stacking_model

def main():
    logging.info("Feature engineering started")
    
    csv_path = os.path.join(config['data_output_path'], 'FXJEFE_Features_with_labels.csv')
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded data from {csv_path} with shape {df.shape}")
    except Exception as e:
        logging.error(f"Failed to load data: {str(e)}")
        raise
    
    df = calculate_indicators(df)
    
    features = config['features']
    available_features = [f for f in features if f in df.columns]
    if not available_features:
        logging.error("No valid features available")
        raise ValueError("No valid features available")
    if len(available_features) < len(features):
        logging.warning(f"Missing features: {[f for f in features if f not in available_features]}")
    
    X = df[available_features].dropna()
    y = df['label'].loc[X.index]
    logging.info(f"Features prepared with shape {X.shape}, target with shape {y.shape}")
    
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    logging.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    try:
        xgboost_params = optimize_xgboost(X_train, y_train)
        lightgbm_params = optimize_lightgbm(X_train, y_train)
    except Exception as e:
        logging.error(f"Hyperparameter tuning failed: {str(e)}")
        raise
    
    try:
        model = train_stacking_model(X_train, y_train, xgboost_params, lightgbm_params)
    except Exception as e:
        logging.error(f"Model training failed: {str(e)}")
        raise
    
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    logging.info(f"Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")
    
    model_path = os.path.join(config['models_path'], 'stacking_model.pkl')
    try:
        joblib.dump(model, model_path)
        logging.info(f"Model saved to {model_path}")
    except Exception as e:
        logging.error(f"Failed to save model: {str(e)}")
        raise
    
    output_path = os.path.join(config['data_output_path'], 'processed_features.csv')
    try:
        df.to_csv(output_path, index=False)
        logging.info(f"Processed data saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save processed data: {str(e)}")
        raise
    
    logging.info("Feature engineering completed successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}")
        raise