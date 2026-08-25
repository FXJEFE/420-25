"""
feature_engineering.py
Reads FXJEFE_Features_with_labels.csv (output of generate_labels.py),
trains an XGBoost + LightGBM stacking classifier on the 28 model features,
and writes:
  models/stacking_model.pkl
  data/processed_features.csv

If raw OHLCV columns (open/high/low/close/volume) are present, additional
indicators are recalculated via TA-Lib or pandas/numpy.  When the data comes
directly from MT5's GenerateFeatures.mq5 (pre-computed indicators), the
recalculation step is skipped and existing feature columns are used as-is.
"""
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

os.makedirs(config['log_path'],    exist_ok=True)
os.makedirs(config['models_path'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config['log_path'], 'feature_engineering.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Optional heavy imports
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("optuna not installed — XGBoost will use default hyperparameters.")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("xgboost not installed — stacking model will use RandomForest as fallback.")

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logging.warning("lightgbm not installed — stacking model will use RandomForest as fallback.")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

OHLCV_COLS = config.get('ohlcv_columns', ['open', 'high', 'low', 'close', 'volume'])


# ── indicator recalculation (only when raw OHLCV is present) ────────────────

def recalculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate technical indicators from raw OHLCV columns."""
    df = df.copy()
    for col in ['close', 'high', 'low']:
        df[col] = df[col].ffill()

    df['price'] = df['close']

    if TALIB_AVAILABLE:
        logging.info("Recalculating indicators via TA-Lib.")
        df['rsi']      = talib.RSI(df['close'], timeperiod=14)
        macd, sig, _   = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd_diff']= macd - sig
        df['atr']      = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        ema_fast       = talib.EMA(df['close'], timeperiod=12)
        ema_slow       = talib.EMA(df['close'], timeperiod=26)
        df['ema_diff'] = ema_fast - ema_slow
        vol_wma        = talib.WMA(df['volume'].astype(float), timeperiod=14)
        df['vwap']     = talib.WMA(df['close'] * df['volume'], timeperiod=14) / vol_wma
    else:
        logging.info("Recalculating indicators via pandas/numpy.")
        delta          = df['close'].diff()
        gain           = delta.where(delta > 0, 0).rolling(14).mean()
        loss           = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi']      = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        ema_fast       = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow       = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_diff']= ema_fast - ema_slow
        df['ema_diff'] = ema_fast - ema_slow
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low']  - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr']      = tr.rolling(14).mean()
        df['vwap']     = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    df['price_vwap_diff'] = df['price'] - df['vwap']
    df['momentum']        = df['close'].diff(10)
    df['volume_delta']    = df['volume'].diff()

    # Fill NaN from leading rolling windows
    for col in ['rsi', 'macd_diff', 'atr', 'ema_diff', 'vwap',
                'price_vwap_diff', 'momentum', 'volume_delta']:
        if col in df.columns:
            df[col] = df[col].bfill().ffill()

    return df


# ── hyperparameter optimisation ──────────────────────────────────────────────

def optimize_xgboost(X_train, y_train) -> dict:
    if OPTUNA_AVAILABLE and XGBOOST_AVAILABLE:
        logging.info("Tuning XGBoost with Optuna (20 trials).")

        def objective(trial):
            params = {
                'n_estimators':  trial.suggest_int('n_estimators', 50, 200),
                'max_depth':     trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample':     trial.suggest_float('subsample', 0.5, 1.0),
                'use_label_encoder': False,
                'eval_metric': 'mlogloss',
            }
            m = XGBClassifier(**params, random_state=42)
            m.fit(X_train, y_train)
            return accuracy_score(y_train, m.predict(X_train))

        study = optuna.create_study(direction='maximize')
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=20)
        logging.info(f"Best XGBoost params: {study.best_params}")
        return study.best_params
    else:
        logging.info("Using default XGBoost params (optuna/xgboost not available).")
        return {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8}


def optimize_lightgbm(X_train, y_train) -> dict:
    if LGBM_AVAILABLE:
        logging.info("Tuning LightGBM with GridSearchCV.")
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth':    [3, 5, 7],
            'learning_rate':[0.01, 0.1, 0.2],
            'subsample':    [0.6, 0.8, 1.0],
        }
        gs = GridSearchCV(LGBMClassifier(random_state=42, verbose=-1),
                          param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        gs.fit(X_train, y_train)
        logging.info(f"Best LightGBM params: {gs.best_params_}")
        return gs.best_params_
    else:
        logging.info("Using default LightGBM params (lightgbm not available).")
        return {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8}


# ── stacking model ───────────────────────────────────────────────────────────

def build_estimators(xgb_params: dict, lgbm_params: dict) -> list:
    from sklearn.ensemble import RandomForestClassifier
    estimators = []
    if XGBOOST_AVAILABLE:
        estimators.append(('xgboost', XGBClassifier(
            **xgb_params, random_state=42,
            use_label_encoder=False, eval_metric='mlogloss'
        )))
    else:
        estimators.append(('rf1', RandomForestClassifier(n_estimators=100, random_state=42)))

    if LGBM_AVAILABLE:
        estimators.append(('lightgbm', LGBMClassifier(**lgbm_params, random_state=42, verbose=-1)))
    else:
        estimators.append(('rf2', RandomForestClassifier(n_estimators=100, random_state=43)))

    return estimators


def train_stacking_model(X_train, y_train, xgb_params: dict, lgbm_params: dict):
    logging.info("Training stacking classifier (XGBoost + LightGBM → LogisticRegression).")
    estimators = build_estimators(xgb_params, lgbm_params)
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
        n_jobs=-1
    )
    stacking.fit(X_train, y_train)
    logging.info("Stacking model trained.")
    return stacking


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    logging.info("feature_engineering.py started.")

    csv_path = os.path.join(config['data_output_path'], 'FXJEFE_Features_with_labels.csv')
    if not os.path.exists(csv_path):
        logging.error(f"Input file not found: {csv_path}")
        logging.error("Run generate_labels.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
    logging.info(f"Loaded {len(df)} rows from {csv_path}")

    if len(df) < 26:
        logging.error(f"Insufficient rows ({len(df)}). Need at least 26.")
        sys.exit(1)

    # Only recalculate indicators if raw OHLCV columns are present
    # (MT5 pre-computed features skip this step)
    has_ohlcv = all(c in df.columns for c in OHLCV_COLS)
    if has_ohlcv:
        logging.info("OHLCV columns detected — recalculating indicators.")
        df = recalculate_indicators(df)
    else:
        logging.info("No raw OHLCV columns — using pre-computed MT5 indicators as-is.")

    # Select features
    features = config['features']   # 28: price + 27 indicators
    available = [f for f in features if f in df.columns]
    missing   = [f for f in features if f not in df.columns]
    if missing:
        logging.warning(f"Missing features (will predict with {len(available)}): {missing}")
    if not available:
        logging.error("No features available — aborting.")
        sys.exit(1)

    if 'label' not in df.columns:
        logging.error("'label' column not found. Run generate_labels.py first.")
        sys.exit(1)

    data = df[available + ['label']].dropna()
    if len(data) < 26:
        logging.error(f"Only {len(data)} usable rows after dropping NaNs.")
        sys.exit(1)

    X = data[available]
    y = data['label']

    # XGBoost requires contiguous integer labels starting at 0.
    # Remap: -1→0, 0→1, 1→2  (internal only — model file stores mapped labels)
    label_map   = {-1: 0, 0: 1, 1: 2}
    y_mapped    = y.map(label_map)
    logging.info(f"Training on {len(X)} rows × {len(available)} features.")

    split = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y_mapped.iloc[:split], y_mapped.iloc[split:]

    xgb_params  = optimize_xgboost(X_train, y_train)
    lgbm_params = optimize_lightgbm(X_train, y_train)
    model       = train_stacking_model(X_train, y_train, xgb_params, lgbm_params)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    logging.info(f"Train accuracy: {train_acc:.4f}  |  Test accuracy: {test_acc:.4f}")

    model_path = os.path.join(config['models_path'], 'stacking_model.pkl')
    joblib.dump(model, model_path)
    logging.info(f"Stacking model saved → {model_path}")

    out_path = os.path.join(config['data_output_path'], 'processed_features.csv')
    df.to_csv(out_path, index=False, encoding='utf-8')
    logging.info(f"Processed features saved → {out_path}")

    logging.info("feature_engineering.py completed successfully.")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f"feature_engineering.py failed: {e}")
        raise
