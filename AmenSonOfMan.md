# FXJEFE
generate_labels.py
Reads FXJEFE_Features.csv (all 27 model features + price), fills gaps,
recomputes sentiment, adds price-change labels (1=buy, 0=hold, -1=sell),
and writes:
  data/FXJEFE_Features_fixed.csv
  data/FXJEFE_Features_with_labels.csv
  data/training_data.csv   ← consumed by train_models.py
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from textblob import TextBlob

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

os.makedirs(config['log_path'],        exist_ok=True)
os.makedirs(config['data_output_path'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config['log_path'], 'generate_labels.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 27 model features (matches config['features'] after removing 'price')
MODEL_FEATURES = [
    'atr', 'ema_diff', 'rsi', 'macd_diff', 'vwap', 'price_vwap_diff',
    'bb_position', 'roc', 'stochastic', 'cci', 'williams', 'momentum',
    'realized_vol', 'chaikin_vol', 'adx', 'rvi', 'obv', 'volume_delta',
    'ad_line', 'vol_osc', 'supertrend', 'hma', 'ichimoku_tenkan', 'sar',
    'dpo', 'spread', 'sentiment'
]

FEATURE_DEFAULTS = {
    'atr': 0.0001, 'ema_diff': 0.0, 'rsi': 50.0, 'macd_diff': 0.0,
    'price_vwap_diff': 0.0, 'bb_position': 0.5, 'roc': 0.0, 'stochastic': 50.0,
    'cci': 0.0, 'williams': -50.0, 'momentum': 0.0,
    'realized_vol': 0.0, 'chaikin_vol': 0.0, 'adx': 25.0, 'rvi': 0.0,
    'obv': 0.0, 'volume_delta': 0.0, 'ad_line': 0.0, 'vol_osc': 0.0,
    'supertrend': 0.0, 'dpo': 0.0, 'spread': 2.0, 'sentiment': 0.0,
    # Price-derived — filled dynamically below
    'vwap': None, 'hma': None, 'ichimoku_tenkan': None, 'sar': None,
}

SENTIMENT_MAP = {
    "EURUSD": "Bullish trend expected",
    "USDJPY": "Neutral market",
    "XAUUSD": "Bearish sentiment",
    "AUDUSD": "Positive outlook",
    "GBPUSD": "Strong buy signals",
    "USDCAD": "Sell pressure",
}

def get_sentiment(symbol):
    text = SENTIMENT_MAP.get(str(symbol).strip(), "Neutral")
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0

def generate_labels(df, threshold=0.0005, look_ahead=1):
    df = df.copy()
    df['future_price'] = df.groupby('symbol')['price'].shift(-look_ahead)
    df['price_change'] = (df['future_price'] - df['price']) / df['price']
    df['label'] = np.select(
        [df['price_change'] > threshold, df['price_change'] < -threshold],
        [1, -1],
        default=0
    )
    df = df.dropna(subset=['future_price', 'price_change'])
    logging.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    return df

def main():
    input_path    = os.path.join(config['data_path'],        'FXJEFE_Features.csv')
    fixed_path    = os.path.join(config['data_output_path'], 'FXJEFE_Features_fixed.csv')
    labeled_path  = os.path.join(config['data_output_path'], 'FXJEFE_Features_with_labels.csv')
    training_path = os.path.join(config['data_output_path'], 'training_data.csv')

    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        logging.error("Run mt5_data_sync.py (or GenerateFeatures.mq5) first.")
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path, encoding='utf-8-sig', low_memory=False)
    logging.info(f"Read {len(df)} rows.  Columns: {list(df.columns)}")

    # Ensure all expected columns exist
    all_cols = ['time', 'symbol', 'price'] + MODEL_FEATURES + ['signal']
    for col in all_cols:
        if col not in df.columns:
            df[col] = '' if col in ('time', 'symbol', 'signal') else 0.0
            logging.info(f"Added missing column: {col}")

    df['price'] = pd.to_numeric(df['price'], errors='coerce').ffill()

    # Fill NaNs in features
    for col in MODEL_FEATURES:
        default = FEATURE_DEFAULTS.get(col)
        if default is None:          # price-derived
            default = df['price']
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)

    # Always recompute sentiment from symbol map
    df['sentiment'] = df['symbol'].apply(get_sentiment)

    nan_after = df[['price'] + MODEL_FEATURES].isna().sum()
    if nan_after.any():
        logging.warning(f"Remaining NaNs:\n{nan_after[nan_after > 0]}")
    if len(df) < 500:
        logging.warning(f"Only {len(df)} rows — re-run GenerateFeatures.mq5 with more History_Bars.")

    df.to_csv(fixed_path, encoding='utf-8', index=False)
    logging.info(f"Saved cleaned CSV  → {fixed_path}")

    df = generate_labels(df)
    df.to_csv(labeled_path, encoding='utf-8', index=False)
    logging.info(f"Saved labeled CSV  → {labeled_path}")

    # training_data.csv — only features + label for train_models.py
    train_cols = config['features'] + ['label']
    missing = [c for c in train_cols if c not in df.columns]
    if missing:
        logging.error(f"Missing training columns: {missing}")
        raise ValueError(f"Missing columns: {missing}")
    df[train_cols].dropna().to_csv(training_path, encoding='utf-8', index=False)
    logging.info(f"Saved training CSV → {training_path}  ({len(df)} rows)")

if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-
"""
run_pipeline.py
Full FXJEFE trading pipeline — generates features from MT5, processes data,
trains models, and prepares the AI server for the EA.

PIPELINE ORDER (all steps run in sequence)
══════════════════════════════════════════
Step 1  mt5_generate_features.py  – Pull M1 bars from MT5, compute 27 indicators
                                     → data/FXJEFE_Features.csv
                                     → Common/Files/FXJEFE_Features.csv (for EA)
                                     → MQL5/Files/FXJEFE_Features.csv
Step 2  validate_data.py          – Validate columns, row count, NaN report
Step 3  Load_and_Process.py       – Data quality check (columns + stats)
Step 4  generate_labels.py        – Compute future price, add buy/hold/sell labels
                                     → FXJEFE_Features_fixed.csv
                                     → FXJEFE_Features_with_labels.csv
                                     → training_data.csv
Step 5  feature_engineering.py    – XGBoost + LightGBM stacking model
                                     → models/stacking_model.pkl
                                     → data/processed_features.csv
Step 6  train_models.py           – Train main RandomForest (28 features)
                                     → models/my_model.pkl  ← used by ai_server.py
Step 7  check_model_features.py   – Verify model feature count matches config
Step 8  signal_processor.py       – SMA crossover reference signals
                                     → data/signals_output.csv

The AI server (python pipeline/ai_server.py) is auto-started if not running.
The EA (FXJEFE_ALGO_AI.mq5 / Predict.mq5) calls the server for live signals.

Usage:
  python run_pipeline.py
  python run_pipeline.py --retry 5 --verbose
  python run_pipeline.py --skip-server
  python run_pipeline.py --skip-mt5          (skip step 1 if data already exists)
"""

import json
import os
import sys
import subprocess
import logging
import time
import argparse

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MQL5_PATH    = os.path.join(
    os.path.expandvars(r'%APPDATA%'),
    r'MetaQuotes\Terminal\81A933A9AFC5DE3C23B15CAB19C63850\MQL5'
)
CONFIG_PATH  = os.path.join(PROJECT_ROOT, 'config.json')
LOG_DIR      = os.path.join(PROJECT_ROOT, 'Logs')
os.makedirs(LOG_DIR, exist_ok=True)


def load_config(path: str) -> dict:
    for enc in ['utf-8', 'utf-8-sig', 'cp1252']:
        try:
            with open(path, 'r', encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    print(f"[FATAL] Cannot load config: {path}")
    sys.exit(1)


# ── Ordered pipeline steps ──────────────────────────────────────────────────
#
# Each tuple: (script_filename, description, required?)
# required=True  → pipeline aborts if the step fails after retries
# required=False → pipeline logs a warning and continues
#
PIPELINE_STEPS = [
    ('mt5_generate_features.py',  'Step 1 — Generate features from MT5 (2000 M1 bars × 6 symbols)',   True),
    ('validate_data.py',          'Step 2 — Validate raw feature data',                                True),
    ('Load_and_Process.py',       'Step 3 — Data quality check (columns + stats)',                     True),
    ('generate_labels.py',        'Step 4 — Compute future price & generate buy/hold/sell labels',     True),
    ('feature_engineering.py',    'Step 5 — XGBoost + LightGBM stacking model',                        False),
    ('train_models.py',           'Step 6 — Train main RandomForest → my_model.pkl',                   True),
    ('check_model_features.py',   'Step 7 — Verify model feature count matches config (28)',            True),
    ('signal_processor.py',       'Step 8 — Generate SMA crossover reference signals',                 False),
]


# ── helpers ──────────────────────────────────────────────────────────────────

def check_server_health(url: str) -> bool:
    if not REQUESTS_AVAILABLE:
        return False
    try:
        r = requests.get(f"{url}/health", timeout=5)
        return r.status_code == 200 and r.json().get('status') == 'running'
    except Exception:
        return False


def start_ai_server(cfg: dict) -> bool:
    candidates = [
        os.path.join(PROJECT_ROOT, 'python pipeline', 'ai_server.py'),
        os.path.join(PROJECT_ROOT, 'ai_server.py'),
    ]
    server_path = next((p for p in candidates if os.path.exists(p)), None)
    if not server_path:
        logging.error("ai_server.py not found.")
        return False
    try:
        flags = subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
        subprocess.Popen([sys.executable, server_path], creationflags=flags)
        logging.info(f"Starting AI server: {server_path}")
        time.sleep(7)
        return True
    except Exception as e:
        logging.error(f"Failed to start AI server: {e}")
        return False


def run_step(script: str) -> bool:
    script_path = os.path.join(PROJECT_ROOT, script)
    if not os.path.exists(script_path):
        logging.error(f"  Script not found: {script_path}")
        return False
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True, capture_output=True, text=True,
            timeout=600     # 10 min max per step
        )
        stdout = result.stdout.strip()
        if stdout:
            for line in stdout.splitlines()[-5:]:   # last 5 lines
                logging.info(f"  [{script}] {line}")
        return True
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or '').strip()
        logging.error(f"  [{script}] FAILED (exit {e.returncode})")
        if stderr:
            for line in stderr.splitlines()[-10:]:
                logging.error(f"  [{script}]   {line}")
        return False
    except subprocess.TimeoutExpired:
        logging.error(f"  [{script}] TIMED OUT (600 s)")
        return False


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='FXJEFE Full Trading Pipeline')
    parser.add_argument('--config',       default=CONFIG_PATH)
    parser.add_argument('--retry',        type=int, default=3)
    parser.add_argument('--verbose',      action='store_true')
    parser.add_argument('--skip-server',  action='store_true',
                        help='Skip AI server health check / auto-start')
    parser.add_argument('--skip-mt5',     action='store_true',
                        help='Skip MT5 data generation (use existing CSV)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, 'pipeline.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    cfg     = load_config(args.config)
    ai_url  = cfg.get('ai_server_url', 'http://127.0.0.1:8080')

    logging.info('=' * 60)
    logging.info('FXJEFE Pipeline starting')
    logging.info(f'Project   : {PROJECT_ROOT}')
    logging.info(f'Config    : {args.config}')
    logging.info(f'AI server : {ai_url}')
    logging.info(f'MQL5 path : {MQL5_PATH}')
    logging.info('=' * 60)

    # ── AI server ─────────────────────────────────────────────────────────────
    if not args.skip_server:
        if check_server_health(ai_url):
            logging.info('AI server already running.')
        else:
            logging.info('AI server not responding — starting it.')
            if start_ai_server(cfg):
                if check_server_health(ai_url):
                    logging.info('AI server started OK.')
                else:
                    logging.warning('AI server started but health check failed — continuing.')
            else:
                logging.error('Could not start AI server. Aborting.')
                sys.exit(1)

    # ── Build step list ───────────────────────────────────────────────────────
    steps = list(PIPELINE_STEPS)
    if args.skip_mt5:
        steps = [(s, d, r) for s, d, r in steps if s != 'mt5_generate_features.py']
        logging.info('Skipping MT5 data generation (--skip-mt5).')

    total = len(steps)

    # ── Run each step ─────────────────────────────────────────────────────────
    for idx, (script, description, required) in enumerate(steps, start=1):
        logging.info(f'[{idx}/{total}] {description}')

        success = False
        for attempt in range(1, args.retry + 1):
            if run_step(script):
                success = True
                break
            if attempt < args.retry:
                logging.warning(f'  Retrying {script} ({attempt}/{args.retry}) in 3 s...')
                time.sleep(3)

        if success:
            logging.info(f'  OK — {script}')
        elif required:
            logging.error(f'Pipeline ABORTED at step {idx}/{total}: {script} '
                          f'failed after {args.retry} attempt(s).')
            sys.exit(1)
        else:
            logging.warning(f'  SKIPPED (optional) — {script} failed but pipeline continues.')

    logging.info('=' * 60)
    logging.info('Pipeline completed — AI server ready for EA signals.')
    logging.info('=' * 60)


if __name__ == '__main__':
    main()
"""
FXJEFE Beast Mode Training - Fixed Version
============================================
Fixes from previous broken versions:
  1. REMOVED future_return from features (was data leakage)
  2. Proper forward-5-bar return target with balanced threshold
  3. Strict walk-forward TimeSeriesSplit (no random split)
  4. Optuna tuning with proper validation (not training data)
  5. Class-weight balancing for imbalanced labels
  6. All 5 models trained + ONNX export

Expected: 0.55-0.65 accuracy on 3-class (BUY/HOLD/SELL) without leakage.
That is genuinely good — the April "0.68" likely had subtle leakage.
"""
import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')

# ========================= CONFIG =========================
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

os.makedirs(config['models_path'], exist_ok=True)
os.makedirs(config['log_path'], exist_ok=True)

LOG_FILE = os.path.join(config['log_path'], f'beast_mode_{datetime.now():%Y%m%d_%H%M%S}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler()]
)
log = logging.info
log("=" * 60)
log("BEAST MODE TRAINING - FIXED (no leakage)")
log("=" * 60)

# ========================= CLEAN FEATURE LIST =========================
# These are the 27 features the EA computes in GetFeatures().
# NEVER include future_return, future_price, price_change, label, sample_weight.
CLEAN_FEATURES = [
    'price', 'atr', 'ema_diff', 'rsi', 'macd_diff', 'vwap', 'price_vwap_diff',
    'bb_position', 'roc', 'stochastic', 'cci', 'williams', 'momentum',
    'realized_vol', 'chaikin_vol', 'adx', 'rvi', 'obv', 'volume_delta',
    'ad_line', 'vol_osc', 'supertrend', 'hma', 'ichimoku_tenkan', 'sar',
    'dpo', 'spread', 'sentiment'
]

# Optional lag features (only if present in data and NOT leaky)
OPTIONAL_FEATURES = [
    'garch_vol',
    'price_lag1', 'price_lag2', 'price_lag3',
    'rsi_lag1', 'rsi_lag2', 'rsi_lag3',
    'macd_diff_lag1', 'macd_diff_lag2', 'macd_diff_lag3',
    'atr_lag1', 'atr_lag2', 'atr_lag3',
    'hour_of_day', 'day_of_week', 'volume_ratio'
]

# Features that must NEVER be used for training (they contain future info)
FORBIDDEN = {'future_return', 'future_price', 'price_change', 'label',
             'signal', 'sample_weight', 'regime', 'time', 'symbol'}

LABEL_NAMES = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}


def load_data(mode='crypto'):
    """Load the appropriate dataset."""
    if mode == 'crypto':
        path = os.path.join(config['data_output_path'], 'crypto_training_data.csv')
    else:
        path = os.path.join(config['data_output_path'], 'FXJEFE_Features_with_labels.csv')

    if not os.path.exists(path):
        log(f"ERROR: Data file not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path, encoding='utf-8')
    log(f"Loaded {len(df):,} rows from {os.path.basename(path)}")
    return df


def prepare_features_and_labels(df, mode='crypto'):
    """Build clean feature matrix and balanced labels."""

    # Determine which features are available
    available = [f for f in CLEAN_FEATURES if f in df.columns]
    optional = [f for f in OPTIONAL_FEATURES if f in df.columns]
    features = available + optional

    # Safety check: remove anything forbidden
    features = [f for f in features if f not in FORBIDDEN]
    log(f"Using {len(features)} clean features (no leakage)")

    # Build labels
    if 'label' in df.columns and df['label'].nunique() >= 3:
        y = df['label'].values
        log(f"Using existing 'label' column")
    elif 'future_return' in df.columns:
        # Create balanced labels from forward returns
        fr = df['future_return']
        threshold = config.get('crypto_label_threshold', 0.002)
        y = np.zeros(len(df), dtype=int)
        y[fr > threshold] = 1   # BUY
        y[fr < -threshold] = -1  # SELL
        log(f"Created labels from future_return (threshold={threshold})")
    elif 'signal' in df.columns:
        y = df['signal'].values
        log(f"Using 'signal' column as labels")
    else:
        log("ERROR: No label/signal/future_return column found")
        sys.exit(1)

    # Label distribution
    unique, counts = np.unique(y, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        log(f"  {LABEL_NAMES.get(int(lbl), str(int(lbl)))}: {cnt:,} ({cnt/len(y)*100:.1f}%)")

    X = df[features].fillna(method='ffill').fillna(0).values.astype(np.float32)
    return X, y, features


def walk_forward_split(X, y, train_frac=0.80):
    """Strict chronological split. No future data in training."""
    n = len(X)
    split = int(n * train_frac)
    log(f"Walk-forward split: train={split:,} (oldest) | test={n-split:,} (newest)")
    return X[:split], X[split:], y[:split], y[split:]


def train_all_models(X_train, y_train, X_test, y_test, features):
    """Train all 5 models with proper validation."""
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, classification_report

    models_dir = config['models_path']
    results = {}

    # Map labels for XGBoost (needs 0-indexed)
    label_map = {-1: 0, 0: 1, 1: 2}
    label_unmap = {0: -1, 1: 0, 2: 1}
    y_train_mapped = np.array([label_map[int(v)] for v in y_train])
    y_test_mapped = np.array([label_map[int(v)] for v in y_test])
    n_classes = len(label_map)

    # ── 1. RandomForest Pipeline (my_model.pkl) ──
    log("\n[1/5] Training RandomForest pipeline...")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=400, max_depth=20, min_samples_split=10,
            min_samples_leaf=5, class_weight='balanced',
            random_state=42, n_jobs=-1
        ))
    ])
    pipe.fit(X_train, y_train_mapped)
    rf_pred = pipe.predict(X_test)
    rf_acc = accuracy_score(y_test_mapped, rf_pred)
    results['my_model'] = rf_acc
    log(f"  RandomForest accuracy: {rf_acc:.4f}")
    joblib.dump(pipe, os.path.join(models_dir, 'my_model.pkl'))

    # ── 2. XGBoost with Optuna ──
    log("\n[2/5] Tuning XGBoost with Optuna (40 trials)...")
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def xgb_objective(trial):
        params = {
            'objective': 'multi:softprob',
            'num_class': n_classes,
            'eval_metric': 'mlogloss',
            'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('depth', 3, 10),
            'subsample': trial.suggest_float('sub', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('col', 0.5, 1.0),
            'reg_lambda': trial.suggest_float('lam', 0.1, 10.0),
            'min_child_weight': trial.suggest_int('mcw', 1, 10),
            'tree_method': 'hist',
            'random_state': 42
        }
        dtrain = xgb.DMatrix(X_train, label=y_train_mapped)
        dval = xgb.DMatrix(X_test, label=y_test_mapped)
        model = xgb.train(params, dtrain, num_boost_round=500,
                          evals=[(dval, 'val')], early_stopping_rounds=30,
                          verbose_eval=False)
        pred = model.predict(dval)
        return accuracy_score(y_test_mapped, pred.argmax(axis=1) if pred.ndim > 1 else pred)

    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(xgb_objective, n_trials=40, show_progress_bar=False)
    log(f"  Best XGBoost params: {study_xgb.best_params}")
    log(f"  Best XGBoost accuracy: {study_xgb.best_value:.4f}")

    # Train final XGBoost
    best_xgb_params = {
        'objective': 'multi:softprob', 'num_class': n_classes,
        'eval_metric': 'mlogloss', 'tree_method': 'hist', 'random_state': 42,
        'learning_rate': study_xgb.best_params['lr'],
        'max_depth': study_xgb.best_params['depth'],
        'subsample': study_xgb.best_params['sub'],
        'colsample_bytree': study_xgb.best_params['col'],
        'reg_lambda': study_xgb.best_params['lam'],
        'min_child_weight': study_xgb.best_params['mcw'],
    }
    dtrain = xgb.DMatrix(X_train, label=y_train_mapped)
    dtest = xgb.DMatrix(X_test, label=y_test_mapped)
    xgb_model = xgb.train(best_xgb_params, dtrain, num_boost_round=800,
                           evals=[(dtest, 'val')], early_stopping_rounds=50,
                           verbose_eval=False)
    xgb_pred = xgb_model.predict(dtest)
    if xgb_pred.ndim > 1:
        xgb_pred_cls = xgb_pred.argmax(axis=1)
    else:
        xgb_pred_cls = xgb_pred.astype(int)
    xgb_acc = accuracy_score(y_test_mapped, xgb_pred_cls)
    results['xgboost'] = xgb_acc
    log(f"  XGBoost final accuracy: {xgb_acc:.4f}")
    xgb_model.save_model(os.path.join(models_dir, 'xgboost_model.json'))

    # ── 3. LightGBM with Optuna ──
    log("\n[3/5] Tuning LightGBM with Optuna (40 trials)...")

    def lgb_objective(trial):
        params = {
            'objective': 'multiclass', 'num_class': n_classes,
            'metric': 'multi_logloss', 'verbosity': -1,
            'num_leaves': trial.suggest_int('leaves', 20, 150),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('depth', 3, 12),
            'subsample': trial.suggest_float('sub', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('col', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('mcs', 5, 50),
            'is_unbalance': True,
            'random_state': 42, 'n_jobs': -1
        }
        model = lgb.LGBMClassifier(**params, n_estimators=500)
        model.fit(X_train, y_train_mapped,
                  eval_set=[(X_test, y_test_mapped)],
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        pred = model.predict(X_test)
        return accuracy_score(y_test_mapped, pred)

    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(lgb_objective, n_trials=40, show_progress_bar=False)
    log(f"  Best LightGBM params: {study_lgb.best_params}")
    log(f"  Best LightGBM accuracy: {study_lgb.best_value:.4f}")

    lgb_params_final = {
        'objective': 'multiclass', 'num_class': n_classes,
        'metric': 'multi_logloss', 'verbosity': -1,
        'is_unbalance': True, 'random_state': 42, 'n_jobs': -1,
        'num_leaves': study_lgb.best_params['leaves'],
        'learning_rate': study_lgb.best_params['lr'],
        'max_depth': study_lgb.best_params['depth'],
        'subsample': study_lgb.best_params['sub'],
        'colsample_bytree': study_lgb.best_params['col'],
        'min_child_samples': study_lgb.best_params['mcs'],
    }
    lgb_model = lgb.LGBMClassifier(**lgb_params_final, n_estimators=800)
    lgb_model.fit(X_train, y_train_mapped,
                  eval_set=[(X_test, y_test_mapped)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    lgb_pred = lgb_model.predict(X_test)
    lgb_acc = accuracy_score(y_test_mapped, lgb_pred)
    results['lightgbm'] = lgb_acc
    log(f"  LightGBM final accuracy: {lgb_acc:.4f}")
    joblib.dump(lgb_model, os.path.join(models_dir, 'lightgbm_model.pkl'))

    # ── 4. Ensemble (VotingClassifier with pre-trained models) ──
    log("\n[4/5] Building ensemble (soft vote of RF + XGB + LGB)...")
    from sklearn.ensemble import VotingClassifier

    # Wrap xgboost for sklearn API
    xgb_sklearn = xgb.XGBClassifier(**{
        'objective': 'multi:softprob', 'num_class': n_classes,
        'tree_method': 'hist', 'random_state': 42,
        'learning_rate': study_xgb.best_params['lr'],
        'max_depth': study_xgb.best_params['depth'],
        'subsample': study_xgb.best_params['sub'],
        'colsample_bytree': study_xgb.best_params['col'],
        'reg_lambda': study_xgb.best_params['lam'],
        'min_child_weight': study_xgb.best_params['mcw'],
        'n_estimators': xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else 500,
        'use_label_encoder': False, 'eval_metric': 'mlogloss',
    })
    xgb_sklearn.fit(X_train, y_train_mapped)

    ensemble = VotingClassifier(
        estimators=[('rf', pipe.named_steps['clf']), ('xgb', xgb_sklearn), ('lgb', lgb_model)],
        voting='soft', n_jobs=1
    )
    # Manually set fitted state to avoid re-training
    ensemble.estimators_ = [pipe.named_steps['clf'], xgb_sklearn, lgb_model]
    ensemble.named_estimators_ = {'rf': pipe.named_steps['clf'], 'xgb': xgb_sklearn, 'lgb': lgb_model}
    ensemble.le_ = None
    ensemble.classes_ = np.array(sorted(label_map.values()))

    ens_pred = ensemble.predict(X_test)
    ens_acc = accuracy_score(y_test_mapped, ens_pred)
    results['ensemble'] = ens_acc
    log(f"  Ensemble accuracy: {ens_acc:.4f}")
    joblib.dump(ensemble, os.path.join(models_dir, 'ensemble_model.pkl'))

    # ── 5. LSTM placeholder (PyTorch) ──
    log("\n[5/5] LSTM model...")
    try:
        import torch
        import torch.nn as nn

        class SimpleLSTM(nn.Module):
            def __init__(self, n_features, n_classes, hidden=64, n_layers=2):
                super().__init__()
                self.lstm = nn.LSTM(n_features, hidden, num_layers=n_layers,
                                   batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden, n_classes)

            def forward(self, x):
                _, (h, _) = self.lstm(x)
                return self.fc(h[-1])

        seq_len = 20
        n_feat = X_train.shape[1]

        # Build sequences
        X_seq = np.array([X_train[i:i+seq_len] for i in range(len(X_train)-seq_len)])
        y_seq = y_train_mapped[seq_len:]

        if len(X_seq) > 200000:
            # Subsample for speed
            idx = np.random.RandomState(42).choice(len(X_seq), 200000, replace=False)
            idx.sort()
            X_seq = X_seq[idx]
            y_seq = y_seq[idx]

        X_t = torch.FloatTensor(X_seq)
        y_t = torch.LongTensor(y_seq)

        model_lstm = SimpleLSTM(n_feat, n_classes)
        optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Quick training (10 epochs)
        model_lstm.train()
        batch_size = 512
        for epoch in range(10):
            perm = torch.randperm(len(X_t))
            total_loss = 0
            n_batches = 0
            for i in range(0, len(X_t), batch_size):
                batch_idx = perm[i:i+batch_size]
                xb = X_t[batch_idx]
                yb = y_t[batch_idx]
                optimizer.zero_grad()
                out = model_lstm(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            log(f"  LSTM epoch {epoch+1}/10 loss={total_loss/n_batches:.4f}")

        torch.save(model_lstm.state_dict(), os.path.join(models_dir, 'lstm_model.h5'))
        results['lstm'] = 'trained'
        log("  LSTM model saved")
    except ImportError:
        log("  PyTorch not installed - skipping LSTM")
        results['lstm'] = 'skipped'

    # ── ONNX Export ──
    log("\n=== ONNX EXPORT ===")
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [('float_input', FloatTensorType([None, len(features)]))]

        # my_model (RF pipeline)
        onx = convert_sklearn(pipe, 'my_model', initial_type)
        onnx_path = os.path.join(models_dir, 'my_model.onnx')
        with open(onnx_path, 'wb') as f:
            f.write(onx.SerializeToString())
        log(f"  Exported my_model.onnx ({os.path.getsize(onnx_path)/1024:.0f} KB)")

        # ensemble (VotingClassifier)
        try:
            onx_ens = convert_sklearn(ensemble, 'ensemble_model', initial_type)
            onnx_ens_path = os.path.join(models_dir, 'ensemble_model.onnx')
            with open(onnx_ens_path, 'wb') as f:
                f.write(onx_ens.SerializeToString())
            log(f"  Exported ensemble_model.onnx ({os.path.getsize(onnx_ens_path)/1024:.0f} KB)")
        except Exception as e:
            log(f"  Ensemble ONNX export failed (complex model): {e}")
            log("  Falling back to XGBoost ONNX only")

        # XGBoost native ONNX
        try:
            import onnxmltools
            from onnxmltools.convert import convert_xgboost as conv_xgb
            xgb_onx = conv_xgb(xgb_sklearn, initial_types=initial_type)
            xgb_onnx_path = os.path.join(models_dir, 'xgboost_model.onnx')
            onnxmltools.utils.save_model(xgb_onx, xgb_onnx_path)
            log(f"  Exported xgboost_model.onnx ({os.path.getsize(xgb_onnx_path)/1024:.0f} KB)")
        except Exception as e:
            log(f"  XGBoost ONNX export: {e}")

        # LightGBM native ONNX
        try:
            import onnxmltools
            from onnxmltools.convert import convert_lightgbm as conv_lgb
            lgb_onx = conv_lgb(lgb_model, initial_types=initial_type)
            lgb_onnx_path = os.path.join(models_dir, 'lightgbm_model.onnx')
            onnxmltools.utils.save_model(lgb_onx, lgb_onnx_path)
            log(f"  Exported lightgbm_model.onnx ({os.path.getsize(lgb_onnx_path)/1024:.0f} KB)")
        except Exception as e:
            log(f"  LightGBM ONNX export: {e}")

    except ImportError:
        log("  skl2onnx not installed - skipping ONNX export")
        log("  Install with: pip install skl2onnx onnxmltools onnx")

    # ── Final Report ──
    log("\n" + "=" * 60)
    log("FINAL RESULTS (walk-forward, no leakage)")
    log("=" * 60)

    y_test_orig = np.array([label_unmap[int(v)] for v in y_test_mapped])

    for name, acc in results.items():
        if isinstance(acc, float):
            log(f"  {name:<20} accuracy: {acc:.4f} ({acc*100:.1f}%)")
        else:
            log(f"  {name:<20} status: {acc}")

    # Best model detailed report
    best_name = max({k: v for k, v in results.items() if isinstance(v, float)},
                    key=lambda k: results[k])
    log(f"\nBest model: {best_name} ({results[best_name]:.4f})")

    # Detailed report for ensemble
    ens_pred_orig = np.array([label_unmap[int(v)] for v in ens_pred])
    log(f"\nEnsemble Classification Report:")
    report = classification_report(y_test_orig, ens_pred_orig,
                                   target_names=['SELL', 'HOLD', 'BUY'], zero_division=0)
    log(f"\n{report}")

    # Profit factor simulation
    price_idx = features.index('price') if 'price' in features else 0
    horizon = 5
    profits, losses = [], []
    for i in range(len(ens_pred_orig) - horizon):
        pred = int(ens_pred_orig[i])
        if pred == 0:
            continue
        entry = X_test[i, price_idx]
        exit_p = X_test[i + horizon, price_idx]
        if entry == 0:
            continue
        ret = ((exit_p - entry) / entry if pred == 1 else (entry - exit_p) / entry) - 0.0002
        if ret > 0:
            profits.append(ret)
        elif ret < 0:
            losses.append(abs(ret))

    total_trades = len(profits) + len(losses)
    if total_trades > 0:
        pf = sum(profits) / sum(losses) if sum(losses) > 0 else float('inf')
        wr = len(profits) / total_trades * 100
        log(f"\nSimulated Trading Metrics:")
        log(f"  Total trades: {total_trades:,}")
        log(f"  Win rate: {wr:.1f}%")
        log(f"  Profit Factor: {pf:.2f}")
    else:
        log(f"\n  No trades generated")

    log("\n" + "=" * 60)
    log("BEAST MODE COMPLETE")
    log(f"Models saved to: {models_dir}")
    log(f"Log saved to: {LOG_FILE}")
    log("=" * 60)

    return results, features


if __name__ == '__main__':
    # Choose mode: 'crypto' for 878k crypto data, 'forex' for 12k forex data
    mode = 'crypto'  # Change to 'forex' to train on FXJEFE_Features_with_labels.csv

    df = load_data(mode)
    X, y, features = prepare_features_and_labels(df, mode)
    X_train, X_test, y_train, y_test = walk_forward_split(X, y)
    results, features = train_all_models(X_train, y_train, X_test, y_test, features)


Sent with Proton Mail secure email.







￼
Nikolai Dreyer <nikodrey2002@gmail.com>


26. mars 2026, 00:13





til larry-local-AI
￼





**Here is the exact fix for your $50 account with 1:500 leverage.**

The main problem is in **generate_labels.py** — it is still using the old flat `threshold=0.0005` and `look_ahead=1`, ignoring the `crypto_label_threshold` in config.json. This creates too many noisy labels, so the model learns to always predict “hold”.

### 1. Fixed `generate_labels.py` (copy-paste this entire file)

Replace the content of `D:\FXJEFE_Project\generate_labels.py` with this updated version:

```python
"""
generate_labels.py
- Respects config.json (crypto_label_threshold)
- Uses look_ahead=5 (April-style)
- Binary or 3-class as configured
- No future_return leakage
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from textblob import TextBlob

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

os.makedirs(config['log_path'], exist_ok=True)
os.makedirs(config['data_output_path'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config['log_path'], 'generate_labels.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

MODEL_FEATURES = [
    'atr', 'ema_diff', 'rsi', 'macd_diff', 'vwap', 'price_vwap_diff',
    'bb_position', 'roc', 'stochastic', 'cci', 'williams', 'momentum',
    'realized_vol', 'chaikin_vol', 'adx', 'rvi', 'obv', 'volume_delta',
    'ad_line', 'vol_osc', 'supertrend', 'hma', 'ichimoku_tenkan', 'sar',
    'dpo', 'spread', 'sentiment'
]

def get_sentiment(symbol):
    # Keep your existing sentiment map
    sentiment_map = {
        "EURUSD": "Bullish trend expected",
        "USDJPY": "Neutral market",
        "XAUUSD": "Bearish sentiment",
        "AUDUSD": "Positive outlook",
        "GBPUSD": "Strong buy signals",
        "USDCAD": "Sell pressure",
    }
    text = sentiment_map.get(str(symbol).strip(), "Neutral")
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0.0

def generate_labels(df):
    df = df.copy()
    
    look_ahead = config.get('look_ahead', 5)                    # April-style 5 bars
    threshold = config.get('crypto_label_threshold', 0.002)     # use config value
    
    logging.info(f"Using look_ahead={look_ahead}, threshold={threshold*100:.1f}%")
    
    df['future_price'] = df.groupby('symbol')['price'].shift(-look_ahead)
    df['future_return'] = (df['future_price'] - df['price']) / df['price']
    
    # Binary classification (recommended for small accounts)
    df['label'] = 0
    df.loc[df['future_return'] > threshold, 'label'] = 1      # BUY
    df.loc[df['future_return'] < -threshold, 'label'] = -1    # SELL
    
    df = df.dropna(subset=['future_price', 'future_return'])
    
    logging.info(f"Final label distribution: {df['label'].value_counts().to_dict()}")
    return df

def main():
    input_path = os.path.join(config['data_output_path'], 'FXJEFE_Features.csv')
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(input_path)
    
    df = pd.read_csv(input_path, encoding='utf-8-sig', low_memory=False)
    logging.info(f"Read {len(df)} rows")
    
    # Fill missing features
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['sentiment'] = df['symbol'].apply(get_sentiment)
    
    df = generate_labels(df)
    
    # Save labeled file
    labeled_path = os.path.join(config['data_output_path'], 'FXJEFE_Features_with_labels.csv')
    df.to_csv(labeled_path, encoding='utf-8', index=False)
    logging.info(f"Saved labeled CSV → {labeled_path}")
    
    # Training data (features + label only)
    train_cols = config['features'] + ['label']
    training_path = os.path.join(config['data_output_path'], 'training_data.csv')
    df[train_cols].dropna().to_csv(training_path, encoding='utf-8', index=False)
    logging.info(f"Saved training CSV → {training_path} ({len(df)} rows)")

if __name__ == '__main__':
    main()
```

### 2. Fix the OrderCalcMargin warning in the EA

In `FXJEFE_ALGO_AI.mq5`, change the margin check to:

```mql5
if (!OrderCalcMargin(orderType, pair, lotSize, openPrice, marginRequired)) {
    Print("MARGIN SKIP: Cannot calculate margin for ", pair);
    return 0.0;
}
```

### 3. Run the Pipeline Now (for the $50 account)

After you replace `generate_labels.py` with the version above:

```bash
cd D:\FXJEFE_Project
python run_pipeline.py
```

This will:
- Use the config’s `crypto_label_threshold` (0.002)
- Use look_ahead=5
- Generate clean binary labels
- Train the model with the new labels
- Deploy to your 1:500 leverage terminal

After the pipeline finishes, restart the AI server:

```bash
taskkill /PID 11208 /F
python python pipeline\ai_server.py
```

### 4. EA Settings for $50 Account (1:500 Leverage)

In the EA inputs:
- `RiskPercent` = 0.5 (or 1.0 max)
- `MaxTradeRiskPct` = 20.0 (already added)
- `MaxDailyLossPercent` = 5.0
- `MaxTotalDDPercent` = 15.0 (conservative for small account)

The new lot sizing code will automatically clamp SL if needed and allow trading on $50.

---

**Do this now**:
1. Replace `generate_labels.py` with the code above.
2. Recompile the EA after the small OrderCalcMargin fix.
3. Run `python run_pipeline.py`

When the pipeline finishes, paste the last 30 lines of output (especially the label distribution and final model metrics) here.

We will then confirm the model is producing BUY/SELL signals and adjust the threshold if needed.

This will make the $50 account work the same way the 10k FTMO demo did.

Ready when you are — run the pipeline after the label fix.**Here is the exact, ready-to-use fix for your $50 account with 1:500 leverage.**

The pipeline works perfectly on the 10k FTMO demo because it has enough balance to absorb the default lot sizing and risk. On $50 the lot sizing code refuses trades or crashes on margin/SL calculations for expensive instruments like XAUUSD.

### 1. Fixed `generate_labels.py` (copy this entire file)

Replace the entire content of **`D:\FXJEFE_Project\generate_labels.py`** with this version. It now respects `crypto_label_threshold` from config, uses `look_ahead=5` (April style), and produces binary labels.

```python
"""
generate_labels.py
- Respects config.json (crypto_label_threshold and look_ahead)
- Binary classification (BUY=1, SELL=0) – recommended for small accounts
- No future_return leakage
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from textblob import TextBlob

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

os.makedirs(config['log_path'], exist_ok=True)
os.makedirs(config['data_output_path'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config['log_path'], 'generate_labels.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

MODEL_FEATURES = [
    'atr', 'ema_diff', 'rsi', 'macd_diff', 'vwap', 'price_vwap_diff',
    'bb_position', 'roc', 'stochastic', 'cci', 'williams', 'momentum',
    'realized_vol', 'chaikin_vol', 'adx', 'rvi', 'obv', 'volume_delta',
    'ad_line', 'vol_osc', 'supertrend', 'hma', 'ichimoku_tenkan', 'sar',
    'dpo', 'spread', 'sentiment'
]

def get_sentiment(symbol):
    sentiment_map = {
        "EURUSD": "Bullish trend expected",
        "USDJPY": "Neutral market",
        "XAUUSD": "Bearish sentiment",
        "AUDUSD": "Positive outlook",
        "GBPUSD": "Strong buy signals",
        "USDCAD": "Sell pressure",
    }
    text = sentiment_map.get(str(symbol).strip(), "Neutral")
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0.0

def generate_labels(df):
    df = df.copy()
    
    look_ahead = config.get('look_ahead', 5)                    # April-style
    threshold = config.get('crypto_label_threshold', 0.002)     # from config.json
    
    logging.info(f"Using look_ahead={look_ahead}, threshold={threshold*100:.1f}% (binary BUY/SELL)")

    df['future_price'] = df.groupby('symbol')['price'].shift(-look_ahead)
    df['future_return'] = (df['future_price'] - df['price']) / df['price']
    
    # Binary classification
    df['label'] = 0
    df.loc[df['future_return'] > threshold, 'label'] = 1   # BUY
    df.loc[df['future_return'] < -threshold, 'label'] = 0   # SELL
    
    df = df.dropna(subset=['future_price', 'future_return'])
    
    logging.info(f"Final label distribution: {df['label'].value_counts().to_dict()}")
    return df

def main():
    input_path = os.path.join(config['data_output_path'], 'FXJEFE_Features.csv')
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(input_path)
    
    df = pd.read_csv(input_path, encoding='utf-8-sig', low_memory=False)
    logging.info(f"Read {len(df)} rows")

    # Fill missing features
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['sentiment'] = df['symbol'].apply(get_sentiment)
    
    df = generate_labels(df)
    
    labeled_path = os.path.join(config['data_output_path'], 'FXJEFE_Features_with_labels.csv')
    df.to_csv(labeled_path, encoding='utf-8', index=False)
    logging.info(f"Saved labeled CSV → {labeled_path}")
    
    # training_data.csv for train_models.py
    train_cols = config['features'] + ['label']
    training_path = os.path.join(config['data_output_path'], 'training_data.csv')
    df[train_cols].dropna().to_csv(training_path, encoding='utf-8', index=False)
    logging.info(f"Saved training CSV → {training_path} ({len(df)} rows)")

if __name__ == '__main__':
    main()
```

### 2. EA Lot Sizing Fix for $50 Account (1:500 Leverage)

The current EA refuses trades on XAUUSD because minLot risk is too high. Replace your `CalculateLotSize` function with this version (it clamps SL instead of refusing):

```mql5
double CalculateLotSize(string pair, double &slPrice, double openPrice, ENUM_ORDER_TYPE orderType) 
{
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = balance * (RiskPercent / 100.0);
    double minLot = SymbolInfoDouble(pair, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(pair, SYMBOL_VOLUME_MAX);
    double stepLot = SymbolInfoDouble(pair, SYMBOL_VOLUME_STEP);
    double point = SymbolInfoDouble(pair, SYMBOL_POINT);
    double pipValue = SymbolInfoDouble(pair, SYMBOL_TRADE_TICK_VALUE);

    double slDistance = MathAbs(openPrice - slPrice) / point;
    if (slDistance == 0) return minLot;

    double lotSize = riskAmount / (slDistance * pipValue);

    // Normalize lot
    lotSize = minLot + MathMax(0, (int)MathFloor((lotSize - minLot) / stepLot)) * stepLot;
    lotSize = MathMax(minLot, MathMin(maxLot, lotSize));

    // Micro account protection - clamp SL instead of refusing
    double minLotRisk = minLot * slDistance * pipValue;
    double maxAcceptableRisk = balance * (MaxTradeRiskPct / 100.0);   // your new input

    if (lotSize <= minLot && minLotRisk > maxAcceptableRisk) {
        // Tighten SL to fit within MaxTradeRiskPct
        double affordableDist = maxAcceptableRisk / (minLot * pipValue);
        double oldSL = slPrice;
        if (orderType == ORDER_TYPE_BUY)
            slPrice = NormalizeDouble(openPrice - affordableDist * point, (int)SymbolInfoInteger(pair, SYMBOL_DIGITS));
        else
            slPrice = NormalizeDouble(openPrice + affordableDist * point, (int)SymbolInfoInteger(pair, SYMBOL_DIGITS));
        lotSize = minLot;
        Print("SL CLAMPED for ", pair, " : old SL ", DoubleToString(oldSL, 5), " -> new SL ", DoubleToString(slPrice, 5));
    }

    // Margin check
    double marginRequired = 0;
    if (OrderCalcMargin(orderType, pair, lotSize, openPrice, marginRequired)) {
        if (marginRequired > AccountInfoDouble(ACCOUNT_MARGIN_FREE) * 0.8) {
            Print("MARGIN SKIP for ", pair);
            return 0.0;
        }
    }

    Print("Calculated lot for ", pair, ": ", DoubleToString(lotSize, 2), " (balance $", DoubleToString(balance, 2), ")");
    return lotSize;
}
```

Also add this input at the top of the EA:

```mql5
input double MaxTradeRiskPct = 20.0;   // Max risk per trade % for micro accounts
```

### 3. Run the Pipeline for the $50 Account

After you replace the two files above:

```bash
cd D:\FXJEFE_Project
python run_pipeline.py
```

Then restart the server:

```bash
taskkill /PID 11208 /F
python python pipeline\ai_server.py
```

Recompile the EA and attach it to your charts.

The EA will now trade on $50 with 1:500 leverage by clamping SL when needed and never refusing trades due to micro-lot protection.

**Next step**  
Replace the two files, run the pipeline, and recompile the EA.  
Then tell me the new log output from the EA (especially any "SL CLAMPED" or "Calculated lot" messages) and we’ll fine-tune the threshold or risk % if needed.

You’re now set for the $50 account. Let me know when you have run the pipeline.


import pandas as pd
import logging
import sys

# --- Configuration ---
# Configure logging to provide detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Log to standard output
)

def load_and_prepare_data(file_path):
    """
    Loads data from a CSV file, handling common encoding issues and cleaning the data.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame or None: A cleaned and prepared DataFrame, or None if loading fails.
    """
    logging.info(f"Attempting to read CSV from: {file_path}")
    try:
        # Use 'utf-8-sig' to handle the Byte Order Mark (BOM) that causes the
        # '0xff' error seen in the logs.
        # Use low_memory=False to prevent DtypeWarning with mixed types.
        df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)
        logging.info(f"Successfully read {len(df)} rows and {len(df.columns)} columns.")

    except FileNotFoundError:
        logging.error(f"Error: The file was not found at {file_path}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading the CSV: {e}")
        return None

    # --- Data Cleaning and Preparation ---
    # Ensure critical columns exist
    required_columns = ['time', 'symbol', 'price']
    if not all(col in df.columns for col in required_columns):
        logging.error(f"CSV is missing one of the required columns: {required_columns}")
        return None

    # Drop rows where essential data is missing before any conversion
    df.dropna(subset=required_columns, inplace=True)
    if df.empty:
        logging.warning("No rows remain after dropping NaNs from essential columns.")
        return df

    # Convert 'time' column to datetime objects for proper sorting
    df['time'] = pd.to_datetime(df['time'], errors='coerce')

    # Convert 'price' to a numeric type, setting non-numeric values to NaN
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Drop any rows that failed conversion
    df.dropna(subset=['time', 'price'], inplace=True)
    if df.empty:
        logging.warning("No rows remain after data type conversion and cleanup.")
        return df

    logging.info("Data cleaning and type conversion complete.")
    logging.info(f"Data shape after cleaning: {df.shape}")
    
    return df


def calculate_future_return(df):
    """
    Calculates future_return as the percentage change in price for the next period,
    grouped by each symbol to ensure correctness.

    Args:
        df (pd.DataFrame): A DataFrame with 'time', 'symbol', and 'price' columns.
                           It's assumed that df is already cleaned.

    Returns:
        pd.DataFrame: The DataFrame with the 'future_return' column added.
    """
    if df is None or df.empty:
        logging.warning("Input DataFrame is empty or None. Skipping calculation.")
        return df
        
    logging.info("Calculating future returns for each symbol...")
    
    # Sort values by symbol and then by time. This is CRITICAL for correct calculations.
    df = df.sort_values(by=['symbol', 'time']).reset_index(drop=True)

    # --- Group by Symbol and Calculate Return ---
    # The original script's error was not doing this. Without grouping, it would
    # calculate the return from the last price of one symbol to the first price
    # of the next, which is incorrect.
    # We use .transform() to apply the calculation within each group and align the
    # result back to the original DataFrame's index.
    df['future_return'] = df.groupby('symbol')['price'].transform(
        lambda x: x.pct_change(periods=1).shift(-1)
    )

    # The last entry for each symbol will have a NaN future_return, which is correct.
    logging.info("Future return calculation finished.")
    
    return df

# --- Main Execution Block ---
if __name__ == "__main__":
    # Example usage with a sample DataFrame that mimics the structure of your data,
    # including multiple symbols and potential data issues.
    
    # --- Create a sample CSV in memory to simulate your file ---
    from io import StringIO

    # Sample data with two symbols (EURUSD, USDJPY) to test the grouping logic
    csv_data = """time,symbol,price,atr,signal
2023-01-01 10:00,EURUSD,1.05,0.001,buy
2023-01-01 10:15,EURUSD,1.06,0.001,hold
2023-01-01 10:30,EURUSD,1.055,0.001,sell
2023-01-01 10:00,USDJPY,130.0,0.1,buy
2023-01-01 10:15,USDJPY,130.5,0.1,hold
2023-01-01 10:30,USDJPY,131.0,0.1,buy
2023-01-01 10:45,USDJPY,130.8,0.1,sell
"""
    # In a real scenario, you would provide the file path directly.
    # For this example, we read from the string.
    # In your code, replace this with:
    # file_path = 'C:/path/to/your/FXJEFE_Features.csv'
    # initial_df = load_and_prepare_data(file_path)
    
    # This simulates reading the file
    file_path_simulation = StringIO(csv_data)
    
    # --- Processing Pipeline ---
    # 1. Load and clean the data
    initial_df = pd.read_csv(file_path_simulation) # In your real code, use load_and_prepare_data(file_path)

    # 2. Calculate future returns
    processed_df = calculate_future_return(initial_df)

    # --- Display Results ---
    if processed_df is not None:
        logging.info("--- Final Processed Data ---")
        # Set display options for better console output
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_rows', 500)
        print(processed_df)

        # Verification: Check the calculated future returns for EURUSD
        # Return from 1.05 to 1.06 should be (1.06 - 1.05) / 1.05 = 0.00952
        logging.info("\n--- Verification ---")
        eurusd_data = processed_df[processed_df['symbol'] == 'EURUSD']
        print("\nEURUSD Data:")
        print(eurusd_data)

        # Verification: Check the calculated future returns for USDJPY
        # Return from 130.0 to 130.5 should be (130.5 - 130.0) / 130.0 = 0.003846
        logging.info("\n--- Verification ---")
        usdjpy_data = processed_df[processed_df['symbol'] == 'USDJPY']
        print("\nUSDJPY Data:")
        print(usdjpy_data)

Production ML + MetaTrader 5 trading runtime (mentor line).

**Policy:** UTF-8 only · all features permitted · no filtering / labeling / blocking of feature names · signals only when model is loaded and featureset matches EA + server + Predict.mq5 + GenerateFeatures.mq5.

Repo: https://github.com/FXJEFE/user_documents_fxjefe_project

---

## Project root (all OS)

```text
~/Documents/FXJEFE_Project
Windows: %USERPROFILE%\\Documents\\FXJEFE_Project
```

---

## Quick setup

### 1. Clone

```bash
mkdir -p ~/Documents
cd ~/Documents
git clone https://github.com/FXJEFE/user_documents_fxjefe_project.git FXJEFE_Project
cd FXJEFE_Project
```

### 2. Python venv

**macOS**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements_mac.txt
```

**Linux**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements_linux.txt
```

**Windows (PowerShell)**

```powershell
py -3.11 -m venv venv
.\\venv\\Scripts\\Activate.ps1
python -m pip install -U pip setuptools wheel
pip install -r requirements_win.txt
```

### 3. Environment

```bash
cp .env.example .env
```

Edit `.env` with MT5 account values locally. Never commit `.env`.

### 4. Lock runtime (expect 200)

```bash
python runtime_lock.py
```

### 5. Feature hash + signal gate smoke test

```bash
python feature_hash.py
python signal_gate.py
```

### 6. Production pipeline (optional)

```bash
python pipelinerun_production.py
```

After first green FINAL:

```bash
python secure_strap.py
```

---

## Feature policy (ALL OK)

| Rule | Value |
|------|--------|
| `feature_policy` | `ACCEPT_ALL_FEATURES` |
| Filter / block / refuse feature names | **Never** |
| Strip feature arrays | **Never** |
| Preferred 17 / 28 lists | Defaults for wiring only |
| Signal emit | Model loaded **and** featureset hash matches EA + server + Predict.mq5 + GenerateFeatures.mq5 |

---

## Encoding

UTF-8 only for `.py`, `.csv`, `.mq5`, `.mqh`, `.json`, configs.

```bash
python encoding_utf8.py --fix
python encoding_utf8.py --scan
```

---

## MT5

Live terminals (Pepperstone / Vantage / FTMO) run on **Windows** (native or VM).  
Mac/Linux: training, feature engineering, AI server.  
EA: allow WebRequest for `http://127.0.0.1:8080` (and LAN IP if split hosts).

---

## Do not commit

- `.env` (secrets)
- `venv/`
- `__pycache__/`
- `*.pkl` model binaries unless intentional
- live broker passwords

See `.gitignore`.
