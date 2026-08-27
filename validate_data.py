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
"""
validate_data.py
Validates FXJEFE_Features.csv after mt5_data_sync:
  - checks all 27 model feature columns are present
  - reports NaN counts per column
  - reports row count, symbol list
  - raises SystemExit(1) on critical failure (file missing, no rows)
"""
import os
import sys
import json
import logging
import pandas as pd

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

os.makedirs(config['log_path'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config['log_path'], 'validate_data.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

MODEL_FEATURES = [c for c in config['features'] if c != 'price']  # 27 indicators


def main():
    csv_path = os.path.join(config['data_path'], 'FXJEFE_Features.csv')

    if not os.path.exists(csv_path):
        logging.error(f"Data file not found: {csv_path}")
        logging.error("Run mt5_data_sync.py (or GenerateFeatures.mq5) first.")
        sys.exit(1)

    df = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)
    logging.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {csv_path}")

    if len(df) == 0:
        logging.error("CSV file is empty.")
        sys.exit(1)

    if len(df) < 500:
        logging.warning(f"Only {len(df)} rows — re-run GenerateFeatures.mq5 with more History_Bars.")

    # Check required columns
    required = ['time', 'symbol', 'price'] + MODEL_FEATURES
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        logging.warning(f"Missing columns (will be filled by generate_labels.py): {missing_cols}")
    else:
        logging.info("All required columns present.")

    # NaN report for numeric features
    numeric_cols = [c for c in config['features'] if c in df.columns]
    if numeric_cols:
        null_counts = df[numeric_cols].apply(pd.to_numeric, errors='coerce').isnull().sum()
        has_nulls = null_counts[null_counts > 0]
        if not has_nulls.empty:
            logging.warning(f"NaN counts per feature:\n{has_nulls.to_string()}")
        else:
            logging.info("No NaN values in feature columns.")

    # Symbol report
    if 'symbol' in df.columns:
        symbols = df['symbol'].unique().tolist()
        logging.info(f"Symbols found: {symbols}")
        counts = df['symbol'].value_counts().to_dict()
        logging.info(f"Rows per symbol: {counts}")

    logging.info("Data validation complete.")


if __name__ == '__main__':
    main()
