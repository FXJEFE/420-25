#!/usr/bin/env python3
"""
FXJEFE Local Model Tester
- Loads trained models from models/ folder
- Reads recent data from MT5-synced CSVs
- Shows expected features + importance (when available)
- Runs batch predictions and shows signal statistics
"""

import json
import os
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

# ────────────────────────────────────────────────
# Project imports (dynamic paths)
# ────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from path_resolver import get_paths, get_config

paths = get_paths()
config = get_config() or {}

# ────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────
log_file = paths.get_log_path('model_tester.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────
# Model loader helpers
# ────────────────────────────────────────────────
def load_xgboost_model(path: Path):
    """Load XGBoost model from JSON file"""
    try:
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(str(path))
        logger.info(f"Loaded XGBoost model: {path.name}")
        return {'type': 'xgboost', 'model': model, 'path': path}
    except Exception as e:
        logger.error(f"Failed to load XGBoost {path.name}: {e}")
        return None

def load_generic_model(path: Path):
    """Attempt to load pickle / joblib / torch / other model"""
    ext = path.suffix.lower()
    name = path.stem

    try:
        if ext in ['.pkl', '.joblib']:
            import joblib
            model = joblib.load(str(path))
            logger.info(f"Loaded pickled model: {name}")
            return {'type': 'sklearn-like', 'model': model, 'path': path}

        elif ext == '.pth' or ext == '.pt':
            import torch
            model = torch.load(str(path), map_location='cpu')
            logger.info(f"Loaded PyTorch model: {name}")
            return {'type': 'pytorch', 'model': model, 'path': path}

        else:
            logger.warning(f"Unknown model format: {path}")
            return None

    except Exception as e:
        logger.error(f"Failed to load {path.name}: {e}")
        return None

def discover_models(models_dir: Path):
    """Find all model files in models/ folder"""
    models = []
    for ext in ['*.json', '*.pkl', '*.joblib', '*.pth', '*.pt']:
        for p in models_dir.glob(ext):
            if 'xgboost' in p.stem.lower():
                m = load_xgboost_model(p)
            else:
                m = load_generic_model(p)
            if m:
                models.append(m)
    return models

# ────────────────────────────────────────────────
# Data loading & feature inspection
# ────────────────────────────────────────────────
def load_recent_data(csv_path: Path, rows: int = 500):
    """Load most recent rows from CSV (MT5 synced file)"""
    if not csv_path.exists():
        logger.error(f"Data file not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df = df.tail(rows).copy()
        df.columns = [c.strip().lower() for c in df.columns]
        logger.info(f"Loaded {len(df)} recent rows from {csv_path.name}")
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV {csv_path}: {e}")
        return None

def show_expected_features(model_info: Dict):
    """Print what features the model expects / uses"""
    mtype = model_info['type']
    model = model_info['model']
    name  = model_info['path'].stem

    print(f"\n{'─'*70}")
    print(f"Model: {name} ({mtype})")
    print(f"{'─'*70}")

    if mtype == 'xgboost':
        try:
            features = model.feature_names
            print(f"XGBoost expects {len(features)} features:")
            print(', '.join(features))
            # Importance if available
            imp = model.get_score(importance_type='gain')
            if imp:
                print("\nTop 10 features by gain:")
                sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
                for f, v in sorted_imp:
                    print(f"  {f:20} {v:>12.2f}")
        except:
            print("Could not read feature names from XGBoost model")

    elif mtype in ['sklearn-like', 'pytorch']:
        print("Model type does not expose feature names directly.")
        print("Check training script or model card for expected columns.")

    else:
        print("Unknown model type — no feature info available.")

# ────────────────────────────────────────────────
# Prediction & reporting
# ────────────────────────────────────────────────
def run_predictions(df: pd.DataFrame, model_info: Dict, feature_cols: list = None):
    """Run batch prediction and show stats"""
    if df is None or df.empty:
        return

    name = model_info['path'].stem
    mtype = model_info['type']
    model = model_info['model']

    print(f"\nRunning predictions with {name} ...")

    try:
        if mtype == 'xgboost':
            if feature_cols is None:
                feature_cols = model.feature_names
            X = df[feature_cols].fillna(0)
            dmat = xgb.DMatrix(X, feature_names=feature_cols)
            proba = model.predict(dmat)

            # Assuming common output shapes
            if len(proba.shape) == 1:  # binary
                preds = np.where(proba > 0.5, 1, 0)
                probs = proba
            elif proba.shape[1] == 3:  # ternary [-1,0,1]
                preds = np.argmax(proba, axis=1) - 1
                probs = np.max(proba, axis=1)
            else:
                preds = np.argmax(proba, axis=1)
                probs = np.max(proba, axis=1)

        else:
            # Generic fallback — assumes model.predict(X) returns class or prob
            X = df[feature_cols or df.columns].fillna(0)
            raw = model.predict(X)
            if len(raw.shape) == 1:
                preds = np.round(raw).astype(int)
                probs = np.abs(raw - preds)  # crude confidence proxy
            else:
                preds = np.argmax(raw, axis=1)
                probs = np.max(raw, axis=1)

        # Map to signals
        signal_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
        signals = [signal_map.get(p, 'HOLD') for p in preds]

        # Summary
        counts = pd.Series(signals).value_counts()
        print("\nPrediction distribution:")
        for sig, cnt in counts.items():
            print(f"  {sig:6} : {cnt:5d} ({cnt/len(signals):.1%})")

        avg_conf = np.mean(probs) if 'probs' in locals() else 0.5
        print(f"Average confidence: {avg_conf:.1%}")

        # Last 20 predictions
        print("\nLast 20 predictions:")
        for i, (idx, row) in enumerate(df.tail(20).iterrows()):
            print(f"{row.get('time','')[:19]} | {signals[-20+i]:4} ({probs[-20+i]:.1%})")

    except Exception as e:
        logger.error(f"Prediction failed for {name}: {e}")

# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='FXJEFE Local Model Tester')
    parser.add_argument('--data', type=str, default='FXJEFE_Features.csv',
                        help='CSV file name in MT5 Files or data folder')
    parser.add_argument('--rows', type=int, default=500,
                        help='How many recent rows to test')
    parser.add_argument('--features', nargs='*',
                        help='Override feature columns (space separated)')
    args = parser.parse_args()

    df = load_recent_data(paths.realtime_data_path / args.data, args.rows)
    if df is None:
        df = load_recent_data(paths.data_path / args.data, args.rows)
    if df is None:
        logger.error("No data found — exiting")
        return

    models = discover_models(paths.models_path)
    if not models:
        logger.warning("No models found in models/ folder")
        return

    print(f"\nFound {len(models)} model(s):")
    for m in models:
        print(f"  - {m['path'].name}")

    feature_override = args.features if args.features else None

    for model_info in models:
        show_expected_features(model_info)
        run_predictions(df, model_info, feature_override)

    print("\nTester finished.")

if __name__ == '__main__':
    main()
