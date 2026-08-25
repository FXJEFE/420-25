#!/usr/bin/env python3
"""
FXJEFE Model Tester + SHAP Explanations
- Loads XGBoost model
- Uses recent MT5-synced data
- Computes SHAP values
- Shows global importance + individual force plots
- Saves plots and SHAP CSV
"""

import json
import os
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

# ────────────────────────────────────────────────
# Project imports
# ────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from path_resolver import get_paths, get_config

paths = get_paths()
config = get_config() or {}

# ────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────
log_file = paths.get_log_path('shap_tester.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("SHAP tester started")

# ────────────────────────────────────────────────
# Configurable paths & settings
# ────────────────────────────────────────────────
DEFAULT_MODEL   = paths.get_model_path('xgboost_model.json')
DEFAULT_DATA    = paths.realtime_data_path / 'FXJEFE_Features.csv'
DEFAULT_ROWS    = 300           # last N rows to explain
DEFAULT_OUTPUT  = paths.logs_path / 'shap_results'

# ────────────────────────────────────────────────
# Load model & data
# ────────────────────────────────────────────────
def load_model(model_path: Path):
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    try:
        model = xgb.Booster()
        model.load_model(str(model_path))
        logger.info(f"XGBoost model loaded: {model_path.name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

def load_data(csv_path: Path, n_rows: int):
    if not csv_path.exists():
        logger.error(f"Data file not found: {csv_path}")
        sys.exit(1)
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.tail(n_rows).copy()
        logger.info(f"Loaded {len(df)} recent rows from {csv_path.name}")
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        sys.exit(1)

# ────────────────────────────────────────────────
# SHAP analysis
# ────────────────────────────────────────────────
def run_shap_analysis(model, df: pd.DataFrame, feature_cols: list = None):
    if feature_cols is None:
        try:
            feature_cols = model.feature_names
        except:
            feature_cols = [c for c in df.columns if c not in ['time', 'symbol', 'signal']]
            logger.warning("No feature_names in model → using all numeric columns except time/symbol/signal")

    # Prepare data
    X = df[feature_cols].fillna(0).astype(float)
    dmat = xgb.DMatrix(X, feature_names=feature_cols)

    # SHAP explainer (TreeExplainer is fast & exact for XGBoost)
    logger.info("Creating TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    logger.info("Computing SHAP values...")
    shap_values = explainer.shap_values(dmat)

    # ────────────────────────────────────────────────
    # Global importance summary plot
    # ────────────────────────────────────────────────
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("Global Feature Importance (SHAP values)")
    summary_path = DEFAULT_OUTPUT / 'shap_summary.png'
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved global SHAP summary: {summary_path}")

    # ────────────────────────────────────────────────
    # Top N most important features
    # ────────────────────────────────────────────────
    shap_importance = np.abs(shap_values).mean(0)
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': shap_importance
    }).sort_values('mean_abs_shap', ascending=False)

    print("\nTop 15 features by mean |SHAP| value:")
    print(imp_df.head(15).to_string(index=False))

    # ────────────────────────────────────────────────
    # Individual force plots (last 5 predictions)
    # ────────────────────────────────────────────────
    print("\nGenerating force plots for last 5 predictions...")
    for i in range(-5, 0):
        idx = len(df) + i
        row = df.iloc[idx]
        pred_shap = shap_values[idx] if len(shap_values.shape) == 2 else shap_values[idx]

        force_path = DEFAULT_OUTPUT / f'force_plot_{idx}.png'
        plt.figure(figsize=(14, 4))
        shap.force_plot(
            explainer.expected_value,
            pred_shap,
            X.iloc[idx],
            matplotlib=True,
            show=False
        )
        plt.title(f"Prediction {idx} | {row.get('time','')} | close={row.get('close',0):.5f}")
        plt.tight_layout()
        plt.savefig(force_path, dpi=120, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved force plot: {force_path}")

    # ────────────────────────────────────────────────
    # Save SHAP values + features for later analysis
    # ────────────────────────────────────────────────
    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in feature_cols])
    output_df = pd.concat([df.reset_index(drop=True), shap_df], axis=1)
    csv_out = DEFAULT_OUTPUT / f'shap_values_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
    output_df.to_csv(csv_out, index=False, encoding='utf-8')
    logger.info(f"Saved full SHAP + features CSV: {csv_out}")

    return shap_values, feature_cols

# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='FXJEFE SHAP Model Tester')
    parser.add_argument('--model', type=str, default=str(DEFAULT_MODEL),
                        help='Path to XGBoost JSON model')
    parser.add_argument('--data', type=str, default=str(DEFAULT_DATA),
                        help='Path to feature CSV')
    parser.add_argument('--rows', type=int, default=DEFAULT_ROWS,
                        help='Number of recent rows to analyze')
    parser.add_argument('--features', nargs='*',
                        help='Specific feature columns to use (overrides model)')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT),
                        help='Where to save plots and CSV')
    args = parser.parse_args()

    # Prepare output folder
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    global DEFAULT_OUTPUT
    DEFAULT_OUTPUT = out_dir

    logger.info(f"Output directory: {out_dir}")

    # Load model & data
    model = load_model(Path(args.model))
    df    = load_data(Path(args.data), args.rows)

    if df is None:
        logger.error("No valid data — exiting")
        return

    feature_cols = args.features if args.features else None

    # Run SHAP
    shap_values, used_features = run_shap_analysis(model, df, feature_cols)

    print("\n" + "═"*70)
    print("SHAP analysis complete.")
    print(f"  Model:     {args.model}")
    print(f"  Data rows: {len(df)}")
    print(f"  Features:  {len(used_features)}")
    print(f"  Plots & CSV saved to: {out_dir}")
    print("═"*70)

if __name__ == '__main__':
    main()
