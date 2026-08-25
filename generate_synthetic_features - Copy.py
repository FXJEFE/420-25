# -*- coding: utf-8 -*-
"""generate_synthetic_features.py — synthetic OHLCV+features generator for testing."""
import json
import os
import sys
import logging
import pandas as pd
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

log_dir = config.get('log_path')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'generate_synthetic_features.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def main():
    out_path = os.path.join(config['data_output_path'], 'FXJEFE_Features.csv')
    if os.path.exists(out_path):
        logging.info(f"{out_path} already exists; not overwriting (use --force to regen)")
        return

    np.random.seed(42)
    n = 1000
    data = {
        "time":   pd.date_range(start='2025-01-01', periods=n, freq='15min'),
        "symbol": ["EURUSD"] * n,
        "open":   np.random.uniform(1.0, 2.0, n),
        "high":   np.random.uniform(1.0, 2.0, n),
        "low":    np.random.uniform(1.0, 2.0, n),
        "close":  np.random.uniform(1.0, 2.0, n),
        "volume": np.random.randint(100, 1000, n),
        "price":  np.random.uniform(1.0, 2.0, n),
        "atr":    np.random.uniform(0.001, 0.1, n),
        "ema_diff": np.random.uniform(-0.05, 0.05, n),
        "rsi":      np.random.uniform(20, 80, n),
        "garch_vol": np.random.uniform(0.001, 0.05, n),
        "macd_diff": np.random.uniform(-0.01, 0.01, n),
        "vwap":      np.random.uniform(1.0, 2.0, n),
        "price_vwap_diff": np.random.uniform(-0.05, 0.05, n),
        "bb_position":     np.random.uniform(0.0, 1.0, n),
        "roc":             np.random.uniform(-1.0, 1.0, n),
        "stochastic":      np.random.uniform(0.0, 100.0, n),
        "cci":             np.random.uniform(-200.0, 200.0, n),
        "williams":        np.random.uniform(-100.0, 0.0, n),
        "momentum":        np.random.uniform(-0.1, 0.1, n),
        "realized_vol":    np.random.uniform(0.001, 0.05, n),
        "chaikin_vol":     np.random.uniform(-50.0, 50.0, n),
        "adx":             np.random.uniform(0.0, 100.0, n),
        "rvi":             np.random.uniform(-1.0, 1.0, n),
        "obv":             np.random.uniform(-1e6, 1e6, n),
        "volume_delta":    np.random.uniform(-1000, 1000, n),
        "ad_line":         np.random.uniform(-1e6, 1e6, n),
        "vol_osc":         np.random.uniform(-100.0, 100.0, n),
        "supertrend":      np.random.uniform(1.0, 2.0, n),
        "hma":             np.random.uniform(1.0, 2.0, n),
        "ichimoku_tenkan": np.random.uniform(1.0, 2.0, n),
        "sar":             np.random.uniform(1.0, 2.0, n),
        "dpo":             np.random.uniform(-0.05, 0.05, n),
        "spread":          np.random.uniform(0.0001, 0.001, n),
        "sentiment":       np.random.uniform(-1, 1, n),
    }
    df = pd.DataFrame(data)
    df['signal'] = 0
    df.loc[df['price'].diff().fillna(0) > 0.0001, 'signal'] = 1
    df.loc[df['price'].diff().fillna(0) < -0.0001, 'signal'] = -1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding='utf-8')
    logging.info(f"Synthetic features saved -> {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"generate_synthetic_features failed: {e}", exc_info=True)
        sys.exit(1)
