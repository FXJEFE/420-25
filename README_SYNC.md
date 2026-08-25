# FXJEFE sync (Aug 2026)

## Venv
Always name: `fxjefe` (not `venv`)

## Core
- ai_server_golden.py — /health /predict, length-safe vectors
- integrity.py — SHA256 + JSON + package signatures
- check_stack.py — required OK / optional SKIP
- runtime_lock.py — status 200

## Features / vol
- realized_vol_features.py
- garch_async.py / garch_multistep.py
- feature_importance_nan.py
- optimize_talib_optuna.py — reads historical_csv from config.json

## MT5 (Windows)
- margin_precheck.py
- mt5_latency_bench.py
- trade_policy_config.py

## Config
- historical_csv: data/raw_ohlcv.csv
- talib_defaults: rsi/atr/macd
- feature_policy: ACCEPT_ALL_FEATURES
- min_confidence_threshold: 0.77

## Mac
- brew install libomp ta-lib
- pandas-ta needs Python>=3.12 — use TA-Lib on 3.11
