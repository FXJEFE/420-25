# -*- coding: utf-8 -*-
"""
One-shot XGBoost signal smoke test (pipeline-safe — does NOT hang as a server).
For the live API use ai_server.py instead.
"""
from fxjefe_paths import load_config, setup_logging, features_path, models_file
import logging
import os
import numpy as np
import pandas as pd
import xgboost as xgb

config = load_config()
setup_logging(config, "fxjefe_xgboost_api")

FEATURE_COLS = ["price", "atr", "ema_diff", "rsi", "garch_vol", "macd_diff"]


def main() -> None:
    # resolve model
    model_path = None
    for name in ("xgboost_model.json", "xgboost_model (1).json", "xgboost_best_sharpe.json"):
        p = models_file(config, name)
        if os.path.isfile(p) and os.path.getsize(p) > 64:
            with open(p, "rb") as f:
                if f.read(1) == b"{":
                    model_path = p
                    break
    if not model_path:
        # try any *_xgb.json
        mdir = config["models_path"]
        for fn in sorted(os.listdir(mdir)) if os.path.isdir(mdir) else []:
            if fn.endswith("_xgb.json"):
                p = os.path.join(mdir, fn)
                with open(p, "rb") as f:
                    if f.read(1) == b"{":
                        model_path = p
                        break
    if not model_path:
        raise FileNotFoundError("No XGBoost model found under models_path")

    model = xgb.Booster()
    model.load_model(model_path)
    n = int(model.num_features())
    logging.info("Loaded %s (n_features=%s)", model_path, n)

    src = features_path(config, "FXJEFE_Features_fixed.csv")
    if not os.path.isfile(src):
        src = features_path(config)
    if not os.path.isfile(src):
        raise FileNotFoundError(src)

    df = pd.read_csv(src, encoding="utf-8", low_memory=False)
    df.columns = [str(c).lower().strip() for c in df.columns]
    cols = list(FEATURE_COLS)
    if n != len(cols):
        cfg_feats = list(config.get("features") or cols)
        cols = (cfg_feats + [f"pad_{i}" for i in range(50)])[:n]
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # smoke predict last row
    row = df[cols].astype(float).tail(1)
    dmat = xgb.DMatrix(row, feature_names=cols)
    pred = float(np.asarray(model.predict(dmat)).reshape(-1)[0])
    thr = float(config.get("min_confidence_threshold", 0.65))
    if pred >= thr:
        signal = "buy"
    elif pred <= 1 - thr:
        signal = "sell"
    else:
        signal = "hold"
    logging.info("smoke predict last row: prob=%.4f signal=%s", pred, signal)
    print({"signal": signal, "probability": pred, "model": os.path.basename(model_path)})
    # health ping optional
    try:
        import requests
        url = (config.get("ai_server_url") or "http://127.0.0.1:8080").rstrip("/") + "/health"
        r = requests.get(url, timeout=2)
        logging.info("ai_server health: %s %s", r.status_code, r.text[:200])
    except Exception as e:
        logging.info("ai_server not reachable (ok for pipeline): %s", e)


if __name__ == "__main__":
    main()
