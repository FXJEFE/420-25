# -*- coding: utf-8 -*-
"""Generate signals with XGBoost model using config models/data paths."""
from fxjefe_paths import load_config, setup_logging, features_path, models_file, write_feature_csv
import logging
import os
import numpy as np
import pandas as pd
import xgboost as xgb

config = load_config()
setup_logging(config, "generate_signals_with_xgboost")

FEATURE_COLS_6 = ["price", "atr", "ema_diff", "rsi", "garch_vol", "macd_diff"]
FEATURE_COLS_CFG = list(config.get("features") or FEATURE_COLS_6)


def resolve_model() -> str:
    for name in (
        "xgboost_model.json",
        "xgboost_model (1).json",
        "xgboost_best_sharpe.json",
        "ensamble_model.pkl.json",
    ):
        p = models_file(config, name)
        if os.path.isfile(p) and os.path.getsize(p) > 64:
            with open(p, "rb") as f:
                if f.read(1) == b"{":
                    return p
    # any binary xgb in models
    mdir = config["models_path"]
    if os.path.isdir(mdir):
        for fn in sorted(os.listdir(mdir)):
            if fn.endswith("_xgb.json") or fn.endswith("xgboost_model.json"):
                p = os.path.join(mdir, fn)
                try:
                    with open(p, "rb") as f:
                        if f.read(1) == b"{":
                            return p
                except OSError:
                    pass
    raise FileNotFoundError("No loadable xgboost_model.json under models_path")


def main() -> None:
    model_path = resolve_model()
    logging.info("Using model: %s", model_path)
    model = xgb.Booster()
    model.load_model(model_path)
    n_model = int(model.num_features())

    src = features_path(config, "FXJEFE_Features_fixed.csv")
    if not os.path.isfile(src):
        src = features_path(config)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"No features CSV: {src}")

    df = pd.read_csv(src, encoding="utf-8", low_memory=False)
    df.columns = [str(c).lower().strip() for c in df.columns]

    # pick feature list matching model n_features
    if n_model == 6:
        cols = FEATURE_COLS_6
    elif n_model <= len(FEATURE_COLS_CFG):
        cols = FEATURE_COLS_CFG[:n_model]
    else:
        # use whatever numeric cols available, pad
        cols = [c for c in FEATURE_COLS_CFG if c in df.columns][:n_model]
        while len(cols) < n_model:
            cols.append(f"pad_{len(cols)}")
            df[cols[-1]] = 0.0

    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    X = df[cols].astype(float)
    dmat = xgb.DMatrix(X, feature_names=cols)
    pred = np.asarray(model.predict(dmat))
    logging.info("predictions shape=%s", pred.shape)

    if pred.ndim == 2 and pred.shape[1] > 1:
        # multiclass probs
        cls = np.argmax(pred, axis=1)
        # map 0,1,2 → -1,0,1 if 3 classes else 0/1
        if pred.shape[1] == 3:
            signal = cls - 1
            conf = pred.max(axis=1)
        else:
            signal = (cls > 0).astype(int)
            conf = pred.max(axis=1)
    else:
        p = pred.reshape(-1)
        thr = float(config.get("min_confidence_threshold", 0.65))
        signal = np.where(p >= thr, 1, np.where(p <= 1 - thr, -1, 0))
        conf = np.where(p >= 0.5, p, 1 - p)

    df["signal"] = signal
    df["confidence"] = conf
    # OG: always write signals CSV to ALL destinations (never skip)
    written = write_feature_csv(df, config, "FXJEFE_Features_with_signals.csv")
    logging.info("Wrote signals → %s locations (%s rows)", len(written), len(df))


if __name__ == "__main__":
    main()
