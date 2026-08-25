#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train GridSearchCV + neural net + XGBoost + LightGBM + gradient boosters
+ LSTM + LTDM + Hidden Markov on live+historic CSV.

Writes ONLY under models/og333_runs/ (never overwrites OG models).
Requires garch_vol in the feature matrix.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import accuracy_score
import joblib

from fxjefe_paths import load_config, setup_logging, features_path, safe_model_out_path, assert_not_og_model

warnings.filterwarnings("ignore")

FEATURE_CORE = [
    "price", "atr", "ema_diff", "rsi", "garch_vol", "macd_diff",
    "vwap", "price_vwap_diff", "bb_position", "roc", "stochastic",
    "cci", "williams", "momentum", "realized_vol", "adx", "spread", "sentiment",
]


def _label(df: pd.DataFrame) -> pd.Series:
    if "label" in df.columns:
        y = pd.to_numeric(df["label"], errors="coerce")
    elif "signal" in df.columns:
        raw = df["signal"]
        if raw.dtype == object:
            m = raw.astype(str).str.lower().map({"buy": 1, "sell": -1, "hold": 0, "1": 1, "-1": -1, "0": 0})
            y = pd.to_numeric(m, errors="coerce")
        else:
            y = pd.to_numeric(raw, errors="coerce")
    else:
        px = pd.to_numeric(df["price"], errors="coerce")
        fut = px.shift(-1)
        y = pd.Series(0, index=df.index, dtype=float)
        y = y.mask(fut > px, 1).mask(fut < px, -1)
    return y.fillna(0).astype(int)


def _xy(df: pd.DataFrame, feats: list) -> tuple[pd.DataFrame, pd.Series, list]:
    use = [c for c in feats if c in df.columns]
    if "garch_vol" not in use and "garch_vol" in df.columns:
        use = ["garch_vol"] + use
    if "garch_vol" not in df.columns:
        raise SystemExit("garch_vol missing from CSV — cannot train zoo")
    if "garch_vol" not in use:
        use.insert(0, "garch_vol")
    X = df[use].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = _label(df)
    mask = y.notna()
    return X.loc[mask], y.loc[mask], use


def _save(cfg, name: str, obj, meta: dict) -> str:
    path = safe_model_out_path(cfg, name)
    assert_not_og_model(path)
    joblib.dump(obj, path)
    meta_path = path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path


def _gs_fit(est, grid, Xtr, ytr, name: str):
    gs = GridSearchCV(est, grid, cv=2, scoring="accuracy", n_jobs=1, refit=True)
    gs.fit(Xtr, ytr)
    print(f"  GridSearchCV {name} best={gs.best_params_} cv={gs.best_score_:.4f}", flush=True)
    return gs.best_estimator_, float(gs.best_score_)


def _train_lstm(X, y, cfg, feats):
    """Small torch LSTM; 2 epochs. Fallback skip if no torch."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  LSTM skip: torch not installed", flush=True)
        return None
    seq = 8
    arr = X.values.astype(np.float32)
    lab = y.values.astype(np.int64)
    # map -1,0,1 -> 0,1,2
    lab_m = np.where(lab < 0, 0, np.where(lab > 0, 2, 1))
    xs, ys = [], []
    for i in range(seq, len(arr)):
        xs.append(arr[i - seq : i])
        ys.append(lab_m[i])
    if len(xs) < 50:
        print("  LSTM skip: not enough sequence rows", flush=True)
        return None
    xt = torch.tensor(np.stack(xs))
    yt = torch.tensor(np.array(ys), dtype=torch.long)

    class SmallLSTM(nn.Module):
        def __init__(self, n_in, hid=32):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hid, num_layers=1, batch_first=True)
            self.fc = nn.Linear(hid, 3)

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.fc(o[:, -1, :])

    model = SmallLSTM(arr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(2):
        opt.zero_grad()
        logits = model(xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(xt).argmax(1).numpy()
    acc = float((pred == yt.numpy()).mean())
    blob = {"state": model.state_dict(), "n_in": arr.shape[1], "seq": seq, "features": feats}
    path = _save(cfg, "zoo_lstm.pkl", blob, {"type": "lstm", "acc": acc, "features": feats, "garch_vol": True})
    print(f"  LSTM acc={acc:.4f} -> {path}", flush=True)
    return path


def _train_hmm(X, y, cfg, feats):
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("  HMM skip: hmmlearn not installed — fitting sklearn GMM-like fallback", flush=True)
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(n_components=3, covariance_type="diag", random_state=42)
        gm.fit(X.values)
        path = _save(cfg, "zoo_hmm.pkl", gm, {"type": "gmm_hmm_fallback", "features": feats, "garch_vol": True})
        print(f"  GMM-HMM fallback -> {path}", flush=True)
        return path
    hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=20, random_state=42)
    hmm.fit(X.values)
    path = _save(cfg, "zoo_hmm.pkl", hmm, {"type": "hmm", "n_states": 3, "features": feats, "garch_vol": True})
    print(f"  HMM 3-state fitted -> {path}", flush=True)
    return path


def main() -> None:
    cfg = load_config()
    setup_logging(cfg, "train_model_zoo")
    src = features_path(cfg, "training_data.csv")
    if not os.path.isfile(src):
        src = features_path(cfg, "FXJEFE_Features_with_labels.csv")
    if not os.path.isfile(src):
        src = features_path(cfg)
    print(f"train_model_zoo source={src}", flush=True)
    df = pd.read_csv(src, encoding="utf-8", low_memory=False)
    feats = list(cfg.get("features") or FEATURE_CORE)
    X, y, use = _xy(df, feats)
    if "garch_vol" not in use:
        raise SystemExit("garch_vol not in training matrix")
    # cap for speed
    if len(X) > 8000:
        X = X.iloc[-8000:]
        y = y.iloc[-8000:]
    print(f"rows={len(X)} feats={len(use)} garch_vol_mean={float(X['garch_vol'].mean()):.6f}", flush=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)

    trained = []

    # XGBoost + GridSearchCV
    try:
        from xgboost import XGBClassifier
        ytr_m = ytr.map({-1: 0, 0: 1, 1: 2}).fillna(1).astype(int)
        yte_m = yte.map({-1: 0, 0: 1, 1: 2}).fillna(1).astype(int)
        est, cv = _gs_fit(
            XGBClassifier(n_estimators=40, max_depth=3, eval_metric="mlogloss", verbosity=0, n_jobs=1),
            {"learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
            Xtr, ytr_m, "xgboost",
        )
        acc = accuracy_score(yte_m, est.predict(Xte))
        p = _save(cfg, "zoo_xgboost.pkl", est, {"type": "xgb", "acc": acc, "cv": cv, "features": use, "garch_vol": True})
        print(f"  XGBoost test_acc={acc:.4f} -> {p}", flush=True)
        trained.append("xgboost")
    except Exception as e:
        print(f"  XGBoost fail: {e}", flush=True)

    # LightGBM + GridSearchCV
    try:
        from lightgbm import LGBMClassifier
        est, cv = _gs_fit(
            LGBMClassifier(n_estimators=40, verbose=-1, random_state=42),
            {"num_leaves": [15, 31], "learning_rate": [0.05, 0.1]},
            Xtr, ytr, "lightgbm",
        )
        acc = accuracy_score(yte, est.predict(Xte))
        p = _save(cfg, "zoo_lightgbm.pkl", est, {"type": "lgb", "acc": acc, "cv": cv, "features": use, "garch_vol": True})
        print(f"  LightGBM test_acc={acc:.4f} -> {p}", flush=True)
        trained.append("lightgbm")
    except Exception as e:
        print(f"  LightGBM fail: {e}", flush=True)

    # Gradient boosters
    try:
        est, cv = _gs_fit(
            HistGradientBoostingClassifier(max_depth=3, random_state=42),
            {"learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
            Xtr, ytr, "hist_gb",
        )
        acc = accuracy_score(yte, est.predict(Xte))
        p = _save(cfg, "zoo_histgb.pkl", est, {"type": "histgb", "acc": acc, "cv": cv, "features": use, "garch_vol": True})
        print(f"  HistGB test_acc={acc:.4f} -> {p}", flush=True)
        trained.append("histgb")
    except Exception as e:
        print(f"  HistGB fail: {e}", flush=True)

    try:
        gb = GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=42)
        gb.fit(Xtr, ytr)
        acc = accuracy_score(yte, gb.predict(Xte))
        p = _save(cfg, "zoo_gb.pkl", gb, {"type": "gb", "acc": acc, "features": use, "garch_vol": True})
        print(f"  GradientBoosting test_acc={acc:.4f} -> {p}", flush=True)
        trained.append("gb")
    except Exception as e:
        print(f"  GB fail: {e}", flush=True)

    # Neural network (sklearn MLP; TF used if present for extra net)
    try:
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=80, random_state=42)
        mlp.fit(Xtr, ytr)
        acc = accuracy_score(yte, mlp.predict(Xte))
        p = _save(cfg, "zoo_mlp.pkl", mlp, {"type": "mlp", "acc": acc, "features": use, "garch_vol": True})
        print(f"  MLP neural net test_acc={acc:.4f} -> {p}", flush=True)
        trained.append("mlp")
    except Exception as e:
        print(f"  MLP fail: {e}", flush=True)

    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
        ytr_m = ytr.map({-1: 0, 0: 1, 1: 2}).fillna(1).astype(int)
        yte_m = yte.map({-1: 0, 0: 1, 1: 2}).fillna(1).astype(int)
        model = keras.Sequential([
            keras.layers.Input(shape=(Xtr.shape[1],)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(3, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(Xtr.values, ytr_m.values, epochs=3, batch_size=256, verbose=0)
        acc = float(model.evaluate(Xte.values, yte_m.values, verbose=0)[1])
        path = safe_model_out_path(cfg, "zoo_tf_mlp.keras")
        assert_not_og_model(path)
        model.save(path)
        print(f"  TensorFlow net test_acc={acc:.4f} -> {path}", flush=True)
        trained.append("tf_mlp")
    except Exception as e:
        print(f"  TensorFlow skip/fail: {e}", flush=True)

    # LTDM — longer-horizon booster on lagged garch/price
    try:
        Xl = X.copy()
        for c in ("price", "garch_vol", "rsi", "adx"):
            if c in Xl.columns:
                Xl[f"{c}_lt"] = Xl[c].rolling(20, min_periods=1).mean()
        ltdm = GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=42)
        ltdm.fit(Xl.iloc[: len(ytr)], ytr)
        acc = accuracy_score(yte, ltdm.predict(Xl.iloc[len(ytr) :]))
        p = _save(cfg, "zoo_ltdm.pkl", ltdm, {"type": "ltdm", "acc": acc, "features": list(Xl.columns), "garch_vol": True})
        print(f"  LTDM test_acc={acc:.4f} -> {p}", flush=True)
        trained.append("ltdm")
    except Exception as e:
        print(f"  LTDM fail: {e}", flush=True)

    _train_lstm(X, y, cfg, use)
    trained.append("lstm_attempted")
    _train_hmm(X, y, cfg, use)
    trained.append("hmm_attempted")

    print(f"OK train_model_zoo trained={trained} (OG models untouched)", flush=True)
    if not trained:
        raise SystemExit("no zoo models trained")


if __name__ == "__main__":
    main()
