#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optimize key TA-Lib periods using historical CSV from config.json."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
import talib
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

def project_root() -> Path:
    return Path.home() / "Documents" / "FXJEFE_Project"

def load_config(root: Path) -> dict:
    return json.loads((root / "config.json").read_text(encoding="utf-8"))

def resolve_csv(root: Path, cfg: dict) -> Path:
    raw = cfg.get("historical_csv") or cfg.get("data_csv") or "data/raw_ohlcv.csv"
    path = Path(str(raw))
    for c in [path if path.is_absolute() else None, root / path, root / "data" / path.name]:
        if c is not None and c.is_file():
            return c.resolve()
    return (root / "data" / "raw_ohlcv.csv").resolve()

def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    need = ["open", "high", "low", "close"]
    for n in need:
        if n not in cols:
            raise SystemExit(f"CSV missing column {n}: {path}")
    out = pd.DataFrame({n: pd.to_numeric(df[cols[n]], errors="coerce") for n in need})
    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(out) < 500:
        raise SystemExit(f"need >=500 bars, got {len(out)}")
    return out

def make_labels(close: np.ndarray, horizon: int = 3) -> np.ndarray:
    fwd = np.roll(close, -horizon) / close - 1.0
    y = (fwd > 0).astype(int)
    y[-horizon:] = -1
    return y

def features_talib(df, rsi_p, atr_p, fast, slow, sig):
    c = df["close"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    rsi = talib.RSI(c, timeperiod=rsi_p)
    atr = talib.ATR(h, l, c, timeperiod=atr_p)
    _, _, hist = talib.MACD(c, fastperiod=fast, slowperiod=slow, signalperiod=sig)
    adx = talib.ADX(h, l, c, timeperiod=14)
    return np.column_stack([c, atr, rsi, hist, adx])

def main():
    root = project_root()
    cfg = load_config(root)
    csv_path = resolve_csv(root, cfg)
    print("csv", csv_path)
    if not csv_path.is_file():
        raise SystemExit(f"missing CSV: {csv_path}")
    df = load_ohlcv(csv_path)
    y_all = make_labels(df["close"].to_numpy(dtype=float), horizon=3)
    def objective(trial):
        rsi_p = trial.suggest_categorical("rsi_period", [7, 10, 14, 21])
        atr_p = trial.suggest_categorical("atr_period", [10, 14, 20])
        fast = trial.suggest_categorical("macd_fast", [8, 12, 16])
        slow = trial.suggest_categorical("macd_slow", [17, 26, 32])
        sig = trial.suggest_categorical("macd_signal", [5, 9])
        if fast >= slow:
            return 1e9
        X = features_talib(df, rsi_p, atr_p, fast, slow, sig)
        valid = np.isfinite(X).all(axis=1) & (y_all >= 0)
        Xv, yv = X[valid], y_all[valid]
        if len(yv) < 300:
            return 1e9
        losses = []
        for tr, te in TimeSeriesSplit(n_splits=3).split(Xv):
            model = XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, subsample=0.9, n_jobs=2, random_state=42, eval_metric="logloss")
            model.fit(Xv[tr], yv[tr])
            proba = model.predict_proba(Xv[te])[:, 1]
            losses.append(log_loss(yv[te], proba, labels=[0, 1]))
        return float(np.mean(losses))
    study = optuna.create_study(direction="minimize", study_name="talib_periods")
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    print("best_value", study.best_value)
    print("best_params", study.best_params)
    out = root / "production" / "optuna_talib_best.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"csv": str(csv_path), "best_value": study.best_value, "best_params": study.best_params}, indent=2), encoding="utf-8")
    cfg.setdefault("talib_defaults", {}).update({
        "rsi_period": study.best_params.get("rsi_period", 14),
        "atr_period": study.best_params.get("atr_period", 14),
        "macd_fast": study.best_params.get("macd_fast", 12),
        "macd_slow": study.best_params.get("macd_slow", 26),
        "macd_signal": study.best_params.get("macd_signal", 9),
    })
    cfg.setdefault("historical_csv", "data/raw_ohlcv.csv")
    (root / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)

if __name__ == "__main__":
    main()
