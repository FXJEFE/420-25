#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FXJEFE Optuna hyperparameter optimization
- Uses locked 28 training features from feature_registry
- Time-series aware splits
- Saves best params + full study under models/
- Never mutates the feature list
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("fxjefe.optuna")

# ---------------------------------------------------------------------------
# Paths (username-agnostic)
# ---------------------------------------------------------------------------
def project_root() -> Path:
    env = os.environ.get("FXJEFE_PROJECT_ROOT")
    if env and env.strip():
        return Path(env.strip()).resolve()
    if sys.platform == "win32":
        home = Path(os.environ.get("USERPROFILE") or Path.home())
    else:
        home = Path.home()
    root = home / "Documents" / "FXJEFE_Project"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()

ROOT = project_root()
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"

# ---------------------------------------------------------------------------
# Feature registry lock
# ---------------------------------------------------------------------------
def load_registry():
    candidates = [
        ROOT / "feature_registry.py",
        ROOT / "__pycache__" / "feature_registry.cpython-312.pyc",
        ROOT / "__pycache__" / "feature_registry.cpython-314.pyc",
    ]
    for p in candidates:
        if p.is_file():
            import importlib.util
            spec = importlib.util.spec_from_file_location("feature_registry", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            log.info("Registry loaded from %s  TRAIN=%s PREDICT=%s", p, mod.TRAIN_COUNT, mod.PREDICT_COUNT)
            return mod
    raise RuntimeError("feature_registry not found")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_training_data(reg) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    candidates = [
        DATA / "training_data.csv",
        DATA / "FXJEFE_Features_with_labels.csv",
        DATA / "processed_features.csv",
    ]
    path = None
    for c in candidates:
        if c.is_file():
            path = c
            break
    if path is None:
        raise FileNotFoundError("No training CSV found under data/")

    df = pd.read_csv(path)
    log.info("Loaded %s rows from %s", len(df), path.name)

    # Use only the locked training features
    features = list(reg.TRAIN_FEATURES)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise RuntimeError(f"Missing locked features: {missing}")

    X = df[features].copy()
    # Label column
    if "label" in df.columns:
        y_raw = df["label"]
    elif "signal" in df.columns:
        y_raw = df["signal"]
    else:
        raise RuntimeError("No label/signal column found")

    # Encode to 0/1/2 if needed
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    log.info("Classes: %s", dict(zip(le.classes_, range(len(le.classes_)))))
    log.info("Label distribution: %s", pd.Series(y).value_counts().to_dict())

    return X, y, features

# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def make_objective(X: pd.DataFrame, y: np.ndarray, n_splits: int = 4):
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=20)  # gap reduces leakage on M1

    def objective(trial: optuna.Trial) -> float:
        import xgboost as xgb

        params = {
            "objective": "multi:softprob",
            "num_class": len(np.unique(y)),
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "verbosity": 0,
            "n_jobs": -1,

            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "gamma": trial.suggest_float("gamma", 0.0, 4.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 120, 600),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 8.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 8.0, log=True),
        }

        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            model = xgb.XGBClassifier(**params, early_stopping_rounds=25)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )
            pred = model.predict(X_va)
            scores.append(f1_score(y_va, pred, average="macro"))

        return float(np.mean(scores))

    return objective

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_trials: int = 60, study_name: str = "fxjefe_xgb_28f"):
    reg = load_registry()
    X, y, features = load_training_data(reg)

    storage = f"sqlite:///{MODELS / 'optuna_xgb.db'}"
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(seed=42),
    )

    log.info("Starting Optuna study '%s' – %d trials", study_name, n_trials)
    study.optimize(make_objective(X, y), n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["objective"] = "multi:softprob"
    best["num_class"] = int(len(np.unique(y)))
    best["eval_metric"] = "mlogloss"
    best["tree_method"] = "hist"
    best["verbosity"] = 0

    # Persist
    params_path = MODELS / "best_xgb_params.json"
    params_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    log.info("Best params saved → %s", params_path)
    log.info("Best macro-F1: %.4f", study.best_value)

    # Retrain final model on full data with best params
    import xgboost as xgb
    final = xgb.XGBClassifier(**best)
    final.fit(X, y)

    # Versioned filename (never overwrite OG)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = MODELS / f"xgb_optuna_v{stamp}.pkl"
    joblib.dump(final, model_path)
    log.info("Final model saved → %s", model_path)

    # Also write a small report
    report = {
        "best_value_macro_f1": study.best_value,
        "n_trials": len(study.trials),
        "best_params": best,
        "features_used": features,
        "n_features": len(features),
        "model_path": str(model_path),
        "at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (MODELS / f"optuna_report_v{stamp}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== OPTUNA COMPLETE ===")
    print(f"Best macro-F1 : {study.best_value:.4f}")
    print(f"Params        : {params_path}")
    print(f"Model         : {model_path}")
    return 0

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--study", default="fxjefe_xgb_28f")
    args = ap.parse_args()
    sys.exit(main(n_trials=args.trials, study_name=args.study))