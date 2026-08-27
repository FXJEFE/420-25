#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FXJEFE Optuna + MLflow + MedianPruner
- Locked 28-feature registry
- TimeSeriesSplit + gap
- Per-fold early stopping + Optuna pruning
- Final model trained with early stopping
- Versioned artifacts (never overwrites OG)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from optuna.exceptions import TrialPruned
from optuna.integration.mlflow import MLflowCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("fxjefe.optuna")

# ---------------------------------------------------------------------------
# Paths
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
MLRUNS = ROOT / "mlruns"
MLRUNS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def load_registry():
    candidates = [
        ROOT / "feature_registry.py",
        ROOT / "__pycache__" / "feature_registry.cpython-314.pyc",
        ROOT / "__pycache__" / "feature_registry.cpython-312.pyc",
    ]
    for p in candidates:
        if p.is_file():
            import importlib.util
            spec = importlib.util.spec_from_file_location("feature_registry", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            log.info("Registry loaded  TRAIN=%s PREDICT=%s", mod.TRAIN_COUNT, mod.PREDICT_COUNT)
            return mod
    raise RuntimeError("feature_registry not found — run pipelinerun first")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_training_data(reg):
    candidates = [
        DATA / "training_data.csv",
        DATA / "FXJEFE_Features_with_labels.csv",
        DATA / "processed_features.csv",
    ]
    path = next((c for c in candidates if c.is_file()), None)
    if path is None:
        raise FileNotFoundError("No training CSV found under data/")

    df = pd.read_csv(path)
    log.info("Loaded %d rows from %s", len(df), path.name)

    features = list(reg.TRAIN_FEATURES)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise RuntimeError(f"Missing locked features: {missing}")

    X = df[features].copy()
    y_raw = df["label"] if "label" in df.columns else df["signal"]
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    log.info("Classes: %s", dict(zip(le.classes_, range(len(le.classes_)))))
    return X, y, features, le

# ---------------------------------------------------------------------------
# Objective with pruning
# ---------------------------------------------------------------------------
def make_objective(X: pd.DataFrame, y: np.ndarray, n_splits: int = 4):
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=20)

    def objective(trial: optuna.Trial) -> float:
        import xgboost as xgb

        params = {
            "objective": "multi:softprob",
            "num_class": int(len(np.unique(y))),
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "verbosity": 0,
            "n_jobs": -1,
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "gamma": trial.suggest_float("gamma", 0.0, 4.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 150, 700),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 8.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 8.0, log=True),
        }

        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            model = xgb.XGBClassifier(**params, early_stopping_rounds=30)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            pred = model.predict(X_va)
            fold_score = f1_score(y_va, pred, average="macro")
            scores.append(fold_score)

            # Pruning report after every fold
            intermediate = float(np.mean(scores))
            trial.report(intermediate, step=fold_idx)
            if trial.should_prune():
                raise TrialPruned()

        return float(np.mean(scores))

    return objective

# ---------------------------------------------------------------------------
# Final model with early stopping
# ---------------------------------------------------------------------------
def train_final_model(X, y, best_params: dict, val_fraction: float = 0.15):
    import xgboost as xgb

    split = int(len(X) * (1 - val_fraction))
    X_tr, X_va = X.iloc[:split], X.iloc[split:]
    y_tr, y_va = y[:split], y[split:]

    params = dict(best_params)
    params["n_estimators"] = max(params.get("n_estimators", 400), 400)

    model = xgb.XGBClassifier(**params, early_stopping_rounds=40)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    pred = model.predict(X_va)
    macro_f1 = f1_score(y_va, pred, average="macro")
    acc = accuracy_score(y_va, pred)
    best_iteration = getattr(model, "best_iteration", params["n_estimators"])

    log.info("Final model  macro-F1=%.4f  acc=%.4f  best_iteration=%s",
             macro_f1, acc, best_iteration)
    return model, {"macro_f1": macro_f1, "accuracy": acc, "best_iteration": int(best_iteration)}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_trials: int = 50, study_name: str = "fxjefe_xgb_28f"):
    reg = load_registry()
    X, y, features, le = load_training_data(reg)

    mlflow.set_tracking_uri(f"file:///{MLRUNS.as_posix()}")
    mlflow.set_experiment("FXJEFE_Optuna_XGBoost")

    storage = f"sqlite:///{MODELS / 'optuna_xgb.db'}"
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(
            n_startup_trials=8,
            n_warmup_steps=1,
            interval_steps=1,
        ),
    )

    mlflc = MLflowCallback(
        tracking_uri=f"file:///{MLRUNS.as_posix()}",
        metric_name="macro_f1",
        create_experiment=False,
        mlflow_kwargs={"nested": True},
    )

    log.info("Starting Optuna + MLflow + Pruner  study=%s  trials=%d", study_name, n_trials)

    with mlflow.start_run(run_name=f"optuna_{study_name}_{datetime.now().strftime('%Y%m%d_%H%M')}") as parent_run:
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("n_trials_requested", n_trials)
        mlflow.log_param("feature_registry_version", getattr(reg, "ACTIVE_VERSION", "v1"))
        mlflow.log_param("min_confidence_gate", getattr(reg, "MIN_CONFIDENCE", 0.77))
        mlflow.log_param("pruner", "MedianPruner")

        study.optimize(
            make_objective(X, y),
            n_trials=n_trials,
            callbacks=[mlflc],
            show_progress_bar=True,
        )

        best = study.best_params
        best["objective"] = "multi:softprob"
        best["num_class"] = int(len(np.unique(y)))
        best["eval_metric"] = "mlogloss"
        best["tree_method"] = "hist"
        best["verbosity"] = 0

        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metric("best_macro_f1", study.best_value)
        mlflow.log_metric("n_trials_completed", len(study.trials))
        mlflow.log_metric("n_pruned", len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]))

        final_model, final_metrics = train_final_model(X, y, best)
        mlflow.log_metrics({f"final_{k}": v for k, v in final_metrics.items()})

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        params_path = MODELS / "best_xgb_params.json"
        model_path = MODELS / f"xgb_optuna_v{stamp}.pkl"
        report_path = MODELS / f"optuna_report_v{stamp}.json"

        params_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
        joblib.dump(final_model, model_path)

        report = {
            "best_value_macro_f1": study.best_value,
            "final_metrics": final_metrics,
            "n_trials": len(study.trials),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "best_params": best,
            "features_used": features,
            "n_features": len(features),
            "model_path": str(model_path),
            "mlflow_run_id": parent_run.info.run_id,
            "at_utc": datetime.now(timezone.utc).isoformat(),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(params_path))
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(report_path))
        mlflow.xgboost.log_model(final_model, "model")

        log.info("Best CV macro-F1 : %.4f", study.best_value)
        log.info("Final model      → %s", model_path)
        log.info("MLflow run id    → %s", parent_run.info.run_id)

    print("\n=== OPTUNA + MLFLOW + PRUNER COMPLETE ===")
    print(f"Best CV macro-F1 : {study.best_value:.4f}")
    print(f"Final macro-F1   : {final_metrics['macro_f1']:.4f}")
    print(f"Best iteration   : {final_metrics['best_iteration']}")
    print(f"Pruned trials    : {report['n_pruned']}")
    print(f"Model            : {model_path}")
    print(f"MLflow UI        : mlflow ui --backend-store-uri file:///{MLRUNS.as_posix()}")
    return 0

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--study", default="fxjefe_xgb_28f")
    args = ap.parse_args()
    sys.exit(main(n_trials=args.trials, study_name=args.study))