#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rolling-window imputation + feature importance (NaN-safe)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def rolling_impute(df: pd.DataFrame, cols: list, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
            continue
        s = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        roll = s.rolling(window, min_periods=1).median()
        out[c] = s.fillna(roll).ffill().fillna(0.0)
    return out

def feature_importance_nan_safe(df: pd.DataFrame, features: list, y_col: str):
    data = rolling_impute(df, list(features) + [y_col])
    X = data[features].to_numpy(dtype=float)
    y = data[y_col].to_numpy()
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    return pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False), clf
