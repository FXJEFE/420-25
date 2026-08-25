#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Realized volatility features for FXJEFE."""
from __future__ import annotations
import numpy as np
import pandas as pd

def log_returns(close: pd.Series) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    r = np.log(c / c.shift(1))
    return r.replace([np.inf, -np.inf], np.nan)

def realized_vol(close: pd.Series, window: int = 20, annualize: bool = False,
                 periods_per_year: float = 252.0 * 24 * 12) -> pd.Series:
    r = log_returns(close)
    rv = np.sqrt(r.pow(2).rolling(window, min_periods=max(2, window // 5)).sum())
    if annualize:
        rv = rv * np.sqrt(periods_per_year / window)
    return rv

def realized_vol_std(close: pd.Series, window: int = 20) -> pd.Series:
    return log_returns(close).rolling(window, min_periods=max(2, window // 5)).std()

def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    hl = np.log(h / l).replace([np.inf, -np.inf], np.nan)
    const = 1.0 / (4.0 * np.log(2.0))
    var = (hl.pow(2) * const).rolling(window, min_periods=max(2, window // 5)).mean()
    return np.sqrt(var)

def garman_klass_vol(open_: pd.Series, high: pd.Series, low: pd.Series,
                     close: pd.Series, window: int = 20) -> pd.Series:
    o = pd.to_numeric(open_, errors="coerce")
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    log_hl = np.log(h / l).replace([np.inf, -np.inf], np.nan)
    log_co = np.log(c / o).replace([np.inf, -np.inf], np.nan)
    term = 0.5 * log_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)
    var = term.rolling(window, min_periods=max(2, window // 5)).mean()
    return np.sqrt(var.clip(lower=0))

def add_realized_vol_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    cols = {str(c).lower(): c for c in out.columns}
    def col(name):
        return cols.get(name)
    c_close = col("close")
    if c_close is None:
        out["realized_vol"] = 0.0
        return out
    out["realized_vol"] = realized_vol(out[c_close], window=window)
    out["realized_vol_std"] = realized_vol_std(out[c_close], window=window)
    c_high, c_low, c_open = col("high"), col("low"), col("open")
    if c_high is not None and c_low is not None:
        out["parkinson_vol"] = parkinson_vol(out[c_high], out[c_low], window=window)
    if all(x is not None for x in (c_open, c_high, c_low, c_close)):
        out["garman_klass_vol"] = garman_klass_vol(
            out[c_open], out[c_high], out[c_low], out[c_close], window=window)
    for c in list(out.columns):
        if "vol" in str(c).lower() or c == "realized_vol":
            out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return out

def last_realized_vol(df: pd.DataFrame, window: int = 20) -> float:
    tmp = add_realized_vol_features(df, window=window)
    v = tmp["realized_vol"].iloc[-1]
    return float(v) if np.isfinite(v) else 0.0
