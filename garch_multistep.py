#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GARCH(1,1) multi-step forecast comparison."""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
try:
    from arch import arch_model
except ImportError:
    arch_model = None

def garch_multistep_compare(returns, horizons=None, window=100, scale=100.0) -> Dict[str, float]:
    horizons = horizons or [1, 3, 5, 10]
    out = {f"vol_h{h}": 0.0 for h in horizons}
    out.update(vol_last_insample=0.0, long_run_vol=0.0, omega=0.0, alpha=0.0, beta=0.0, persistence=0.0, ok=0.0)
    if arch_model is None:
        return out
    r = returns.astype(float) if isinstance(returns, pd.Series) else pd.Series(np.asarray(returns, dtype=float))
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) > window:
        r = r.iloc[-window:]
    if len(r) < max(20, window // 5):
        return out
    try:
        res = arch_model(r * scale, vol="Garch", p=1, q=1, mean="Zero", rescale=False).fit(disp="off", show_warning=False)
        out["vol_last_insample"] = float(res.conditional_volatility.iloc[-1]) / scale
        params = res.params
        omega = float(params.get("omega", params.iloc[0]))
        alpha = float(params.get("alpha[1]", params.get("alpha", 0.0)))
        beta = float(params.get("beta[1]", params.get("beta", 0.0)))
        pers = alpha + beta
        out.update(omega=omega, alpha=alpha, beta=beta, persistence=pers)
        if pers < 0.999 and (1.0 - pers) > 1e-8:
            out["long_run_vol"] = float(np.sqrt(max(omega / (1.0 - pers), 0.0)) / scale)
        else:
            out["long_run_vol"] = out["vol_last_insample"]
        var = np.asarray(res.forecast(horizon=max(horizons), reindex=False).variance.values[-1], dtype=float)
        for h in horizons:
            if h - 1 < len(var) and var[h - 1] >= 0:
                out[f"vol_h{h}"] = float(np.sqrt(var[h - 1]) / scale)
        out["ok"] = 1.0
    except Exception:
        pass
    return out

def pick_garch_feature(result: Dict[str, float], mode: str = "h1") -> float:
    if result.get("ok", 0.0) < 1.0:
        return 0.0
    if mode == "h5":
        return float(result.get("vol_h5", result.get("vol_h1", 0.0)))
    if mode == "last":
        return float(result.get("vol_last_insample", 0.0))
    if mode == "blend":
        return 0.7 * float(result.get("vol_h1", 0.0)) + 0.3 * float(result.get("long_run_vol", 0.0))
    return float(result.get("vol_h1", 0.0))
