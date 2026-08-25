#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Central optional/mandatory imports — never crash on missing extras."""
from __future__ import annotations
import logging
from typing import Any
log = logging.getLogger("fxjefe.optional")
try:
    import numpy as np
except ImportError:
    np = None
try:
    import pandas as pd
except ImportError:
    pd = None
HAS_TALIB = False
talib = None
try:
    import talib as _talib
    talib = _talib
    HAS_TALIB = True
except Exception as e:
    log.debug("talib unavailable: %s", e)
HAS_XGB = False
xgb = None
XGBClassifier = None
try:
    import xgboost as _xgb
    from xgboost import XGBClassifier as _XGBC
    xgb = _xgb
    XGBClassifier = _XGBC
    HAS_XGB = True
except Exception as e:
    log.debug("xgboost unavailable: %s", e)
HAS_ARCH = False
arch_model = None
try:
    from arch import arch_model as _am
    arch_model = _am
    HAS_ARCH = True
except Exception as e:
    log.debug("arch unavailable: %s", e)
HAS_SKLEARN = False
try:
    import sklearn
    HAS_SKLEARN = True
except Exception:
    pass
HAS_JOBLIB = False
joblib = None
try:
    import joblib as _joblib
    joblib = _joblib
    HAS_JOBLIB = True
except Exception:
    pass
HAS_FLASK = False
try:
    import flask
    HAS_FLASK = True
except Exception:
    pass
HAS_FASTAPI = False
try:
    import fastapi
    HAS_FASTAPI = True
except Exception:
    pass
HAS_PANDAS_TA = False
pandas_ta = None
try:
    import pandas_ta as _pta
    pandas_ta = _pta
    HAS_PANDAS_TA = True
except Exception:
    try:
        import pandas_ta_classic as _pta
        pandas_ta = _pta
        HAS_PANDAS_TA = True
    except Exception:
        pass
HAS_TF = False
try:
    import tensorflow
    HAS_TF = True
except Exception:
    pass
HAS_MT5 = False
mt5 = None
try:
    import MetaTrader5 as _mt5
    mt5 = _mt5
    HAS_MT5 = True
except Exception:
    pass
HAS_OPTUNA = False
optuna = None
try:
    import optuna as _optuna
    optuna = _optuna
    HAS_OPTUNA = True
except Exception:
    pass

def require(*names):
    mapping = {"numpy": np is not None, "pandas": pd is not None, "sklearn": HAS_SKLEARN, "flask": HAS_FLASK, "joblib": HAS_JOBLIB}
    missing = [n for n in names if n in mapping and not mapping[n]]
    if missing:
        raise ImportError("mandatory packages missing: " + ", ".join(missing))

def status():
    return {"numpy": np is not None, "pandas": pd is not None, "sklearn": HAS_SKLEARN, "flask": HAS_FLASK,
            "joblib": HAS_JOBLIB, "talib": HAS_TALIB, "xgboost": HAS_XGB, "arch": HAS_ARCH,
            "pandas_ta": HAS_PANDAS_TA, "fastapi": HAS_FASTAPI, "tensorflow": HAS_TF,
            "MetaTrader5": HAS_MT5, "optuna": HAS_OPTUNA}

if __name__ == "__main__":
    import json
    print(json.dumps(status(), indent=2))
