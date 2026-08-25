#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FXJEFE stack smoke test with robust error handling."""
from __future__ import annotations
import importlib, sys, traceback
from typing import List, Tuple

REQUIRED = [
    ("numpy", "numpy"), ("pandas", "pandas"), ("flask", "flask"),
    ("sklearn", "sklearn"), ("joblib", "joblib"), ("requests", "requests"),
]
OPTIONAL = [
    ("xgboost", "xgboost"), ("arch", "arch"), ("pandas_ta", "pandas_ta"),
    ("fastapi", "fastapi"), ("tensorflow", "tensorflow"), ("MetaTrader5", "MetaTrader5"),
    ("talib", "talib"), ("optuna", "optuna"),
]

def try_import(label: str, module: str) -> Tuple[str, str]:
    try:
        importlib.import_module(module)
        return "OK", ""
    except ImportError as e:
        return "FAIL", f"{type(e).__name__}: {e}"
    except OSError as e:
        return "FAIL", f"OSError: {e}"
    except Exception as e:
        return "FAIL", f"{type(e).__name__}: {e}"

def main() -> int:
    print(f"Python {sys.version.split()[0]}  executable={sys.executable}")
    hard_fail = False
    for label, mod in REQUIRED:
        status, detail = try_import(label, mod)
        if status == "OK":
            print(f"OK   {label}")
        else:
            hard_fail = True
            print(f"FAIL {label}  {detail}")
    for label, mod in OPTIONAL:
        status, detail = try_import(label, mod)
        if status == "OK":
            print(f"OK   {label}")
        else:
            hint = ""
            if label == "xgboost" and "libomp" in detail:
                hint = "  -> brew install libomp"
            if label == "talib":
                hint = "  -> brew install ta-lib && pip install TA-Lib"
            if label == "MetaTrader5":
                hint = "  -> Windows only"
            if label == "pandas_ta":
                hint = "  -> needs Python>=3.12 or skip (use TA-Lib)"
            print(f"SKIP {label}  {detail}{hint}")
    if hard_fail:
        print("stack INCOMPLETE (required packages missing)")
        return 1
    print("stack OK (required packages present)")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
