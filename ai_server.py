# -*- coding: utf-8 -*-
"""
FXJEFE AI Server entrypoint — Golden multi-model ensemble.

All /predict traffic uses ai_server_golden789:
  * multi-group ensemble (xgb-6 / 9-feat / full / per-symbol)
  * hard confidence gate: buy/sell ONLY if conf >= 0.77
  * consensus gate across active model groups

Does not modify model files on disk.
"""
from __future__ import annotations

import importlib.util
import os
import sys

# UTF-8
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _s in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
    try:
        if _s is not None and hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_GOLDEN_PATH = os.path.join(_HERE, "ai_server_golden789.py")
if not os.path.isfile(_GOLDEN_PATH):
    # fallback name variants
    for alt in ("ai_server_golden28_43.py", "ai_server_golden.py"):
        p = os.path.join(_HERE, alt)
        if os.path.isfile(p):
            _GOLDEN_PATH = p
            break

_spec = importlib.util.spec_from_file_location("ai_server_golden_impl", _GOLDEN_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load golden AI server from {_GOLDEN_PATH}")
_golden = importlib.util.module_from_spec(_spec)
# Register before exec so relative imports / flask app name work
sys.modules["ai_server_golden_impl"] = _golden
_spec.loader.exec_module(_golden)

# Force immutable confidence band: [0.77, 0.9888]
try:
    if float(getattr(_golden, "GATE", 0.77)) < 0.77:
        _golden.GATE = 0.77
    if float(getattr(_golden, "_HARD_FLOOR", 0.77)) < 0.77:
        _golden._HARD_FLOOR = 0.77
    _golden.GATE = max(float(_golden.GATE), 0.77)
    if not hasattr(_golden, "MAX_CONF") or float(getattr(_golden, "MAX_CONF", 0)) <= 0:
        _golden.MAX_CONF = 0.9888
    else:
        _golden.MAX_CONF = min(float(_golden.MAX_CONF), 0.9888) if float(_golden.MAX_CONF) > 0.9888 else float(_golden.MAX_CONF)
        if float(_golden.MAX_CONF) < float(_golden.GATE):
            _golden.MAX_CONF = 0.9888
    # Allow both multi-model voting and singular models
    _golden.ALLOW_SINGULAR = True
    _golden.ALLOW_MULTI_VOTE = True
except Exception:
    _golden.GATE = 0.77
    _golden._HARD_FLOOR = 0.77
    _golden.MAX_CONF = 0.9888
    _golden.ALLOW_SINGULAR = True
    _golden.ALLOW_MULTI_VOTE = True

# Load multi-model registry once at import (waitress / pipeline import path)
if hasattr(_golden, "load_all"):
    _golden.load_all()

# Public surface expected by waitress / pipeline: `from ai_server import app`
app = _golden.app
GATE = getattr(_golden, "GATE", 0.77)
load_all = getattr(_golden, "load_all", None)


if __name__ == "__main__":
    import logging

    log = logging.getLogger("ai_server")
    log.info("=" * 64)
    log.info("FXJEFE AI Server → Golden multi-model ensemble")
    log.info("Implementation: %s", os.path.basename(_GOLDEN_PATH))
    log.info("Confidence gate: %.2f (buy/sell only if conf >= gate)", float(GATE))
    log.info("=" * 64)

    port = int(os.environ.get("AI_SERVER_PORT", "8080"))
    # Prefer config api_port
    try:
        port = int(getattr(_golden, "config", {}).get("api_port") or port)
    except Exception:
        pass

    try:
        from waitress import serve

        log.info("Serving golden ensemble with waitress on 0.0.0.0:%s", port)
        serve(app, host="0.0.0.0", port=port, threads=8)
    except Exception as e:
        log.warning("waitress unavailable (%s); Flask threaded", e)
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
