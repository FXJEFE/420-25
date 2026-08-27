# -*- coding: utf-8 -*-
"""
FXJEFE AI Server entrypoint — Golden OG + multi-model ensemble.

Loads ai_server_golden789 (OG 6/9/28 + M15 specialists + og333 voters).
Does not overwrite files in Documents\\models except via golden skip-zero logic.
"""
from __future__ import annotations

import os
import sys
import logging

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE in sys.path:
    sys.path.remove(_HERE)
sys.path.insert(0, _HERE)
_DOCS = os.path.join(os.environ.get("USERPROFILE", r"C:\Users\locallarry"), "Documents")
if _DOCS not in sys.path:
    sys.path.append(_DOCS)

import ai_server_golden789 as _golden  # noqa: E402

_golden.load_all()
app = _golden.app
GATE = _golden.GATE
MAX_CONF = _golden.MAX_CONF

log = logging.getLogger("ai_server")


if __name__ == "__main__":
    host = os.environ.get("AI_SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("AI_SERVER_PORT", "8080"))
    log.info("=" * 64)
    log.info("FXJEFE AI Server → Golden OG + multi-model")
    log.info("Confidence band: [%.4f, %.4f]", float(GATE), float(MAX_CONF))
    log.info("Golden models: %s  symbol specialists: %s", len(_golden._loaded), len(_golden._symbol_models))
    log.info("=" * 64)
    try:
        from waitress import serve

        log.info("Serving with waitress on %s:%s", host, port)
        serve(app, host=host, port=port, threads=8)
    except Exception as e:
        log.warning("waitress unavailable (%s); Flask threaded", e)
        app.run(host=host, port=port, debug=False, threaded=True)
