# -*- coding: utf-8 -*-
"""
FXJEFE HFT Signal Server — FastAPI + ZeroMQ (low-latency path for MT5).

Runs alongside (or instead of) waitress golden on 8080:
  - FastAPI / Uvicorn   → http://0.0.0.0:8081  (HFT HTTP)
  - ZeroMQ REP          → tcp://127.0.0.1:5555 (Predict_ZeroMQ.mq5)
  - ZeroMQ PUB          → tcp://127.0.0.1:5556 (signal fan-out)

Prediction backends (auto):
  1) In-process golden ensemble if import succeeds
  2) Else HTTP proxy to GOLDEN_URL (default http://127.0.0.1:8080/predict)

Usage:
  runtime\\Python311\\python.exe hft_signal_server.py
  runtime\\Python311\\python.exe hft_signal_server.py --port 8081 --zmq-rep 5555

MT5:
  - WebRequest allow: http://127.0.0.1:8081
  - Predict_ZeroMQ / PredictZeroMQPS → tcp://127.0.0.1:5555
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "OG_pipeline222"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(ROOT, "Logs", "hft_signal_server.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("hft")

GOLDEN_URL = os.environ.get("GOLDEN_URL", "http://127.0.0.1:8080/predict").rstrip("/")
if not GOLDEN_URL.endswith("/predict"):
    # allow host-only
    GOLDEN_URL = GOLDEN_URL.rstrip("/") + "/predict"

# ── Optional: TA-Lib probe ───────────────────────────────────────────────────
TALIB_OK = False
try:
    import talib  # noqa: F401

    TALIB_OK = True
    log.info("TA-Lib available: %s", getattr(talib, "__version__", "?"))
except Exception as e:
    log.warning("TA-Lib not importable (%s) — pipeline scripts that need it still fail until wheel installs", e)

# ── Prediction core ──────────────────────────────────────────────────────────
_USE_INPROC = False
_predict_fn = None
_http = None


def _normalize_payload(data: dict) -> dict:
    out = dict(data or {})
    try:
        px = float(out.get("price") or 0)
    except Exception:
        px = 0.0
    try:
        cl = float(out.get("close") or 0)
    except Exception:
        cl = 0.0
    if px <= 0 and cl > 0:
        out["price"] = cl
        px = cl
    elif px > 0 and ("close" not in out or not out.get("close")):
        out["close"] = px
    try:
        gv = float(out.get("garch_vol") or 0)
    except Exception:
        gv = 0.0
    if gv <= 0:
        try:
            rv = float(out.get("realized_vol") or 0)
        except Exception:
            rv = 0.0
        if rv > 0:
            out["garch_vol"] = rv
        elif px > 0:
            try:
                atr0 = float(out.get("atr") or 0)
            except Exception:
                atr0 = 0.0
            if atr0 > 0:
                out["garch_vol"] = atr0 / px
    tf = str(out.get("timeframe", "M15")).strip().upper()
    if tf.startswith("PERIOD_"):
        tf = tf[7:]
    out["timeframe"] = tf
    return out


def _init_backend() -> None:
    global _USE_INPROC, _predict_fn, _http
    # Prefer proxy to already-running golden (your log shows waitress :8080 up)
    try:
        import httpx

        _http = httpx.Client(timeout=2.5)
        r = _http.get(GOLDEN_URL.replace("/predict", "/health"))
        if r.status_code == 200:
            log.info("Backend: HTTP proxy → %s (golden healthy)", GOLDEN_URL)
            _USE_INPROC = False
            return
    except Exception as e:
        log.warning("Golden health check failed: %s — will try in-process import", e)

    try:
        # Import after path setup; may re-load models (slow) — only if golden not up
        import ai_server_golden as golden  # type: ignore

        def _inproc(data: dict) -> dict:
            data = _normalize_payload(data)
            symbol = golden.normalize_symbol(str(data.get("symbol", "")))
            timeframe = str(data.get("timeframe", "M15")).upper()
            price = float(data.get("price") or 0)
            atr = float(data.get("atr") or 0.001)
            ens_prob, model_probs, group_signals = golden.ensemble(data, symbol, timeframe)
            signal, confidence, gate_info = golden.apply_gates(ens_prob, group_signals)
            stop_loss = 0.0
            if signal == "buy":
                stop_loss = price - 2 * atr
            elif signal == "sell":
                stop_loss = price + 2 * atr
            return {
                "signal": signal,
                "confidence": round(float(confidence), 4),
                "probability": round(float(ens_prob), 6),
                "n_models": len(model_probs),
                "stop_loss": round(stop_loss, 5),
                "symbol": symbol,
                "timeframe": timeframe,
                "group_signals": group_signals,
                "gate_passed": bool(gate_info.get("stat_ok") and gate_info.get("consensus_ok")),
                "gate_info": gate_info,
                "model_probs": model_probs,
                "server": "hft_inproc_golden",
                "latency_ms": None,
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            }

        _predict_fn = _inproc
        _USE_INPROC = True
        log.info("Backend: in-process ai_server_golden ensemble")
    except Exception as e:
        log.error("In-process golden import failed: %s — proxy mode forced", e)
        _USE_INPROC = False
        if _http is None:
            import httpx

            _http = httpx.Client(timeout=2.5)


def predict_signal(data: dict) -> dict:
    """Single entry used by FastAPI + ZeroMQ."""
    t0 = time.perf_counter()
    data = _normalize_payload(data)
    try:
        if _USE_INPROC and _predict_fn is not None:
            out = _predict_fn(data)
        else:
            if _http is None:
                import httpx

                client = httpx.Client(timeout=2.5)
            else:
                client = _http
            r = client.post(GOLDEN_URL, json=data)
            r.raise_for_status()
            out = r.json()
            out["server"] = out.get("server") or "hft_proxy_golden"
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        out["ts"] = out.get("ts") or datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        out["hft"] = True
        out["talib"] = TALIB_OK
        # LOSS GUARD: re-apply 0.77 gate at HFT edge (never trust upstream alone)
        try:
            conf = float(out.get("confidence") or 0)
        except Exception:
            conf = 0.0
        sig = str(out.get("signal") or "hold").lower()
        if sig in ("buy", "sell") and conf < 0.77:
            out["signal"] = "hold"
            out["gate_passed"] = False
            out["hft_gate_blocked"] = True
            out["min_conf_gate"] = 0.77
        return out
    except Exception as e:
        log.exception("predict_signal failed")
        return {
            "signal": "hold",
            "confidence": 0.0,
            "probability": 0.5,
            "n_models": 0,
            "error": str(e),
            "server": "hft_error",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "hft": True,
            "talib": TALIB_OK,
        }


# ── ZeroMQ REP worker (Predict_ZeroMQ.mq5) ───────────────────────────────────
class ZmqRepServer(threading.Thread):
    def __init__(self, endpoint: str = "tcp://127.0.0.1:5555", pub_endpoint: Optional[str] = None):
        super().__init__(daemon=True, name="zmq-rep")
        self.endpoint = endpoint
        self.pub_endpoint = pub_endpoint
        self._stop = threading.Event()

    def run(self) -> None:
        try:
            import zmq
        except ImportError:
            log.error("pyzmq not installed — ZeroMQ disabled")
            return
        ctx = zmq.Context.instance()
        rep = ctx.socket(zmq.REP)
        rep.setsockopt(zmq.LINGER, 0)
        rep.setsockopt(zmq.RCVTIMEO, 1000)
        rep.setsockopt(zmq.SNDTIMEO, 1000)
        rep.bind(self.endpoint)
        pub = None
        if self.pub_endpoint:
            pub = ctx.socket(zmq.PUB)
            pub.setsockopt(zmq.LINGER, 0)
            pub.bind(self.pub_endpoint)
            log.info("ZeroMQ PUB bound %s", self.pub_endpoint)
        log.info("ZeroMQ REP bound %s (MT5 Predict_ZeroMQ)", self.endpoint)

        while not self._stop.is_set():
            try:
                raw = rep.recv()
            except zmq.Again:
                continue
            except Exception as e:
                log.warning("ZMQ recv: %s", e)
                continue
            try:
                if isinstance(raw, bytes):
                    text = raw.decode("utf-8", errors="replace")
                else:
                    text = str(raw)
                data = json.loads(text)
            except Exception as e:
                reply = json.dumps({"signal": "hold", "confidence": 0.0, "error": f"bad_json:{e}"})
                try:
                    rep.send_string(reply)
                except Exception:
                    pass
                continue

            result = predict_signal(data)
            # Compact reply for MQL5 parsers (signal + confidence required)
            compact = {
                "signal": result.get("signal", "hold"),
                "confidence": result.get("confidence", 0.0),
                "probability": result.get("probability"),
                "n_models": result.get("n_models"),
                "stop_loss": result.get("stop_loss"),
                "latency_ms": result.get("latency_ms"),
                "server": result.get("server"),
                "gate_passed": result.get("gate_passed"),
            }
            try:
                rep.send_string(json.dumps(compact, separators=(",", ":")))
            except Exception as e:
                log.warning("ZMQ send failed: %s", e)
                try:
                    # recover REP state
                    rep.send_string(json.dumps({"signal": "hold", "confidence": 0.0, "error": str(e)}))
                except Exception:
                    pass
            if pub is not None:
                try:
                    topic = str(result.get("symbol") or data.get("symbol") or "UNK")
                    pub.send_string(f"{topic} {json.dumps(compact, separators=(',', ':'))}")
                except Exception:
                    pass
        rep.close(0)
        if pub is not None:
            pub.close(0)
        log.info("ZeroMQ REP stopped")

    def stop(self) -> None:
        self._stop.set()


# ── FastAPI app ──────────────────────────────────────────────────────────────
def build_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import ORJSONResponse, PlainTextResponse

    app = FastAPI(
        title="FXJEFE HFT Signal Server",
        version="1.0.0",
        default_response_class=ORJSONResponse,
    )

    @app.get("/health")
    def health():
        return {
            "status": "running",
            "server": "hft_signal_server",
            "backend": "inproc" if _USE_INPROC else "proxy",
            "golden_url": GOLDEN_URL,
            "talib": TALIB_OK,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    @app.get("/")
    def root():
        return {
            "service": "FXJEFE HFT",
            "http_predict": "POST /predict",
            "zmq_rep": "tcp://127.0.0.1:5555",
            "zmq_pub": "tcp://127.0.0.1:5556",
            "golden": GOLDEN_URL,
        }

    @app.post("/predict")
    async def predict(request: Request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        return predict_signal(data if isinstance(data, dict) else {})

    @app.post("/hft/predict")
    async def hft_predict(request: Request):
        """Alias optimized path (same handler)."""
        try:
            data = await request.json()
        except Exception:
            data = {}
        return predict_signal(data if isinstance(data, dict) else {})

    @app.get("/ping", response_class=PlainTextResponse)
    def ping():
        return "pong"

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description="FXJEFE HFT FastAPI + ZeroMQ signal server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("HFT_PORT", "8081")))
    ap.add_argument("--zmq-rep", default=os.environ.get("HFT_ZMQ_REP", "tcp://127.0.0.1:5555"))
    ap.add_argument("--zmq-pub", default=os.environ.get("HFT_ZMQ_PUB", "tcp://127.0.0.1:5556"))
    ap.add_argument("--no-zmq", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.join(ROOT, "Logs"), exist_ok=True)
    log.info("=" * 64)
    log.info("FXJEFE HFT Signal Server")
    log.info("FastAPI  : http://%s:%s", args.host, args.port)
    log.info("ZMQ REP  : %s", args.zmq_rep)
    log.info("ZMQ PUB  : %s", args.zmq_pub)
    log.info("Golden   : %s", GOLDEN_URL)
    log.info("TA-Lib   : %s", TALIB_OK)
    log.info("=" * 64)

    _init_backend()

    zmq_thread = None
    if not args.no_zmq:
        zmq_thread = ZmqRepServer(args.zmq_rep, args.zmq_pub)
        zmq_thread.start()

    import uvicorn

    app = build_app()
    # Single worker — models/ZMQ not multi-process safe by default
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)
    if zmq_thread:
        zmq_thread.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
