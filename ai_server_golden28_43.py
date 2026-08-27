# -*- coding: utf-8 -*-
import os as _os_utf8, sys as _sys_utf8
_os_utf8.environ.setdefault("PYTHONUTF8", "1")
_os_utf8.environ.setdefault("PYTHONIOENCODING", "utf-8")
# -*- coding: utf-8 -*-
"""
ai_server_golden.py — FXJEFE Comprehensive Golden AI Server (port 8080)

Combines:
  * Locked golden multi-feature ensemble (6 / 9 / 28+ features)
  * Optional bonus voters (stacking, lightgbm, sharpe-xgb, extra pkls)
  * Auto-discovered per-symbol XGB/LGB models (when feature JSON is valid)
  * Dual gates: statistical confidence + multi-group consensus
  * Broker suffix stripping (.r / .a / .m)
  * Hot reload, audit log, rich health payload

Endpoints:
  GET  /health
  GET  /models
  POST /predict
  GET  /sentiment  |  GET /predict/sentiment
  POST /reload

Usage:
  python ai_server_golden.py
  set GOLDEN_LOAD_HEAVY=1   # also load huge optional models (e.g. crypto_model.pkl)
"""
from __future__ import annotations

import csv
import glob
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import xgboost as xgb
from flask import Flask, jsonify, request

warnings.filterwarnings("ignore")

# Optional torch (LSTM .h5 that is actually torch.save)
try:
    import torch
    import torch.nn as _nn

    _TORCH = True
except ImportError:
    torch = None
    _nn = None
    _TORCH = False


class _LSTM2Layer3Class(_nn.Module if _nn is not None else object):
    def __init__(self, input_size=45, hidden_size=64, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = _nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = _nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── Paths / config ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
    config = json.load(_f)

MODELS_DIR = config.get("models_path") or os.path.join(PROJECT_ROOT, "models")
LOG_DIR = config.get("log_path") or os.path.join(PROJECT_ROOT, "Logs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
# Prefer project root first — many files under models/ are zero-filled on this machine
SEARCH_DIRS = [
    PROJECT_ROOT,
    MODELS_DIR,
    os.path.join(PROJECT_ROOT, "config"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "ai_server_golden.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("golden")

LOAD_HEAVY = os.environ.get("GOLDEN_LOAD_HEAVY", "0").strip() in ("1", "true", "True", "yes")

# ── Feature contracts ─────────────────────────────────────────────────────────
FEATURES_6 = ["price", "atr", "ema_diff", "rsi", "garch_vol", "macd_diff"]
FEATURES_9 = FEATURES_6 + ["vwap", "price_vwap_diff", "bb_position"]
FEATURES_FULL: List[str] = list(config.get("features") or FEATURES_9)
FEATURES_28 = FEATURES_FULL[:28] if len(FEATURES_FULL) >= 28 else FEATURES_FULL
FEATURES_43 = FEATURES_FULL if len(FEATURES_FULL) >= 28 else (
    FEATURES_FULL + [f"f_{i}" for i in range(len(FEATURES_FULL), 43)]
)

_w = config.get("golden_weights") or {}
W_XGB = float(_w.get("xgb_6", 0.30))
W_NINE = float(_w.get("avg_9feat", 0.30))
W_FULL = float(_w.get("rf_28", 0.25))
W_SYM = float(_w.get("symbol_model", 0.15))  # per-symbol specialist

# Persistent hard gate: NEVER allow buy/sell below 0.77 (loss protection)
_HARD_FLOOR = float(config.get("hard_confidence_floor", 0.77))
if _HARD_FLOOR < 0.77:
    _HARD_FLOOR = 0.77  # immutable floor — cannot lower via config below 0.77
GATE = max(float(config.get("min_confidence_threshold", 0.77)), _HARD_FLOOR)
REQUIRE_CONSENSUS = bool(config.get("golden_require_consensus", True))
STRIP_SUFFIXES = list(
    config.get("symbol_suffixes_strip")
    or [".r", ".R", ".a", ".A", ".m", ".M", ".pro", ".i", ".raw"]
)

# ── Model registries (paths that are VALID on this Admin PC) ──────────────────
# Note: models\my_model.pkl / models\xgboost_model.json are often zero-filled.
# Prefer project-root copies discovered via audit (2026-07).
GOLDEN_CORE = {
    # 6-feat XGB filter
    "xgb_6": ("xgboost_model (1).json", "xgb", FEATURES_6, "xgb"),
    "xgb_6_alt": ("ensamble_model.pkl.json", "xgb", FEATURES_6, "xgb"),
    "xgb_6_copy": ("xgboost_model - Copy.json", "xgb", FEATURES_6, "xgb"),
    # 9-feat voters
    "ensemble_9a": ("11_feature_rf.pkl", "pkl", FEATURES_9, "nine"),
    "ensemble_9b": ("ensemble_model_new.pkl", "pkl", FEATURES_9, "nine"),
    "ensemble_9c": ("my_model (1).pkl", "pkl", FEATURES_9, "nine"),
    "ensemble_9d": ("my_model (5).pkl", "pkl", FEATURES_9, "nine"),
    "forex_9": ("forex_model_2025.pkl", "pkl", FEATURES_9, "nine"),
    # 11-feat
    "ltdm_11": ("ltdm_model.pkl", "pkl", FEATURES_9 + ["spread", "sentiment"], "nine"),
    "rf_xrp_11": ("11_feature_rf_XRP_2025_FINAL.pkl", "pkl", FEATURES_9 + ["spread", "sentiment"], "nine"),
    # 28 / 45 full
    "rf_28": ("my_model (8).pkl", "pkl", FEATURES_28, "full"),
    "xgb_45": ("xgboost_model.json", "xgb", FEATURES_43, "full"),
    "xgb_sharpe": ("xgboost_best_sharpe.json", "xgb", FEATURES_43, "full"),
    "pipe_45": ("my_model.pkl", "pkl", FEATURES_43, "full"),
}

GOLDEN_OPTIONAL = {
    "lgbm_onnx": ("lightgbm_model.onnx", "onnx", FEATURES_43, "full"),
    "lstm": ("lstm_model.h5", "torch", FEATURES_43, "nine"),
    "ensemble_9e": ("my_model (6).pkl", "pkl", FEATURES_9, "nine"),
    "ensemble_9f": ("my_model (7).pkl", "pkl", FEATURES_9, "nine"),
    "xgb_45_alt": ("xgboost_model (2).json", "xgb", FEATURES_43, "full"),
}

if LOAD_HEAVY:
    GOLDEN_OPTIONAL["crypto_rf"] = ("crypto_model.pkl", "pkl", FEATURES_28, "full")
    GOLDEN_OPTIONAL["pipe_45_big"] = ("my_model (4).pkl", "pkl", FEATURES_43, "full")

_loaded: Dict[str, dict] = {}
_symbol_models: Dict[Tuple[str, str], dict] = {}


def _file_looks_valid(path: str) -> bool:
    """Reject zero-filled / empty corrupt artifacts common on USB copies."""
    try:
        if not os.path.isfile(path) or os.path.getsize(path) < 64:
            return False
        with open(path, "rb") as f:
            head = f.read(16)
        if not head or all(b == 0 for b in head):
            return False
        # XGB JSON must start with '{'
        if path.lower().endswith(".json") and head[:1] != b"{":
            return False
        # pickle protocol
        if path.lower().endswith(".pkl") and head[:1] not in (b"\x80", b"c", b"("):
            return False
        return True
    except OSError:
        return False


def _resolve_model_path(fname: str) -> Optional[str]:
    candidates = []
    for d in SEARCH_DIRS:
        candidates.append(os.path.join(d, fname))
    # de-dupe preserve order
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if _file_looks_valid(p):
            return p
    return None


def _detect_n_features(model, mtype: str, declared: list) -> int:
    try:
        if mtype == "pkl":
            n = getattr(model, "n_features_in_", None)
            if n is not None:
                return int(n)
            if hasattr(model, "estimators_") and model.estimators_:
                first = model.estimators_[0]
                if hasattr(first, "n_features_in_"):
                    return int(first.n_features_in_)
        elif mtype == "xgb":
            return int(model.num_features())
        elif mtype == "onnx":
            for dim in reversed(model.get_inputs()[0].shape):
                if isinstance(dim, int) and dim > 0:
                    return dim
        elif mtype == "torch" and _TORCH:
            for mod in model.modules():
                if isinstance(mod, _nn.LSTM):
                    return int(mod.input_size)
    except Exception:
        pass
    return len(declared)


def _load_one(key: str, fname: str, mtype: str, feats: list, group: str) -> Optional[dict]:
    path = _resolve_model_path(fname)
    if not path:
        log.warning(f"  MISSING [{key}]: {fname}")
        return None
    try:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > 120 and not LOAD_HEAVY:
            log.warning(f"  SKIP [{key}] heavy {size_mb:.0f}MB (set GOLDEN_LOAD_HEAVY=1): {fname}")
            return None
        if mtype == "xgb":
            m = xgb.Booster()
            m.load_model(path)
        elif mtype == "pkl":
            m = joblib.load(path)
        elif mtype == "onnx":
            import onnxruntime as ort

            m = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        elif mtype == "torch":
            if not _TORCH:
                log.warning(f"  SKIP [{key}]: torch not installed")
                return None
            state = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and any(k.startswith("lstm.") or k.startswith("fc.") for k in state):
                m = _LSTM2Layer3Class()
                m.load_state_dict(state)
                m.eval()
            else:
                log.warning(f"  SKIP [{key}]: unrecognized torch payload")
                return None
        else:
            log.warning(f"  SKIP [{key}]: unknown type {mtype}")
            return None

        n_actual = _detect_n_features(m, mtype, feats)
        # Align declared list length to actual when possible by pad/truncate names
        feats_use = list(feats)
        if n_actual != len(feats_use):
            if n_actual < len(feats_use):
                feats_use = feats_use[:n_actual]
            else:
                feats_use = feats_use + [f"pad_{i}" for i in range(len(feats_use), n_actual)]
            mismatched = True
        else:
            mismatched = False

        log.info(
            f"  Loaded [{key}] {os.path.basename(path)} type={mtype} "
            f"n={n_actual} group={group}{' MISMATCH-adj' if mismatched else ''}"
        )
        return {
            "model": m,
            "type": mtype,
            "features": feats_use,
            "n_features": n_actual,
            "group": group,
            "mismatched": mismatched,
            "path": path,
        }
    except Exception as e:
        log.error(f"  FAILED [{key}] {fname}: {e}")
        return None


def _load_golden() -> None:
    global _loaded
    _loaded = {}
    for key, (fname, mtype, feats, group) in GOLDEN_CORE.items():
        entry = _load_one(key, fname, mtype, feats, group)
        if entry:
            _loaded[key] = entry
    for key, (fname, mtype, feats, group) in GOLDEN_OPTIONAL.items():
        entry = _load_one(key, fname, mtype, feats, group)
        if entry:
            _loaded[key] = entry
    log.info(f"Golden registry: {len(_loaded)} models")


def _load_symbol_models() -> None:
    """Auto-discover valid per-symbol binary models from root + models/ + config/."""
    global _symbol_models
    _symbol_models = {}
    feat_files = []
    for d in SEARCH_DIRS:
        feat_files.extend(glob.glob(os.path.join(d, "*_binary_features.json")))
        # also bare xgb without features json — invent FEATURES_FULL-sized pad
        for xp in glob.glob(os.path.join(d, "*_binary_xgb.json")):
            if not _file_looks_valid(xp):
                continue
            # derive symbol_tf from filename
            base = os.path.basename(xp).replace("_binary_xgb.json", "")
            parts = base.rsplit("_", 1)
            if len(parts) != 2:
                continue
            symbol, tf = parts[0].upper(), parts[1].upper()
            key = (symbol, tf)
            if key in _symbol_models:
                continue
            try:
                m = xgb.Booster()
                m.load_model(xp)
                n = int(m.num_features())
                feats = (FEATURES_FULL + [f"f_{i}" for i in range(len(FEATURES_FULL), n)])[:n]
                _symbol_models[key] = {
                    "model": m,
                    "type": "xgb",
                    "features": feats,
                    "group": "symbol",
                    "path": xp,
                }
            except Exception:
                continue

    for fpath in feat_files:
        if not _file_looks_valid(fpath):
            continue
        try:
            raw = open(fpath, "r", encoding="utf-8", errors="ignore").read().strip()
            if not raw or raw[0] != "{":
                continue
            meta = json.loads(raw)
            symbol = str(meta.get("symbol", "")).upper()
            tf = str(meta.get("timeframe", "D1")).upper()
            features = meta.get("features") or []
            if not symbol or not features:
                continue
            key = (symbol, tf)
            base = fpath[: -len("_features.json")]
            model = None
            mtype = None
            xgb_path = base + "_xgb.json"
            lgb_path = base + "_lgb.pkl"
            if _file_looks_valid(xgb_path):
                try:
                    m = xgb.Booster()
                    m.load_model(xgb_path)
                    model, mtype = m, "xgb"
                except Exception:
                    pass
            if model is None and _file_looks_valid(lgb_path):
                try:
                    model = joblib.load(lgb_path)
                    mtype = "lgb"
                except Exception:
                    pass
            if model is None:
                continue
            _symbol_models[key] = {
                "model": model,
                "type": mtype,
                "features": features,
                "group": "symbol",
                "path": fpath,
            }
        except Exception as e:
            log.debug(f"symbol model skip {fpath}: {e}")
    log.info(f"Symbol specialists: {len(_symbol_models)} combinations")


def load_all() -> None:
    _load_golden()
    _load_symbol_models()


# ── Prediction helpers ────────────────────────────────────────────────────────
def normalize_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    for suf in sorted(STRIP_SUFFIXES, key=len, reverse=True):
        u = suf.upper()
        if s.endswith(u):
            return s[: -len(u)]
    return s


def _clamp01(v: Any) -> Optional[float]:
    try:
        v = float(v)
    except Exception:
        return None
    if v != v:
        return None
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0 if v >= 2 else 0.5
    return v


def _extract(data: dict, features: list) -> np.ndarray:
    vals = []
    for f in features:
        try:
            vals.append(float(data.get(f, 0.0) or 0.0))
        except Exception:
            vals.append(0.0)
    return np.array(vals, dtype=np.float32).reshape(1, -1)


def _prob_buy(key: str, entry: dict, data: dict) -> Optional[float]:
    model = entry["model"]
    feats = list(entry["features"])
    # Align feature vector length to what the model expects
    try:
        if entry["type"] == "xgb":
            want = int(model.num_features())
            if want != len(feats):
                feats = (feats + [f"f_{i}" for i in range(len(feats), want)])[:want]
        elif entry["type"] in ("pkl", "lgb"):
            want = getattr(model, "n_features_in_", None)
            if want and int(want) != len(feats):
                want = int(want)
                feats = (feats + [f"f_{i}" for i in range(len(feats), want)])[:want]
    except Exception:
        pass
    arr = _extract(data, feats)
    try:
        mtype = entry["type"]
        if mtype == "xgb":
            # Booster models trained with feature_names require them on DMatrix
            # (error: "data did not contain feature names, but the following fields are expected")
            try:
                dmat = xgb.DMatrix(arr, feature_names=list(feats))
            except Exception:
                dmat = xgb.DMatrix(arr)
            try:
                raw = np.asarray(model.predict(dmat))
            except Exception as e1:
                # Fallback: retry opposite naming strategy
                msg = str(e1).lower()
                if "feature name" in msg:
                    if "did not contain feature names" in msg:
                        dmat = xgb.DMatrix(arr, feature_names=list(feats))
                    else:
                        dmat = xgb.DMatrix(arr)
                    raw = np.asarray(model.predict(dmat))
                else:
                    raise
            if raw.ndim == 2 and raw.shape[1] >= 2:
                return _clamp01(raw[0, -1])
            return _clamp01(raw.ravel()[0])
        if mtype in ("pkl", "lgb"):
            if hasattr(model, "predict_proba"):
                proba = np.asarray(model.predict_proba(arr))
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    return _clamp01(proba[0, -1])
                return _clamp01(proba.ravel()[0])
            pred = float(np.asarray(model.predict(arr)).ravel()[0])
            return 1.0 if pred > 0 else 0.0
        if mtype == "onnx":
            name = model.get_inputs()[0].name
            outs = model.run(None, {name: arr})
            for out in outs:
                a = np.asarray(out)
                if a.ndim == 2 and a.shape[1] >= 2 and a.dtype.kind == "f":
                    return _clamp01(a[0, -1])
            return _clamp01(np.asarray(outs[0]).ravel()[0])
        if mtype == "torch" and _TORCH:
            with torch.no_grad():
                t = torch.from_numpy(arr).float().unsqueeze(0)
                logits = model(t)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                return _clamp01(float(probs[0, 2]))
    except Exception as e:
        log.warning(f"Predict fail [{key}]: {e}")
        return None
    return None


def _sig(p: float) -> str:
    if p >= 0.55:
        return "buy"
    if p <= 0.45:
        return "sell"
    return "hold"


def _find_symbol_model(symbol: str, timeframe: str) -> Optional[Tuple[str, dict]]:
    key = (symbol, timeframe)
    if key in _symbol_models:
        return f"{symbol}_{timeframe}", _symbol_models[key]
    for tf in ("H4", "H1", "D1", "M15"):
        k = (symbol, tf)
        if k in _symbol_models:
            return f"{symbol}_{tf}", _symbol_models[k]
    return None


def ensemble(data: dict, symbol: str, timeframe: str):
    probs: Dict[str, float] = {}
    for key, entry in _loaded.items():
        p = _prob_buy(key, entry, data)
        if p is not None:
            probs[key] = round(float(p), 6)

    sym_hit = _find_symbol_model(symbol, timeframe)
    if sym_hit:
        sk, sentry = sym_hit
        # Map features from payload (missing → 0)
        p = _prob_buy(sk, sentry, data)
        if p is not None:
            probs[sk] = round(float(p), 6)

    if not probs:
        return 0.5, {}, (None, None, None, None)

    xgb_keys = [k for k in probs if k.startswith("xgb_") or k == "xgb_6"]
    nine_keys = [k for k, e in _loaded.items() if e.get("group") == "nine" and k in probs]
    full_keys = [k for k, e in _loaded.items() if e.get("group") == "full" and k in probs]
    # any remaining symbol specialist keys
    sym_keys = [k for k in probs if k not in _loaded]

    groups = []
    weights = []
    group_sigs: List[Optional[str]] = [None, None, None, None]

    def _mean(keys):
        return float(np.mean([probs[k] for k in keys])) if keys else None

    mx = _mean(xgb_keys)
    if mx is not None:
        groups.append(mx)
        weights.append(W_XGB)
        group_sigs[0] = _sig(mx)
    m9 = _mean(nine_keys)
    if m9 is not None:
        groups.append(m9)
        weights.append(W_NINE)
        group_sigs[1] = _sig(m9)
    mf = _mean(full_keys)
    if mf is not None:
        groups.append(mf)
        weights.append(W_FULL)
        group_sigs[2] = _sig(mf)
    ms = _mean(sym_keys)
    if ms is not None:
        groups.append(ms)
        weights.append(W_SYM)
        group_sigs[3] = _sig(ms)

    total_w = sum(weights) or 1.0
    ens = sum(c * w / total_w for c, w in zip(groups, weights))
    return float(ens), probs, tuple(group_sigs)


def apply_gates(ens_prob: float, group_signals: tuple) -> Tuple[str, float, dict]:
    """
    Statistical gate + optional consensus across active groups.
    HARD RULE: conf < GATE (0.77+) always forces hold — no bypass.
    """
    direction = "buy" if ens_prob >= 0.5 else "sell"
    conf = ens_prob if direction == "buy" else (1.0 - ens_prob)
    # Clamp conf to [0,1]
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    if conf != conf:  # NaN
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    active = [s for s in group_signals if s is not None and s != "hold"]
    consensus_ok = True
    if REQUIRE_CONSENSUS and len(active) >= 2:
        consensus_ok = all(s == direction for s in active)
    # Single active group is OK; zero active → no trade
    if len(active) == 0:
        consensus_ok = False

    # Persistent 0.77+ gate (GATE is at least hard floor)
    stat_ok = conf + 1e-12 >= GATE
    if not stat_ok or not consensus_ok:
        return (
            "hold",
            conf,
            {
                "stat_ok": stat_ok,
                "consensus_ok": consensus_ok,
                "gate": GATE,
                "hard_floor": _HARD_FLOOR,
                "active_groups": active,
            },
        )
    # Final belt-and-suspenders: never emit buy/sell under gate
    if conf < GATE:
        return "hold", conf, {
            "stat_ok": False,
            "consensus_ok": consensus_ok,
            "gate": GATE,
            "hard_floor": _HARD_FLOOR,
            "active_groups": active,
        }
    return direction, conf, {
        "stat_ok": True,
        "consensus_ok": True,
        "gate": GATE,
        "hard_floor": _HARD_FLOOR,
        "active_groups": active,
    }


def _audit(row: dict) -> None:
    try:
        path = os.path.join(LOG_DIR, f"audit_golden_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(
                    [
                        "ts",
                        "symbol",
                        "tf",
                        "signal",
                        "conf",
                        "prob",
                        "n_models",
                        "groups",
                        "gates",
                    ]
                )
            w.writerow(
                [
                    row.get("ts"),
                    row.get("symbol"),
                    row.get("tf"),
                    row.get("signal"),
                    row.get("conf"),
                    row.get("prob"),
                    row.get("n_models"),
                    row.get("groups"),
                    row.get("gates"),
                ]
            )
    except Exception as e:
        log.warning(f"audit failed: {e}")


# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    models_info = {
        k: {
            "type": v["type"],
            "n_features": v.get("n_features", len(v["features"])),
            "group": v.get("group"),
            "mismatched": bool(v.get("mismatched")),
        }
        for k, v in _loaded.items()
    }
    sym = [f"{s}_{t}" for (s, t) in sorted(_symbol_models.keys())]
    return jsonify(
        {
            "status": "running",
            "server": "ai_server_golden_comprehensive",
            "loaded_models": len(_loaded),
            "symbol_models": len(_symbol_models),
            "models": models_info,
            "symbol_list": sym,
            "gate": GATE,
            "require_consensus": REQUIRE_CONSENSUS,
            "weights": {"xgb": W_XGB, "nine": W_NINE, "full": W_FULL, "symbol": W_SYM},
            "load_heavy": LOAD_HEAVY,
            "models_dir": MODELS_DIR,
        }
    )


@app.route("/models", methods=["GET"])
def models_ep():
    return jsonify(
        {
            "golden": list(_loaded.keys()),
            "symbol": [f"{s}_{t}" for (s, t) in sorted(_symbol_models.keys())],
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True) or {}
        # ── Normalize live / MQ5 payloads to golden feature contracts ──
        # Predict/EA may send close without price; FEATURES_6/9 need price + garch_vol
        try:
            px = float(data.get("price") or 0)
        except Exception:
            px = 0.0
        try:
            cl = float(data.get("close") or 0)
        except Exception:
            cl = 0.0
        if px <= 0 and cl > 0:
            data["price"] = cl
            px = cl
        elif px > 0 and cl <= 0:
            data["close"] = px
        try:
            gv = float(data.get("garch_vol") or 0)
        except Exception:
            gv = 0.0
        if gv <= 0:
            # fall back: realized_vol or atr/price as return-vol proxy
            try:
                rv = float(data.get("realized_vol") or 0)
            except Exception:
                rv = 0.0
            if rv > 0:
                data["garch_vol"] = rv
            elif px > 0:
                try:
                    atr0 = float(data.get("atr") or 0)
                except Exception:
                    atr0 = 0.0
                if atr0 > 0:
                    data["garch_vol"] = atr0 / px

        symbol = normalize_symbol(str(data.get("symbol", "")))
        timeframe = str(data.get("timeframe", "M15")).strip().upper()
        if timeframe.startswith("PERIOD_"):
            timeframe = timeframe[7:]
        price = float(data.get("price", 0) or 0)
        atr = float(data.get("atr", 0.001) or 0.001)

        ens_prob, model_probs, group_signals = ensemble(data, symbol, timeframe)
        signal, confidence, gate_info = apply_gates(ens_prob, group_signals)

        # LOSS PROTECTION: never return tradeable signal under 0.77 gate
        if signal in ("buy", "sell") and float(confidence) < GATE:
            log.warning(
                "GATE OVERRIDE blocked %s conf=%.4f < %.2f for %s",
                signal, confidence, GATE, symbol,
            )
            signal = "hold"
            gate_info = dict(gate_info or {})
            gate_info["stat_ok"] = False
            gate_info["blocked_under_gate"] = True

        stop_loss = 0.0
        if signal == "buy" and price > 0 and atr > 0:
            stop_loss = price - 2 * atr
        elif signal == "sell" and price > 0 and atr > 0:
            stop_loss = price + 2 * atr

        log.info(
            f"[{symbol} {timeframe}] prob={ens_prob:.4f} groups={group_signals} "
            f"-> {signal.upper()} conf={confidence:.3f} gate={GATE:.2f} "
            f"stat={gate_info.get('stat_ok')} cons={gate_info.get('consensus_ok')} "
            f"n={len(model_probs)}"
        )
        _audit(
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "symbol": symbol,
                "tf": timeframe,
                "signal": signal,
                "conf": round(confidence, 4),
                "prob": round(ens_prob, 6),
                "n_models": len(model_probs),
                "groups": str(group_signals),
                "gates": json.dumps(gate_info),
            }
        )

        return jsonify(
            {
                "signal": signal,
                "confidence": round(confidence, 4),
                "probability": round(ens_prob, 6),
                "n_models": len(model_probs),
                "stop_loss": round(stop_loss, 5),
                "symbol": symbol,
                "timeframe": timeframe,
                "model_probs": model_probs,
                "group_signals": list(group_signals),
                "min_conf_gate": GATE,
                "gate_passed": signal != "hold",
                "gate_info": gate_info,
                "server": "ai_server_golden_comprehensive",
            }
        )
    except Exception as e:
        log.error(f"predict error: {e}", exc_info=True)
        return (
            jsonify(
                {
                    "signal": "hold",
                    "confidence": 0.0,
                    "probability": 0.5,
                    "n_models": 0,
                    "error": str(e),
                }
            ),
            500,
        )


@app.route("/predict/sentiment", methods=["GET"])
@app.route("/sentiment", methods=["GET"])
def sentiment():
    symbol = normalize_symbol(request.args.get("symbol", ""))
    try:
        from textblob import TextBlob

        texts = {
            "EURUSD": "Bullish trend expected",
            "USDJPY": "Neutral market",
            "XAUUSD": "Bearish sentiment",
            "AUDUSD": "Positive outlook",
            "GBPUSD": "Strong buy signals",
            "USDCAD": "Sell pressure",
            "BTCUSD": "Volatile bullish momentum",
            "XRPUSD": "Speculative neutral",
            "ETHUSD": "Mixed crypto signals",
        }
        score = float(TextBlob(texts.get(symbol, "Neutral")).sentiment.polarity)
    except Exception:
        score = 0.0
    return jsonify({"sentiment": score, "symbol": symbol})


@app.route("/reload", methods=["POST"])
def reload_models():
    load_all()
    return jsonify(
        {
            "status": "reloaded",
            "loaded_models": len(_loaded),
            "symbol_models": len(_symbol_models),
        }
    )


if __name__ == "__main__":
    log.info("=" * 64)
    log.info("FXJEFE Comprehensive Golden AI Server")
    log.info(f"Models dir : {MODELS_DIR}")
    log.info(f"Gate       : {GATE}  hard_floor={_HARD_FLOOR}  consensus={REQUIRE_CONSENSUS}")
    log.info("LOSS GUARD : buy/sell REQUIRES conf>=%.2f + consensus (no stale bypass)", GATE)
    log.info(f"Weights    : xgb={W_XGB} nine={W_NINE} full={W_FULL} symbol={W_SYM}")
    log.info(f"Heavy load : {LOAD_HEAVY}")
    log.info("=" * 64)
    load_all()
    port = int(os.environ.get("AI_SERVER_PORT", "8080"))
    # Prefer waitress for production robustness
    try:
        from waitress import serve

        log.info(f"Serving with waitress on 0.0.0.0:{port}")
        serve(app, host="0.0.0.0", port=port, threads=8)
    except Exception as e:
        log.warning(f"waitress unavailable ({e}); using Flask dev server")
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
