#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FXJEFE Golden Server — hardened
- Resolve config.json safely (project root, env, cwd)
- Optional schema + SHA256 validation of config
- Feature vector length mismatch / OOB guards on every model predict
- ACCEPT_ALL features: missing → 0.0; extras ignored; never crash on length
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from flask import Flask, jsonify, request
except ImportError:
    print("pip install flask")
    raise

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def project_root() -> Path:
    env = os.environ.get("FXJEFE_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    if sys.platform == "win32":
        home = Path(os.environ.get("USERPROFILE", str(Path.home())))
    else:
        home = Path.home()
    return (home / "Documents" / "FXJEFE_Project").resolve()


def find_config() -> Path:
    """Locate config.json: env → project root → cwd → alongside this file."""
    candidates = []
    if os.environ.get("FXJEFE_CONFIG"):
        candidates.append(Path(os.environ["FXJEFE_CONFIG"]).expanduser())
    root = project_root()
    candidates.append(root / "config.json")
    candidates.append(Path.cwd() / "config.json")
    candidates.append(Path(__file__).resolve().parent / "config.json")
    for p in candidates:
        try:
            if p.is_file():
                return p.resolve()
        except OSError:
            continue
    return (root / "config.json").resolve()


CONFIG_SCHEMA_REQUIRED = ("feature_policy", "min_confidence_threshold")
CONFIG_SCHEMA_TYPES = {
    "feature_policy": str,
    "min_confidence_threshold": (int, float),
    "features": list,
    "api_port": (int, type(None)),
    "allow_all_features": (bool, type(None)),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_config_schema(cfg: dict) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(cfg, dict):
        return False, ["config root must be object"]
    for key in CONFIG_SCHEMA_REQUIRED:
        if key not in cfg:
            errors.append(f"missing required key: {key}")
    for key, types in CONFIG_SCHEMA_TYPES.items():
        if key in cfg and cfg[key] is not None and not isinstance(cfg[key], types):
            errors.append(f"key {key} has wrong type: {type(cfg[key]).__name__}")
    feats = cfg.get("features")
    if feats is not None and not all(isinstance(x, str) for x in feats):
        errors.append("features must be a list of strings")
    return len(errors) == 0, errors


def validate_config_sha256(path: Path, cfg: dict) -> Tuple[bool, str]:
    """If checksums.json lists config.json, enforce match. Else soft OK."""
    root = path.parent
    sums_path = root / "checksums.json"
    if not sums_path.is_file():
        return True, "no checksums.json — skip sha256 gate"
    try:
        sums = json.loads(sums_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"checksums.json parse error: {e}"
    want = sums.get("config.json") or sums.get(str(path.name))
    if not want:
        return True, "config.json not in checksums — skip"
    got = sha256_file(path)
    if got != want:
        return False, f"config.json sha256 mismatch want={want[:16]}… got={got[:16]}…"
    return True, "config.json sha256 OK"


def load_config() -> Tuple[dict, dict]:
    """
    Returns (config, meta) where meta has path, schema_ok, sha_ok, errors.
    Never raises for missing optional keys.
    """
    meta: Dict[str, Any] = {
        "path": None,
        "schema_ok": False,
        "sha_ok": True,
        "errors": [],
        "warnings": [],
    }
    path = find_config()
    meta["path"] = str(path)
    if not path.is_file():
        meta["errors"].append(f"config not found at {path}")
        meta["warnings"].append("using empty config defaults")
        return {}, meta
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        meta["errors"].append(f"JSON parse error: {e}")
        return {}, meta
    except OSError as e:
        meta["errors"].append(f"read error: {e}")
        return {}, meta

    ok, errs = validate_config_schema(cfg)
    meta["schema_ok"] = ok
    meta["errors"].extend(errs)

    sha_ok, sha_msg = validate_config_sha256(path, cfg)
    meta["sha_ok"] = sha_ok
    if not sha_ok:
        meta["errors"].append(sha_msg)
    else:
        meta["warnings"].append(sha_msg)

    return cfg if isinstance(cfg, dict) else {}, meta


# ---------------------------------------------------------------------------
# Feature vector (length-safe)
# ---------------------------------------------------------------------------

FEATURES_29: List[str] = [
    "price", "atr", "ema_diff", "rsi", "garch_vol", "macd_diff",
    "vwap", "price_vwap_diff", "bb_position", "roc", "stochastic",
    "cci", "williams", "momentum", "realized_vol", "chaikin_vol",
    "adx", "rvi", "obv", "volume_delta", "ad_line", "vol_osc",
    "supertrend", "hma", "ichimoku_tenkan", "sar", "dpo", "spread", "sentiment",
]
FEATURES_9 = FEATURES_29[:9]
FEATURES_6 = ["price", "atr", "ema_diff", "rsi", "macd_diff", "vwap"]

GATE = 0.77
HOST = "127.0.0.1"
PORT = 8080

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GOLDEN] %(levelname)s %(message)s",
)
log = logging.getLogger("golden")

app = Flask(__name__)
LOADED: Dict[str, Any] = {}
OPTIONAL: Dict[str, Any] = {}
CONFIG_META: Dict[str, Any] = {}


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def vector_from_payload(
    data: dict,
    feature_names: List[str],
    expected_len: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build model input row.
    - Missing keys → 0.0 (ACCEPT_ALL, no refuse)
    - Extra keys in payload ignored
    - expected_len: pad or truncate with explicit mismatch report (never OOB into model)
    """
    info: Dict[str, Any] = {
        "requested_len": len(feature_names),
        "expected_len": expected_len,
        "mismatch": False,
        "padded": False,
        "truncated": False,
        "missing_keys": [],
    }
    row: List[float] = []
    for name in feature_names:
        if name not in data:
            info["missing_keys"].append(name)
        row.append(safe_float(data.get(name), 0.0))

    target = expected_len if expected_len is not None else len(feature_names)
    if len(row) < target:
        row.extend([0.0] * (target - len(row)))
        info["padded"] = True
        info["mismatch"] = True
    elif len(row) > target:
        row = row[:target]
        info["truncated"] = True
        info["mismatch"] = True

    arr = np.asarray(row, dtype=np.float64).reshape(1, -1)
    # Final OOB guard
    if arr.shape[1] != target:
        info["mismatch"] = True
        fixed = np.zeros((1, target), dtype=np.float64)
        n = min(arr.shape[1], target)
        fixed[0, :n] = arr[0, :n]
        arr = fixed
        log.warning("vector length force-fixed to %s", target)
    info["final_len"] = int(arr.shape[1])
    return arr, info


def infer_n_features(model: Any, fallback: int) -> int:
    """Best-effort feature count from sklearn / xgb / generic."""
    for attr in ("n_features_in_", "n_features_"):
        if hasattr(model, attr):
            try:
                return int(getattr(model, attr))
            except Exception:
                pass
    if hasattr(model, "feature_names_in_"):
        try:
            return len(list(model.feature_names_in_))
        except Exception:
            pass
    return fallback


def _joblib_load(path: Path):
    import joblib
    return joblib.load(path)


def _xgb_load(path: Path):
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def register_models(cfg: dict) -> None:
    models_dir = Path(cfg.get("models_path") or project_root())
    if not models_dir.is_dir():
        models_dir = project_root()

    core = {
        "xgb_6": ("xgboost_model.json", "xgb", FEATURES_6),
        "ensemble_9a": ("ensamble_model.pkl", "pkl", FEATURES_9),
        "rf_9b": ("my_model (2).pkl", "pkl", FEATURES_9),
        "voting_9c": ("my_model (3).pkl", "pkl", FEATURES_9),
        "rf_9d": ("my_model - Copy.pkl", "pkl", FEATURES_9),
        "rf_28": ("my_model.pkl", "pkl", FEATURES_29[:28]),
    }
    for key, (fname, kind, feats) in core.items():
        path = models_dir / fname
        if not path.is_file():
            path = project_root() / fname
        if not path.is_file():
            log.warning("core missing %s (%s)", key, fname)
            continue
        try:
            if kind == "xgb":
                model = _xgb_load(path)
            else:
                model = _joblib_load(path)
            n_exp = infer_n_features(model, len(feats))
            LOADED[key] = {
                "model": model,
                "kind": kind,
                "features": feats,
                "n_features_expected": n_exp,
            }
            log.info("loaded core %s n_features_expected=%s file=%s", key, n_exp, path.name)
        except Exception as e:
            log.warning("failed core %s: %s", key, e)


def predict_one(entry: dict, data: dict) -> Tuple[str, float, Dict[str, Any]]:
    """Length-safe predict. Never raises out-of-bounds into caller."""
    model = entry["model"]
    kind = entry["kind"]
    feats = list(entry["features"])
    n_exp = int(entry.get("n_features_expected") or len(feats))
    diag: Dict[str, Any] = {"model_kind": kind}

    try:
        X, vinfo = vector_from_payload(data, feats, expected_len=n_exp)
        diag["vector"] = vinfo
        if vinfo.get("mismatch"):
            log.warning(
                "feature length mismatch kind=%s final=%s expected=%s padded=%s truncated=%s",
                kind, vinfo.get("final_len"), n_exp, vinfo.get("padded"), vinfo.get("truncated"),
            )

        if kind == "xgb":
            import xgboost as xgb
            # DMatrix: use numeric only; feature names optional if length matches
            try:
                dmat = xgb.DMatrix(X)
            except Exception:
                dmat = xgb.DMatrix(np.zeros((1, n_exp), dtype=np.float64))
            proba = float(model.predict(dmat)[0])
            return ("buy" if proba >= 0.5 else "sell"), (proba if proba >= 0.5 else 1.0 - proba), diag

        if hasattr(model, "predict_proba"):
            # Guard sklearn shape
            if X.shape[1] != n_exp:
                X = vector_from_payload(data, feats, expected_len=n_exp)[0]
            proba_arr = model.predict_proba(X)[0]
            classes = list(getattr(model, "classes_", list(range(len(proba_arr)))))
            if len(proba_arr) == 0:
                return "hold", 0.0, diag
            buy_idx = classes.index(1) if 1 in classes else int(np.argmax(proba_arr))
            buy_idx = max(0, min(buy_idx, len(proba_arr) - 1))  # OOB guard
            proba = float(proba_arr[buy_idx])
            pred = model.predict(X)[0]
            label = str(pred).lower()
            if label in ("1", "buy", "long"):
                return "buy", proba, diag
            if label in ("0", "-1", "sell", "short"):
                return "sell", (proba if proba <= 0.5 else 1.0 - proba), diag
            return ("buy" if proba >= 0.5 else "sell"), max(proba, 1.0 - proba), diag

        pred = model.predict(X)[0]
        label = str(pred).lower()
        if label in ("1", "buy", "long"):
            return "buy", 0.6, diag
        if label in ("0", "-1", "sell", "short"):
            return "sell", 0.6, diag
        return "hold", 0.5, diag
    except Exception as e:
        log.warning("predict_one error: %s", e)
        diag["error"] = str(e)
        return "hold", 0.0, diag


def group_vote(keys: List[str], store: dict, data: dict) -> Tuple[str, float, int, List[dict]]:
    votes, probs, diags = [], [], []
    for k in keys:
        if k not in store:
            continue
        sig, p, d = predict_one(store[k], data)
        d = dict(d)
        d["key"] = k
        diags.append(d)
        if sig != "hold":
            votes.append(sig)
            probs.append(p)
    if not votes:
        return "hold", 0.0, 0, diags
    buy_n = sum(1 for v in votes if v == "buy")
    sell_n = sum(1 for v in votes if v == "sell")
    if buy_n > sell_n:
        return "buy", float(np.mean([p for v, p in zip(votes, probs) if v == "buy"])), len(votes), diags
    if sell_n > buy_n:
        return "sell", float(np.mean([p for v, p in zip(votes, probs) if v == "sell"])), len(votes), diags
    return "hold", float(np.mean(probs)), len(votes), diags


def consensus_predict(data: dict, gate: float) -> dict:
    xgb_keys = [k for k in LOADED if k.startswith("xgb")]
    nine_keys = [k for k in LOADED if k in ("ensemble_9a", "rf_9b", "voting_9c", "rf_9d")]
    full_keys = [k for k in LOADED if k == "rf_28"]

    g1, p1, n1, d1 = group_vote(xgb_keys, LOADED, data)
    g2, p2, n2, d2 = group_vote(nine_keys, LOADED, data)
    g3, p3, n3, d3 = group_vote(full_keys, LOADED, data)

    groups = [g1, g2, g3]
    active = [s for s in groups if s != "hold"]
    n_models = n1 + n2 + n3
    mismatches = [
        d for d in (d1 + d2 + d3)
        if (d.get("vector") or {}).get("mismatch")
    ]

    if len(active) == 3 and len(set(active)) == 1:
        final_signal = active[0]
        conf = float(np.mean([p for p in (p1, p2, p3) if p > 0] or [0.0]))
    else:
        final_signal = "hold"
        conf = float(np.mean([p for p in (p1, p2, p3) if p > 0] or [0.0]))

    if final_signal != "hold" and conf < gate:
        final_signal = "hold"

    atr = safe_float(data.get("atr"))
    price = safe_float(data.get("price"))
    stop_loss = None
    if atr > 0 and price > 0 and final_signal in ("buy", "sell"):
        stop_loss = round(price - 1.5 * atr if final_signal == "buy" else price + 1.5 * atr, 5)

    return {
        "signal": final_signal,
        "confidence": round(conf, 4),
        "probability": round(conf, 4),
        "n_models": n_models,
        "stop_loss": stop_loss,
        "groups": {"xgb": g1, "nine": g2, "full": g3},
        "gate": gate,
        "features_count": 29,
        "symbol": data.get("symbol"),
        "vector_mismatches": len(mismatches),
        "config_path": CONFIG_META.get("path"),
        "config_schema_ok": CONFIG_META.get("schema_ok"),
        "config_sha_ok": CONFIG_META.get("sha_ok"),
    }


@app.route("/health", methods=["GET"])
def health():
    cfg, meta = load_config()
    gate = float(cfg.get("min_confidence_threshold", GATE)) if cfg else GATE
    return jsonify({
        "status": "running",
        "loaded_models": list(LOADED.keys()),
        "core_count": len(LOADED),
        "gate": gate,
        "features_count": 29,
        "port": PORT,
        "policy": "ACCEPT_ALL_FEATURES",
        "config_path": meta.get("path"),
        "config_schema_ok": meta.get("schema_ok"),
        "config_sha_ok": meta.get("sha_ok"),
        "config_errors": meta.get("errors") or [],
        "config_warnings": meta.get("warnings") or [],
    })


@app.route("/predict", methods=["POST"])
def predict():
    cfg, meta = load_config()
    global CONFIG_META
    CONFIG_META = meta
    gate = float(cfg.get("min_confidence_threshold", GATE)) if cfg else GATE
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        return jsonify({"error": "JSON object required", "signal": "hold"}), 400
    try:
        out = consensus_predict(data, gate)
        log.info(
            "predict symbol=%s signal=%s conf=%.3f mismatches=%s",
            out.get("symbol"), out["signal"], out["confidence"], out.get("vector_mismatches"),
        )
        return jsonify(out)
    except Exception as e:
        log.error("predict failed: %s\n%s", e, traceback.format_exc())
        return jsonify({
            "signal": "hold",
            "confidence": 0.0,
            "error": str(e),
            "config_path": meta.get("path"),
            "config_schema_ok": meta.get("schema_ok"),
            "config_sha_ok": meta.get("sha_ok"),
        }), 500


def main():
    global GATE, PORT, CONFIG_META
    cfg, meta = load_config()
    CONFIG_META = meta
    log.info("config path=%s schema_ok=%s sha_ok=%s errors=%s",
             meta.get("path"), meta.get("schema_ok"), meta.get("sha_ok"), meta.get("errors"))
    if meta.get("errors") and not meta.get("schema_ok"):
        log.warning("config schema issues — continuing with defaults where needed")
    GATE = float(cfg.get("min_confidence_threshold", 0.77)) if cfg else 0.77
    PORT = int(cfg.get("api_port") or 8080)
    register_models(cfg or {})
    log.info("GOLDEN on %s:%s gate=%.2f models=%s", HOST, PORT, GATE, list(LOADED))
    app.run(host=HOST, port=PORT, threaded=True, debug=False)


if __name__ == "__main__":
    main()
