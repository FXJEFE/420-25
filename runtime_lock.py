#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FXJEFE runtime lock — enforce ACCEPT_ALL_FEATURES across .env + config.json,
compile confirmed scripts to .pyc, write RUNTIME_LOCK.json.

Exit 0 = 200 OK style green. Exit 1 = not ready.
Never refuses features. Never overwrites user scripts beyond policy files.
"""
from __future__ import annotations

import json
import os
import py_compile
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def log(m: str) -> None:
    print(f"[FXJEFE-LOCK] {m}", flush=True)


def project_root() -> Path:
    env = os.environ.get("FXJEFE_PROJECT_ROOT")
    if env:
        return Path(env)
    if sys.platform == "win32":
        home = Path(os.environ.get("USERPROFILE", str(Path.home())))
    else:
        home = Path.home()
    return home / "Documents" / "FXJEFE_Project"


def load_dotenv(root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in (".env", "env.template"):
        p = root / name
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip('"').strip("'")
        if name == ".env":
            break
    return out


def ensure_env(root: Path) -> Path:
    env_path = root / ".env"
    if env_path.exists():
        log(f"exists_preserved {env_path}")
        # merge missing policy keys only
        existing = load_dotenv(root)
        defaults = {
            "FXJEFE_FEATURE_POLICY": "ACCEPT_ALL_FEATURES",
            "FXJEFE_ALLOW_ALL_FEATURES": "1",
            "FXJEFE_STRIP_FEATURES": "0",
            "FXJEFE_REFUSE_FEATURES": "0",
            "FXJEFE_FEATURE_REGISTRY_VERSION": "v1_april2025",
            "FXJEFE_MIN_CONFIDENCE": "0.77",
            "FXJEFE_AI_SERVER_URL": "http://127.0.0.1:8080",
        }
        lines = env_path.read_text(encoding="utf-8", errors="replace").splitlines()
        changed = False
        for k, v in defaults.items():
            if k not in existing:
                lines.append(f"{k}={v}")
                changed = True
        if changed:
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            log("merged policy keys into .env")
        return env_path

    tpl = root / "env.template"
    body = """# FXJEFE environment — all features permitted
FXJEFE_FEATURE_POLICY=ACCEPT_ALL_FEATURES
FXJEFE_FEATURE_REGISTRY_VERSION=v1_april2025
FXJEFE_DEMO_MODE=0
FXJEFE_MIN_CONFIDENCE=0.77
FXJEFE_AI_SERVER_URL=http://127.0.0.1:8080
FXJEFE_BRIDGE_URL=http://127.0.0.1:8000
FXJEFE_ZMQ_PORT=5555
FXJEFE_ALLOW_ALL_FEATURES=1
FXJEFE_STRIP_FEATURES=0
FXJEFE_REFUSE_FEATURES=0
MT5_ACCOUNT=
MT5_PASSWORD=
MT5_SERVER=
"""
    env_path.write_text(body, encoding="utf-8")
    log(f"wrote {env_path}")
    return env_path


def ensure_config(root: Path) -> Dict[str, Any]:
    cfg_path = root / "config.json"
    policy_block = {
        "feature_policy": "ACCEPT_ALL_FEATURES",
        "allow_all_features": True,
        "strip_features": False,
        "refuse_features": False,
        "features_forbidden": [],
        "feature_registry_locked": True,
        "feature_registry_version": "v1_april2025",
        "min_confidence_threshold": 0.77,
        "ohlcv_columns": ["open", "high", "low", "close", "volume"],
        "policy_note": "Preferred lists are defaults only. All features accepted. Never strip or refuse.",
    }
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
        log(f"exists_preserved {cfg_path} — merging policy keys only")
    else:
        cfg = {
            "project_root": str(root),
            "scripts_path": str(root),
            "models_path": str(root / "models"),
            "data_path": str(root / "data"),
            "data_output_path": str(root / "data"),
            "log_path": str(root / "Logs"),
            "ai_server_url": "http://127.0.0.1:8080",
            "ai_server_script": "ai_server.py",
            "features_predict": [
                "price", "atr", "ema_diff", "rsi", "macd_diff", "vwap",
                "price_vwap_diff", "bb_position", "roc", "stochastic", "cci",
                "williams", "momentum", "adx", "rvi", "spread", "sentiment",
            ],
            "features": [
                "price", "atr", "ema_diff", "rsi", "macd_diff", "vwap",
                "price_vwap_diff", "bb_position", "roc", "stochastic", "cci",
                "williams", "momentum", "realized_vol", "chaikin_vol", "adx",
                "rvi", "obv", "volume_delta", "ad_line", "vol_osc", "supertrend",
                "hma", "ichimoku_tenkan", "sar", "dpo", "spread", "sentiment",
            ],
        }
        log(f"wrote new {cfg_path}")

    cfg.update(policy_block)
    # force empty forbidden always
    cfg["features_forbidden"] = []
    cfg["allow_all_features"] = True
    cfg["strip_features"] = False
    cfg["refuse_features"] = False
    cfg["feature_policy"] = "ACCEPT_ALL_FEATURES"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def compile_confirmed(root: Path) -> List[str]:
    """Compile policy-critical scripts only (confirmed runtime set)."""
    confirmed = [
        "feature_registry.py",
        "runtime_lock.py",
        "path_validate.py",
        "path_loader.py",
        "script_inventory.py",
        "pipelinerun_production.py",
        "secure_strap.py",
        "fxjefe_smart.py",
    ]
    ok = []
    for name in confirmed:
        p = root / name
        if not p.is_file():
            log(f"skip compile missing {name}")
            continue
        try:
            py_compile.compile(str(p), doraise=True)
            ok.append(name)
            log(f"compiled {name}")
        except Exception as e:
            log(f"compile fail {name}: {e}")
    return ok


def verify_registry(root: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(root))
    import feature_registry as reg  # type: ignore

    # force accept-all semantics
    if hasattr(reg, "FORBIDDEN_FEATURES") and len(getattr(reg, "FORBIDDEN_FEATURES")):
        log("WARN: registry still has forbidden set — expected empty for GitHub policy")
    a = reg.get_active() if hasattr(reg, "get_active") else {}
    sample = list(getattr(reg, "PREDICT_FEATURES", a.get("predict", []))) + [
        "garch_vol", "regime", "price_lag1",
    ]
    v = reg.validate_predict_keys(sample) if hasattr(reg, "validate_predict_keys") else {"ok": True}
    m = reg.model_features_match(sample) if hasattr(reg, "model_features_match") else {"accepted": True}
    return {
        "version": a.get("version") or getattr(reg, "ACTIVE_VERSION", "?"),
        "policy": a.get("policy", "ACCEPT_ALL_FEATURES"),
        "validate_ok": bool(v.get("ok", True)),
        "validate_accepted": bool(v.get("accepted", True)),
        "model_accepted": bool(m.get("accepted", True)),
        "forbidden_empty": len(getattr(reg, "FORBIDDEN_FEATURES", ())) == 0,
        "keys_preserved_len": len(v.get("keys_preserved", sample)),
        "predict_count": a.get("predict_count") or getattr(reg, "PREDICT_COUNT", 0),
        "train_count": a.get("train_count") or getattr(reg, "TRAIN_COUNT", 0),
    }


def main() -> int:
    root = project_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "state").mkdir(parents=True, exist_ok=True)
    (root / "production").mkdir(parents=True, exist_ok=True)

    log(f"root = {root}")
    ensure_env(root)
    cfg = ensure_config(root)
    compiled = compile_confirmed(root)

    checks: Dict[str, Any] = {
        "env_policy": None,
        "config_policy": cfg.get("feature_policy"),
        "allow_all": cfg.get("allow_all_features") is True,
        "refuse_false": cfg.get("refuse_features") is False,
        "strip_false": cfg.get("strip_features") is False,
        "forbidden_empty": cfg.get("features_forbidden") == [],
        "compiled": compiled,
        "registry": None,
    }

    env = load_dotenv(root)
    checks["env_policy"] = env.get("FXJEFE_FEATURE_POLICY", "")
    checks["env_allow"] = env.get("FXJEFE_ALLOW_ALL_FEATURES", "") in ("1", "true", "True", "yes")

    try:
        checks["registry"] = verify_registry(root)
    except Exception as e:
        log(f"registry verify failed: {e}")
        checks["registry"] = {"error": str(e)}

    # 200 OK style gate
    reg = checks.get("registry") or {}
    green = all([
        checks["config_policy"] == "ACCEPT_ALL_FEATURES",
        checks["allow_all"],
        checks["refuse_false"],
        checks["strip_false"],
        checks["forbidden_empty"],
        checks["env_policy"] == "ACCEPT_ALL_FEATURES",
        checks["env_allow"],
        reg.get("validate_ok") is True,
        reg.get("model_accepted") is True,
        reg.get("forbidden_empty") is True,
        "feature_registry.py" in compiled,
    ])

    status = 200 if green else 503
    lock = {
        "status": status,
        "status_text": "OK" if green else "NOT_READY",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "ACCEPT_ALL_FEATURES",
        "checks": checks,
        "message": (
            "Runtime locked — all features permitted; preferred lists are defaults only."
            if green else
            "Runtime not ready — fix checks below before repo completion."
        ),
    }
    out = root / "production" / "RUNTIME_LOCK.json"
    out.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    (root / "state" / "runtime_lock_latest.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")

    log(f"status={status} {lock['status_text']}")
    log(f"wrote {out}")
    if not green:
        log(f"checks={json.dumps(checks, indent=2)}")
        return 1
    log("200 OK — policy env+config+registry+pyc confirmed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
