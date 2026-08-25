#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated FXJEFE config.json validation.
Exit 0 = OK, 1 = errors, 2 = fatal (missing/unreadable).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path.home() / "Documents" / "FXJEFE_Project"
if not ROOT.is_dir():
    ROOT = Path(__file__).resolve().parent

REQUIRED_KEYS = {
    "feature_policy": str,
    "min_confidence_threshold": (int, float),
}

RECOMMENDED_KEYS = {
    "features": list,
    "historical_csv": str,
    "log_path": str,
    "data_path": str,
    "models_path": str,
    "ai_server_url": str,
    "api_port": (int, str),
    "talib_defaults": dict,
    "allow_all_features": (bool, int, str),
}

ALLOWED_POLICIES = {
    "ACCEPT_ALL_FEATURES",
    "ACCEPT_ALL",
    "STRICT",
    "LOCKED",
}


def config_path() -> Path:
    env = os.environ.get("FXJEFE_CONFIG", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (ROOT / "config.json").resolve()


def load(path: Path) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    if not path.is_file():
        return {}, [f"missing file: {path}"]
    raw = path.read_text(encoding="utf-8")
    if raw.startswith("\ufeff"):
        errors.append("BOM present — strip UTF-8 BOM")
        raw = raw.lstrip("\ufeff")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return {}, [f"invalid JSON: {e}"]
    if not isinstance(data, dict):
        return {}, ["config root must be a JSON object"]
    return data, errors


def check_types(data: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    for key, typ in REQUIRED_KEYS.items():
        if key not in data:
            errs.append(f"missing required key: {key}")
            continue
        if not isinstance(data[key], typ):
            errs.append(f"key {key!r} type {type(data[key]).__name__}, expected {typ}")
    return errs


def check_values(data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errs: List[str] = []
    warns: List[str] = []
    pol = data.get("feature_policy")
    if isinstance(pol, str) and pol not in ALLOWED_POLICIES:
        warns.append(f"feature_policy {pol!r} not in {sorted(ALLOWED_POLICIES)}")
    gate = data.get("min_confidence_threshold")
    if isinstance(gate, (int, float)):
        if not (0.0 <= float(gate) <= 1.0):
            errs.append(f"min_confidence_threshold out of range: {gate}")
        if float(gate) < 0.5:
            warns.append(f"gate {gate} is low for production")
    feats = data.get("features")
    if isinstance(feats, list):
        if len(feats) == 0:
            warns.append("features list empty")
        if len(feats) != len(set(map(str, feats))):
            warns.append("features list has duplicates")
    for key in ("log_path", "data_path", "models_path", "scripts_path", "project_root"):
        val = data.get(key)
        if val is None:
            continue
        if isinstance(val, str) and val.strip() == "":
            warns.append(f"{key} is empty string (will use project defaults)")
        if isinstance(val, str) and (":\\" in val or val.startswith("C:")):
            warns.append(f"{key} looks like Windows path on this host: {val}")
    return errs, warns


def check_paths(data: Dict[str, Any], root: Path) -> Tuple[List[str], List[str]]:
    errs: List[str] = []
    warns: List[str] = []
    hist = data.get("historical_csv") or "data/raw_ohlcv.csv"
    if isinstance(hist, str):
        p = Path(hist)
        if not p.is_absolute():
            p = root / p
        if not p.is_file():
            warns.append(f"historical_csv not found (ok until Optuna/train): {p}")
    log_p = data.get("log_path") or "Logs"
    if isinstance(log_p, str) and log_p.strip():
        lp = Path(log_p) if Path(log_p).is_absolute() else root / log_p
        try:
            lp.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errs.append(f"cannot create log_path {lp}: {e}")
    return errs, warns


def main() -> int:
    path = config_path()
    print(f"[CONFIG-VALIDATE] path={path}")
    data, load_errs = load(path)
    if load_errs and not data:
        for e in load_errs:
            print(f"[ERROR] {e}")
        return 2

    errors: List[str] = list(load_errs)
    warnings: List[str] = []
    errors.extend(check_types(data))
    e2, w2 = check_values(data)
    errors.extend(e2)
    warnings.extend(w2)
    e3, w3 = check_paths(data, ROOT)
    errors.extend(e3)
    warnings.extend(w3)

    for key in RECOMMENDED_KEYS:
        if key not in data:
            warnings.append(f"recommended key missing: {key}")

    for w in warnings:
        print(f"[WARN] {w}")
    for e in errors:
        print(f"[ERROR] {e}")

    summary = {
        "path": str(path),
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "policy": data.get("feature_policy"),
        "gate": data.get("min_confidence_threshold"),
        "feature_count": len(data["features"]) if isinstance(data.get("features"), list) else None,
    }
    out = ROOT / "production" / "config_validate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[CONFIG-VALIDATE] wrote {out}")
    if errors:
        print("[CONFIG-VALIDATE] FAIL")
        return 1
    print("[CONFIG-VALIDATE] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
