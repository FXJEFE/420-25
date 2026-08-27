#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FXJEFE feature lock — April 2025 restore contract (17 features).

ONLY these may be sent to /predict:
  price, atr, ema_diff, rsi, macd_diff, vwap, price_vwap_diff, bb_position,
  roc, stochastic, cci, williams, momentum, adx, rvi, spread, sentiment

Anything in features_forbidden must never appear in the /predict JSON payload.
garch_vol may be computed for live filters but MUST NOT be sent to the model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

try:
    from feature_registry import PREDICT_FEATURES as CANONICAL_17, FORBIDDEN_FEATURES as _FF, MIN_CONFIDENCE
    CANONICAL_17 = list(CANONICAL_17)
    FORBIDDEN_DEFAULT = list(_FF)
except Exception:
    CANONICAL_17 = [
    "price",
    "atr",
    "ema_diff",
    "rsi",
    "macd_diff",
    "vwap",
    "price_vwap_diff",
    "bb_position",
    "roc",
    "stochastic",
    "cci",
    "williams",
    "momentum",
    "adx",
    "rvi",
    "spread",
    "sentiment",
]

FORBIDDEN_DEFAULT = [
    "garch_vol",
    "future_price",
    "future_return",
    "price_change",
    "regime",
    "price_lag1",
    "price_lag2",
    "price_lag3",
    "rsi_lag1",
    "rsi_lag2",
    "rsi_lag3",
    "macd_diff_lag",
    "atr_lag",
    "hour_of_day",
    "day_of_week",
    "volume_ratio",
]


def _project_root() -> Path:
    import sys
    home = Path(os.environ.get("USERPROFILE", str(Path.home()))) if sys.platform == "win32" else Path.home()
    return home / "Documents" / "FXJEFE_Project"


def load_feature_config(root: Optional[Path] = None) -> Dict[str, Any]:
    root = root or _project_root()
    for cand in (root / "config" / "config.json", root / "config.json"):
        if cand.is_file():
            with open(cand, "r", encoding="utf-8") as f:
                return json.load(f)
    return {
        "features": list(CANONICAL_17),
        "features_forbidden": list(FORBIDDEN_DEFAULT),
        "features_count": 17,
    }


def allowed_features(cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    cfg = cfg or load_feature_config()
    feats = list(cfg.get("features") or CANONICAL_17)
    if len(feats) != 17:
        # hard lock: refuse silent expansion
        raise ValueError(
            f"Feature lock requires exactly 17 features, found {len(feats)}. "
            "Edit config/config.json only if intentionally unlocking."
        )
    return feats


def forbidden_features(cfg: Optional[Dict[str, Any]] = None) -> Set[str]:
    cfg = cfg or load_feature_config()
    return set(cfg.get("features_forbidden") or FORBIDDEN_DEFAULT)


def validate_predict_payload(payload: Mapping[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Validate a /predict feature dict.
    Returns {ok, errors, stripped_payload} where stripped has ONLY allowed keys.
    """
    cfg = cfg or load_feature_config()
    allowed = allowed_features(cfg)
    forbidden = forbidden_features(cfg)
    errors: List[str] = []
    keys = set(payload.keys())

    extra = keys - set(allowed)
    missing = set(allowed) - keys
    banned = keys & forbidden

    if banned:
        errors.append(f"forbidden keys present (must not send to /predict): {sorted(banned)}")
    if extra - forbidden:
        # extra that aren't in forbidden list still violate lock
        errors.append(f"non-canonical keys present: {sorted(extra)}")
    if missing:
        errors.append(f"missing required features: {sorted(missing)}")

    stripped = {k: payload[k] for k in allowed if k in payload}
    return {
        "ok": len(errors) == 0 and len(stripped) == 17,
        "errors": errors,
        "stripped_payload": stripped,
        "allowed": allowed,
    }


def build_predict_json(feature_values: Mapping[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a clean /predict body with only the 17 features (order preserved)."""
    cfg = cfg or load_feature_config()
    allowed = allowed_features(cfg)
    body = {}
    missing = []
    for k in allowed:
        if k not in feature_values:
            missing.append(k)
        else:
            body[k] = feature_values[k]
    if missing:
        raise KeyError(f"Cannot build /predict JSON; missing: {missing}")
    # safety strip
    v = validate_predict_payload(body, cfg)
    if not v["ok"]:
        raise ValueError(v["errors"])
    return body


if __name__ == "__main__":
    cfg = load_feature_config()
    print("allowed", allowed_features(cfg))
    print("forbidden sample", sorted(list(forbidden_features(cfg)))[:5], "...")
    demo = {k: 0.0 for k in CANONICAL_17}
    print("validate", validate_predict_payload(demo, cfg))
