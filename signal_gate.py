#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FXJEFE signal gate
==================
Default: everything auto-enabled (ACCEPT_ALL_FEATURES, servers, pipeline).

Signals are emitted ONLY when ALL of these hold:
  1. Model is loaded
  2. Model feature set matches the current EA feature set
  3. Server feature set matches the same set
  4. Predict.mq5 / generatefeatures.mq5 (or GenerateFeatures.mq5) agree

Demo is a COMMENT / label only — does not disable systems.

Usage:
  from signal_gate import can_emit_signal, load_featureset_bundle
  ok, info = can_emit_signal(model_features=..., ea_features=..., server_features=..., mq5_features=...)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Auto-enabled defaults (repo policy)
AUTO_ENABLED = True
# DEMO: comment-only marker — does not turn systems off
DEMO_COMMENT = (
    "# DEMO: optional label for research runs. "
    "Does not disable models, server, EA, or feature acceptance."
)


def _norm(names: Optional[Iterable[str]]) -> List[str]:
    if not names:
        return []
    return [str(x).strip() for x in names if str(x).strip()]


def same_featureset(a: Sequence[str], b: Sequence[str], *, order_matters: bool = False) -> bool:
    if order_matters:
        return list(a) == list(b)
    return set(a) == set(b)


def load_featureset_bundle(root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load preferred + optional live feature sets from config / state.
    Auto-enables missing policy keys.
    """
    if root is None:
        home = Path(os.environ.get("USERPROFILE", str(Path.home()))) if os.name == "nt" else Path.home()
        root = Path(os.environ.get("FXJEFE_PROJECT_ROOT", home / "Documents" / "FXJEFE_Project"))

    cfg: Dict[str, Any] = {}
    cfg_path = root / "config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}

    # auto-enable policy defaults
    cfg.setdefault("feature_policy", "ACCEPT_ALL_FEATURES")
    cfg.setdefault("allow_all_features", True)
    cfg.setdefault("auto_enabled", True)
    cfg.setdefault("refuse_features", False)
    cfg.setdefault("strip_features", False)
    cfg.setdefault("features_forbidden", [])
    cfg.setdefault("signal_require_featureset_match", True)
    cfg.setdefault("signal_require_model_loaded", True)

    preferred_predict = _norm(cfg.get("features_predict") or cfg.get("features") or [])
    preferred_train = _norm(cfg.get("features") or preferred_predict)

    # optional live snapshots written by EA/server sync
    state = root / "state"
    live = {}
    for name in ("ea_featureset.json", "server_featureset.json", "mq5_featureset.json", "model_featureset.json"):
        p = state / name
        if p.is_file():
            try:
                live[name] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                live[name] = {}

    return {
        "root": str(root),
        "config": cfg,
        "preferred_predict": preferred_predict,
        "preferred_train": preferred_train,
        "live": live,
        "demo_comment": DEMO_COMMENT,
        "auto_enabled": True,
    }


def can_emit_signal(
    *,
    model_loaded: bool,
    model_features: Optional[Sequence[str]] = None,
    ea_features: Optional[Sequence[str]] = None,
    server_features: Optional[Sequence[str]] = None,
    mq5_features: Optional[Sequence[str]] = None,
    predict_mq5_features: Optional[Sequence[str]] = None,
    generatefeatures_mq5_features: Optional[Sequence[str]] = None,
    order_matters: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Gate for sending trading signals.

    Required:
      - model_loaded
      - model features match EA featureset
      - model/EA match server featureset
      - model/EA match Predict.mq5 + GenerateFeatures.mq5 featureset

    Feature *acceptance* remains open (ACCEPT_ALL). This gate only controls
    whether a live signal is allowed out.
    """
    info: Dict[str, Any] = {
        "auto_enabled": AUTO_ENABLED,
        "demo_comment": DEMO_COMMENT,
        "model_loaded": bool(model_loaded),
        "emit": False,
        "reasons": [],
        "matches": {},
    }

    if not model_loaded:
        info["reasons"].append("model not loaded")
        return False, info

    model_f = _norm(model_features)
    ea_f = _norm(ea_features)
    server_f = _norm(server_features)

    # Merge mq5 sources: explicit args or combined mq5_features
    pred_f = _norm(predict_mq5_features) or _norm(mq5_features)
    gen_f = _norm(generatefeatures_mq5_features) or _norm(mq5_features)

    if not model_f:
        info["reasons"].append("model feature set empty")
        return False, info
    if not ea_f:
        info["reasons"].append("EA feature set empty / unknown")
        return False, info
    if not server_f:
        info["reasons"].append("server feature set empty / unknown")
        return False, info
    if not pred_f:
        info["reasons"].append("Predict.mq5 feature set empty / unknown")
        return False, info
    if not gen_f:
        info["reasons"].append("GenerateFeatures.mq5 feature set empty / unknown")
        return False, info

    checks = {
        "model_vs_ea": same_featureset(model_f, ea_f, order_matters=order_matters),
        "model_vs_server": same_featureset(model_f, server_f, order_matters=order_matters),
        "ea_vs_server": same_featureset(ea_f, server_f, order_matters=order_matters),
        "model_vs_predict_mq5": same_featureset(model_f, pred_f, order_matters=order_matters),
        "model_vs_generatefeatures_mq5": same_featureset(model_f, gen_f, order_matters=order_matters),
        "predict_vs_generatefeatures_mq5": same_featureset(pred_f, gen_f, order_matters=order_matters),
    }
    info["matches"] = checks
    info["sets"] = {
        "model": model_f,
        "ea": ea_f,
        "server": server_f,
        "predict_mq5": pred_f,
        "generatefeatures_mq5": gen_f,
    }

    if not all(checks.values()):
        bad = [k for k, v in checks.items() if not v]
        info["reasons"].append(f"featureset mismatch: {bad}")
        return False, info

    info["emit"] = True
    info["reasons"].append("model loaded + featureset aligned (EA, server, Predict.mq5, GenerateFeatures.mq5)")
    return True, info


def write_featureset_snapshot(root: Path, kind: str, features: Sequence[str], extra: Optional[dict] = None) -> Path:
    """Persist a live featureset for gate checks (ea|server|mq5|model)."""
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    name = {
        "ea": "ea_featureset.json",
        "server": "server_featureset.json",
        "mq5": "mq5_featureset.json",
        "predict_mq5": "mq5_featureset.json",
        "generatefeatures_mq5": "mq5_featureset.json",
        "model": "model_featureset.json",
    }.get(kind, f"{kind}_featureset.json")
    payload = {"kind": kind, "features": _norm(features), "count": len(_norm(features))}
    if extra:
        payload.update(extra)
    path = state / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    # self-check: aligned → emit; mismatched → block signal only
    feats = ["price", "atr", "rsi"]
    ok, info = can_emit_signal(
        model_loaded=True,
        model_features=feats,
        ea_features=feats,
        server_features=feats,
        predict_mq5_features=feats,
        generatefeatures_mq5_features=feats,
    )
    print("aligned", ok, info["reasons"])
    ok2, info2 = can_emit_signal(
        model_loaded=True,
        model_features=feats,
        ea_features=feats + ["garch_vol"],
        server_features=feats,
        predict_mq5_features=feats,
        generatefeatures_mq5_features=feats,
    )
    print("mismatch", ok2, info2["reasons"])
    print(DEMO_COMMENT)
