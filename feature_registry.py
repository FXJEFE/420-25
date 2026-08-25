# -*- coding: utf-8 -*-
"""
FXJEFE feature registry — GitHub / open repo policy.

POLICY (strict):
  - Do NOT disable, forbid, flag, or refuse any features.
  - Canonical lists are preferred defaults / documentation only.
  - Extra, experimental, lag, regime, garch_vol, etc. are ALL accepted.
  - validate_* and model_features_match never reject; they only report.
  - Feature arrays are never stripped or destroyed.

Versions still exist so later packs can be added without deleting history.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

# Preferred defaults (documentation + default model wiring) — NOT a ban list
OG_PREDICT_17: Tuple[str, ...] = (
    "price", "atr", "ema_diff", "rsi", "macd_diff", "vwap",
    "price_vwap_diff", "bb_position", "roc", "stochastic", "cci",
    "williams", "momentum", "adx", "rvi", "spread", "sentiment",
)

OG_TRAIN_28: Tuple[str, ...] = (
    "price", "atr", "ema_diff", "rsi", "macd_diff", "vwap",
    "price_vwap_diff", "bb_position", "roc", "stochastic", "cci",
    "williams", "momentum", "realized_vol", "chaikin_vol", "adx",
    "rvi", "obv", "volume_delta", "ad_line", "vol_osc", "supertrend",
    "hma", "ichimoku_tenkan", "sar", "dpo", "spread", "sentiment",
)

# Historical names people used — documented, never refused
DOCUMENTED_ALSO: Tuple[str, ...] = (
    "garch_vol", "future_price", "future_return", "price_change", "regime",
    "price_lag1", "price_lag2", "price_lag3",
    "rsi_lag1", "rsi_lag2", "rsi_lag3",
    "macd_diff_lag", "macd_diff_lag1", "macd_diff_lag2", "macd_diff_lag3",
    "atr_lag", "atr_lag1", "atr_lag2", "atr_lag3",
    "hour_of_day", "day_of_week", "volume_ratio",
)

REGISTRY: Dict[str, Dict[str, Any]] = {
    "v1_april2025": {
        "description": "April 2025 preferred defaults — predict 17 / train 28 (nothing refused)",
        "predict": OG_PREDICT_17,
        "train": OG_TRAIN_28,
        "min_confidence": 0.77,
        "ohlcv": ("open", "high", "low", "close", "volume"),
        "demo": False,
    },
    "demo_og_17": {
        "description": "DEMO pack — same OG lists; all features accepted",
        "predict": OG_PREDICT_17,
        "train": OG_TRAIN_28,
        "min_confidence": 0.50,
        "ohlcv": ("open", "high", "low", "close", "volume"),
        "demo": True,
    },
    "v2_example_expanded": {
        "description": "EXAMPLE expanded pack template — inactive until you fill + activate",
        "predict": (),
        "train": (),
        "min_confidence": 0.77,
        "ohlcv": ("open", "high", "low", "close", "volume"),
        "inactive": True,
        "demo": False,
    },
}

ACTIVE_VERSION: str = "v1_april2025"
DEMO_MODE: bool = False

# Empty on purpose — policy forbids a refuse-list
FORBIDDEN_FEATURES = frozenset()  # nothing is forbidden


def list_versions() -> List[str]:
    return list(REGISTRY.keys())


def get_pack(version: str | None = None) -> Dict[str, Any]:
    ver = version or ACTIVE_VERSION
    if ver not in REGISTRY:
        raise KeyError(f"Unknown feature registry version: {ver}")
    pack = REGISTRY[ver]
    if pack.get("inactive"):
        raise RuntimeError(f"Registry version {ver} is marked inactive")
    return pack


def get_active() -> Dict[str, Any]:
    pack = get_pack(ACTIVE_VERSION)
    predict = tuple(pack["predict"])
    train = tuple(pack["train"])
    # empty predict/train only illegal for inactive templates
    is_demo = bool(pack.get("demo")) or DEMO_MODE
    return {
        "version": ACTIVE_VERSION,
        "description": pack.get("description", ""),
        "predict": predict,
        "train": train,
        "predict_count": len(predict),
        "train_count": len(train),
        "min_confidence": float(pack.get("min_confidence", 0.77)),
        "ohlcv": tuple(pack.get("ohlcv", ("open", "high", "low", "close", "volume"))),
        "forbidden": frozenset(),  # always empty
        "demo": is_demo,
        "policy": "ACCEPT_ALL_FEATURES",
    }


def enable_demo(on: bool = True) -> None:
    global DEMO_MODE, ACTIVE_VERSION
    DEMO_MODE = bool(on)
    ACTIVE_VERSION = "demo_og_17" if on else "v1_april2025"
    _refresh_module_snapshots()


def _refresh_module_snapshots() -> None:
    global PREDICT_FEATURES, TRAIN_FEATURES, PREDICT_COUNT, TRAIN_COUNT
    global MIN_CONFIDENCE, OHLCV_HEADERS
    a = get_active()
    PREDICT_FEATURES = a["predict"]
    TRAIN_FEATURES = a["train"]
    PREDICT_COUNT = a["predict_count"]
    TRAIN_COUNT = a["train_count"]
    MIN_CONFIDENCE = a["min_confidence"]
    OHLCV_HEADERS = a["ohlcv"]


_a0 = get_active()
PREDICT_FEATURES: Tuple[str, ...] = _a0["predict"]
TRAIN_FEATURES: Tuple[str, ...] = _a0["train"]
PREDICT_COUNT: int = _a0["predict_count"]
TRAIN_COUNT: int = _a0["train_count"]
MIN_CONFIDENCE: float = _a0["min_confidence"]
OHLCV_HEADERS: Tuple[str, ...] = _a0["ohlcv"]


def predict_feature_list() -> List[str]:
    return list(PREDICT_FEATURES)


def train_feature_list() -> List[str]:
    return list(TRAIN_FEATURES)


def is_forbidden(name: str) -> bool:
    """Always False — nothing is forbidden."""
    return False


def feature_bytes(mode: str = "predict") -> bytes:
    names = PREDICT_FEATURES if mode == "predict" else TRAIN_FEATURES
    return ("\n".join(names) + "\n").encode("utf-8")


def feature_sha256(mode: str = "predict") -> str:
    return hashlib.sha256(feature_bytes(mode)).hexdigest()


def print_feature_box(mode: str = "predict") -> None:
    raw = feature_bytes(mode)
    names = raw.decode("utf-8").splitlines()
    digest = feature_sha256(mode)
    width = max((len(n) for n in names), default=8) + 4
    bar = "═" * (width + 2)
    print(f"╔{bar}╗")
    print(f"║{f' FXJEFE FEATURES [{mode}] n={len(names)} ':<{width+2}}║")
    print(f"║{f' version={ACTIVE_VERSION} demo={DEMO_MODE} ':<{width+2}}║")
    print(f"║{f' policy=ACCEPT_ALL_FEATURES ':<{width+2}}║")
    print(f"╠{bar}╣")
    for i, n in enumerate(names, 1):
        print(f"║{f' {i:02d}. {n}':<{width+2}}║")
    print(f"╠{bar}╣")
    print(f"║{f' sha256={digest[:16]}… ':<{width+2}}║")
    print(f"║{f' bytes={len(raw)} encoding=utf-8 LF ':<{width+2}}║")
    print(f"╚{bar}╝")
    print(f"[byte-for-byte] {raw!r}")


def validate_predict_keys(keys, *, demo: Optional[bool] = None) -> Dict[str, Any]:
    """
    Report-only. NEVER refuses, forbids, flags hard-fail, or strips keys.
    Always ok=True; keys_preserved = full original list.
    """
    keys_list = list(keys)
    preferred = list(PREDICT_FEATURES)
    extra = sorted(set(keys_list) - set(preferred))
    missing_preferred = sorted(set(preferred) - set(keys_list))
    notes = []
    if extra:
        notes.append(f"extra beyond preferred defaults (accepted): {extra}")
    if missing_preferred:
        notes.append(f"preferred defaults not present (still accepted): {missing_preferred}")
    return {
        "ok": True,
        "accepted": True,
        "demo": bool(demo if demo is not None else (DEMO_MODE or get_active().get("demo"))),
        "errors": [],
        "notes": notes,
        "keys_preserved": keys_list,
        "preferred": preferred,
        "policy": "ACCEPT_ALL_FEATURES",
        "message": "all features accepted — nothing refused or stripped",
    }


def model_features_match(model_features, mode: str = "predict", *, demo: Optional[bool] = None) -> Dict[str, Any]:
    """
    Report-only. Always accepted=True. Never refuse a model feature list.
    """
    preferred = list(PREDICT_FEATURES if mode == "predict" else TRAIN_FEATURES)
    got = list(model_features or [])
    exact = got == preferred
    same_set = set(got) == set(preferred)
    if exact:
        reason = "exact order match to preferred defaults"
    elif same_set:
        reason = "same set as preferred defaults, different order"
    else:
        reason = (
            f"differs from preferred defaults "
            f"(missing_preferred={sorted(set(preferred)-set(got))} "
            f"extra={sorted(set(got)-set(preferred))}) — still accepted"
        )
    return {
        "accepted": True,
        "ok": True,
        "demo": bool(demo if demo is not None else (DEMO_MODE or get_active().get("demo"))),
        "reason": reason,
        "features": got,
        "preferred": preferred,
        "policy": "ACCEPT_ALL_FEATURES",
    }


def register_new_version(
    version_id: str,
    *,
    predict: Tuple[str, ...],
    train: Tuple[str, ...],
    min_confidence: float = 0.77,
    description: str = "",
    activate: bool = False,
    demo: bool = False,
) -> None:
    if version_id in REGISTRY and not REGISTRY[version_id].get("inactive"):
        raise ValueError(f"{version_id} already exists — use a new id")
    REGISTRY[version_id] = {
        "description": description or version_id,
        "predict": tuple(predict),
        "train": tuple(train),
        "min_confidence": float(min_confidence),
        "ohlcv": ("open", "high", "low", "close", "volume"),
        "inactive": not activate,
        "demo": demo,
    }
    if activate:
        global ACTIVE_VERSION
        ACTIVE_VERSION = version_id
        _refresh_module_snapshots()


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        enable_demo(True)
    a = get_active()
    print(
        "ACTIVE", a["version"],
        "predict", a["predict_count"],
        "train", a["train_count"],
        "conf", a["min_confidence"],
        "demo", a["demo"],
        "policy", a["policy"],
    )
    print_feature_box("predict")
    sample = list(OG_PREDICT_17) + ["garch_vol", "regime", "price_lag1"]
    print("validate:", validate_predict_keys(sample))
    print("model_match:", model_features_match(sample))
