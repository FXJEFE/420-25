# -*- coding: utf-8 -*-
"""
FXJEFE Feature Registry — locked April 2025 contract + original 2025 models
NEVER mutate the v1 keys. New experiments get a new version key.
"""
from __future__ import annotations
from typing import List, Dict

# === ORIGINAL 2025 WORKING MODELS (protected) ===
FEATURES_OG_7_9 = [          # XGBOOST_model.json family
    "price", "atr", "ema_diff", "rsi", "macd_diff",
    "bb_position", "roc", "stochastic", "cci"
]

FEATURES_OG_27_28 = [        # my_model.pkl / ensemble family (27 if price omitted)
    "price", "atr", "ema_diff", "rsi", "garch_vol", "macd_diff",
    "vwap", "price_vwap_diff", "bb_position", "roc", "stochastic",
    "cci", "williams", "momentum", "realized_vol", "chaikin_vol",
    "adx", "rvi", "obv", "volume_delta", "ad_line", "vol_osc",
    "supertrend", "hma", "ichimoku_tenkan", "sar", "dpo", "spread"
]

# === CURRENT LOCKED CONTRACT (v1_april2025) ===
FEATURES_TRAIN_28 = FEATURES_OG_27_28[:]          # train contract
FEATURES_PREDICT_17 = FEATURES_TRAIN_28[:17]      # live /predict contract

MIN_CONFIDENCE = 0.77
MAX_CONFIDENCE = 0.9888

# Forbidden for NEW models only
FORBIDDEN_NEW = {"future_price", "future_return", "label_lookahead"}

# Allowed for OG models only (legacy)
LEGACY_ALLOWED = {"garch_vol", "volume_delta", "ad_line", "vol_osc", "chaikin_vol"}

REGISTRY: Dict[str, Dict] = {
    "v1_april2025": {
        "train": FEATURES_TRAIN_28,
        "predict": FEATURES_PREDICT_17,
        "min_confidence": MIN_CONFIDENCE,
        "max_confidence": MAX_CONFIDENCE,
    },
    "og_7_9": {
        "train": FEATURES_OG_7_9,
        "predict": FEATURES_OG_7_9,
        "min_confidence": 0.77,
    },
    "og_27_28": {
        "train": FEATURES_OG_27_28,
        "predict": FEATURES_OG_27_28[:17] if len(FEATURES_OG_27_28) > 17 else FEATURES_OG_27_28,
        "min_confidence": 0.77,
    },
}

def get_features(version: str = "v1_april2025", mode: str = "train") -> List[str]:
    return list(REGISTRY[version][mode])

def is_forbidden(feat: str, version: str = "v1_april2025") -> bool:
    if version.startswith("og_"):
        return feat in FORBIDDEN_NEW
    return feat in FORBIDDEN_NEW or feat in LEGACY_ALLOWED