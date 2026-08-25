#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Signal gate: emit only if model loaded + featureset matches EA, server, Predict.mq5, GenerateFeatures.mq5."""
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from feature_hash import feature_hash, hashes_equal
except ImportError:
    feature_hash = hashes_equal = None

AUTO_ENABLED = True
DEMO_COMMENT = "# DEMO: label only; does not disable systems."

def _norm(names: Optional[Iterable[str]]) -> List[str]:
    if not names:
        return []
    return [str(x).strip() for x in names if str(x).strip()]

def same_featureset(a: Sequence[str], b: Sequence[str], *, order_matters: bool = False) -> bool:
    if hashes_equal is not None:
        return hashes_equal(a, b, mode="ordered" if order_matters else "set")
    return (list(a) == list(b)) if order_matters else (set(a) == set(b))

def can_emit_signal(*, model_loaded: bool, model_features=None, ea_features=None, server_features=None,
                    predict_mq5_features=None, generatefeatures_mq5_features=None, mq5_features=None,
                    order_matters: bool = False) -> Tuple[bool, dict]:
    info = {"auto_enabled": AUTO_ENABLED, "demo_comment": DEMO_COMMENT, "model_loaded": bool(model_loaded), "emit": False, "reasons": []}
    if not model_loaded:
        info["reasons"].append("model not loaded"); return False, info
    model_f, ea_f, server_f = _norm(model_features), _norm(ea_features), _norm(server_features)
    pred_f = _norm(predict_mq5_features) or _norm(mq5_features)
    gen_f = _norm(generatefeatures_mq5_features) or _norm(mq5_features)
    for label, val in [("model", model_f), ("EA", ea_f), ("server", server_f), ("Predict.mq5", pred_f), ("GenerateFeatures.mq5", gen_f)]:
        if not val:
            info["reasons"].append(f"{label} feature set empty"); return False, info
    checks = {
        "model_vs_ea": same_featureset(model_f, ea_f, order_matters=order_matters),
        "model_vs_server": same_featureset(model_f, server_f, order_matters=order_matters),
        "model_vs_predict_mq5": same_featureset(model_f, pred_f, order_matters=order_matters),
        "model_vs_generatefeatures_mq5": same_featureset(model_f, gen_f, order_matters=order_matters),
        "predict_vs_generatefeatures_mq5": same_featureset(pred_f, gen_f, order_matters=order_matters),
    }
    info["matches"] = checks
    if feature_hash:
        info["hashes"] = {k: feature_hash(v, length=16) for k, v in [("model", model_f), ("ea", ea_f), ("server", server_f), ("predict_mq5", pred_f), ("generatefeatures_mq5", gen_f)]}
    if not all(checks.values()):
        info["reasons"].append("featureset mismatch: " + str([k for k,v in checks.items() if not v])); return False, info
    info["emit"] = True
    info["reasons"].append("model loaded + featureset aligned")
    return True, info

if __name__ == "__main__":
    f = ["price", "atr", "rsi"]
    print(can_emit_signal(model_loaded=True, model_features=f, ea_features=f, server_features=f, predict_mq5_features=f, generatefeatures_mq5_features=f))
