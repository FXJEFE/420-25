#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FXJEFE feature hashing — scalable featureset identity (UTF-8)."""
from __future__ import annotations
import hashlib, json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

def normalize_features(names: Optional[Iterable[str]]) -> List[str]:
    if not names:
        return []
    return [str(x).strip() for x in names if str(x).strip()]

def canonical_bytes(names: Sequence[str], *, mode: str = "set") -> bytes:
    feats = normalize_features(names)
    if mode == "ordered":
        payload = "\n".join(feats) + ("\n" if feats else "")
    else:
        payload = "\n".join(sorted(set(feats))) + ("\n" if feats else "")
    return payload.encode("utf-8")

def feature_hash(names: Optional[Iterable[str]], *, mode: str = "set", length: Optional[int] = None) -> str:
    digest = hashlib.sha256(canonical_bytes(list(names or []), mode=mode)).hexdigest()
    return digest[:int(length)] if length else digest

def feature_hash_pair(names: Optional[Iterable[str]]) -> Dict[str, str]:
    feats = normalize_features(names)
    return {
        "count": str(len(feats)),
        "unique": str(len(set(feats))),
        "hash_set": feature_hash(feats, mode="set"),
        "hash_set_short": feature_hash(feats, mode="set", length=16),
        "hash_ordered": feature_hash(feats, mode="ordered"),
        "hash_ordered_short": feature_hash(feats, mode="ordered", length=16),
    }

def hashes_equal(a, b, *, mode: str = "set") -> bool:
    return feature_hash(a, mode=mode) == feature_hash(b, mode=mode)

def all_roles_match(sets: Dict[str, Sequence[str]], *, mode: str = "set") -> Tuple[bool, Dict[str, Any]]:
    order = ["model", "ea", "server", "predict_mq5", "generatefeatures_mq5", "mq5"]
    hashes = {k: feature_hash(v, mode=mode) for k, v in sets.items()}
    non_empty = {k: h for k, h in hashes.items() if h != feature_hash([], mode=mode)}
    if len(non_empty) < 2:
        return False, {"hashes": hashes, "reason": "need two non-empty sets"}
    ref_key = next((k for k in order if k in non_empty), next(iter(non_empty)))
    ref = non_empty[ref_key]
    mismatches = [k for k, h in non_empty.items() if h != ref]
    return len(mismatches) == 0, {"ok": not mismatches, "reference": ref_key, "reference_hash": ref, "hashes": hashes, "mismatches": mismatches}

def snapshot_dict(names: Sequence[str], *, kind: str = "features") -> Dict[str, Any]:
    feats = normalize_features(names)
    pair = feature_hash_pair(feats)
    return {"kind": kind, "features": feats, "count": len(feats), "unique": len(set(feats)), **{k: pair[k] for k in pair if k.startswith("hash")}}

if __name__ == "__main__":
    demo = ["price", "atr", "rsi", "macd_diff"]
    print("set", feature_hash(demo, mode="set", length=16))
    print("ordered", feature_hash(demo, mode="ordered", length=16))
    print(all_roles_match({"model": demo, "ea": list(reversed(demo)), "server": demo}))
