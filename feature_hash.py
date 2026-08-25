#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FXJEFE feature hashing — scalable featureset identity.

Use cases:
  - Fast model/EA/server/mq5 equality without O(n²) set diffs every tick
  - Cache keys for loaded models
  - Snapshot IDs in state/*.json
  - Signal gate short-circuit: hash match ⇒ featureset match

Canonical forms
---------------
  ordered : UTF-8 names joined by \\n (byte-for-byte list identity)
  set     : sorted unique names joined by \\n (order-independent)

Hashes: sha256 hex (full) + short prefix for logs.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def normalize_features(names: Optional[Iterable[str]]) -> List[str]:
    if not names:
        return []
    out: List[str] = []
    for x in names:
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def canonical_bytes(names: Sequence[str], *, mode: str = "set") -> bytes:
    """
    mode='set'     → sorted unique (order-independent)
    mode='ordered' → preserve order, allow duplicates as given
    """
    feats = normalize_features(names)
    if mode == "ordered":
        payload = "\n".join(feats) + ("\n" if feats else "")
    else:
        payload = "\n".join(sorted(set(feats))) + ("\n" if feats else "")
    return payload.encode("utf-8")


def feature_hash(
    names: Optional[Iterable[str]],
    *,
    mode: str = "set",
    length: Optional[int] = None,
) -> str:
    """
    SHA-256 hex of canonical feature bytes.
    length=16 → short id for logs / filenames.
    """
    digest = hashlib.sha256(canonical_bytes(list(names or []), mode=mode)).hexdigest()
    if length:
        return digest[: int(length)]
    return digest


def feature_hash_pair(names: Optional[Iterable[str]]) -> Dict[str, str]:
    """Both set and ordered hashes + counts."""
    feats = normalize_features(names)
    return {
        "count": str(len(feats)),
        "unique": str(len(set(feats))),
        "hash_set": feature_hash(feats, mode="set"),
        "hash_set_short": feature_hash(feats, mode="set", length=16),
        "hash_ordered": feature_hash(feats, mode="ordered"),
        "hash_ordered_short": feature_hash(feats, mode="ordered", length=16),
    }


def hashes_equal(
    a: Optional[Iterable[str]],
    b: Optional[Iterable[str]],
    *,
    mode: str = "set",
) -> bool:
    return feature_hash(a, mode=mode) == feature_hash(b, mode=mode)


def bundle_hashes(sets: Dict[str, Sequence[str]], *, mode: str = "set") -> Dict[str, str]:
    """Map role → hash for model/ea/server/mq5."""
    return {k: feature_hash(v, mode=mode) for k, v in sets.items()}


def all_roles_match(
    sets: Dict[str, Sequence[str]],
    *,
    mode: str = "set",
) -> Tuple[bool, Dict[str, Any]]:
    """
    Scalable multi-party match: compute one hash per role, compare to reference.
    Reference = first non-empty role in stable order preference.
    """
    order = ["model", "ea", "server", "predict_mq5", "generatefeatures_mq5", "mq5"]
    hashes = {}
    for k in order:
        if k in sets:
            hashes[k] = feature_hash(sets[k], mode=mode)
    for k, v in sets.items():
        if k not in hashes:
            hashes[k] = feature_hash(v, mode=mode)

    non_empty = {k: h for k, h in hashes.items() if h != feature_hash([], mode=mode)}
    if len(non_empty) < 2:
        return False, {"hashes": hashes, "reason": "need at least two non-empty featuresets"}

    ref_key = next(k for k in order if k in non_empty)
    ref = non_empty[ref_key]
    mismatches = [k for k, h in non_empty.items() if h != ref]
    ok = len(mismatches) == 0
    return ok, {
        "ok": ok,
        "mode": mode,
        "reference": ref_key,
        "reference_hash": ref,
        "hashes": hashes,
        "mismatches": mismatches,
    }


def snapshot_dict(names: Sequence[str], *, kind: str = "features") -> Dict[str, Any]:
    feats = normalize_features(names)
    pair = feature_hash_pair(feats)
    return {
        "kind": kind,
        "features": feats,
        "count": len(feats),
        "unique": len(set(feats)),
        "hash_set": pair["hash_set"],
        "hash_set_short": pair["hash_set_short"],
        "hash_ordered": pair["hash_ordered"],
        "hash_ordered_short": pair["hash_ordered_short"],
    }


def dump_snapshot_json(names: Sequence[str], path: str, *, kind: str = "features") -> str:
    data = snapshot_dict(names, kind=kind)
    Path = __import__("pathlib").Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data["hash_set_short"]


# Cache: hash → last known feature list (process-local, scalable lookups)
_HASH_CACHE: Dict[str, Tuple[str, ...]] = {}


def cache_put(names: Sequence[str], *, mode: str = "set") -> str:
    h = feature_hash(names, mode=mode)
    _HASH_CACHE[h] = tuple(normalize_features(names) if mode == "ordered" else sorted(set(normalize_features(names))))
    return h


def cache_get(h: str) -> Optional[Tuple[str, ...]]:
    return _HASH_CACHE.get(h)


if __name__ == "__main__":
    demo = ["price", "atr", "rsi", "macd_diff"]
    print("set     ", feature_hash(demo, mode="set", length=16))
    print("ordered ", feature_hash(demo, mode="ordered", length=16))
    print("pair    ", feature_hash_pair(demo))
    # order independence for set mode
    assert feature_hash(["rsi", "price", "atr", "macd_diff"], mode="set") == feature_hash(demo, mode="set")
    ok, info = all_roles_match(
        {
            "model": demo,
            "ea": list(reversed(demo)),
            "server": demo,
            "predict_mq5": demo,
            "generatefeatures_mq5": demo,
        }
    )
    print("all_match", ok, info["reference_hash"][:16], "mismatches", info["mismatches"])
    ok2, info2 = all_roles_match({"model": demo, "ea": demo + ["garch_vol"], "server": demo})
    print("mismatch", ok2, info2["mismatches"])
