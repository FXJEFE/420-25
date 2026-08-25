#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify AI server + M15 models + feature CSV + paths are aligned.
Does NOT write or overwrite any OG model files.
"""
from __future__ import annotations

import json
import os
import sys

from fxjefe_paths import load_config, setup_logging, features_path

try:
    import requests
except ImportError:
    requests = None
import pandas as pd


def main() -> None:
    cfg = load_config()
    setup_logging(cfg, "verify_m15_align")
    import logging

    url = (cfg.get("ai_server_url") or "http://127.0.0.1:8080").rstrip("/")
    feat_path = features_path(cfg)
    models_dir = cfg.get("models_path")
    tf = str(cfg.get("preferred_timeframe") or "M15").upper()

    print("=== ALIGNMENT CONTRACT ===", flush=True)
    print(f"timeframe={tf}", flush=True)
    print(f"features_csv={feat_path} exists={os.path.isfile(feat_path)}", flush=True)
    print(f"models_path={models_dir}", flush=True)
    print(f"ai_server={url}", flush=True)
    print(f"historic={cfg.get('historical_data_path')}", flush=True)

    if not os.path.isfile(feat_path):
        raise SystemExit(f"FAIL: missing feature CSV {feat_path}")

    df = pd.read_csv(feat_path, encoding="utf-8", low_memory=False)
    print(f"feature rows={len(df)} cols={len(df.columns)}", flush=True)
    if "symbol" not in df.columns:
        raise SystemExit("FAIL: feature CSV has no symbol column")

    required = ["price", "atr", "ema_diff", "rsi", "macd_diff"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"FAIL: feature CSV missing {missing}")

    # M15 specialist files must exist (read-only check)
    m15_names = []
    if os.path.isdir(models_dir):
        for fn in sorted(os.listdir(models_dir)):
            if "_M15_" in fn and (fn.endswith("_xgb.json") or fn.endswith("_features.json")):
                p = os.path.join(models_dir, fn)
                m15_names.append(f"{fn}:{os.path.getsize(p)}")
    print(f"M15 model artifacts on disk: {len(m15_names)}", flush=True)
    for n in m15_names[:20]:
        print(f"  {n}", flush=True)

    if requests is None:
        raise SystemExit("FAIL: requests not installed in venv")

    try:
        h = requests.get(url + "/health", timeout=8)
        health = h.json()
    except Exception as e:
        raise SystemExit(f"FAIL: /health unreachable {e}")

    print("=== /health ===", flush=True)
    print(
        f"server={health.get('server')} loaded={health.get('loaded_models')} "
        f"symbol_models={health.get('symbol_models')} gate={health.get('gate')} "
        f"preferred={health.get('preferred_timeframe')} "
        f"m15={health.get('m15_symbol_models')}",
        flush=True,
    )
    if health.get("models_dir") and os.path.normcase(health.get("models_dir")) != os.path.normcase(models_dir):
        print(
            f"WARN models_dir mismatch health={health.get('models_dir')} config={models_dir}",
            flush=True,
        )

    symbols = []
    for col in df["symbol"].astype(str).str.upper().str.replace(r"\.R$", "", regex=True).unique():
        symbols.append(str(col))
    want = list(cfg.get("forex_symbols") or []) + list(cfg.get("crypto_symbols") or [])
    for s in want:
        if s not in symbols and s in set(symbols):
            pass
    test_syms = [s for s in want if s in set(symbols)] or symbols[:4]
    if not test_syms:
        raise SystemExit("FAIL: no symbols to predict")

    ok = 0
    for sym in test_syms:
        row = df[df["symbol"].astype(str).str.upper().str.replace(r"\.R$", "", regex=True) == sym]
        if row.empty:
            print(f"FAIL {sym}: no feature row", flush=True)
            continue
        payload = row.iloc[-1].to_dict()
        payload["symbol"] = sym
        payload["timeframe"] = tf
        # json-safe
        clean = {}
        for k, v in payload.items():
            if k.startswith("_"):
                continue
            try:
                if pd.isna(v):
                    continue
            except Exception:
                pass
            if hasattr(v, "item"):
                v = v.item()
            clean[k] = v
        try:
            r = requests.post(url + "/predict", json=clean, timeout=20)
            body = r.json()
        except Exception as e:
            print(f"FAIL {sym}: predict error {e}", flush=True)
            continue
        sig = body.get("signal")
        conf = body.get("confidence")
        n = body.get("n_models")
        probs = body.get("model_probs") or {}
        server = body.get("server")
        valid = (
            r.status_code == 200
            and isinstance(conf, (int, float))
            and 0.0 <= float(conf) <= 1.0
            and int(n or 0) >= 1
            and server == "ai_server_golden_comprehensive"
            and sig in ("buy", "sell", "hold")
        )
        print(
            f"{'OK' if valid else 'FAIL'} {sym} {tf} signal={sig} conf={conf} "
            f"n_models={n} specialists={[k for k in probs if 'M15' in k]}",
            flush=True,
        )
        if valid:
            ok += 1

    print(f"=== M15 PREDICT {ok}/{len(test_syms)} valid ===", flush=True)
    if ok < 1:
        raise SystemExit("FAIL: no valid M15 predictions")
    print("ALIGN OK — models, features, paths, M15 predictions", flush=True)


if __name__ == "__main__":
    main()
