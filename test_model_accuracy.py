#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-pipeline M15 accuracy test.
Reads live+historic FXJEFE_Features.csv, labels from next-bar return,
scores AI-server /predict. Does NOT write or overwrite OG models.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

from fxjefe_paths import load_config, setup_logging, features_path


def next_bar_label(g: pd.DataFrame, atr_mult: float = 0.15) -> pd.Series:
    px = pd.to_numeric(g["price"], errors="coerce")
    fut = px.shift(-1)
    ret = fut - px
    atr = pd.to_numeric(g["atr"], errors="coerce") if "atr" in g.columns else pd.Series(0.0, index=g.index)
    thr = atr.fillna(px.abs() * 0.0005) * atr_mult
    thr = thr.replace(0, px.abs() * 0.0005)
    lab = pd.Series(0, index=g.index, dtype=int)
    lab = lab.mask(ret > thr, 1)
    lab = lab.mask(ret < -thr, -1)
    return lab


def main() -> None:
    cfg = load_config()
    setup_logging(cfg, "test_model_accuracy")
    import logging

    src = features_path(cfg)
    url = (cfg.get("ai_server_url") or "http://127.0.0.1:8080").rstrip("/")
    tf = str(cfg.get("preferred_timeframe") or "M15")
    print(f"features={src}", flush=True)
    print(f"server={url} tf={tf}", flush=True)

    df = pd.read_csv(src, encoding="utf-8", low_memory=False)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.replace(r"\.R$", "", regex=True)
    if "label" not in df.columns:
        parts = []
        for _, g in df.groupby("symbol", sort=False):
            gg = g.copy()
            gg["label"] = next_bar_label(gg)
            parts.append(gg)
        df = pd.concat(parts, ignore_index=True)

    # last 80 labeled bars per symbol (skip final bar with no future)
    samples = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.dropna(subset=["price"]).copy()
        g = g.iloc[:-1]
        if len(g) < 10:
            continue
        take = g.tail(80)
        for _, row in take.iterrows():
            samples.append(row)

    print(f"scoring {len(samples)} M15 rows", flush=True)
    h = requests.get(url + "/health", timeout=8)
    print("health", h.status_code, h.json().get("server"), "loaded", h.json().get("loaded_models"), flush=True)

    stats = defaultdict(lambda: {"n": 0, "hit": 0, "dir_n": 0, "dir_hit": 0, "conf": []})
    overall = {"n": 0, "hit": 0, "dir_n": 0, "dir_hit": 0}

    for row in samples:
        payload = {}
        for k, v in row.items():
            if str(k).startswith("_"):
                continue
            try:
                if pd.isna(v):
                    continue
            except Exception:
                pass
            if hasattr(v, "item"):
                v = v.item()
            payload[k] = v
        payload["symbol"] = str(row["symbol"])
        payload["timeframe"] = tf
        try:
            r = requests.post(url + "/predict", json=payload, timeout=20)
            body = r.json()
        except Exception as e:
            logging.warning("predict fail %s: %s", row.get("symbol"), e)
            continue
        sig = str(body.get("signal") or "hold").lower()
        pred = 1 if sig == "buy" else (-1 if sig == "sell" else 0)
        y = int(row["label"]) if not pd.isna(row["label"]) else 0
        sym = payload["symbol"]
        stats[sym]["n"] += 1
        overall["n"] += 1
        if pred == y:
            stats[sym]["hit"] += 1
            overall["hit"] += 1
        if pred != 0 and y != 0:
            stats[sym]["dir_n"] += 1
            overall["dir_n"] += 1
            if pred == y:
                stats[sym]["dir_hit"] += 1
                overall["dir_hit"] += 1
        try:
            stats[sym]["conf"].append(float(body.get("confidence") or 0))
        except Exception:
            pass

    print("\n=== M15 ACCURACY (next-bar label vs /predict) ===", flush=True)
    print(f"{'symbol':10} {'n':>5} {'acc':>7} {'dir_n':>6} {'dir_acc':>8} {'mean_conf':>9}", flush=True)
    for sym in sorted(stats):
        s = stats[sym]
        acc = s["hit"] / s["n"] if s["n"] else 0
        dacc = s["dir_hit"] / s["dir_n"] if s["dir_n"] else float("nan")
        mc = float(np.mean(s["conf"])) if s["conf"] else 0
        dacc_s = f"{dacc:.3f}" if s["dir_n"] else "n/a"
        print(f"{sym:10} {s['n']:5d} {acc:7.3f} {s['dir_n']:6d} {dacc_s:>8} {mc:9.3f}", flush=True)
    oacc = overall["hit"] / overall["n"] if overall["n"] else 0
    odacc = overall["dir_hit"] / overall["dir_n"] if overall["dir_n"] else 0
    print(
        f"{'OVERALL':10} {overall['n']:5d} {oacc:7.3f} {overall['dir_n']:6d} {odacc:8.3f}",
        flush=True,
    )
    print("Accuracy test complete (OG models not modified)", flush=True)
    if overall["n"] < 1:
        raise SystemExit("no scored rows")


if __name__ == "__main__":
    main()
