#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fill every live + historical CSV destination so OG333 can run end-to-end.
Never skips a destination. Converts UTF-16. Adds missing headers.
"""
from __future__ import annotations

import os as _os_utf8
import sys as _sys_utf8

_os_utf8.environ.setdefault("PYTHONUTF8", "1")
_os_utf8.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _s in (getattr(_sys_utf8, "stdout", None), getattr(_sys_utf8, "stderr", None)):
    try:
        if _s is not None and hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import io
import os
import shutil
from pathlib import Path

import pandas as pd

from fxjefe_paths import load_config, setup_logging, feature_write_targets, write_feature_csv

REQUIRED = [
    "FXJEFE_Features.csv",
    "FXJEFE_Features_fixed.csv",
    "FXJEFE_Features_with_labels.csv",
    "FXJEFE_Features_with_signals.csv",
    "training_data.csv",
    "training_data_updated.csv",
    "processed_features.csv",
    "signals_output.csv",
    "parsed_log_signals.csv",
    "realtime_data.csv",
    "candle_data.csv",
    "FXJEFE_trades.csv",
    "FXJEFE_trades_outcomes.csv",
    "FXJEFE_merged.csv",
]

TRADES_HEADER = ["ticket", "time", "symbol", "comment", "type", "lots", "price", "sl", "tp"]
OUTCOMES_HEADER = ["ticket", "time", "symbol", "comment", "profit"]


def _read_any(path: Path) -> pd.DataFrame | None:
    if not path.is_file() or path.stat().st_size < 4:
        return None
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16", errors="replace")
        try:
            return pd.read_csv(io.StringIO(text), low_memory=False)
        except Exception:
            return None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            if df is not None and len(df.columns) >= 1:
                return df
        except Exception:
            continue
    # no-header trades-like
    try:
        df = pd.read_csv(path, encoding="utf-8", header=None, low_memory=False)
        return df
    except Exception:
        return None


def _best_source(name: str, search: list[Path]) -> tuple[Path | None, pd.DataFrame | None]:
    best_path, best_df, best_n = None, None, -1
    for folder in search:
        p = folder / name
        df = _read_any(p)
        if df is None:
            continue
        n = len(df)
        if n > best_n:
            best_path, best_df, best_n = p, df, n
    return best_path, best_df


def _normalize_trades(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    header = TRADES_HEADER if kind == "trades" else OUTCOMES_HEADER
    if list(df.columns)[:1] == [0] or str(df.columns[0]).startswith("Unnamed"):
        # numeric headerless
        cols = header[: len(df.columns)]
        while len(cols) < len(df.columns):
            cols.append(f"col_{len(cols)}")
        df = df.copy()
        df.columns = cols
    missing = [c for c in header if c not in df.columns]
    if len(missing) == len(header) and df.shape[1] >= len(header):
        df = df.copy()
        df.columns = list(header) + [f"col_{i}" for i in range(len(header), df.shape[1])]
    for c in header:
        if c not in df.columns:
            df[c] = ""
    return df[header]


def _from_features(features: pd.DataFrame, name: str) -> pd.DataFrame:
    df = features.copy()
    if name == "training_data.csv" or name == "training_data_updated.csv":
        if "signal" in df.columns:
            sig = pd.to_numeric(df["signal"], errors="coerce").fillna(0)
            df["label"] = sig.map({1: 1, -1: -1, 0: 0}).fillna(0)
        elif "label" not in df.columns:
            df["label"] = 0
        return df
    if name == "processed_features.csv":
        return df
    if name == "signals_output.csv":
        if "signal" not in df.columns:
            df["signal"] = "hold"
        return df
    if name == "FXJEFE_Features_fixed.csv":
        return df
    if name == "FXJEFE_Features_with_labels.csv":
        if "label" not in df.columns:
            df["label"] = 0
        return df
    if name == "FXJEFE_Features_with_signals.csv":
        if "signal" not in df.columns:
            df["signal"] = 0
        return df
    if name == "realtime_data.csv":
        return df.tail(50).copy()
    if name == "candle_data.csv":
        keep = [c for c in ("time", "symbol", "price", "atr", "spread") if c in df.columns]
        return df[keep].tail(200).copy() if keep else df.tail(200).copy()
    if name == "parsed_log_signals.csv":
        out = pd.DataFrame(columns=["time", "symbol", "signal", "source"])
        if "time" in df.columns and "symbol" in df.columns:
            out = df[["time", "symbol"]].copy()
            out["signal"] = df["signal"] if "signal" in df.columns else "hold"
            out["source"] = "fill_all_csvs"
        return out
    return df


def main() -> None:
    cfg = load_config()
    setup_logging(cfg, "fill_all_csvs")
    import logging

    data = Path(cfg["data_path"])
    project = Path(cfg["project_root"])
    docs = Path(cfg.get("scripts_path") or str(Path.home() / "Documents"))
    mt5 = Path(cfg["mt5_files_path"])
    common = Path(cfg.get("mt5_common_path") or "")
    hist = Path(cfg.get("historical_data_path") or (project / "HISTORIC--DATA"))
    search = [data, project, docs, mt5, common, hist, hist / "pipeline"]

    logging.info("Searching CSV sources in %s", [str(p) for p in search])

    feat_src, features = _best_source("FXJEFE_Features.csv", search)
    if features is None:
        raise SystemExit("No FXJEFE_Features.csv found in any location")
    logging.info("Best features: %s rows=%s from %s", list(features.columns)[:6], len(features), feat_src)

    written_total = 0
    for name in REQUIRED:
        src, df = _best_source(name, search)
        if df is None or len(df) == 0:
            if name == "FXJEFE_trades.csv":
                df = pd.DataFrame(columns=TRADES_HEADER)
            elif name == "FXJEFE_trades_outcomes.csv":
                df = pd.DataFrame(columns=OUTCOMES_HEADER)
            elif name == "FXJEFE_merged.csv":
                df = features.copy()
            else:
                df = _from_features(features, name)
            logging.info("Built %s from features/fallback (%s rows)", name, len(df))
        else:
            logging.info("Using existing %s from %s (%s rows)", name, src, len(df))

        if name == "FXJEFE_trades.csv":
            df = _normalize_trades(df, "trades")
        elif name == "FXJEFE_trades_outcomes.csv":
            df = _normalize_trades(df, "outcomes")
        elif name == "FXJEFE_merged.csv" and "ticket" not in df.columns:
            # merge features + trades on time+symbol if possible
            tsrc, trades = _best_source("FXJEFE_trades.csv", search)
            if trades is not None and len(trades):
                trades = _normalize_trades(trades, "trades")
                if "time" in features.columns and "symbol" in features.columns:
                    df = pd.merge(features, trades, on=["time", "symbol"], how="left")
                else:
                    df = features.copy()
            else:
                df = features.copy()

        paths = write_feature_csv(df, cfg, name)
        written_total += len(paths)
        print(f"OK {name}: {len(df)} rows -> {len(paths)} locations", flush=True)

    # Also copy historic OHLCV as UTF-8 (rewrite in place if needed) + pipeline index
    if hist.is_dir():
        index_rows = []
        for p in sorted(hist.glob("*.csv")):
            if p.name in REQUIRED:
                continue
            raw = p.read_bytes()[:4]
            if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
                text = p.read_bytes().decode("utf-16", errors="replace")
                p.write_bytes(text.encode("utf-8"))
                logging.info("Converted historic UTF-16 -> UTF-8: %s", p.name)
            index_rows.append(
                {
                    "file": p.name,
                    "bytes": p.stat().st_size,
                    "symbol": p.name.split("_")[0] if "_" in p.name else "",
                }
            )
        if index_rows:
            idx = pd.DataFrame(index_rows)
            write_feature_csv(idx, cfg, "historic_ohlcv_index.csv")
            print(f"OK historic_ohlcv_index.csv: {len(idx)} files", flush=True)

    # logs used by test_encoding / parse
    for log_name in ("log.txt", "FXJEFE_log.txt"):
        src = None
        for folder in search:
            cand = folder / log_name
            if cand.is_file() and cand.stat().st_size > 20:
                src = cand
                break
        if src is None:
            continue
        for dest_dir in feature_write_targets(cfg, log_name):
            dest = Path(dest_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.resolve() == src.resolve():
                continue
            try:
                shutil.copy2(src, dest)
            except OSError as e:
                logging.warning("log copy %s -> %s failed: %s", src, dest, e)

    print(f"fill_all_csvs complete: {written_total} file writes", flush=True)
    logging.info("fill_all_csvs complete writes=%s", written_total)


if __name__ == "__main__":
    main()
