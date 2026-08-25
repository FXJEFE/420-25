# -*- coding: utf-8 -*-
"""Shared path + config bootstrap for all FXJEFE pipeline scripts."""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, Optional

# UTF-8 process defaults
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _s in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
    try:
        if _s is not None and hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

USERPROFILE = os.environ.get("USERPROFILE", r"C:\Users\locallarry")
DOCUMENTS = os.path.join(USERPROFILE, "Documents")
PROJECT = os.path.join(DOCUMENTS, "FXJEFE_Project")
DEFAULT_CONFIG_CANDIDATES = [
    os.path.join(DOCUMENTS, "config.json"),
    os.path.join(PROJECT, "config.json"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"),
]

MT5_MQL5 = os.path.join(
    os.environ.get("APPDATA", os.path.join(USERPROFILE, "AppData", "Roaming")),
    "MetaQuotes",
    "Terminal",
    "D0E8209F77C8CF37AD8BF550E51FF075",
    "MQL5",
)


def find_config_path(explicit: Optional[str] = None) -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    env = os.environ.get("FXJEFE_CONFIG")
    if env and os.path.isfile(env):
        return env
    for p in DEFAULT_CONFIG_CANDIDATES:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "No config.json found. Tried: " + ", ".join(DEFAULT_CONFIG_CANDIDATES)
    )


def load_config(explicit: Optional[str] = None) -> Dict[str, Any]:
    path = find_config_path(explicit)
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_path"] = path
    return ensure_paths(cfg)


def ensure_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing path keys with locallarry defaults and create dirs."""
    defaults = {
        "project_root": PROJECT,
        "scripts_path": DOCUMENTS,
        "project_scripts_path": os.path.join(PROJECT, "Scripts"),
        "data_path": os.path.join(PROJECT, "data"),
        "data_output_path": os.path.join(PROJECT, "data"),
        "log_path": os.path.join(DOCUMENTS, "logs"),
        "models_path": os.path.join(DOCUMENTS, "models"),
        "mt5_path": MT5_MQL5,
        "mt5_mql5_path": MT5_MQL5,
        "mt5_files_path": os.path.join(MT5_MQL5, "Files"),
        "mt5_common_path": os.path.join(
            os.environ.get("APPDATA", ""),
            "MetaQuotes",
            "Terminal",
            "Common",
            "Files",
        ),
        "mt5_experts_path": os.path.join(MT5_MQL5, "Experts"),
        "mt5_scripts_path": os.path.join(MT5_MQL5, "Scripts"),
        "historical_data_path": os.path.join(PROJECT, "HISTORIC--DATA"),
        "historic_data_path": os.path.join(PROJECT, "HISTORIC--DATA"),
        "features_csv": "FXJEFE_Features.csv",
        "features_fixed_csv": "FXJEFE_Features_fixed.csv",
        "features_signals_csv": "FXJEFE_Features_with_signals.csv",
        "trades_csv": "FXJEFE_trades.csv",
        "trades_outcomes_csv": "FXJEFE_trades_outcomes.csv",
        "ai_server_url": "http://127.0.0.1:8080",
        "api_port": 8080,
    }
    for k, v in defaults.items():
        if not cfg.get(k):
            cfg[k] = v

    # Normalize aliases
    cfg["data_output_path"] = cfg.get("data_output_path") or cfg["data_path"]
    cfg["mt5_mql5_path"] = cfg.get("mt5_mql5_path") or cfg.get("mt5_path") or MT5_MQL5
    cfg["mt5_path"] = cfg["mt5_mql5_path"]
    if not cfg.get("mt5_files_path"):
        cfg["mt5_files_path"] = os.path.join(cfg["mt5_mql5_path"], "Files")

    for key in (
        "project_root",
        "scripts_path",
        "project_scripts_path",
        "data_path",
        "data_output_path",
        "log_path",
        "models_path",
        "mt5_files_path",
        "mt5_common_path",
        "historical_data_path",
    ):
        p = cfg.get(key)
        if p:
            try:
                os.makedirs(p, exist_ok=True)
            except OSError:
                pass

    return cfg


def setup_logging(cfg: Dict[str, Any], script_name: str) -> None:
    log_dir = cfg.get("log_path") or os.path.join(DOCUMENTS, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{script_name}.log")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    logging.info("Loaded config: %s", cfg.get("_config_path"))


def features_path(cfg: Dict[str, Any], name: Optional[str] = None) -> str:
    return os.path.join(cfg["data_path"], name or cfg.get("features_csv") or "FXJEFE_Features.csv")


MODEL_EXTS = (".pkl", ".json", ".cbm", ".onnx", ".h5", ".pt", ".pth", ".joblib", ".lgb", ".txt")


def safe_model_out_path(cfg: Dict[str, Any], basename: str) -> str:
    """
    Write-only path for NEW pipeline models.
    Never overwrites an existing OG artifact (my_model.pkl, xgboost_model.json, …).
    Always lands under models/og333_runs/<stem>_YYYYmmdd_HHMMSS<ext>.
    """
    from datetime import datetime

    models_root = cfg.get("models_path") or os.path.join(DOCUMENTS, "models")
    out_dir = os.path.join(models_root, cfg.get("model_write_dir") or "og333_runs")
    os.makedirs(out_dir, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(basename) or "model.pkl")
    if not ext:
        ext = ".pkl"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{stem}_{stamp}{ext}")
    # belt: if somehow exists, add counter
    n = 1
    while os.path.exists(path):
        path = os.path.join(out_dir, f"{stem}_{stamp}_{n}{ext}")
        n += 1
    return path


def assert_not_og_model(path: str) -> None:
    """Raise if caller is about to overwrite a protected existing model file."""
    if not path:
        return
    if os.path.isfile(path):
        raise PermissionError(
            f"Refusing to overwrite existing OG model: {path}. "
            "Write to models/og333_runs/ instead."
        )


def snapshot_model_files(cfg: Dict[str, Any]) -> dict:
    """mtime+size of every model file under known model dirs (OG protection)."""
    roots = [
        cfg.get("models_path"),
        os.path.join(DOCUMENTS, "models"),
        os.path.join(cfg.get("project_root") or PROJECT, "models"),
        DOCUMENTS,
        cfg.get("project_root") or PROJECT,
    ]
    snap = {}
    seen = set()
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        key = os.path.normcase(os.path.abspath(root))
        if key in seen:
            continue
        seen.add(key)
        try:
            names = os.listdir(root)
        except OSError:
            continue
        for fn in names:
            if not fn.lower().endswith(MODEL_EXTS):
                continue
            p = os.path.join(root, fn)
            if not os.path.isfile(p):
                continue
            try:
                st = os.stat(p)
                snap[os.path.abspath(p)] = (st.st_mtime_ns, st.st_size)
            except OSError:
                continue
    return snap


def verify_og_models_untouched(before: dict) -> list:
    """Return list of OG model paths that changed size/mtime. Empty = safe."""
    changed = []
    for path, (mtime_ns, size) in before.items():
        if not os.path.isfile(path):
            changed.append(path + " (DELETED)")
            continue
        try:
            st = os.stat(path)
        except OSError:
            changed.append(path + " (UNREADABLE)")
            continue
        if st.st_mtime_ns != mtime_ns or st.st_size != size:
            changed.append(path)
    return changed


def models_file(cfg: Dict[str, Any], name: str) -> str:
    """Resolve model file under models_path, then Documents, then project."""
    candidates = [
        os.path.join(cfg["models_path"], name),
        os.path.join(DOCUMENTS, name),
        os.path.join(cfg["project_root"], name),
        os.path.join(cfg["project_root"], "models", name),
    ]
    for p in candidates:
        if os.path.isfile(p) and os.path.getsize(p) > 64:
            with open(p, "rb") as f:
                head = f.read(8)
            if head and not all(b == 0 for b in head):
                return p
    return candidates[0]


def mt5_file(cfg: Dict[str, Any], name: str) -> str:
    return os.path.join(cfg["mt5_files_path"], name)


def feature_write_targets(cfg: Dict[str, Any], filename: str) -> list:
    """
    OG-style destinations: ALWAYS write feature CSVs to every known location.
    Never skip a destination just because a file already exists.
    """
    name = filename or cfg.get("features_csv") or "FXJEFE_Features.csv"
    hist = cfg.get("historical_data_path") or cfg.get("historic_data_path") or ""
    targets = [
        os.path.join(cfg["data_path"], name),
        os.path.join(cfg.get("data_output_path") or cfg["data_path"], name),
        os.path.join(cfg["project_root"], name),
        os.path.join(cfg.get("scripts_path") or DOCUMENTS, name),
        os.path.join(cfg["mt5_files_path"], name),
        os.path.join(cfg.get("mt5_common_path") or "", name),
        os.path.join(hist, name) if hist else "",
        os.path.join(hist, "pipeline", name) if hist else "",
    ]
    # de-dupe preserve order, drop empties
    out, seen = [], set()
    for t in targets:
        if not t or t in seen:
            continue
        # skip if parent path empty (common path missing APPDATA)
        parent = os.path.dirname(t)
        if not parent:
            continue
        seen.add(t)
        out.append(t)
    return out


def write_feature_csv(df, cfg: Dict[str, Any], filename: str, index: bool = False) -> list:
    """
    Force-write a feature DataFrame to ALL OG destinations.
    Returns list of paths successfully written.
    """
    import pandas as pd  # local import OK

    if df is None:
        raise ValueError("write_feature_csv: df is None")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("write_feature_csv: expected DataFrame")

    written = []
    for path in feature_write_targets(cfg, filename):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # atomic-ish write: temp then replace
            tmp = path + ".tmp_write"
            df.to_csv(tmp, index=index, encoding="utf-8")
            os.replace(tmp, path)
            written.append(path)
            logging.info(
                "FEATURE CSV WRITE → %s (%s rows, %s cols, %.1f KB)",
                path,
                len(df),
                len(df.columns),
                os.path.getsize(path) / 1024.0,
            )
        except Exception as e:
            logging.error("FEATURE CSV WRITE FAILED → %s : %s", path, e)
    if not written:
        raise RuntimeError(f"Failed to write {filename} to any OG destination")
    return written
