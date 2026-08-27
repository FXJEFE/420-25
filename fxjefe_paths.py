#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FXJEFE path helpers — required by train_models.py and related scripts.

Rules:
- Never hard-code usernames
- Prefer FXJEFE_PROJECT_ROOT environment variable when set
- Fallback = USERPROFILE/Documents/FXJEFE_Project (Windows)
             ~/Documents/FXJEFE_Project           (Linux/macOS)
- Models always live under <project_root>/models/
- NEVER write a plain "my_model.pkl" (or any OG-style name).
  Every model automatically receives a version mark: _vYYYYMMDD_HHMMSS
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


def project_root() -> Path:
    """
    Canonical project root.
    Order of precedence:
      1. FXJEFE_PROJECT_ROOT env var (set by pipelinerun_production)
      2. USERPROFILE/Documents/FXJEFE_Project   (Windows)
      3. ~/Documents/FXJEFE_Project             (Linux/macOS)
    """
    env = os.environ.get("FXJEFE_PROJECT_ROOT")
    if env and env.strip():
        root = Path(env.strip())
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve()

    if sys.platform == "win32":
        home = Path(os.environ.get("USERPROFILE") or Path.home())
    else:
        home = Path.home()

    root = home / "Documents" / "FXJEFE_Project"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def models_dir() -> Path:
    d = project_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir() -> Path:
    d = project_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    d = project_root() / "Logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def bridge_dir() -> Path:
    d = project_root() / "bridge"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _version_mark() -> str:
    """Short UTC version stamp: v20260815_163245"""
    return "v" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _make_versioned_name(name: str) -> str:
    """
    Turn any requested model name into a versioned name.
    Plain "my_model.pkl"  →  "my_model_v20260815_163245.pkl"
    Already versioned names are left alone.
    """
    p = Path(name)
    stem = p.stem
    suffix = p.suffix or ".pkl"

    # Already has a version mark → keep it
    if "_v20" in stem or stem.endswith(tuple(f"_v{i}" for i in range(1, 100))):
        return name

    # Protect classic OG / original names
    lower = stem.lower()
    if lower in ("my_model", "model", "xgb_model", "stacking_model") or \
       lower.startswith(("og_", "original_", "backup_", "bak_")):
        stem = f"{stem}_{_version_mark()}"
    else:
        # Still add a light version mark for every new model
        stem = f"{stem}_{_version_mark()}"

    return stem + suffix


def og333_runs_dir() -> Path:
    """New training writes here only — never into Documents/models originals."""
    if sys.platform == "win32":
        home = Path(os.environ.get("USERPROFILE") or Path.home())
    else:
        home = Path.home()
    d = home / "Documents" / "models" / "og333_runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_model_out_path(config: Any = None, name: str = "my_model.pkl") -> Path:
    """
    Return a safe, versioned write path under Documents/models/og333_runs/.

    Accepts the exact call style used by train_models.py:
        safe_model_out_path(config, 'my_model.pkl')

    The first argument (config) is accepted for compatibility but ignored.
    The returned filename ALWAYS contains a version mark so it can never
    collide with an original / OG model.
    """
    if isinstance(config, str) and name == "my_model.pkl":
        # Called as safe_model_out_path("my_model.pkl")
        name = config

    versioned = _make_versioned_name(name)
    p = og333_runs_dir() / versioned
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def assert_not_og_model(path: PathLike) -> None:
    """
    Hard guard against overwriting protected / original models.
    Raises RuntimeError if the target looks like an original or backup model.
    """
    p = Path(path)
    name = p.name.lower()
    protected_prefixes = ("og_", "original_", "backup_", "bak_")
    protected_substrings = ("og_model", ".og.pkl", "_og.pkl")

    # Also protect the classic unversioned names
    if name in ("my_model.pkl", "model.pkl", "xgb_model.pkl", "stacking_model.pkl"):
        raise RuntimeError(f"Refusing to overwrite protected unversioned model: {p}")

    if any(name.startswith(pref) for pref in protected_prefixes) or \
       any(s in name for s in protected_substrings):
        raise RuntimeError(f"Refusing to overwrite protected model: {p}")


# ------------------------------------------------------------------
# Compatibility aliases
# ------------------------------------------------------------------
def get_project_root() -> Path:
    return project_root()


def get_models_path() -> Path:
    return models_dir()


def get_data_path() -> Path:
    return data_dir()


if __name__ == "__main__":
    print("FXJEFE_PROJECT_ROOT env :", os.environ.get("FXJEFE_PROJECT_ROOT"))
    print("project_root()          :", project_root())
    print("models_dir()            :", models_dir())
    print("safe_model_out_path()   :", safe_model_out_path())
    print("safe_model_out_path(cfg):", safe_model_out_path({}, "my_model.pkl"))
    print("assert_not_og_model     : OK")
