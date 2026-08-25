#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve FXJEFE paths from config.json only — no hardcoded usernames."""
from __future__ import annotations
import json, os, sys
from pathlib import Path

def project_root() -> Path:
    env = os.environ.get("FXJEFE_PROJECT_ROOT")
    if env:
        return Path(env)
    if sys.platform == "win32":
        home = Path(os.environ.get("USERPROFILE", str(Path.home())))
    else:
        home = Path.home()
    return home / "Documents" / "FXJEFE_Project"

def load_config(root: Path | None = None) -> dict:
    root = root or project_root()
    p = root / "config.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def resolve(root: Path | None = None) -> dict:
    root = root or project_root()
    cfg = load_config(root)
    def pick(key, default_rel):
        v = (cfg.get(key) or "").strip()
        if v:
            return str(Path(v))
        return str(root / default_rel) if default_rel else str(root)
    return {
        "project_root": str(root),
        "venv_dir": str(root / (cfg.get("venv_name") or cfg.get("venv_dir") or "fxjefe")),
        "scripts_path": pick("scripts_path", ""),
        "models_path": pick("models_path", ""),
        "data_path": pick("data_path", "data"),
        "log_path": pick("log_path", "Logs"),
        "config_path": str(root / "config.json"),
    }

if __name__ == "__main__":
    print(json.dumps(resolve(), indent=2))
