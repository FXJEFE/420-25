#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FXJEFE smart terminal helper
============================
Finds the project by filename (not by fragile cwd), runs setup steps,
and prints exact next commands. Error-resistant for zsh/bash.

Usage (from ANY directory):
  python3 fxjefe_smart.py
  python3 fxjefe_smart.py doctor
  python3 fxjefe_smart.py find feature_registry.py
  python3 fxjefe_smart.py run inventory
  python3 fxjefe_smart.py run production
  python3 fxjefe_smart.py run strap
  python3 fxjefe_smart.py ports
  python3 fxjefe_smart.py framework

Never replaces existing project scripts.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


MARKER_FILES = (
    "feature_registry.py",
    "pipelinerun_production.py",
    "config.json",
    "pipelinerun.py",
)


def log(msg: str) -> None:
    print(f"[FXJEFE] {msg}", flush=True)


def home() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("USERPROFILE", str(Path.home())))
    return Path.home()


def candidate_roots() -> List[Path]:
    h = home()
    roots = [
        h / "Documents" / "FXJEFE_Project",
        h / "Documents" / "FXJEFE",
        h / "FXJEFE_Project",
        Path.cwd(),
        Path.cwd().resolve(),
    ]
    # recent Downloads installers often leave user in wrong folder
    roots.append(h / "Downloads")
    return roots


def find_by_name(filename: str, limit: int = 15) -> List[Path]:
    """Search common trees for a filename. Fast-ish, skips venv/node."""
    skip = {".venv", "venv", "node_modules", "__pycache__", ".git", "Library", "Application Support"}
    hits: List[Path] = []
    search_roots = [
        home() / "Documents",
        home() / "Downloads",
        home() / "Desktop",
        Path.cwd(),
    ]
    for base in search_roots:
        if not base.is_dir():
            continue
        try:
            for p in base.rglob(filename):
                if any(s in p.parts for s in skip):
                    continue
                hits.append(p)
                if len(hits) >= limit:
                    return hits
        except PermissionError:
            continue
    return hits


def detect_project() -> Optional[Path]:
    # 1) explicit env
    env = os.environ.get("FXJEFE_PROJECT_ROOT")
    if env and Path(env).is_dir():
        return Path(env)

    # 2) known candidates that contain markers
    for root in candidate_roots():
        if not root.is_dir():
            continue
        for m in MARKER_FILES:
            if (root / m).is_file():
                return root.resolve()

    # 3) search by marker filename
    for m in MARKER_FILES:
        hits = find_by_name(m, limit=5)
        for h in hits:
            return h.parent.resolve()
    return None


def ensure_dirs(root: Path) -> None:
    for rel in (
        "data", "data/raw_ohlcv", "data/hist", "data/Historical",
        "models", "Logs", "logs", "bridge", "tests", "state",
        "production", "config", "features", "runs",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)


def run_in_project(root: Path, script: str, extra: List[str] | None = None) -> int:
    path = root / script
    if not path.is_file():
        log(f"missing {path}")
        # try find by name
        hits = find_by_name(script, limit=3)
        if hits:
            log(f"found elsewhere: {hits[0]}")
            path = hits[0]
        else:
            return 1
    cmd = [sys.executable, "-u", str(path)] + (extra or [])
    log(f"exec: {' '.join(cmd)}")
    log(f"cwd : {root}")
    return subprocess.call(cmd, cwd=str(root))


def cmd_doctor(root: Optional[Path]) -> int:
    log(f"python = {sys.executable} ({sys.version.split()[0]})")
    log(f"cwd    = {Path.cwd()}")
    if root is None:
        log("project root NOT found")
        log("searched Documents/FXJEFE_Project and by filename markers")
        for m in MARKER_FILES:
            hits = find_by_name(m, limit=5)
            for h in hits:
                log(f"  hit {m}: {h}")
        log("FIX: cd ~/Documents/FXJEFE_Project")
        return 1
    log(f"project root = {root}")
    ensure_dirs(root)
    for name in (
        "feature_registry.py",
        "script_inventory.py",
        "pipelinerun_production.py",
        "secure_strap.py",
        "pipelinerun.py",
        "config.json",
        "venv",
    ):
        p = root / name
        status = "OK" if (p.is_file() or p.is_dir()) else "MISSING"
        log(f"  [{status:7}] {name}")
    last = root / "state" / "last_production_run.json"
    log(f"  last_production_run = {'yes' if last.is_file() else 'no (run production first)'}")
    strap = root / "production" / "STRAP_SEAL.json"
    log(f"  strap_seal = {'yes' if strap.is_file() else 'no'}")
    return 0


def cmd_ports() -> int:
    """Show firewall / WebRequest port guidance for MT5 + AI server."""
    log("MT5 WebRequest + AI server ports")
    log("AI server default: 8080 (http://127.0.0.1:8080)")
    log("Bridge / FastAPI often: 8000, ZMQ 5555")
    log("Trading: 443, 8443")
    if sys.platform == "darwin":
        log("macOS: System Settings → Network → Firewall → Options")
        log("  Allow python3 / Terminal for incoming if server binds 0.0.0.0")
        log("  MT5 itself runs in Windows VM — open ports on the VM guest")
        log("MT5 terminal: Tools → Options → Expert Advisors")
        log("  Allow WebRequest for listed URL: http://127.0.0.1:8080")
        log("  Add also http://192.168.x.x:8080 if EA is on VM and server on Mac")
    elif sys.platform == "win32":
        log("Run in Admin PowerShell:")
        for p in (8080, 8000, 5555, 443, 8443):
            log(
                f'  New-NetFirewallRule -DisplayName "FXJEFE-{p}" '
                f"-Direction Inbound -Protocol TCP -LocalPort {p} -Action Allow"
            )
        log("MT5: Tools → Options → Expert Advisors → Allow WebRequest URL")
    else:
        log("Linux:")
        for p in (8080, 8000, 5555):
            log(f"  sudo ufw allow {p}/tcp comment FXJEFE")
    return 0


def cmd_framework(root: Path) -> int:
    ensure_dirs(root)
    # seed config.json if missing without overwriting
    cfg = root / "config.json"
    if not cfg.is_file():
        tpl = root / "config" / "config_template.json"
        import json
        data = {
            "project_root": str(root),
            "scripts_path": str(root),
            "models_path": str(root / "models"),
            "data_path": str(root / "data"),
            "data_output_path": str(root / "data"),
            "log_path": str(root / "Logs"),
            "ai_server_url": "http://127.0.0.1:8080",
            "min_confidence_threshold": 0.77,
            "ohlcv_columns": ["open", "high", "low", "close", "volume"],
            "feature_registry_locked": True,
        }
        if tpl.is_file():
            try:
                data.update(json.loads(tpl.read_text(encoding="utf-8")))
            except Exception:
                pass
        data["project_root"] = str(root)
        cfg.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log(f"wrote {cfg}")
    else:
        log(f"exists_preserved {cfg}")
    log("framework folders ready")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="FXJEFE smart path helper")
    ap.add_argument(
        "command",
        nargs="?",
        default="doctor",
        choices=["doctor", "find", "run", "ports", "framework", "where"],
    )
    ap.add_argument("target", nargs="?", default="")
    ap.add_argument("extra", nargs="*", default=[])
    args = ap.parse_args()

    root = detect_project()

    if args.command in ("doctor", "where"):
        return cmd_doctor(root)

    if args.command == "find":
        name = args.target or "feature_registry.py"
        hits = find_by_name(name)
        if not hits:
            log(f"no hits for {name}")
            return 1
        for h in hits:
            log(str(h))
        return 0

    if args.command == "ports":
        return cmd_ports()

    if root is None:
        log("Cannot find FXJEFE project. Create/install first:")
        log("  mkdir -p ~/Documents/FXJEFE_Project")
        log("  python3 ~/Downloads/FXJEFE_install_kit.py")
        return 1

    if args.command == "framework":
        return cmd_framework(root)

    if args.command == "run":
        target = (args.target or "").lower()
        mapping = {
            "inventory": ("script_inventory.py", []),
            "production": ("pipelinerun_production.py", list(args.extra)),
            "strap": ("secure_strap.py", []),
            "pipeline": ("pipelinerun.py", list(args.extra)),
            "validate": ("path_validate.py", ["--print-map"]),
        }
        if target not in mapping:
            log("run targets: inventory | production | strap | pipeline | validate")
            return 1
        script, extra = mapping[target]
        if target == "strap":
            last = root / "state" / "last_production_run.json"
            if not last.is_file():
                log("No last_production_run.json yet.")
                log("Run production to success FIRST:")
                log(f"  cd {root}")
                log("  python3 pipelinerun_production.py")
                log("Then:")
                log("  python3 secure_strap.py")
                return 1
        return run_in_project(root, script, extra)

    return cmd_doctor(root)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[FXJEFE] interrupted", flush=True)
        raise SystemExit(130)
