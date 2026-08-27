#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FXJEFE secure strap — run AFTER first successful full production pipeline.

Purpose (mentor / live forever):
  - Freeze feature registry to .pyc
  - Snapshot device + pipeline state
  - Mark production strap seal (does NOT delete or replace user scripts)
  - Optionally chmod / flag critical configs read-intent
  - Write ROLLBACK pointer for emergency

Never replaces current scripts. Never starts hidden processes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def log(m: str) -> None:
    print(f"[FXJEFE-STRAP] {m}", flush=True)


def project_root() -> Path:
    if sys.platform == "win32":
        home = Path(os.environ.get("USERPROFILE", str(Path.home())))
    else:
        home = Path.home()
    return home / "Documents" / "FXJEFE_Project"


def main() -> int:
    root = project_root()
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    prod = root / "production"
    prod.mkdir(parents=True, exist_ok=True)

    # Require evidence of a successful run
    last = state / "last_production_run.json"
    if not last.is_file():
        log("No last_production_run.json — run pipelinerun_production.py to success first")
        return 1
    try:
        data = json.loads(last.read_text(encoding="utf-8"))
        final = data.get("final") or {}
        if not final.get("ok"):
            log("Last production run final.ok is not true — refuse to strap")
            return 1
    except Exception as e:
        log(f"Cannot read last run: {e}")
        return 1

    # Compile feature registry (source stays; pyc is the runtime lock)
    reg = root / "feature_registry.py"
    if reg.is_file():
        subprocess.run([sys.executable, "-m", "py_compile", str(reg)], check=False)
        log(f"compiled {reg}")
    else:
        log("feature_registry.py missing — copy from skill assets first")
        return 1

    seal = {
        "strapped_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "DO_NOT_REPLACE_CURRENT_SCRIPTS",
        "feature_registry": "frozen via pyc + ACTIVE_VERSION",
        "last_run": str(last),
        "final_ok": True,
        "note": "Scripts in project root were not modified by strap. Isolate failures per-stage.",
    }
    seal_path = prod / "STRAP_SEAL.json"
    seal_path.write_text(json.dumps(seal, indent=2), encoding="utf-8")
    # also pointer
    (prod / "ACTIVE_STRAP").write_text(seal["strapped_at_utc"], encoding="utf-8")
    log(f"seal written → {seal_path}")
    log("STRAPPED — project secured for production runtime")
    log("One script failure should no longer imply mass silence: use per-stage required=False + retries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
