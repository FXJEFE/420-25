#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Idempotent FXJEFE port alignment: 127.0.0.1:8080 → 127.0.0.1:8080
- Only touches text files under project root
- Skips binary / huge files
- No-op when already correct
- Prints a clear summary
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

OLD = "127.0.0.1:8080"
NEW = "127.0.0.1:8080"

# Only scan these relative roots (keeps the walk focused)
SCAN_DIRS = (
    ".",
    "config",
    "bridge",
    "pipeline",
    "state",
    "scripts",
)

# File extensions we rewrite
TEXT_EXT = {".py", ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".env", ".md", ".txt", ".sh", ".ps1"}

# Never touch these path fragments
SKIP_PARTS = {
    ".git", "__pycache__", "venv", ".venv", "node_modules",
    "mlruns", "models", "data", "runs", "production",
}

MAX_BYTES = 2_000_000  # skip huge files


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & SKIP_PARTS:
        return True
    if path.suffix.lower() not in TEXT_EXT:
        return True
    try:
        if path.stat().st_size > MAX_BYTES:
            return True
    except OSError:
        return True
    return False


def fix_file(path: Path) -> str:
    """Return 'updated' | 'ok' | 'skip' | 'error'."""
    if should_skip(path):
        return "skip"
    try:
        raw = path.read_bytes()
    except OSError:
        return "error"
    # Skip obvious binary
    if b"\x00" in raw[:4096]:
        return "skip"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return "skip"

    if OLD not in text:
        return "ok"

    new_text = text.replace(OLD, NEW)
    if new_text == text:
        return "ok"

    try:
        path.write_text(new_text, encoding="utf-8", newline="\n")
    except OSError:
        return "error"
    return "updated"


def main() -> int:
    root = Path(os.environ.get("FXJEFE_PROJECT_ROOT") or Path.cwd()).resolve()
    if not (root / "feature_registry.py").exists() and not (root / "pipelinerun_production.py").exists():
        print(f"[WARN] {root} does not look like FXJEFE_Project — continuing anyway")

    counts = {"updated": 0, "ok": 0, "skip": 0, "error": 0}
    updated_files: list[str] = []

    for rel in SCAN_DIRS:
        base = (root / rel).resolve()
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            # Stay under root
            try:
                path.relative_to(root)
            except ValueError:
                continue
            status = fix_file(path)
            counts[status] += 1
            if status == "updated":
                updated_files.append(str(path.relative_to(root)))

    print("=== FXJEFE port fix (idempotent) ===")
    print(f"root     : {root}")
    print(f"replace  : {OLD}  →  {NEW}")
    print(f"updated  : {counts['updated']}")
    print(f"already  : {counts['ok']}")
    print(f"skipped  : {counts['skip']}")
    print(f"errors   : {counts['error']}")
    if updated_files:
        print("files changed:")
        for f in updated_files:
            print(f"  - {f}")
    else:
        print("No changes needed (already aligned or nothing matched).")
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())