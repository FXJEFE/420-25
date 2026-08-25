#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scan FXJEFE_Project text files for non-UTF-8 encodings."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(r"C:\Users\locallarry\Documents\FXJEFE_Project")
SKIP_DIRS = {
    "venv",
    ".venv",
    "__pycache__",
    "node_modules",
    ".git",
    "_FLAT_all_exe",
    "_FLAT_all_ex5",
    "_FLAT_all_ex4",
    "libzmq",
    "vendor",
    "numpy-mkl-wheels-main",
    "talib-build-main",
}
TEXT_EXT = {
    ".py",
    ".pyw",
    ".json",
    ".csv",
    ".txt",
    ".md",
    ".mq5",
    ".mqh",
    ".mq4",
    ".log",
    ".yml",
    ".yaml",
    ".ini",
    ".cfg",
    ".xml",
    ".ps1",
    ".bat",
    ".cmd",
    ".toml",
}


def skip_dir(p: Path) -> bool:
    return any(part in SKIP_DIRS for part in p.parts)


def classify(raw: bytes) -> str:
    if not raw:
        return "empty"
    if raw.startswith(b"\xff\xfe"):
        return "utf-16-le-bom"
    if raw.startswith(b"\xfe\xff"):
        return "utf-16-be-bom"
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            raw.decode("utf-8-sig")
            return "utf-8-bom"
        except UnicodeDecodeError:
            return "utf-8-bom-invalid"
    # UTF-16 LE without BOM: lots of NULs in even/odd pattern
    nul = raw.count(b"\x00")
    if nul > 8 and nul / max(len(raw), 1) > 0.20:
        try:
            raw.decode("utf-16-le")
            return "utf-16-le"
        except UnicodeDecodeError:
            try:
                raw.decode("utf-16-be")
                return "utf-16-be"
            except UnicodeDecodeError:
                return f"binary-nuls({nul})"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    for enc in ("cp1252", "latin-1"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "unknown"


def main() -> None:
    counts: dict[str, int] = {}
    bad: list[tuple[str, str, int]] = []
    n = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dp = Path(dirpath)
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if skip_dir(dp):
            continue
        for name in filenames:
            p = dp / name
            if p.suffix.lower() not in TEXT_EXT:
                continue
            try:
                raw = p.read_bytes()
            except OSError as e:
                print(f"READFAIL {p}: {e}")
                continue
            kind = classify(raw)
            counts[kind] = counts.get(kind, 0) + 1
            n += 1
            if kind != "utf-8" and kind != "empty":
                bad.append((kind, str(p.relative_to(ROOT)), len(raw)))
    print(f"scanned {n} text files")
    print("counts:")
    for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:20} {v}")
    print(f"\nnon-utf8 files: {len(bad)}")
    for kind, rel, size in sorted(bad)[:200]:
        print(f"  [{kind}] {rel} ({size} bytes)")
    if len(bad) > 200:
        print(f"  ... +{len(bad) - 200} more")


if __name__ == "__main__":
    main()
