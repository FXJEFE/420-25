#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FXJEFE encoding policy — UTF-8 ONLY. See repo README."""
from pathlib import Path
import argparse, json, sys, os

TEXT_SUFFIXES = {".py", ".csv", ".mq5", ".mqh", ".json", ".md", ".txt", ".env", ".tsv"}

def project_root():
    if sys.platform == "win32":
        return Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Documents" / "FXJEFE_Project"
    return Path.home() / "Documents" / "FXJEFE_Project"

def detect_encoding(raw: bytes) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        raw.decode("utf-8"); return "utf-8"
    except UnicodeDecodeError:
        pass
    for enc in ("cp1252", "latin-1", "cp437"):
        try:
            raw.decode(enc); return enc
        except UnicodeDecodeError:
            continue
    return "unknown"

def to_utf8(raw: bytes) -> bytes:
    enc = detect_encoding(raw)
    if enc in ("utf-8", "ascii"):
        return raw[3:] if raw.startswith(b"\xef\xbb\xbf") else raw
    if enc == "utf-8-sig":
        return raw[3:]
    if enc == "unknown":
        return raw.decode("utf-8", errors="replace").encode("utf-8")
    return raw.decode(enc).encode("utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--root", default="")
    args = ap.parse_args()
    root = Path(args.root) if args.root else project_root()
    report = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(x in p.parts for x in (".git", "venv", "__pycache__")):
            continue
        raw = p.read_bytes()
        enc = detect_encoding(raw)
        ok = enc in ("utf-8", "ascii")
        report.append({"path": str(p.relative_to(root)), "encoding": enc, "utf8_ok": ok})
        if args.fix and not ok:
            p.write_bytes(to_utf8(raw))
    print(json.dumps({"total": len(report), "non_utf8": sum(1 for r in report if not r["utf8_ok"]), "sample": report[:20]}, indent=2))
    return 0 if all(r["utf8_ok"] for r in report) else 1

if __name__ == "__main__":
    raise SystemExit(main())
