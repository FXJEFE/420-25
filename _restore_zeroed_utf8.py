#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Restore zero-filled project text files from known good UTF-8 copies."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\locallarry\Documents\FXJEFE_Project")
SEARCH = [
    ROOT / "_FLAT_all_mqh",
    ROOT / "_FLAT_all_mq5",
    Path(r"C:\Users\locallarry\Documents\Include"),
    Path(r"C:\Users\locallarry\Documents"),
    Path(r"C:\Users\locallarry\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Include"),
    Path(r"C:\Users\locallarry\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Experts"),
    Path(r"C:\Users\locallarry\Documents\mq5_source\usb_new"),
]
TEXT_EXT = {".py", ".json", ".csv", ".txt", ".md", ".mq5", ".mqh", ".mq4", ".log"}
SKIP_DIRS = {"venv", ".venv", "__pycache__", "_FLAT_all_exe", "libzmq", "vendor"}


def good_text(raw: bytes) -> bool:
    if not raw or raw.count(b"\x00") / len(raw) > 0.05:
        return False
    if raw.startswith(b"\x00\x05\x16\x07"):
        return False
    try:
        t = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            t = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return False
    printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in t[:400])
    return printable >= 20


def find_good(name: str, dest: Path) -> Path | None:
    dest_res = dest.resolve()
    for folder in SEARCH:
        if not folder.is_dir():
            continue
        cand = folder / name
        if cand.is_file() and cand.resolve() != dest_res and good_text(cand.read_bytes()):
            return cand
    return None


def main() -> None:
    restored = []
    still = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        dp = Path(dirpath)
        if any(part in SKIP_DIRS for part in dp.parts):
            continue
        # do not overwrite the good copies we search
        if dp.name in ("_FLAT_all_mqh", "_FLAT_all_mq5"):
            continue
        for name in filenames:
            p = dp / name
            if p.suffix.lower() not in TEXT_EXT:
                continue
            try:
                raw = p.read_bytes()
            except OSError:
                continue
            if not raw or raw.count(b"\x00") / len(raw) < 0.95:
                continue
            src = find_good(name, p)
            if src is None:
                still.append(str(p.relative_to(ROOT)))
                continue
            bak = p.with_suffix(p.suffix + ".zero.bak")
            if not bak.exists():
                shutil.copy2(p, bak)
            data = src.read_bytes()
            # persist as UTF-8 no BOM
            if data.startswith(b"\xef\xbb\xbf"):
                data = data[3:]
            text = data.decode("utf-8", errors="replace")
            if text and not text.endswith("\n"):
                text += "\n"
            p.write_bytes(text.encode("utf-8"))
            restored.append((str(p.relative_to(ROOT)), str(src)))
    print("restored", len(restored))
    for dest, src in restored:
        print(f"  {dest} <- {src}")
    print("still_zeroed", len(still))
    for s in still:
        print("  ", s)


if __name__ == "__main__":
    main()
