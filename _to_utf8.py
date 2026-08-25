#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rewrite recoverable FXJEFE_Project text files as UTF-8 (no BOM)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\locallarry\Documents\FXJEFE_Project")
DOCS = Path(r"C:\Users\locallarry\Documents")

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


def looks_zeroed(raw: bytes) -> bool:
    if not raw:
        return False
    nuls = raw.count(b"\x00")
    return nuls / len(raw) >= 0.95


def looks_appledouble(path: Path, raw: bytes) -> bool:
    if path.name.startswith("._"):
        return True
    return raw.startswith(b"\x00\x05\x16\x07")


def _odd_even_nul_ratio(raw: bytes) -> tuple[float, float]:
    """Return (odd-index NUL ratio, even-index NUL ratio)."""
    if len(raw) < 4:
        return 0.0, 0.0
    odd = raw[1::2]
    even = raw[0::2]
    return odd.count(b"\x00") / len(odd), even.count(b"\x00") / len(even)


def decode_utf16(raw: bytes, enc: str) -> str:
    return raw.decode(enc, errors="replace").lstrip("\ufeff").replace("\x00", "")


def decode_text(raw: bytes) -> tuple[str | None, str]:
    if not raw:
        return "", "empty"
    if looks_zeroed(raw):
        return None, "zeroed"
    if raw.startswith(b"\xff\xfe"):
        return decode_utf16(raw, "utf-16-le"), "utf-16-le-bom"
    if raw.startswith(b"\xfe\xff"):
        return decode_utf16(raw, "utf-16-be"), "utf-16-be-bom"
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            return raw.decode("utf-8-sig"), "utf-8-bom"
        except UnicodeDecodeError:
            return None, "utf-8-bom-bad"

    odd_nul, even_nul = _odd_even_nul_ratio(raw)
    # Typical UTF-16 LE ASCII/source: nearly every odd byte is NUL
    if odd_nul >= 0.40 and odd_nul > even_nul + 0.20:
        return decode_utf16(raw, "utf-16-le"), "utf-16-le"
    if even_nul >= 0.40 and even_nul > odd_nul + 0.20:
        return decode_utf16(raw, "utf-16-be"), "utf-16-be"

    try:
        text = raw.decode("utf-8")
        if "\x00" in text:
            nuls = text.count("\x00")
            if nuls / max(len(text), 1) > 0.2:
                return decode_utf16(raw, "utf-16-le"), "utf-16-le"
        return text, "utf-8"
    except UnicodeDecodeError:
        pass
    nuls = raw.count(b"\x00")
    if nuls > 8 and nuls / len(raw) > 0.15:
        for enc in ("utf-16-le", "utf-16-be"):
            try:
                t = decode_utf16(raw, enc)
                printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in t[:400])
                if printable >= 20:
                    return t, enc
            except Exception:
                continue
        return None, "binary-nuls"
    for enc in ("cp1252", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return None, "unknown"


def write_utf8(path: Path, text: str) -> None:
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_bytes(text.encode("utf-8"))


def find_restore(path: Path) -> Path | None:
    """Find a non-zeroed UTF-8-ish copy in Documents or project root."""
    name = path.name
    cands = [
        DOCS / name,
        ROOT / name,
        DOCS / "FXJEFE_Project" / name,
    ]
    seen = {path.resolve()}
    for c in cands:
        try:
            if not c.is_file():
                continue
            if c.resolve() in seen:
                continue
            raw = c.read_bytes()
            if not raw or looks_zeroed(raw) or looks_appledouble(c, raw):
                continue
            text, kind = decode_text(raw)
            if text is None:
                continue
            # require some printable content
            printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in text[:400])
            if printable < 20:
                continue
            if kind in ("utf-8", "utf-8-bom", "cp1252", "latin-1", "utf-16-le", "utf-16-le-bom"):
                return c
        except OSError:
            continue
    return None


def main() -> None:
    converted = []
    restored = []
    already = 0
    skipped_zero = []
    skipped_other = []
    errors = []

    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        dp = Path(dirpath)
        if skip_dir(dp):
            continue
        for name in filenames:
            p = dp / name
            if p.suffix.lower() not in TEXT_EXT:
                continue
            try:
                raw = p.read_bytes()
            except OSError as e:
                errors.append((str(p), str(e)))
                continue
            if looks_appledouble(p, raw):
                skipped_other.append((str(p.relative_to(ROOT)), "appledouble"))
                continue
            text, kind = decode_text(raw)
            if kind == "utf-8":
                already += 1
                continue
            if kind == "empty":
                already += 1
                continue
            if kind == "zeroed":
                src = find_restore(p)
                if src is not None:
                    try:
                        src_raw = src.read_bytes()
                        src_text, src_kind = decode_text(src_raw)
                        if src_text is None:
                            skipped_zero.append(str(p.relative_to(ROOT)))
                            continue
                        bak = p.with_suffix(p.suffix + ".zero.bak")
                        if not bak.exists():
                            shutil.copy2(p, bak)
                        write_utf8(p, src_text)
                        restored.append((str(p.relative_to(ROOT)), str(src), src_kind))
                    except OSError as e:
                        errors.append((str(p), str(e)))
                else:
                    skipped_zero.append(str(p.relative_to(ROOT)))
                continue
            if text is None:
                skipped_other.append((str(p.relative_to(ROOT)), kind))
                continue
            try:
                write_utf8(p, text)
                converted.append((str(p.relative_to(ROOT)), kind, len(raw)))
            except OSError as e:
                errors.append((str(p), str(e)))

    print("already_utf8_or_empty", already)
    print("converted", len(converted))
    for rel, kind, size in converted:
        print(f"  CONV [{kind}] {rel} ({size})")
    print("restored_from_copy", len(restored))
    for rel, src, kind in restored:
        print(f"  REST [{kind}] {rel} <- {src}")
    print("skipped_zeroed_unrecoverable", len(skipped_zero))
    for rel in skipped_zero:
        print(f"  ZERO {rel}")
    print("skipped_other", len(skipped_other))
    for rel, kind in skipped_other[:40]:
        print(f"  SKIP [{kind}] {rel}")
    if len(skipped_other) > 40:
        print(f"  ... +{len(skipped_other) - 40} more")
    print("errors", len(errors))
    for p, e in errors:
        print(f"  ERR {p}: {e}")


if __name__ == "__main__":
    main()
