#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import hashlib, json, sys
from pathlib import Path

def main() -> int:
    root = Path(__file__).resolve().parent
    sums = root / "checksums.json"
    if sums.is_file():
        expected = json.loads(sums.read_text(encoding="utf-8"))
    else:
        alt = root / "SHA256SUMS"
        if not alt.is_file():
            print("[SHA256] no checksums file"); return 1
        expected = {}
        for line in alt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                expected[parts[-1]] = parts[0]
    ok, bad, missing = 0, [], []
    for name, want in sorted(expected.items()):
        p = root / name
        if not p.is_file():
            missing.append(name); continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got == want:
            ok += 1; print(f"[SHA256] OK  {name}")
        else:
            bad.append(name); print(f"[SHA256] BAD {name}")
    print(f"[SHA256] ok={ok} bad={len(bad)} missing={len(missing)}")
    return 0 if not bad and not missing else 1

if __name__ == "__main__":
    raise SystemExit(main())
