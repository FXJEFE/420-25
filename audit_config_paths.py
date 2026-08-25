#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scan all project .py files for config path usage."""
from __future__ import annotations
import json, os, re
from pathlib import Path

ROOT = Path.home() / "Documents" / "FXJEFE_Project"
if not ROOT.is_dir():
    ROOT = Path(__file__).resolve().parent

# Simple substring markers — avoid fragile regex escaping
MARKERS = [
    "config.json",
    "FXJEFE_CONFIG",
    "config_path",
    "CONFIG_PATH",
    "load_config",
    "read_config",
]

def effective_config() -> Path:
    env = os.environ.get("FXJEFE_CONFIG", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (ROOT / "config.json").resolve()

def scan_file(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"file": str(path.relative_to(ROOT)), "error": str(e)}
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        low = line.lower()
        if any(m.lower() in low or m in line for m in MARKERS):
            hits.append({"line": i, "text": line.strip()[:120]})
    return {
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "mentions_config": len(hits) > 0,
        "hit_count": len(hits),
        "hits": hits[:8],
    }

def main():
    print("ROOT", ROOT)
    cfg = effective_config()
    print("EFFECTIVE_CONFIG", cfg)
    print("EXISTS", cfg.is_file())
    if cfg.is_file():
        try:
            d = json.loads(cfg.read_text(encoding="utf-8"))
            print("KEYS", len(d))
            print("policy", d.get("feature_policy"))
            print("gate", d.get("min_confidence_threshold"))
            for k in ("log_path", "data_path", "models_path", "scripts_path", "historical_csv", "ai_server_url", "api_port"):
                if k in d:
                    print(f"  cfg[{k}]={d.get(k)!r}")
        except Exception as e:
            print("JSON_ERROR", e)
    print("---")
    rows = []
    for p in sorted(ROOT.rglob("*.py")):
        if any(x in p.parts for x in (".git", "fxjefe", "venv", "__pycache__", "site-packages")):
            continue
        rows.append(scan_file(p))
    with_cfg = [r for r in rows if r.get("mentions_config")]
    without = [r for r in rows if not r.get("mentions_config") and not r.get("error")]
    print(f"scripts_total={len(rows)} with_config_ref={len(with_cfg)} without={len(without)}")
    print()
    print("=== SCRIPTS THAT REFERENCE CONFIG ===")
    for r in with_cfg:
        print(f"\n{r['file']}  hits={r['hit_count']}")
        for h in r.get("hits") or []:
            print(f"  L{h['line']}: {h['text']}")
    print()
    print("=== NO CONFIG REFERENCE ===")
    for r in without:
        print(r["file"])
    out = ROOT / "production" / "config_path_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "root": str(ROOT), "effective_config": str(cfg), "exists": cfg.is_file(),
        "with_config": with_cfg, "without_config": [r["file"] for r in without],
    }, indent=2), encoding="utf-8")
    print("\nwrote", out)

if __name__ == "__main__":
    main()
