#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Versioned .pyc compile — NEVER replaces OG .py scripts.
Writes under production/pyc-vYYYYMMDD-HHMMSS-vN/ + VERSION.json
"""
from __future__ import annotations
import hashlib, json, os, py_compile, sys
from datetime import datetime, timezone
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

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def next_version_dir(prod: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    n = len(list(prod.glob("pyc-v*"))) + 1
    d = prod / f"pyc-v{stamp}-v{n}"
    d.mkdir(parents=True, exist_ok=False)
    return d

def load_manifest(root: Path) -> list:
    man = root / "pipeline_manifest.json"
    if man.is_file():
        data = json.loads(man.read_text(encoding="utf-8"))
        return [s["script"] for s in data.get("stages", [])]
    return []

def main() -> int:
    root = project_root()
    if not root.is_dir():
        root = Path.cwd()
    prod = root / "production"
    prod.mkdir(parents=True, exist_ok=True)
    last = root / "state" / "last_production_run.json"
    force = "--force" in sys.argv
    if last.is_file() and not force:
        try:
            final = (json.loads(last.read_text(encoding="utf-8")).get("final") or {})
            if not final.get("ok"):
                print("[COMPILE] last run not ok — use --force to compile anyway")
                return 1
        except Exception:
            pass
    elif not last.is_file() and not force:
        print("[COMPILE] no last_production_run.json — run pipeline first or pass --force")
        return 1
    scripts = load_manifest(root)
    extra = [
        "feature_registry.py", "feature_hash.py", "signal_gate.py",
        "runtime_lock.py", "path_resolver.py", "pipelinerun_production.py",
        "secure_strap.py", "ai_server_golden.py", "ai_server.py",
        "compile_versioned_pyc.py",
    ]
    names = list(dict.fromkeys(scripts + extra))
    out_dir = next_version_dir(prod)
    print(f"[COMPILE] version dir = {out_dir}")
    print("[COMPILE] policy = NEVER replace OG .py scripts")
    compiled, skipped, sources_meta = [], [], {}
    for name in names:
        src = root / name
        if not src.is_file():
            skipped.append(name)
            continue
        cfile = out_dir / (src.stem + ".pyc")
        try:
            py_compile.compile(str(src), cfile=str(cfile), doraise=True)
            sources_meta[name] = {"sha256": sha256_file(src), "pyc": cfile.name, "bytes": src.stat().st_size}
            compiled.append(name)
            print(f"[COMPILE] ok {name} -> {cfile.name}")
        except Exception as e:
            print(f"[COMPILE] fail {name}: {e}")
            skipped.append(name)
    version = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "NEVER_REPLACE_OG_SCRIPTS",
        "venv_name": "fxjefe",
        "version_dir": str(out_dir.relative_to(root)),
        "compiled_count": len(compiled),
        "compiled": compiled,
        "skipped_missing": skipped,
        "sources": sources_meta,
    }
    (out_dir / "VERSION.json").write_text(json.dumps(version, indent=2), encoding="utf-8")
    (prod / "LATEST_PYC_VERSION").write_text(out_dir.name + "\n", encoding="utf-8")
    print(f"[COMPILE] VERSION.json ({len(compiled)} pyc) LATEST -> {out_dir.name}")
    return 0 if compiled else 1

if __name__ == "__main__":
    raise SystemExit(main())
