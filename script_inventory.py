# -*- coding: utf-8 -*-
"""
FXJEFE script inventory — NEVER REPLACE CURRENT SCRIPTS by default.

Policy (mentor / live-forever):
  - Existing files in Documents/FXJEFE_Project are SACRED.
  - Default action for an existing dest = SKIP (not overwrite).
  - Missing scripts may be installed from discovery sources.
  - Optional --force-backup-replace only after explicit user intent;
    always writes a timestamped .bak first.
  - Production shadow tree production/script_mirrors/ can hold copies
    without touching live scripts.

One broken script must not require mass-overwrite of the whole tree.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEST_MAP: Dict[str, str] = {
    "pipelinerun.py": ".",
    "pipelinerun_production.py": ".",
    "run_pipeline.py": ".",
    "path_resolver.py": ".",
    "path_loader.py": ".",
    "path_validate.py": ".",
    "feature_registry.py": ".",
    "feature_lock.py": ".",
    "fxjefe_bootstrap.py": ".",
    "fxjefe_local_framework.py": ".",
    "secure_strap.py": ".",
    "script_inventory.py": ".",
    "mt5_data_sync.py": ".",
    "mt5_generate_features.py": ".",
    "mt5_generate_crypto_features.py": ".",
    "validate_data.py": ".",
    "Load_and_Process.py": ".",
    "generate_labels.py": ".",
    "feature_engineering.py": ".",
    "full_pipeline.py": ".",
    "clean_training_data.py": ".",
    "fix_csv.py": ".",
    "fix_csv_encoding.py": ".",
    "adjust_headers.py": ".",
    "convert_encoding.py": ".",
    "generate_training_data.py": ".",
    "generate_new_csv.py": ".",
    "train_models.py": ".",
    "check_model_features.py": ".",
    "ensemble_predictions.py": ".",
    "generate_signals_with_xgboost.py": ".",
    "model_deploy.py": ".",
    "ai_server.py": "bridge",
    "fxjefe_xgboost_api.py": "bridge",
    "waitress server.py": "bridge",
    "signal_processor.py": ".",
    "risk_management.py": ".",
    "process_trades.py": ".",
    "analyze_outcomes.py": ".",
    "merge_datasets.py": ".",
    "update_database.py": ".",
    "test_local_trading_models.py": "tests",
    "test_models_with_shap.py": "tests",
    "test_server.py": "tests",
    "test_encoding.py": "tests",
    "test_regex.py": "tests",
    "check_integrity.py": "tests",
    "check_labels.py": "tests",
    "log_summary.py": "tests",
    "logging_utils.py": ".",
    "create_structure.py": ".",
    "update_scripts.py": "tests",
    "parse_log_to_csv.py": "tests",
    "get_lstm_prediction.py": "tests",
    "mt5_signal_script.py": "bridge",
}

PRODUCTION_ORDER: List[str] = [
    "path_validate.py",
    "mt5_data_sync.py",
    "mt5_generate_features.py",
    "validate_data.py",
    "Load_and_Process.py",
    "adjust_headers.py",
    "generate_labels.py",
    "feature_engineering.py",
    "train_models.py",
    "check_model_features.py",
    "full_pipeline.py",
    "signal_processor.py",
    "model_deploy.py",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def project_root() -> Path:
    if sys.platform == "win32":
        home = Path(os.environ.get("USERPROFILE", str(Path.home())))
    else:
        home = Path.home()
    return home / "Documents" / "FXJEFE_Project"


def discover_scripts(search_roots: List[Path]) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for root in search_roots:
        if not root or not Path(root).exists():
            continue
        for p in Path(root).rglob("*.py"):
            if any(x in p.parts for x in (".venv", "venv", "__pycache__", "site-packages", "production")):
                continue
            name = p.name
            prev = found.get(name)
            if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
                found[name] = p
    return found


def relocate(
    found: Dict[str, Path],
    root: Optional[Path] = None,
    *,
    dry_run: bool = False,
    force_backup_replace: bool = False,
    mirror_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    DO NOT REPLACE CURRENT SCRIPTS unless force_backup_replace=True.
    mirror_only=True writes under production/script_mirrors/ only.
    """
    root = root or project_root()
    results = []
    for name, src in sorted(found.items()):
        rel = DEST_MAP.get(name, "scripts_extra")
        if mirror_only:
            dest_dir = root / "production" / "script_mirrors" / (rel if rel != "." else "root")
        else:
            dest_dir = root / rel if rel != "." else root
        dest = dest_dir / name
        entry: Dict[str, Any] = {
            "name": name,
            "src": str(src),
            "dest": str(dest),
            "action": "skip",
        }
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if _sha256(dest) == _sha256(src):
                    entry["action"] = "identical_skip"
                    results.append(entry)
                    print(f"[FXJEFE-INVENTORY] identical_skip {name}", flush=True)
                    continue
                if not force_backup_replace and not mirror_only:
                    # SACRED — never overwrite live script
                    entry["action"] = "exists_preserved"
                    entry["note"] = "DO NOT REPLACE CURRENT SCRIPTS — left untouched"
                    results.append(entry)
                    print(f"[FXJEFE-INVENTORY] exists_preserved {name} (live file kept)", flush=True)
                    continue
                # explicit force path only
                bak = dest.with_suffix(dest.suffix + f".bak_{datetime.now().strftime('%Y%m%d%H%M%S')}")
                if not dry_run:
                    shutil.copy2(dest, bak)
                entry["backup"] = str(bak)
                entry["action"] = "force_replaced_with_backup"
            else:
                entry["action"] = "installed_missing"

            if not dry_run and entry["action"] in ("installed_missing", "force_replaced_with_backup"):
                text = src.read_text(encoding="utf-8", errors="replace")
                if name == "mt5_signal_script.py":
                    text = text.replace('password = "*pb5BEU?s4f"', 'password = os.environ.get("MT5_PASSWORD", "")')
                    text = text.replace("account = 1512751258", 'account = int(os.environ.get("MT5_ACCOUNT", "0"))')
                    dest.write_text(text, encoding="utf-8")
                else:
                    shutil.copy2(src, dest)
            elif not dry_run and mirror_only and not dest.exists():
                shutil.copy2(src, dest)
                entry["action"] = "mirrored"
        except Exception as e:
            entry["action"] = "error"
            entry["error"] = str(e)
        results.append(entry)
        print(f"[FXJEFE-INVENTORY] {entry['action']:28} {name}", flush=True)
    return results


def verify_present(root: Optional[Path] = None) -> Dict[str, Any]:
    root = root or project_root()
    present, missing = [], []
    for name in sorted(set(list(DEST_MAP) + PRODUCTION_ORDER)):
        rel = DEST_MAP.get(name, ".")
        dest = (root / rel / name) if rel != "." else (root / name)
        if dest.is_file():
            present.append(name)
        else:
            missing.append(name)
    prod_ok = all(
        ((root / DEST_MAP.get(n, ".") / n) if DEST_MAP.get(n, ".") != "." else (root / n)).is_file()
        for n in PRODUCTION_ORDER
    )
    return {
        "present": present,
        "missing": missing,
        "present_count": len(present),
        "missing_count": len(missing),
        "production_ready": prod_ok,
        "policy": "DO_NOT_REPLACE_CURRENT_SCRIPTS",
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-backup-replace", action="store_true",
                    help="DANGEROUS: replace existing after .bak — default is NEVER replace")
    ap.add_argument("--mirror-only", action="store_true",
                    help="Only write production/script_mirrors/ copies")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    roots = [
        Path("/home/workdir/attachments"),
        project_root(),
        project_root() / "Scripts",
        Path(__file__).resolve().parent,
    ]
    print("[FXJEFE-INVENTORY] policy=DO_NOT_REPLACE_CURRENT_SCRIPTS", flush=True)
    found = discover_scripts(roots)
    results = relocate(
        found,
        dry_run=args.dry_run,
        force_backup_replace=args.force_backup_replace,
        mirror_only=args.mirror_only,
    )
    ver = verify_present()
    out = project_root() / "state" / "script_inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"relocate": results, "verify": ver}, indent=2), encoding="utf-8")
    print(f"[FXJEFE-INVENTORY] present={ver['present_count']} missing={ver['missing_count']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
