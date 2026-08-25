# -*- coding: utf-8 -*-
"""Ensure pipeline scripts folder exists and stamp UTF-8/config bootstrap if missing."""
from fxjefe_paths import load_config, setup_logging, DOCUMENTS, PROJECT
import logging
import os

config = load_config()
setup_logging(config, "update_scripts")

# Prefer Documents (canonical pipeline scripts) + project Scripts
SCRIPT_DIRS = [
    config.get("scripts_path") or DOCUMENTS,
    config.get("project_scripts_path") or os.path.join(PROJECT, "Scripts"),
    os.path.join(config.get("project_root") or PROJECT, "Scripts"),
]

BOOT_MARK = "fxjefe_paths"
BOOT_SNIPPET = '''# -*- coding: utf-8 -*-
import os as _os_utf8, sys as _sys_utf8
_os_utf8.environ.setdefault("PYTHONUTF8", "1")
_os_utf8.environ.setdefault("PYTHONIOENCODING", "utf-8")
'''


def main() -> None:
    updated = 0
    scanned = 0
    for d in SCRIPT_DIRS:
        if not d:
            continue
        os.makedirs(d, exist_ok=True)
        logging.info("Scanning scripts dir: %s", d)
        try:
            names = os.listdir(d)
        except OSError as e:
            logging.warning("Cannot list %s: %s", d, e)
            continue
        for filename in names:
            if not filename.endswith(".py"):
                continue
            if filename.startswith("_") or filename in ("fxjefe_paths.py", "pipeline_utf8.py"):
                continue
            filepath = os.path.join(d, filename)
            if not os.path.isfile(filepath):
                continue
            scanned += 1
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError as e:
                logging.warning("read fail %s: %s", filepath, e)
                continue
            if "PYTHONUTF8" in content or BOOT_MARK in content:
                continue
            # only light-touch: prepend utf8 env if completely missing
            if content.startswith("\ufeff"):
                content = content.lstrip("\ufeff")
            new = BOOT_SNIPPET + content
            try:
                with open(filepath, "w", encoding="utf-8", newline="\n") as f:
                    f.write(new)
                updated += 1
                logging.info("stamped UTF-8 bootstrap: %s", filepath)
            except OSError as e:
                logging.warning("write fail %s: %s", filepath, e)

    # always ensure project Scripts exists as empty-ok
    ps = config.get("project_scripts_path") or os.path.join(PROJECT, "Scripts")
    os.makedirs(ps, exist_ok=True)
    marker = os.path.join(ps, "README_SCRIPTS.txt")
    if not os.path.isfile(marker):
        with open(marker, "w", encoding="utf-8") as f:
            f.write("Pipeline scripts live in Documents; this folder is for project extras.\n")
    logging.info("update_scripts done scanned=%s updated=%s", scanned, updated)


if __name__ == "__main__":
    main()
