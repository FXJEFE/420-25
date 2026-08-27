# -*- coding: utf-8 -*-
"""Create/verify FXJEFE project folder structure under config project_root."""
from fxjefe_paths import load_config, setup_logging
import logging
import os

config = load_config()
setup_logging(config, "create_structure")

root_dir = config.get("project_root") or os.path.join(
    os.environ.get("USERPROFILE", r"C:\Users\locallarry"), "Documents", "FXJEFE_Project"
)

structure = {
    "config.json": None,  # skip overwrite if missing content
    "models": {},
    "data": {},
    "Logs": {},
    "Scripts": {},
    "logs": {},
}


def create_structure(base_path: str, tree: dict) -> None:
    os.makedirs(base_path, exist_ok=True)
    for name, content in tree.items():
        path = os.path.join(base_path, name)
        if content is None:
            # placeholder file only if absent
            if not os.path.exists(path) and name != "config.json":
                with open(path, "w", encoding="utf-8") as f:
                    f.write("")
            continue
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)


def main() -> None:
    logging.info("Ensuring structure under %s", root_dir)
    create_structure(root_dir, structure)
    # also ensure config paths
    for key in ("data_path", "log_path", "models_path", "project_scripts_path", "mt5_files_path"):
        p = config.get(key)
        if p:
            os.makedirs(p, exist_ok=True)
            logging.info("ok dir: %s", p)
    print(f"Folder structure verified: {root_dir}")
    logging.info("create_structure completed")


if __name__ == "__main__":
    main()
