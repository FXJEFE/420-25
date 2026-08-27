"""
check_model_features.py
Verify a *current-contract* model matches config['features'].

Never treat OG Documents/models/my_model.pkl as the required check —
that file is 29-f legacy and must stay untouched. Prefer the newest
versioned my_model_*.pkl under og333_runs or FXJEFE_Project/models.
"""
import os
import json
import logging
from pathlib import Path

import joblib

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

os.makedirs(config['log_path'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config['log_path'], 'check_model_features.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def _home() -> Path:
    return Path(os.environ.get("USERPROFILE") or Path.home())


def candidate_model_paths():
    roots = [
        _home() / "Documents" / "models" / "og333_runs",
        Path(config.get("project_models_path") or "") / "og333_runs",
        Path(config.get("project_models_path") or ""),
        Path(__file__).resolve().parent / "models",
    ]
    found = []
    for root in roots:
        if not root or not root.is_dir():
            continue
        for pat in ("my_model_v*.pkl", "my_model_20*.pkl"):
            found.extend(root.glob(pat))
    # newest first
    found = sorted({p.resolve() for p in found if p.is_file()},
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return found


def main():
    n_config = len(config['features'])
    og_path = Path(config['models_path']) / "my_model.pkl"
    if og_path.is_file():
        try:
            og_n = getattr(joblib.load(og_path), "n_features_in_", None)
            logging.info("OG my_model.pkl n_features_in_=%s (legacy — not the required check)", og_n)
        except Exception as e:
            logging.warning("OG my_model.pkl unreadable: %s", e)

    candidates = candidate_model_paths()
    if not candidates:
        logging.error("No versioned my_model_*.pkl found under og333_runs or project models.")
        logging.error("Run train_models.py first (it writes a timestamped file, never OG my_model.pkl).")
        raise FileNotFoundError("versioned my_model_*.pkl")

    last_err = None
    for model_path in candidates:
        try:
            model = joblib.load(model_path)
        except Exception as e:
            logging.warning("skip %s: %s", model_path, e)
            last_err = e
            continue
        n_model = getattr(model, "n_features_in_", None)
        status = "OK" if n_model == n_config else "MISMATCH"
        logging.info("Checking %s", model_path)
        logging.info("Model feature count : %s", n_model)
        logging.info("Config feature count: %s  (%s)", n_config, config['features'])
        logging.info("Feature count check : %s", status)
        if status == "OK":
            return
        last_err = ValueError(
            f"Feature count mismatch — {model_path.name} expects {n_model}, config has {n_config}."
        )

    raise last_err or ValueError("No versioned model matched config['features'].")


if __name__ == '__main__':
    main()
