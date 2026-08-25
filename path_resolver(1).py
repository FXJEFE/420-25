#!/usr/bin/env python3
"""
FXJEFE Path Resolver
Provides consistent path resolution and config loading for all scripts.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / 'config.json'


class Paths:
    def __init__(self, config: dict):
        self.project_root    = Path(config.get('project_root',    str(PROJECT_ROOT)))
        self.scripts_path    = Path(config.get('scripts_path',    str(PROJECT_ROOT / 'Scripts')))
        self.models_path     = Path(config.get('models_path',     str(PROJECT_ROOT / 'models')))
        self.data_path       = Path(config.get('data_path',       str(PROJECT_ROOT / 'data')))
        self.logs_path       = Path(config.get('log_path',        str(PROJECT_ROOT / 'Logs')))
        self.mt5_files_path  = Path(config.get('mt5_files_path',  ''))

        # Realtime data lives in MT5 Files folder (primary source)
        self.realtime_data_path = self.mt5_files_path

        # Ensure log directory exists
        self.logs_path.mkdir(parents=True, exist_ok=True)

    def get_log_path(self, filename: str) -> Path:
        return self.logs_path / filename

    def get_model_path(self, filename: str) -> Path:
        return self.models_path / filename

    def get_data_path(self, filename: str) -> Path:
        return self.data_path / filename


_paths = None
_config = None


def get_config() -> dict:
    global _config
    if _config is None:
        for enc in ['utf-8', 'utf-8-sig', 'cp1252', 'latin1']:
            try:
                with open(CONFIG_PATH, 'r', encoding=enc) as f:
                    _config = json.load(f)
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            except FileNotFoundError:
                print(f"config.json not found at {CONFIG_PATH}")
                return {}
        if _config is None:
            print("Could not parse config.json")
            return {}
    return _config


def get_paths() -> Paths:
    global _paths
    if _paths is None:
        _paths = Paths(get_config())
    return _paths
