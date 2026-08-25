# -*- coding: utf-8 -*-
"""
run_pipeline.py
Full FXJEFE trading pipeline — generates features from MT5, processes data,
trains models, and prepares the AI server for the EA.

PIPELINE ORDER (all steps run in sequence)
══════════════════════════════════════════
Step 1  mt5_generate_features.py  – Pull M1 bars from MT5, compute 27 indicators
                                     → data/FXJEFE_Features.csv
Step 2  validate_data.py          – Validate columns, row count, NaN report
Step 3  Load_and_Process.py       – Data quality check (columns + stats)
Step 4  generate_labels.py        – Compute future price, add buy/hold/sell labels
                                     → training_data.csv
Step 5  train_models.py           – Train legacy RandomForest (28 features)
                                     → models/my_model.pkl  (fallback for ai_server)
Step 6  check_model_features.py   – Verify legacy model feature count matches config
Step 7  full_pipeline.py          – Walk-forward XGB/LGB per-symbol models (MAIN)
                                     → models/{SYMBOL}_{TF}_binary_xgb.json
                                     → models/{SYMBOL}_{TF}_binary_features.json
Step 8  signal_processor.py       – SMA crossover reference signals
                                     → data/signals_output.csv

The AI server (python pipeline/ai_server.py) is auto-started if not running.
The EA (FXJEFE_ALGO_AI.mq5 / Predict.mq5) calls the server for live signals.

Usage:
  python run_pipeline.py
  python run_pipeline.py --retry 5 --verbose
  python run_pipeline.py --skip-server
  python run_pipeline.py --skip-mt5          (skip step 1 if data already exists)
"""

import json
import os
import sys
import subprocess
import logging
import time
import argparse

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MQL5_PATH    = os.path.join(
    os.path.expandvars(r'%APPDATA%'),
    r'MetaQuotes\Terminal\81A933A9AFC5DE3C23B15CAB19C63850\MQL5'
)
CONFIG_PATH  = os.path.join(PROJECT_ROOT, 'config.json')
LOG_DIR      = os.path.join(PROJECT_ROOT, 'Logs')
os.makedirs(LOG_DIR, exist_ok=True)


def load_config(path: str) -> dict:
    for enc in ['utf-8', 'utf-8-sig', 'cp1252']:
        try:
            with open(path, 'r', encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    print(f"[FATAL] Cannot load config: {path}")
    sys.exit(1)


# ── Ordered pipeline steps ──────────────────────────────────────────────────
#
# Each tuple: (script_filename, description, required?)
# required=True  → pipeline aborts if the step fails after retries
# required=False → pipeline logs a warning and continues
#
PIPELINE_STEPS = [
    ('mt5_generate_features_wine.py',  'Step 1 — Generate features from MT5 via Wine (2000 M1 bars × 6 symbols)',   True),
    ('validate_data.py',          'Step 2 — Validate raw feature data',                                True),
    ('Load_and_Process.py',       'Step 3 — Data quality check (columns + stats)',                     True),
    ('generate_labels.py',        'Step 4 — Compute future price & generate buy/hold/sell labels',     True),
    ('train_models.py',           'Step 5 — Train legacy RandomForest → my_model.pkl (fallback)',      True),
    ('check_model_features.py',   'Step 6 — Verify legacy model feature count matches config (28)',    True),
    ('full_pipeline.py',          'Step 7 — Walk-forward XGB/LGB per-symbol models (main)',            False),
    ('signal_processor.py',       'Step 8 — Generate SMA crossover reference signals',                 False),
]


# ── helpers ──────────────────────────────────────────────────────────────────

def check_server_health(url: str) -> bool:
    if not REQUESTS_AVAILABLE:
        return False
    try:
        r = requests.get(f"{url}/health", timeout=5)
        return r.status_code == 200 and r.json().get('status') == 'running'
    except Exception:
        return False


def start_ai_server(cfg: dict) -> bool:
    candidates = [
        os.path.join(PROJECT_ROOT, 'python pipeline', 'ai_server.py'),
        os.path.join(PROJECT_ROOT, 'ai_server.py'),
    ]
    server_path = next((p for p in candidates if os.path.exists(p)), None)
    if not server_path:
        logging.error("ai_server.py not found.")
        return False
    try:
        flags = subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
        subprocess.Popen([sys.executable, server_path], creationflags=flags)
        logging.info(f"Starting AI server: {server_path}")
        time.sleep(7)
        return True
    except Exception as e:
        logging.error(f"Failed to start AI server: {e}")
        return False


def run_step(script: str) -> bool:
    script_path = os.path.join(PROJECT_ROOT, script)
    if not os.path.exists(script_path):
        logging.error(f"  Script not found: {script_path}")
        return False
    # full_pipeline.py runs walk-forward + Optuna for many symbols — needs much longer
    timeout = 7200 if script == 'full_pipeline.py' else 600
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True, capture_output=True, text=True,
            timeout=timeout
        )
        stdout = result.stdout.strip()
        if stdout:
            for line in stdout.splitlines()[-5:]:   # last 5 lines
                logging.info(f"  [{script}] {line}")
        return True
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or '').strip()
        logging.error(f"  [{script}] FAILED (exit {e.returncode})")
        if stderr:
            for line in stderr.splitlines()[-10:]:
                logging.error(f"  [{script}]   {line}")
        return False
    except subprocess.TimeoutExpired:
        logging.error(f"  [{script}] TIMED OUT (600 s)")
        return False


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='FXJEFE Full Trading Pipeline')
    parser.add_argument('--config',       default=CONFIG_PATH)
    parser.add_argument('--retry',        type=int, default=3)
    parser.add_argument('--verbose',      action='store_true')
    parser.add_argument('--skip-server',  action='store_true',
                        help='Skip AI server health check / auto-start')
    parser.add_argument('--skip-mt5',     action='store_true',
                        help='Skip MT5 data generation (use existing CSV)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, 'pipeline.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    cfg     = load_config(args.config)
    ai_url  = cfg.get('ai_server_url', 'http://127.0.0.1:8080')

    logging.info('=' * 60)
    logging.info('FXJEFE Pipeline starting')
    logging.info(f'Project   : {PROJECT_ROOT}')
    logging.info(f'Config    : {args.config}')
    logging.info(f'AI server : {ai_url}')
    logging.info(f'MQL5 path : {MQL5_PATH}')
    logging.info('=' * 60)

    # ── AI server ─────────────────────────────────────────────────────────────
    if not args.skip_server:
        if check_server_health(ai_url):
            logging.info('AI server already running.')
        else:
            logging.info('AI server not responding — starting it.')
            if start_ai_server(cfg):
                if check_server_health(ai_url):
                    logging.info('AI server started OK.')
                else:
                    logging.warning('AI server started but health check failed — continuing.')
            else:
                logging.error('Could not start AI server. Aborting.')
                sys.exit(1)

    # ── Build step list ───────────────────────────────────────────────────────
    steps = list(PIPELINE_STEPS)
    if args.skip_mt5:
        steps = [(s, d, r) for s, d, r in steps if s != 'mt5_generate_features_wine.py']
        logging.info('Skipping MT5 data generation (--skip-mt5).')

    total = len(steps)

    # ── Run each step ─────────────────────────────────────────────────────────
    for idx, (script, description, required) in enumerate(steps, start=1):
        logging.info(f'[{idx}/{total}] {description}')

        success = False
        for attempt in range(1, args.retry + 1):
            if run_step(script):
                success = True
                break
            if attempt < args.retry:
                logging.warning(f'  Retrying {script} ({attempt}/{args.retry}) in 3 s...')
                time.sleep(3)

        if success:
            logging.info(f'  OK — {script}')
        elif required:
            logging.error(f'Pipeline ABORTED at step {idx}/{total}: {script} '
                          f'failed after {args.retry} attempt(s).')
            sys.exit(1)
        else:
            logging.warning(f'  SKIPPED (optional) — {script} failed but pipeline continues.')

    logging.info('=' * 60)
    logging.info('Pipeline completed — AI server ready for EA signals.')
    logging.info('=' * 60)


if __name__ == '__main__':
    main()
