"""
pipelinerun.py
Full FXJEFE trading pipeline — all steps in correct order.

PIPELINE ORDER
══════════════
Step 1  mt5_data_sync.py        – copy FXJEFE_Features.csv from MT5 terminal
Step 2  validate_data.py        – validate raw CSV (columns, row count, NaN report)
Step 3  generate_labels.py      – clean data, compute future price, add buy/hold/sell labels
                                   → FXJEFE_Features_fixed.csv
                                   → FXJEFE_Features_with_labels.csv
                                   → training_data.csv
Step 4  feature_engineering.py  – XGBoost + LightGBM stacking model on labeled data
                                   → models/stacking_model.pkl
                                   → data/processed_features.csv
Step 5  train_models.py         – train main RandomForest on training_data.csv
                                   → models/my_model.pkl   ← used by ai_server.py
Step 6  check_model_features.py – verify model feature count matches config['features']
Step 7  signal_processor.py     – SMA crossover signals (supplementary reference)
                                   → data/signals_output.csv

The AI server (python pipeline/ai_server.py) is checked / auto-started first.
The EA calls the AI server to receive live buy/hold/sell signals.

Usage:
  python pipelinerun.py
  python pipelinerun.py --retry 5 --verbose
  python pipelinerun.py --skip-server       (skip server health check)
"""

import json
import os
import subprocess
import logging
import time
import sys
import argparse

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ── config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH  = os.path.join(PROJECT_ROOT, 'config.json')


def load_config(path: str) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[FATAL] Cannot load config: {e}")
        sys.exit(1)


config = load_config(CONFIG_PATH)

os.makedirs(config['log_path'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config['log_path'], 'pipeline.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# ── ordered pipeline ──────────────────────────────────────────────────────────
#
# ALL steps run in sequence.  Each script must exit 0 for the pipeline to
# advance.  Failed steps are retried up to --retry times before aborting.
#
PIPELINE_STEPS = [
    # (script filename,          description)
    ('mt5_data_sync.py',         'Step 1 — Sync FXJEFE_Features.csv from MT5'),
    ('validate_data.py',         'Step 2 — Validate raw feature data'),
    ('generate_labels.py',       'Step 3 — Compute future price & generate labels'),
    ('feature_engineering.py',   'Step 4 — XGBoost/LightGBM stacking model'),
    ('train_models.py',          'Step 5 — Train main RandomForest (my_model.pkl)'),
    ('check_model_features.py',  'Step 6 — Verify model feature count'),
    ('signal_processor.py',      'Step 7 — Generate SMA crossover signals'),
]

# ── helpers ───────────────────────────────────────────────────────────────────

def check_server_health(url: str) -> bool:
    if not REQUESTS_AVAILABLE:
        logging.warning("'requests' not installed — cannot check server health.")
        return False
    try:
        r = requests.get(f"{url}/health", timeout=5)
        return r.status_code == 200 and r.json().get('status') == 'running'
    except Exception:
        return False


def start_ai_server(cfg: dict) -> bool:
    # ai_server.py lives in the 'python pipeline' sub-folder
    candidates = [
        os.path.join(cfg['project_root'], 'python pipeline', 'ai_server.py'),
        os.path.join(cfg.get('scripts_path', PROJECT_ROOT), 'ai_server.py'),
    ]
    server_path = next((p for p in candidates if os.path.exists(p)), None)
    if server_path is None:
        logging.error("ai_server.py not found. Check 'python pipeline/' subfolder.")
        return False
    try:
        flags = subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
        subprocess.Popen([sys.executable, server_path], creationflags=flags)
        logging.info(f"Starting AI server: {server_path}")
        time.sleep(7)   # allow Flask to bind
        return True
    except Exception as e:
        logging.error(f"Failed to start AI server: {e}")
        return False


def run_step(script: str, cfg: dict) -> bool:
    script_path = os.path.join(cfg.get('scripts_path', PROJECT_ROOT), script)
    if not os.path.exists(script_path):
        logging.error(f"Script not found: {script_path}")
        return False
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True, capture_output=True, text=True
        )
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                logging.info(f"  [{script}] {line}")
        return True
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or '').strip()
        logging.error(f"  [{script}] FAILED (exit {e.returncode})")
        if stderr:
            for line in stderr.splitlines()[-10:]:   # last 10 lines of stderr
                logging.error(f"  [{script}]   {line}")
        return False


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='FXJEFE Full Trading Pipeline')
    parser.add_argument('--config',       default=CONFIG_PATH,
                        help='Path to config.json')
    parser.add_argument('--retry',        type=int, default=3,
                        help='Max retries per failed step (default: 3)')
    parser.add_argument('--verbose',      action='store_true',
                        help='Enable DEBUG-level logging')
    parser.add_argument('--skip-server',  action='store_true',
                        help='Skip AI server health check / auto-start')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg     = load_config(args.config)
    ai_url  = cfg.get('ai_server_url', 'http://127.0.0.1:8080')

    logging.info("=" * 60)
    logging.info("FXJEFE Pipeline starting")
    logging.info(f"Project root : {PROJECT_ROOT}")
    logging.info(f"Config       : {args.config}")
    logging.info(f"AI server    : {ai_url}")
    logging.info("=" * 60)

    # ── ensure AI server is running ───────────────────────────────────────────
    if not args.skip_server:
        if check_server_health(ai_url):
            logging.info("AI server already running.")
        else:
            logging.info("AI server not responding — attempting to start it.")
            if start_ai_server(cfg):
                if check_server_health(ai_url):
                    logging.info("AI server started successfully.")
                else:
                    logging.warning("AI server started but health check failed. "
                                    "Pipeline will continue — server may still be warming up.")
            else:
                logging.error("Could not start AI server. Aborting pipeline.")
                sys.exit(1)

    # ── run each step ─────────────────────────────────────────────────────────
    total  = len(PIPELINE_STEPS)
    for idx, (script, description) in enumerate(PIPELINE_STEPS, start=1):
        logging.info(f"[{idx}/{total}] {description}")

        success  = False
        for attempt in range(1, args.retry + 1):
            if run_step(script, cfg):
                success = True
                break
            if attempt < args.retry:
                logging.warning(f"  Retrying {script} (attempt {attempt}/{args.retry}) in 3 s…")
                time.sleep(3)

        if not success:
            logging.error(f"Pipeline ABORTED at step {idx}/{total}: {script} "
                          f"failed after {args.retry} attempt(s).")
            sys.exit(1)

        logging.info(f"  ✓ {script} complete")

    logging.info("=" * 60)
    logging.info("Pipeline completed successfully — AI server ready for EA.")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()
