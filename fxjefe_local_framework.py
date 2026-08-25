#!/usr/bin/env python3
"""
FXJEFE Local Framework - Daily Driver
Handles model testing, SHAP, server start, pipeline run, etc.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Project root = folder containing this script
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from path_resolver import get_paths

paths = get_paths()


def run_command(cmd_list, cwd=None):
    try:
        result = subprocess.run(
            cmd_list,
            check=True,
            cwd=cwd or project_root,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        if result.stdout.strip():
            print(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd_list)}")
        print(e.stderr.strip())
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="FXJEFE Local Framework")
    sub = parser.add_subparsers(dest='command', required=True)

    # Start AI server
    sub.add_parser('start-server', help='Start AI prediction server')

    # Test models
    p_test = sub.add_parser('test-models', help='Run model tester')
    p_test.add_argument('--rows', type=int, default=500)

    # SHAP analysis
    p_shap = sub.add_parser('shap', help='Run SHAP explainer')
    p_shap.add_argument('--rows', type=int, default=300)

    # Run full pipeline
    sub.add_parser('pipeline', help='Run full pipeline')

    args = parser.parse_args()

    if args.command == 'start-server':
        server_script = paths.scripts_path / 'fxjefe_xgboost_server.py'
        print(f"Starting AI server: {server_script}")
        subprocess.Popen(['python', str(server_script)], creationflags=subprocess.CREATE_NEW_CONSOLE)

    elif args.command == 'test-models':
        tester = project_root / 'test_local_trading_models.py'
        print(f"Running model tester with {args.rows} rows...")
        run_command(['python', str(tester), '--rows', str(args.rows)])

    elif args.command == 'shap':
        shap_script = project_root / 'test_models_with_shap.py'
        print(f"Running SHAP analysis with {args.rows} rows...")
        run_command(['python', str(shap_script), '--rows', str(args.rows)])

    elif args.command == 'pipeline':
        runner = project_root / 'run_fxjeje_pipeline.py'
        print("Running full pipeline...")
        run_command(['python', str(runner)])


if __name__ == '__main__':
    main()
