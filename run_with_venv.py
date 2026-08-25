#!/usr/bin/env python3
"""
Pipeline runner with automatic venv activation.

This script activates the project virtual environment and runs the FXJEFE pipeline.
Can be used standalone or imported as a module.

Usage:
    python3 run_with_venv.py                    # Run main pipeline
    python3 run_with_venv.py og333              # Run OG333 pipeline
    python3 run_with_venv.py production         # Run production pipeline
"""

import os
import sys
import subprocess
import venv
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
VENV_PATH = PROJECT_ROOT / ".venv"
VENV_BIN = VENV_PATH / "bin"
VENV_PYTHON = VENV_BIN / "python"


def activate_venv():
    """Activate the project virtual environment."""
    if not VENV_PATH.exists():
        print(f"❌ Virtual environment not found at: {VENV_PATH}")
        print("Please create it first: python3 -m venv .venv")
        sys.exit(1)

    # Set environment variables to use venv
    os.environ['VIRTUAL_ENV'] = str(VENV_PATH)
    os.environ['PATH'] = f"{VENV_BIN}:{os.environ['PATH']}"

    # Import site-packages from venv
    if sys.prefix != str(VENV_PATH):
        sys.prefix = str(VENV_PATH)
        sys.executable = str(VENV_PYTHON)


def run_pipeline(pipeline_type="default"):
    """
    Run the specified pipeline.

    Args:
        pipeline_type: Type of pipeline to run
                      - "default" or "main": run_pipeline.py
                      - "og333": run_pipelineOG333.py
                      - "production": pipelinerun_production.py
    """
    pipeline_scripts = {
        "default": "run_pipeline.py",
        "main": "run_pipeline.py",
        "og333": "run_pipelineOG333.py",
        "production": "pipelinerun_production.py",
    }

    script_name = pipeline_scripts.get(pipeline_type, "run_pipeline.py")
    script_path = PROJECT_ROOT / script_name

    if not script_path.exists():
        print(f"❌ Pipeline script not found: {script_path}")
        print(f"Available scripts:")
        for key, val in pipeline_scripts.items():
            full_path = PROJECT_ROOT / val
            exists = "✓" if full_path.exists() else "✗"
            print(f"  {exists} {key:12} → {val}")
        sys.exit(1)

    # Activate venv
    activate_venv()

    print("=" * 80)
    print("🔄 FXJEFE Pipeline Runner with Virtual Environment")
    print("=" * 80)
    print(f"📂 Project Root  : {PROJECT_ROOT}")
    print(f"🐍 Python        : {sys.executable}")
    print(f"📦 Venv          : {VENV_PATH}")
    print(f"📋 Pipeline Type : {pipeline_type}")
    print(f"📜 Script        : {script_name}")
    print("=" * 80)
    print()

    # Run the pipeline script
    try:
        result = subprocess.run(
            [str(VENV_PYTHON), str(script_path)] + sys.argv[2:],
            cwd=str(PROJECT_ROOT),
            check=False
        )
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        sys.exit(1)


def print_help():
    """Print help information."""
    help_text = """
FXJEFE Pipeline Runner with Virtual Environment Activation

Usage:
    python3 run_with_venv.py [pipeline_type] [args...]

Pipeline Types:
    default, main       Run main FXJEFE pipeline (run_pipeline.py)
    og333               Run OG333 version pipeline (run_pipelineOG333.py)
    production          Run production pipeline (pipelinerun_production.py)

Examples:
    python3 run_with_venv.py
    python3 run_with_venv.py og333
    python3 run_with_venv.py production

Notes:
    - Virtual environment must be at: .venv/
    - All ML dependencies must be installed in venv
    - This script automatically activates the venv before running the pipeline
"""
    print(help_text)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help", "help"]:
        print_help()
        if sys.argv[1:]:  # Only exit with error if help was explicitly requested
            sys.exit(0)
        sys.exit(0)

    pipeline_type = sys.argv[1]
    run_pipeline(pipeline_type)
