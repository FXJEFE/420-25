#!/bin/bash
# Wrapper script to run FXJEFE OG333 pipeline with automatic venv activation
# Usage: ./run_pipelineOG333_venv.sh [args...]

set -e

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PATH="${PROJECT_ROOT}/.venv"

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at: $VENV_PATH"
    echo "Please create it first: python3 -m venv $VENV_PATH"
    exit 1
fi

# Activate venv
source "$VENV_PATH/bin/activate"

# Run the OG333 pipeline with any passed arguments
echo "📋 Running FXJEFE OG333 Pipeline from: $PROJECT_ROOT"
echo "🐍 Using Python: $(which python)"
echo "📦 Using venv: $VENV_PATH"
echo ""

python "$PROJECT_ROOT/run_pipelineOG333.py" "$@"

# Deactivate venv (optional, shell will exit anyway)
deactivate

exit $?
