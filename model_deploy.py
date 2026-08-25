"""
Deploy trained models (.onnx, .pkl, .json) to all configured MT5 terminals.
Copies from models/ to each terminal's MQL5/Files/ folder.
"""
import os
import json
import shutil
from datetime import datetime

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

MODELS_DIR = config['models_path']
EXTENSIONS = ['.onnx', '.json']  # Only deploy inference-ready files
# Add '.pkl' to EXTENSIONS if your server.py loads pkl files

def deploy():
    terminals = config.get('mt5_terminals', [])
    if not terminals:
        # Fallback to single path
        terminals = [{'name': 'Default', 'files_path': config['mt5_files_path']}]

    model_files = [f for f in os.listdir(MODELS_DIR)
                   if any(f.endswith(ext) for ext in EXTENSIONS)]

    if not model_files:
        print("No model files found to deploy.")
        print(f"  Looked in: {MODELS_DIR}")
        print(f"  Extensions: {EXTENSIONS}")
        return

    print(f"Found {len(model_files)} model files to deploy:")
    for f in model_files:
        size = os.path.getsize(os.path.join(MODELS_DIR, f))
        print(f"  {f} ({size/1024:.0f} KB)")

    print()
    for terminal in terminals:
        name = terminal['name']
        dest = terminal['files_path']

        if not os.path.exists(dest):
            os.makedirs(dest, exist_ok=True)
            print(f"  Created: {dest}")

        print(f"Deploying to [{name}]: {dest}")
        for f in model_files:
            src = os.path.join(MODELS_DIR, f)
            dst = os.path.join(dest, f)
            shutil.copy2(src, dst)
            print(f"  Copied {f}")

        print(f"  Done ({len(model_files)} files)\n")

    # Also copy to Common files
    common = config.get('mt5_common_path', '')
    if common and os.path.exists(os.path.dirname(common)):
        os.makedirs(common, exist_ok=True)
        print(f"Deploying to [Common]: {common}")
        for f in model_files:
            shutil.copy2(os.path.join(MODELS_DIR, f), os.path.join(common, f))
            print(f"  Copied {f}")
        print()

    print(f"Deployment complete at {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == '__main__':
    deploy()
