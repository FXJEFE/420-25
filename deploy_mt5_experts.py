#!/usr/bin/env python3
"""
deploy_mt5_experts.py
Deployment script for organizing and deploying MQ5 scripts and compiled EXEs to MT5 via Wine.
Handles macOS → Wine → Windows path mapping for MT5 expert advisors.

Organizes:
- MQ5 source files (advisors, indicators, libraries)
- Compiled EX5 files
- Models and include files
- Creates deployment manifest for tracking versions
"""

import os
import shutil
import json
import logging
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

# Configuration
PROJECT_ROOT = "/Users/localhugo/Desktop/FXJEFE_Project"
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
EXPERTS_ROOT = os.path.join(DATA_PATH, "mt5_experts")

# Subdirectories
ADVISORS_DIR = os.path.join(EXPERTS_ROOT, "advisors")
INCLUDES_DIR = os.path.join(EXPERTS_ROOT, "includes")
INDICATORS_DIR = os.path.join(EXPERTS_ROOT, "indicators")
MODELS_DIR = os.path.join(EXPERTS_ROOT, "models")

# Wine MT5 paths (Wine maps Z: to macOS root)
# Example: C:\Users\LarryLocal\... becomes Z:\Users\localhugo\...
WINE_TERMINAL_PRIMARY = "Z:\\Users\\localhugo\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5"
WINE_TERMINAL_SECONDARY = "Z:\\Users\\localhugo\\AppData\\Roaming\\MetaQuotes\\Terminal\\AE2CC2E013FDE1E3CDF010AA51C60400\\MQL5"

# Logging
LOG_FILE = os.path.join(PROJECT_ROOT, "Logs", "deploy_mt5_experts.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_file_hash(filepath: str) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    return sha256.hexdigest()


def categorize_mq5_files() -> Dict[str, List[str]]:
    """
    Categorize MQ5 files based on naming conventions and content.

    Returns:
        dict: Categorized files {'advisors': [], 'includes': [], 'indicators': []}
    """
    categories = {
        'advisors': [],
        'includes': [],
        'indicators': []
    }

    advisor_keywords = ['Predict', 'GenerateFeatures', 'FXJEFE_ALGO', 'QuantumAlgo', 'SaturnMatrix', 'EA']
    include_keywords = ['.mqh', 'Account', 'Position', 'Trading', 'Data', 'Error', 'HTTPClient', 'Logger', 'Risk', 'Manager', 'CSV', 'Symbol', 'Python', 'JSON', 'Trade', 'Trend', 'Array', 'Math']
    indicator_keywords = ['Chaikin', 'VolumeOsc', 'DPO', 'Hull', 'SuperTrend', 'ROC', 'Stochastic', 'VWAP', 'Indicator']

    for filename in os.listdir(DATA_PATH):
        if not (filename.endswith('.mq5') or filename.endswith('.mqh')):
            continue

        filepath = os.path.join(DATA_PATH, filename)
        if not os.path.isfile(filepath):
            continue

        # Skip copies and backups
        if any(x in filename for x in [' - Copy', '(', ')', '.zero.bak']):
            continue

        # Categorize based on keywords
        is_include = filename.endswith('.mqh') or any(kw in filename for kw in include_keywords)
        is_advisor = any(kw in filename for kw in advisor_keywords)
        is_indicator = any(kw in filename for kw in indicator_keywords)

        if is_include:
            categories['includes'].append(filename)
        elif is_advisor:
            categories['advisors'].append(filename)
        elif is_indicator:
            categories['indicators'].append(filename)
        else:
            # Default to advisors if unclear
            categories['advisors'].append(filename)

    return categories


def find_compiled_ex5_files() -> Dict[str, str]:
    """
    Find compiled EX5 files and match them to MQ5 sources.

    Returns:
        dict: Mapping of ex5 files with their metadata
    """
    ex5_files = {}

    for filename in os.listdir(DATA_PATH):
        if filename.endswith('.ex5'):
            # Skip copies and backups
            if any(x in filename for x in [' - Copy', ' (', '.zero.bak']):
                continue

            filepath = os.path.join(DATA_PATH, filename)
            if os.path.isfile(filepath):
                ex5_files[filename] = {
                    'path': filepath,
                    'size': os.path.getsize(filepath),
                    'mtime': os.path.getmtime(filepath),
                    'hash': get_file_hash(filepath)
                }

    return ex5_files


def find_model_files() -> Dict[str, str]:
    """
    Find all model files (pkl, json, cbm, h5, etc.)

    Returns:
        dict: Model files grouped by symbol/type
    """
    model_files = {}
    model_extensions = ['.pkl', '.json', '.cbm', '.h5', '.onnx', '.xgb']

    for filename in os.listdir(DATA_PATH):
        if any(filename.endswith(ext) for ext in model_extensions):
            # Skip duplicates and backups
            if any(x in filename for x in ['(1)', '(2)', '(3)', ' - Copy', '.corrupt', '.bak']):
                continue

            filepath = os.path.join(DATA_PATH, filename)
            if os.path.isfile(filepath):
                model_files[filename] = {
                    'path': filepath,
                    'size': os.path.getsize(filepath),
                    'type': os.path.splitext(filename)[1]
                }

    return model_files


def organize_source_files():
    """Organize MQ5 source files into appropriate folders."""
    logger.info("Organizing MQ5 source files...")

    categories = categorize_mq5_files()

    for category, files in categories.items():
        if category == 'advisors':
            dest_dir = ADVISORS_DIR
        elif category == 'includes':
            dest_dir = INCLUDES_DIR
        else:
            dest_dir = INDICATORS_DIR

        os.makedirs(dest_dir, exist_ok=True)

        for filename in files:
            src = os.path.join(DATA_PATH, filename)
            dst = os.path.join(dest_dir, filename)

            # Only copy if not already there
            if not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                    logger.info(f"  Organized: {filename} → {category}/")
                except Exception as e:
                    logger.error(f"  Failed to copy {filename}: {e}")
            else:
                logger.debug(f"  Skipped (already exists): {filename}")


def organize_compiled_files():
    """Organize compiled EX5 files into advisors folder."""
    logger.info("Organizing compiled EX5 files...")

    ex5_files = find_compiled_ex5_files()
    os.makedirs(ADVISORS_DIR, exist_ok=True)

    for filename, metadata in ex5_files.items():
        src = metadata['path']
        dst = os.path.join(ADVISORS_DIR, filename)

        if not os.path.exists(dst):
            try:
                shutil.copy2(src, dst)
                logger.info(f"  Compiled: {filename} (size: {metadata['size']} bytes)")
            except Exception as e:
                logger.error(f"  Failed to copy {filename}: {e}")
        else:
            logger.debug(f"  Skipped (already exists): {filename}")


def organize_models():
    """Organize model files."""
    logger.info("Organizing model files...")

    models = find_model_files()
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_count = {}
    for filename, metadata in models.items():
        src = metadata['path']
        dst = os.path.join(MODELS_DIR, filename)

        if not os.path.exists(dst):
            try:
                shutil.copy2(src, dst)
                model_type = metadata['type']
                model_count[model_type] = model_count.get(model_type, 0) + 1
                logger.info(f"  Organized: {filename} ({metadata['type']})")
            except Exception as e:
                logger.error(f"  Failed to copy {filename}: {e}")

    logger.info(f"  Total models organized: {sum(model_count.values())} files")


def create_deployment_manifest():
    """Create a deployment manifest with versions and checksums."""
    logger.info("Creating deployment manifest...")

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "platform": "macOS via Wine on M1 (Tahoe 26)",
        "advisors": {},
        "includes": {},
        "indicators": {},
        "models": {}
    }

    # Add advisors
    for filename in os.listdir(ADVISORS_DIR):
        if filename.endswith(('.mq5', '.ex5')):
            filepath = os.path.join(ADVISORS_DIR, filename)
            manifest["advisors"][filename] = {
                "size": os.path.getsize(filepath),
                "hash": get_file_hash(filepath),
                "mtime": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
            }

    # Add includes
    for filename in os.listdir(INCLUDES_DIR):
        if filename.endswith('.mqh'):
            filepath = os.path.join(INCLUDES_DIR, filename)
            manifest["includes"][filename] = {
                "size": os.path.getsize(filepath),
                "hash": get_file_hash(filepath),
                "mtime": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
            }

    # Add indicators
    for filename in os.listdir(INDICATORS_DIR):
        if filename.endswith(('.mq5', '.ex5')):
            filepath = os.path.join(INDICATORS_DIR, filename)
            manifest["indicators"][filename] = {
                "size": os.path.getsize(filepath),
                "hash": get_file_hash(filepath),
                "mtime": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
            }

    # Add models
    for filename in os.listdir(MODELS_DIR):
        filepath = os.path.join(MODELS_DIR, filename)
        manifest["models"][filename] = {
            "size": os.path.getsize(filepath),
            "type": os.path.splitext(filename)[1],
            "hash": get_file_hash(filepath),
            "mtime": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
        }

    manifest_path = os.path.join(EXPERTS_ROOT, "DEPLOYMENT_MANIFEST.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"  Manifest saved: {manifest_path}")
    return manifest


def create_wine_sync_script():
    """Create a script to sync files to MT5 via Wine."""
    script_path = os.path.join(PROJECT_ROOT, "sync_to_wine_mt5.sh")

    script_content = f'''#!/bin/bash
# Wine MT5 Synchronization Script
# Syncs organized expert advisors to MT5 terminal via Wine

set -e

EXPERTS_SRC="{EXPERTS_ROOT}"
DATA_SRC="{DATA_PATH}"

# Wine terminal paths (Z: maps to macOS root)
WINE_PRIMARY_EXPERTS="${{WINEPREFIX:-$HOME/.wine}}/drive_c/users/LarryLocal/AppData/Roaming/MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Experts"
WINE_SECONDARY_EXPERTS="${{WINEPREFIX:-$HOME/.wine}}/drive_c/users/LarryLocal/AppData/Roaming/MetaQuotes/Terminal/AE2CC2E013FDE1E3CDF010AA51C60400/MQL5/Experts"

WINE_PRIMARY_INCLUDE="${{WINEPREFIX:-$HOME/.wine}}/drive_c/users/LarryLocal/AppData/Roaming/MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Include"
WINE_SECONDARY_INCLUDE="${{WINEPREFIX:-$HOME/.wine}}/drive_c/users/LarryLocal/AppData/Roaming/MetaQuotes/Terminal/AE2CC2E013FDE1E3CDF010AA51C60400/MQL5/Include"

WINE_PRIMARY_INDICATORS="${{WINEPREFIX:-$HOME/.wine}}/drive_c/users/LarryLocal/AppData/Roaming/MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Indicators"
WINE_SECONDARY_INDICATORS="${{WINEPREFIX:-$HOME/.wine}}/drive_c/users/LarryLocal/AppData/Roaming/MetaQuotes/Terminal/AE2CC2E013FDE1E3CDF010AA51C60400/MQL5/Indicators"

echo "🔄 Syncing MT5 Expert Advisors to Wine MT5 Terminal..."
echo "Platform: macOS M1 (Tahoe 26) → Wine → MT5"
echo ""

# Sync Advisors
echo "📁 Syncing Expert Advisors..."
if [ -d "$EXPERTS_SRC/advisors" ]; then
    mkdir -p "$WINE_PRIMARY_EXPERTS"
    cp -v "$EXPERTS_SRC"/advisors/*.ex5 "$WINE_PRIMARY_EXPERTS/" 2>/dev/null || true
    cp -v "$EXPERTS_SRC"/advisors/*.mq5 "$WINE_PRIMARY_EXPERTS/" 2>/dev/null || true
    
    if [ -d "$WINE_SECONDARY_EXPERTS" ]; then
        mkdir -p "$WINE_SECONDARY_EXPERTS"
        cp -v "$EXPERTS_SRC"/advisors/*.ex5 "$WINE_SECONDARY_EXPERTS/" 2>/dev/null || true
        cp -v "$EXPERTS_SRC"/advisors/*.mq5 "$WINE_SECONDARY_EXPERTS/" 2>/dev/null || true
    fi
    echo "✅ Expert Advisors synced"
fi

# Sync Include files
echo "📁 Syncing Include files..."
if [ -d "$EXPERTS_SRC/includes" ]; then
    mkdir -p "$WINE_PRIMARY_INCLUDE"
    cp -v "$EXPERTS_SRC"/includes/*.mqh "$WINE_PRIMARY_INCLUDE/" 2>/dev/null || true
    
    if [ -d "$WINE_SECONDARY_INCLUDE" ]; then
        mkdir -p "$WINE_SECONDARY_INCLUDE"
        cp -v "$EXPERTS_SRC"/includes/*.mqh "$WINE_SECONDARY_INCLUDE/" 2>/dev/null || true
    fi
    echo "✅ Include files synced"
fi

# Sync Indicators
echo "📁 Syncing Indicators..."
if [ -d "$EXPERTS_SRC/indicators" ]; then
    mkdir -p "$WINE_PRIMARY_INDICATORS"
    cp -v "$EXPERTS_SRC"/indicators/*.ex5 "$WINE_PRIMARY_INDICATORS/" 2>/dev/null || true
    cp -v "$EXPERTS_SRC"/indicators/*.mq5 "$WINE_PRIMARY_INDICATORS/" 2>/dev/null || true
    
    if [ -d "$WINE_SECONDARY_INDICATORS" ]; then
        mkdir -p "$WINE_SECONDARY_INDICATORS"
        cp -v "$EXPERTS_SRC"/indicators/*.ex5 "$WINE_SECONDARY_INDICATORS/" 2>/dev/null || true
        cp -v "$EXPERTS_SRC"/indicators/*.mq5 "$WINE_SECONDARY_INDICATORS/" 2>/dev/null || true
    fi
    echo "✅ Indicators synced"
fi

echo ""
echo "🎉 Sync complete! Expert advisors are ready in MT5."
echo ""
echo "📋 Next steps:"
echo "  1. Reload MT5 (Terminal → Restart)"
echo "  2. Check Expert Advisors in Navigator"
echo "  3. Compile any new MQ5 files in MetaEditor"
echo "  4. Attach EAs to charts to test"
'''

    with open(script_path, 'w') as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)
    logger.info(f"  Wine sync script created: {script_path}")


def create_readme():
    """Create a README with deployment instructions."""
    readme_path = os.path.join(EXPERTS_ROOT, "README.md")

    readme_content = '''# MT5 Expert Advisors Deployment Guide

## Platform
- **OS**: macOS M1 (Tahoe 26)
- **MT5 Runtime**: Wine (via Rosetta 2 translation)
- **Organization Date**: ''' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '''

## Folder Structure

```
mt5_experts/
├── advisors/          # Expert Advisors (.mq5, .ex5)
├── includes/          # Include files (.mqh) - shared libraries
├── indicators/        # Indicator scripts (.mq5, .ex5)
├── models/            # ML models (*.pkl, *.json, *.onnx, etc.)
├── DEPLOYMENT_MANIFEST.json
└── README.md
```

## Organization Categories

### 📊 Advisors (Expert Advisors)
- `Predict*.ex5` / `.mq5` - Trading signal generation advisors
- `GenerateFeatures*.ex5` / `.mq5` - Feature generation advisors
- `FXJEFE_ALGO_AI*.mq5` - Main AI trading algorithm
- `QuantumAlgo*.mq5` - Quantum trading algorithm
- `SaturnMatrix*.mq5` - Saturn matrix advisors

### 📚 Includes (Library Files)
- `.mqh` files - MQL5 header libraries
- Shared classes: AccountInfo, Position Manager, CSV Manager, Logger, etc.
- Cross-advisor utility functions

### 📈 Indicators (Analysis Tools)
- Individual indicator scripts
- Can be used standalone or within EAs
- Examples: Chaikin, VolumeOsc, SuperTrend, DPO, etc.

### 🤖 Models (Machine Learning)
- XGBoost models (`.json`, `.onnx`)
- LightGBM models (`.pkl`, `.onnx`)
- CatBoost models (`.cbm`)
- LSTM models (`.h5`)
- Feature configs (`.json`)
- Scalers and preprocessors (`.pkl`)

## Deployment to MT5 via Wine

### Method 1: Automated Sync Script

```bash
cd /Users/localhugo/Desktop/FXJEFE_Project
./sync_to_wine_mt5.sh
```

This script:
1. Copies all EX5 (compiled) files to Wine MT5 Experts folder
2. Copies all MQH (include) files to Wine MT5 Include folder
3. Copies all indicator files to Wine MT5 Indicators folder
4. Handles both primary and secondary MT5 terminals

### Method 2: Manual Deployment (Python)

```python
from deploy_mt5_experts import *

# Run deployment
organize_source_files()
organize_compiled_files()
organize_models()
manifest = create_deployment_manifest()
print(manifest)
```

### Method 3: Wine Direct Copy

Wine mounts macOS filesystem at `Z:`. From Wine terminal (cmd.exe):

```cmd
REM Copy advisors
copy Z:\\Users\\localhugo\\Desktop\\FXJEFE_Project\\data\\mt5_experts\\advisors\\*.ex5 ^
      "C:\\Users\\LarryLocal\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Experts\\"

REM Copy includes
copy Z:\\Users\\localhugo\\Desktop\\FXJEFE_Project\\data\\mt5_experts\\includes\\*.mqh ^
      "C:\\Users\\LarryLocal\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Include\\"
```

## Manifest File

**DEPLOYMENT_MANIFEST.json** contains:
- SHA256 checksums of all deployed files
- File sizes and modification times
- File organization metadata
- Deployment platform information

Use this to verify integrity and track versions.

## Workflow

1. **Edit MQ5**: Edit source files in `advisors/` or `indicators/`
2. **Compile in MetaEditor**: Compile `.mq5` → `.ex5` in Wine MT5
3. **Move EX5 back**: Copy compiled `.ex5` to `advisors/` folder
4. **Sync to MT5**: Run `./sync_to_wine_mt5.sh`
5. **Reload MT5**: Restart Terminal or press F5 in Navigator
6. **Attach EAs**: Right-click chart → Expert Advisors → Select EA

## Troubleshooting

### "File not found in MT5"
- Check that `.ex5` file is in `mt5_experts/advisors/`
- Verify Wine path exists: `~/.wine/drive_c/users/LarryLocal/...`
- Run sync script again: `./sync_to_wine_mt5.sh`

### "EA won't load in MT5"
- Ensure file is `.ex5` (compiled), not `.mq5` (source)
- Check file permissions: `chmod 644 *.ex5`
- Verify no corrupted files: Check DEPLOYMENT_MANIFEST.json hash

### "Include files missing"
- Verify `.mqh` files are in `mt5_experts/includes/`
- Make sure path in MQ5 code matches: `#include <FileName.mqh>`
- Run sync script to ensure Wine copy is updated

### Wine MT5 path not found
- Find your Wine prefix: `echo $WINEPREFIX` or `~/.wine/`
- Verify terminal ID exists: `ls ~/.wine/drive_c/users/LarryLocal/AppData/Roaming/MetaQuotes/Terminal/`
- Update script with correct terminal ID if needed

## File Statistics

Check the latest deployment:

```bash
cat /Users/localhugo/Desktop/FXJEFE_Project/data/mt5_experts/DEPLOYMENT_MANIFEST.json
```

## Next Steps

1. ✅ Organize MQ5/EX5/Model files (DONE)
2. 📋 Compile remaining MQ5 → EX5 in MetaEditor
3. 🔄 Run sync script to push to Wine MT5
4. ✔️ Test EAs in MT5 with live data
5. 📊 Monitor strategy performance in Account History

---
Generated by `deploy_mt5_experts.py` on macOS M1 Tahoe 26
'''

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    logger.info(f"  README created: {readme_path}")


def print_summary():
    """Print deployment summary."""
    print("\n" + "="*70)
    print("📦 MT5 EXPERT ADVISORS DEPLOYMENT SUMMARY")
    print("="*70)
    print(f"Platform: macOS M1 (Tahoe 26) → Wine → MT5")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    advisors = len([f for f in os.listdir(ADVISORS_DIR) if f.endswith(('.mq5', '.ex5'))])
    includes = len([f for f in os.listdir(INCLUDES_DIR) if f.endswith('.mqh')])
    indicators = len([f for f in os.listdir(INDICATORS_DIR) if f.endswith(('.mq5', '.ex5'))])
    models = len([f for f in os.listdir(MODELS_DIR)])

    print(f"\n✅ Organization Results:")
    print(f"   📊 Expert Advisors: {advisors} files")
    print(f"   📚 Include Libraries: {includes} files")
    print(f"   📈 Indicators: {indicators} files")
    print(f"   🤖 Models: {models} files")

    print(f"\n📁 Organized in: {EXPERTS_ROOT}/")
    print(f"   ├── advisors/")
    print(f"   ├── includes/")
    print(f"   ├── indicators/")
    print(f"   └── models/")

    print(f"\n📋 Documentation:")
    print(f"   README: {os.path.join(EXPERTS_ROOT, 'README.md')}")
    print(f"   Manifest: {os.path.join(EXPERTS_ROOT, 'DEPLOYMENT_MANIFEST.json')}")
    print(f"   Logs: {LOG_FILE}")

    print(f"\n🔄 Sync Script:")
    print(f"   {os.path.join(PROJECT_ROOT, 'sync_to_wine_mt5.sh')}")
    print(f"   Usage: ./sync_to_wine_mt5.sh")

    print("\n" + "="*70)
    print("🎯 Next Steps:")
    print("="*70)
    print("1. Run: ./sync_to_wine_mt5.sh")
    print("2. Restart MT5 terminal")
    print("3. Check Expert Advisors in Navigator (Ctrl+N)")
    print("4. Attach EA to chart to test")
    print("="*70 + "\n")


def main():
    """Main deployment function."""
    logger.info("="*70)
    logger.info("Starting MT5 Expert Advisors Deployment")
    logger.info(f"Platform: macOS M1 Tahoe 26 via Wine")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("="*70)

    try:
        # Create directories
        os.makedirs(ADVISORS_DIR, exist_ok=True)
        os.makedirs(INCLUDES_DIR, exist_ok=True)
        os.makedirs(INDICATORS_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Organize files
        organize_source_files()
        organize_compiled_files()
        organize_models()

        # Create deployment artifacts
        manifest = create_deployment_manifest()
        create_wine_sync_script()
        create_readme()

        # Print summary
        print_summary()

        logger.info("✅ Deployment complete!")
        logger.info("="*70)

        return 0

    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
